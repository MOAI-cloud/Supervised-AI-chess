from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from superchess.encoding import POLICY_SIZE
from superchess.model import ChessCNNTransformer, ModelConfig

PACKED_BOARD_BITS = 18 * 8 * 8
POLICY_SQUARE_ORDER = "python-chess"
POLICY_PHASES = (
    ("early", 0, 20),
    ("mid", 20, 60),
    ("endgame", 60, None),
)

def _resolve_device(device_name: str | None = None) -> torch.device:
    if device_name is not None:
        device = torch.device(device_name)
        if device.type == "cuda":
            _check_cuda_device(device)
        return device

    if not torch.cuda.is_available():
        return torch.device("cpu")

    device = torch.device("cuda")
    try:
        _check_cuda_device(device)
    except RuntimeError as error:
        warnings.warn(
            f"CUDA is visible but unusable ({error}); falling back to CPU. "
            "Install a PyTorch build that supports this GPU to train on CUDA.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


def _check_cuda_device(device: torch.device) -> None:
    try:
        probe = torch.empty(1, device=device)
        probe.zero_()
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(
            f"CUDA device {device} cannot run kernels. "
            "Install a PyTorch build compatible with the GPU, or pass --device cpu."
        ) from error


class NPZShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: Path) -> None:
        self.shards = sorted(data_dir.glob("shard-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
        self.lengths: list[int] = []
        for shard in self.shards:
            with np.load(shard) as data:
                self.lengths.append(int(data["policies"].shape[0]))
        self.cumulative = np.cumsum(self.lengths).tolist()
        self._cache_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        shard_index = bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - shard_start
        shard = self._load_shard(shard_index)
        board = np.unpackbits(shard["boards"][local_index], count=PACKED_BOARD_BITS).astype(np.float32, copy=False)
        board = board.reshape(18, 8, 8)
        return (
            torch.from_numpy(board),
            torch.tensor(int(shard["policies"][local_index]), dtype=torch.long),
            torch.tensor(float(shard["values"][local_index]), dtype=torch.float32),
            torch.tensor(int(shard["plies"][local_index]), dtype=torch.long),
        )

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == shard_index and self._cache is not None:
            return self._cache
        with np.load(self.shards[shard_index]) as data:
            self._cache = {name: data[name] for name in ("boards", "policies", "values")}
            self._cache["plies"] = (
                data["plies"] if "plies" in data.files else np.zeros(data["policies"].shape[0], dtype=np.uint16)
            )
        self._cache_index = shard_index
        return self._cache


class NPZShardBatchDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: Path, batch_size: int, *, shards: Sequence[Path] | None = None, shuffle: bool = True) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shards = sorted(data_dir.glob("shard-*.npz")) if shards is None else list(shards)
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths: list[int] = []
        for shard in self.shards:
            with np.load(shard) as data:
                self.lengths.append(int(data["policies"].shape[0]))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shard_indices = np.arange(worker_id, len(self.shards), num_workers)
        rng = np.random.default_rng()
        if self.shuffle:
            rng.shuffle(shard_indices)

        for shard_index in shard_indices:
            with np.load(self.shards[int(shard_index)]) as data:
                shard = _arrays_from_npz(data)
                order = np.arange(shard["policies"].shape[0])
                if self.shuffle:
                    order = rng.permutation(order)
                for start in range(0, order.shape[0], self.batch_size):
                    indices = order[start : start + self.batch_size]
                    yield _batch_from_shard(shard, indices)

    def __len__(self) -> int:
        return sum((length + self.batch_size - 1) // self.batch_size for length in self.lengths)


def _arrays_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    shard = {name: data[name] for name in ("boards", "policies", "values")}
    if "plies" in data.files:
        shard["plies"] = data["plies"]
    else:
        shard["plies"] = np.zeros(data["policies"].shape[0], dtype=np.uint16)
    return shard


def _batch_from_shard(
    shard: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = _unpack_board_batch(shard["boards"][indices])
    policies = torch.from_numpy(shard["policies"][indices].astype(np.int64, copy=False))
    values = torch.from_numpy(shard["values"][indices].astype(np.float32, copy=False))
    plies = torch.from_numpy(shard["plies"][indices].astype(np.int64, copy=False))
    return boards, policies, values, plies


def _unpack_board_batch(packed_boards: np.ndarray) -> torch.Tensor:
    boards = np.unpackbits(packed_boards, axis=1, count=PACKED_BOARD_BITS).astype(np.float32, copy=False)
    return torch.from_numpy(boards.reshape(-1, 18, 8, 8))


def _new_policy_accuracy_stats() -> dict[str, int]:
    stats = {"policy_samples": 0, "policy_top1_correct": 0, "policy_top5_correct": 0}
    for phase, _, _ in POLICY_PHASES:
        stats[f"policy_samples_{phase}"] = 0
        stats[f"policy_top1_correct_{phase}"] = 0
        stats[f"policy_top5_correct_{phase}"] = 0
    return stats


def _update_policy_accuracy_stats(
    stats: dict[str, int],
    policy_logits: torch.Tensor,
    policies: torch.Tensor,
    plies: torch.Tensor,
) -> None:
    with torch.no_grad():
        topk = policy_logits.detach().topk(k=min(5, policy_logits.shape[1]), dim=1).indices
        top1_correct = topk[:, 0].eq(policies)
        top5_correct = topk.eq(policies.unsqueeze(1)).any(dim=1)

        _add_policy_accuracy_counts(stats, "", top1_correct, top5_correct)
        for phase, start, stop in POLICY_PHASES:
            mask = plies >= start
            if stop is not None:
                mask = mask & (plies < stop)
            _add_policy_accuracy_counts(stats, f"_{phase}", top1_correct[mask], top5_correct[mask])


def _add_policy_accuracy_counts(
    stats: dict[str, int],
    suffix: str,
    top1_correct: torch.Tensor,
    top5_correct: torch.Tensor,
) -> None:
    stats[f"policy_samples{suffix}"] += int(top1_correct.numel())
    stats[f"policy_top1_correct{suffix}"] += int(top1_correct.sum().item())
    stats[f"policy_top5_correct{suffix}"] += int(top5_correct.sum().item())


def _policy_accuracy_metrics(stats: dict[str, int]) -> dict[str, float]:
    metrics = {
        "policy_accuracy_top1": _accuracy(stats["policy_top1_correct"], stats["policy_samples"]),
        "policy_accuracy_top5": _accuracy(stats["policy_top5_correct"], stats["policy_samples"]),
    }
    for phase, _, _ in POLICY_PHASES:
        samples = stats[f"policy_samples_{phase}"]
        metrics[f"policy_accuracy_top1_{phase}"] = _accuracy(stats[f"policy_top1_correct_{phase}"], samples)
        metrics[f"policy_accuracy_top5_{phase}"] = _accuracy(stats[f"policy_top5_correct_{phase}"], samples)
    return metrics


def _accuracy(correct: int, total: int) -> float:
    return 0.0 if total == 0 else correct / total


def _split_train_validation_shards(
    data_dir: Path,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[list[Path], list[Path]]:
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0)")

    shards = sorted(data_dir.glob("shard-*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
    if validation_fraction == 0.0 or len(shards) < 2:
        return shards, []

    validation_count = min(len(shards) - 1, max(1, math.ceil(len(shards) * validation_fraction)))
    rng = np.random.default_rng(validation_seed)
    validation_indices = set(rng.permutation(len(shards))[:validation_count].tolist())
    train_shards = [shard for index, shard in enumerate(shards) if index not in validation_indices]
    validation_shards = [shard for index, shard in enumerate(shards) if index in validation_indices]
    return train_shards, validation_shards


def _make_batch_loader(
    data_dir: Path,
    batch_size: int,
    shards: Sequence[Path],
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = NPZShardBatchDataset(data_dir, batch_size, shards=shards, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    policy_loss: nn.Module,
    value_loss: nn.Module,
    use_amp: bool,
    *,
    max_steps: int | None = None,
    desc: str = "evaluate",
) -> dict[str, float]:
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    policy_accuracy_stats = _new_policy_accuracy_stats()
    steps = 0
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        progress = tqdm(loader, desc=desc)
        for boards, policies, values, plies in progress:
            boards = boards.to(device, non_blocking=True)
            if device.type == "cuda":
                boards = boards.contiguous(memory_format=torch.channels_last)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            plies = plies.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(boards)
                loss_policy = policy_loss(outputs["policy"], policies)
                loss_value = value_loss(outputs["value"], values)
                loss = loss_policy + loss_value

            steps += 1
            total_loss += float(loss.detach())
            total_policy += float(loss_policy.detach())
            total_value += float(loss_value.detach())
            _update_policy_accuracy_stats(policy_accuracy_stats, outputs["policy"], policies, plies)
            policy_accuracy = _policy_accuracy_metrics(policy_accuracy_stats)
            progress.set_postfix(
                loss=total_loss / steps,
                policy=total_policy / steps,
                value=total_value / steps,
                top1=policy_accuracy["policy_accuracy_top1"],
                top5=policy_accuracy["policy_accuracy_top5"],
            )
            if max_steps is not None and steps >= max_steps:
                break

    if was_training:
        model.train()

    return {
        "loss": total_loss / steps,
        "policy_loss": total_policy / steps,
        "value_loss": total_value / steps,
        **_policy_accuracy_metrics(policy_accuracy_stats),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def train_supervised(
    data_dir: Path,
    out_path: Path,
    *,
    epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    num_workers: int = 2,
    device_name: str | None = None,
    model_config: ModelConfig = ModelConfig(),
    compile_model: bool = False,
    amp: bool = True,
    max_steps: int | None = None,
    validation_fraction: float = 0.05,
    validation_seed: int = 0,
) -> list[dict[str, float]]:
    device = _resolve_device(device_name)
    train_shards, validation_shards = _split_train_validation_shards(data_dir, validation_fraction, validation_seed)
    loader = _make_batch_loader(data_dir, batch_size, train_shards, num_workers, device, shuffle=True)
    validation_loader = (
        _make_batch_loader(data_dir, batch_size, validation_shards, num_workers, device, shuffle=False)
        if validation_shards
        else None
    )
    model = ChessCNNTransformer(model_config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint_model = model
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        policy_accuracy_stats = _new_policy_accuracy_stats()
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for boards, policies, values, plies in progress:
            boards = boards.to(device, non_blocking=True)
            if device.type == "cuda":
                boards = boards.contiguous(memory_format=torch.channels_last)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            plies = plies.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(boards)
                loss_policy = policy_loss(outputs["policy"], policies)
                loss_value = value_loss(outputs["value"], values)
                loss = loss_policy + loss_value

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            total_loss += float(loss.detach())
            total_policy += float(loss_policy.detach())
            total_value += float(loss_value.detach())
            _update_policy_accuracy_stats(policy_accuracy_stats, outputs["policy"], policies, plies)
            policy_accuracy = _policy_accuracy_metrics(policy_accuracy_stats)
            progress.set_postfix(
                loss=total_loss / steps,
                policy=total_policy / steps,
                value=total_value / steps,
                top1=policy_accuracy["policy_accuracy_top1"],
                top5=policy_accuracy["policy_accuracy_top5"],
            )
            if max_steps is not None and steps >= max_steps:
                break

        epoch_metrics = {
            "loss": total_loss / steps,
            "policy_loss": total_policy / steps,
            "value_loss": total_value / steps,
            **_policy_accuracy_metrics(policy_accuracy_stats),
        }
        if validation_loader is not None:
            validation_metrics = _evaluate_model(
                model,
                validation_loader,
                device,
                policy_loss,
                value_loss,
                use_amp,
                max_steps=max_steps,
                desc=f"validation {epoch + 1}/{epochs}",
            )
            epoch_metrics.update(_prefix_metrics("val", validation_metrics))
        history.append(epoch_metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": checkpoint_model.state_dict(),
            "model_config": asdict(model_config),
            "policy_size": POLICY_SIZE,
            "policy_square_order": POLICY_SQUARE_ORDER,
            "history": history,
            "validation_fraction": validation_fraction,
            "validation_shards": [shard.name for shard in validation_shards],
        },
        out_path,
    )
    (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    return history


def evaluate_supervised(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    batch_size: int = 512,
    num_workers: int = 2,
    device_name: str | None = None,
    amp: bool = True,
    max_steps: int | None = None,
) -> dict[str, float]:
    model, _ = load_model_checkpoint(checkpoint_path, device_name=device_name)
    device = next(model.parameters()).device
    shards = sorted(data_dir.glob("shard-*.npz"))
    loader = _make_batch_loader(data_dir, batch_size, shards, num_workers, device, shuffle=False)
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()
    use_amp = amp and device.type == "cuda"
    return _evaluate_model(model, loader, device, policy_loss, value_loss, use_amp, max_steps=max_steps)


def load_model_checkpoint(checkpoint_path: Path, device_name: str | None = None) -> tuple[ChessCNNTransformer, ModelConfig]:
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("policy_square_order") != POLICY_SQUARE_ORDER:
        warnings.warn(
            "checkpoint was saved without the current policy square-order metadata; "
            "retrain it before using search or the GUI",
            RuntimeWarning,
            stacklevel=2,
        )
    config = ModelConfig(**checkpoint["model_config"])
    model = ChessCNNTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Distillation from the Lichess Stockfish evaluation dump (soft policy + WDL).
# ---------------------------------------------------------------------------

EvalBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class EvalShardBatchDataset(IterableDataset[EvalBatch]):
    """Streams batches from eval-distillation shards (soft policy + WDL targets)."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        *,
        shards: Sequence[Path] | None = None,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shards = sorted(data_dir.glob("shard-*.npz")) if shards is None else list(shards)
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths: list[int] = []
        for shard in self.shards:
            with np.load(shard) as data:
                self.lengths.append(int(data["wdl"].shape[0]))

    def __iter__(self) -> Iterator[EvalBatch]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shard_indices = np.arange(worker_id, len(self.shards), num_workers)
        rng = np.random.default_rng()
        if self.shuffle:
            rng.shuffle(shard_indices)

        for shard_index in shard_indices:
            with np.load(self.shards[int(shard_index)]) as data:
                shard = {name: data[name] for name in ("boards", "wdl", "policy_indices", "policy_probs")}
            order = np.arange(shard["wdl"].shape[0])
            if self.shuffle:
                order = rng.permutation(order)
            for start in range(0, order.shape[0], self.batch_size):
                indices = order[start : start + self.batch_size]
                yield _eval_batch_from_shard(shard, indices)

    def __len__(self) -> int:
        return sum((length + self.batch_size - 1) // self.batch_size for length in self.lengths)


def _eval_batch_from_shard(shard: dict[str, np.ndarray], indices: np.ndarray) -> EvalBatch:
    boards = _unpack_board_batch(shard["boards"][indices])
    wdl = torch.from_numpy(shard["wdl"][indices].astype(np.float32, copy=False))
    policy_indices = torch.from_numpy(shard["policy_indices"][indices].astype(np.int64, copy=False))
    policy_probs = torch.from_numpy(shard["policy_probs"][indices].astype(np.float32, copy=False))
    return boards, wdl, policy_indices, policy_probs


def _make_eval_batch_loader(
    data_dir: Path,
    batch_size: int,
    shards: Sequence[Path],
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = EvalShardBatchDataset(data_dir, batch_size, shards=shards, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _distillation_losses(
    outputs: dict[str, torch.Tensor],
    wdl_target: torch.Tensor,
    policy_indices: torch.Tensor,
    policy_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_log_probs = F.log_softmax(outputs["policy"], dim=1)
    gathered = policy_log_probs.gather(1, policy_indices.clamp(min=0))
    policy_loss = -(policy_probs * gathered).sum(dim=1).mean()
    value_log_probs = F.log_softmax(outputs["wdl"], dim=1)
    value_loss = -(wdl_target * value_log_probs).sum(dim=1).mean()
    return policy_loss, value_loss


def _distillation_target_moves(policy_indices: torch.Tensor, policy_probs: torch.Tensor) -> torch.Tensor:
    best = policy_probs.argmax(dim=1, keepdim=True)
    return policy_indices.gather(1, best).squeeze(1)


def _update_distillation_accuracy(
    stats: dict[str, int],
    policy_logits: torch.Tensor,
    target_moves: torch.Tensor,
) -> None:
    with torch.no_grad():
        topk = policy_logits.detach().topk(k=min(5, policy_logits.shape[1]), dim=1).indices
        stats["policy_samples"] += int(target_moves.numel())
        stats["policy_top1_correct"] += int(topk[:, 0].eq(target_moves).sum().item())
        stats["policy_top5_correct"] += int(topk.eq(target_moves.unsqueeze(1)).any(dim=1).sum().item())


def _distillation_accuracy_metrics(stats: dict[str, int]) -> dict[str, float]:
    return {
        "policy_accuracy_top1": _accuracy(stats["policy_top1_correct"], stats["policy_samples"]),
        "policy_accuracy_top5": _accuracy(stats["policy_top5_correct"], stats["policy_samples"]),
    }


def _evaluate_distillation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    value_weight: float,
    use_amp: bool,
    *,
    max_steps: int | None = None,
    desc: str = "evaluate",
) -> dict[str, float]:
    total_loss = total_policy = total_value = 0.0
    stats = {"policy_samples": 0, "policy_top1_correct": 0, "policy_top5_correct": 0}
    steps = 0
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        progress = tqdm(loader, desc=desc)
        for boards, wdl, policy_indices, policy_probs in progress:
            boards = boards.to(device, non_blocking=True)
            if device.type == "cuda":
                boards = boards.contiguous(memory_format=torch.channels_last)
            wdl = wdl.to(device, non_blocking=True)
            policy_indices = policy_indices.to(device, non_blocking=True)
            policy_probs = policy_probs.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(boards)
                loss_policy, loss_value = _distillation_losses(outputs, wdl, policy_indices, policy_probs)
                loss = loss_policy + value_weight * loss_value

            steps += 1
            total_loss += float(loss.detach())
            total_policy += float(loss_policy.detach())
            total_value += float(loss_value.detach())
            _update_distillation_accuracy(stats, outputs["policy"], _distillation_target_moves(policy_indices, policy_probs))
            accuracy = _distillation_accuracy_metrics(stats)
            progress.set_postfix(loss=total_loss / steps, top1=accuracy["policy_accuracy_top1"], top5=accuracy["policy_accuracy_top5"])
            if max_steps is not None and steps >= max_steps:
                break

    if was_training:
        model.train()

    return {
        "loss": total_loss / steps,
        "policy_loss": total_policy / steps,
        "value_loss": total_value / steps,
        **_distillation_accuracy_metrics(stats),
    }


def train_distillation(
    data_dir: Path,
    out_path: Path,
    *,
    epochs: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    value_weight: float = 1.0,
    num_workers: int = 8,
    device_name: str | None = None,
    model_config: ModelConfig = ModelConfig(),
    compile_model: bool = False,
    amp: bool = True,
    max_steps: int | None = None,
    validation_fraction: float = 0.02,
    validation_seed: int = 0,
) -> list[dict[str, float]]:
    device = _resolve_device(device_name)
    train_shards, validation_shards = _split_train_validation_shards(data_dir, validation_fraction, validation_seed)
    loader = _make_eval_batch_loader(data_dir, batch_size, train_shards, num_workers, device, shuffle=True)
    validation_loader = (
        _make_eval_batch_loader(data_dir, batch_size, validation_shards, num_workers, device, shuffle=False)
        if validation_shards
        else None
    )
    model = ChessCNNTransformer(model_config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint_model = model
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = total_policy = total_value = 0.0
        stats = {"policy_samples": 0, "policy_top1_correct": 0, "policy_top5_correct": 0}
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for boards, wdl, policy_indices, policy_probs in progress:
            boards = boards.to(device, non_blocking=True)
            if device.type == "cuda":
                boards = boards.contiguous(memory_format=torch.channels_last)
            wdl = wdl.to(device, non_blocking=True)
            policy_indices = policy_indices.to(device, non_blocking=True)
            policy_probs = policy_probs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(boards)
                loss_policy, loss_value = _distillation_losses(outputs, wdl, policy_indices, policy_probs)
                loss = loss_policy + value_weight * loss_value

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            total_loss += float(loss.detach())
            total_policy += float(loss_policy.detach())
            total_value += float(loss_value.detach())
            _update_distillation_accuracy(stats, outputs["policy"], _distillation_target_moves(policy_indices, policy_probs))
            accuracy = _distillation_accuracy_metrics(stats)
            progress.set_postfix(loss=total_loss / steps, top1=accuracy["policy_accuracy_top1"], top5=accuracy["policy_accuracy_top5"])
            if max_steps is not None and steps >= max_steps:
                break

        epoch_metrics = {
            "loss": total_loss / steps,
            "policy_loss": total_policy / steps,
            "value_loss": total_value / steps,
            **_distillation_accuracy_metrics(stats),
        }
        if validation_loader is not None:
            validation_metrics = _evaluate_distillation(
                model,
                validation_loader,
                device,
                value_weight,
                use_amp,
                max_steps=max_steps,
                desc=f"validation {epoch + 1}/{epochs}",
            )
            epoch_metrics.update(_prefix_metrics("val", validation_metrics))
        history.append(epoch_metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": checkpoint_model.state_dict(),
            "model_config": asdict(model_config),
            "policy_size": POLICY_SIZE,
            "policy_square_order": POLICY_SQUARE_ORDER,
            "data_format": "evals",
            "value_weight": value_weight,
            "history": history,
            "validation_fraction": validation_fraction,
            "validation_shards": [shard.name for shard in validation_shards],
        },
        out_path,
    )
    (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    return history


def evaluate_distillation(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    batch_size: int = 1024,
    value_weight: float = 1.0,
    num_workers: int = 8,
    device_name: str | None = None,
    amp: bool = True,
    max_steps: int | None = None,
) -> dict[str, float]:
    model, _ = load_model_checkpoint(checkpoint_path, device_name=device_name)
    device = next(model.parameters()).device
    shards = sorted(data_dir.glob("shard-*.npz"))
    loader = _make_eval_batch_loader(data_dir, batch_size, shards, num_workers, device, shuffle=False)
    use_amp = amp and device.type == "cuda"
    return _evaluate_distillation(model, loader, device, value_weight, use_amp, max_steps=max_steps)