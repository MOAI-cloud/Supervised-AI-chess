"""Supervised training for the CNN+Transformer engine.

Two data formats are supported:

* ``games`` shards (CCRL PGNs): one-hot best move + game result.
* ``evals`` shards (Lichess Stockfish dump): raw side-to-move scores for the
  position and for each multi-PV move. Both the current ``evals-v2`` layout and
    the legacy WDL/soft-probability layout are read; legacy scores are reconstructed
    approximately (:func:`superchess.targets.reconstruct_legacy_scores`) so the
  policy temperature and value mapping can be changed without re-preprocessing.

Targets are shaped on the GPU by :mod:`superchess.targets`. Training keeps an
exponential moving average of the weights (the EMA weights are what gets saved
as ``model`` and evaluated), supports resuming, and refuses to train a model
with clock planes on data whose clock planes are constant.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from superchess.data_quality import SPLITS, file_sha256, load_manifest, manifest_shards, verify_shard_checksums
from superchess.encoding import (
    BOARD_CHANNELS,
    FULL_BITPACKED_BOARD_BYTES,
    LEGACY_BOARD_CHANNELS,
    LEGACY_PACKED_BOARD_BITS,
    LEGACY_PACKED_BOARD_BYTES,
    PACKED_BOARD_BITS,
    PACKED_BOARD_BYTES,
    POLICY_SIZE,
)
from superchess.model import ChessCNNTransformer, ModelConfig, model_config_from_checkpoint
from superchess.targets import (
    TargetConfig,
    best_target_moves,
    entropy,
    outcome_value_targets,
    policy_targets,
    reconstruct_legacy_scores,
    scores_to_expected,
    value_targets,
)

POLICY_SQUARE_ORDER = "python-chess"
POLICY_PHASES = (
    ("early", 0, 20),
    ("mid", 20, 60),
    ("endgame", 60, None),
)
EVAL_FORMAT_V2 = "evals-v2"
EVAL_FORMAT_LEGACY = "evals-v1"
CHECKPOINT_VERSION = 2


# ---------------------------------------------------------------------------
# Device / backend helpers
# ---------------------------------------------------------------------------


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


def _configure_backend(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _amp_dtype(device: torch.device) -> torch.dtype:
    """Prefer bfloat16 (no gradient scaling, wider dynamic range) when supported."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------


def _optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Apply weight decay to matrices only; norms, biases, and positional tables skip it."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(("square_embedding", "square_bias")):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _total_training_steps(loader: DataLoader, epochs: int, max_steps: int | None) -> int:
    steps_per_epoch = len(loader)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    return max(1, steps_per_epoch * epochs)


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    *,
    warmup_steps: int | None = None,
    floor: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to ``floor`` of the peak, stepped per optimizer step."""
    if warmup_steps is None:
        warmup_steps = max(1, min(2000, total_steps // 20))
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential moving average of the weights (buffers are copied, not averaged).

    Evaluating and shipping averaged weights is standard practice for policy
    networks (Lc0 uses stochastic weight averaging) and typically plays notably
    stronger than the raw weights at the end of a noisy schedule. The decay ramps
    up from zero so early averages are not dominated by the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        ema_params = [p for p in self.module.parameters()]
        model_params = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_params, model_params, 1.0 - decay)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.updates = int(state.get("updates", 0))
        self.module.load_state_dict(state["module"])


# ---------------------------------------------------------------------------
# Board unpacking and input-plane validation
# ---------------------------------------------------------------------------


def _unpack_board_batch(packed_boards: np.ndarray) -> torch.Tensor:
    width = packed_boards.shape[1]
    if width == FULL_BITPACKED_BOARD_BYTES:
        boards = np.unpackbits(packed_boards, axis=1, count=PACKED_BOARD_BITS).astype(np.float32, copy=False)
        return torch.from_numpy(boards.reshape(-1, BOARD_CHANNELS, 8, 8))
    if width not in (LEGACY_PACKED_BOARD_BYTES, PACKED_BOARD_BYTES):
        raise ValueError(f"unexpected packed board byte count: {width}")

    boards = np.zeros((packed_boards.shape[0], BOARD_CHANNELS, 8, 8), dtype=np.float32)
    binary = np.unpackbits(
        packed_boards[:, :LEGACY_PACKED_BOARD_BYTES],
        axis=1,
        count=LEGACY_PACKED_BOARD_BITS,
    ).astype(np.float32, copy=False)
    boards[:, :LEGACY_BOARD_CHANNELS] = binary.reshape(-1, LEGACY_BOARD_CHANNELS, 8, 8)
    if width == PACKED_BOARD_BYTES:
        aux = packed_boards[:, LEGACY_PACKED_BOARD_BYTES:PACKED_BOARD_BYTES].astype(np.float32) / 255.0
        boards[:, LEGACY_BOARD_CHANNELS:, :, :] = aux[:, :, None, None]
    return torch.from_numpy(boards)


def clock_planes_informative(shards: Sequence[Path], *, sample_shards: int = 3) -> bool:
    """Whether the halfmove/fullmove planes vary across the sampled shards.

    The Lichess evaluation dump stores four-field FENs, so every position gets
    ``halfmove=0, fullmove=1``: the two clock planes are constant during training
    but take real values at play time. A model that consumes them therefore sees
    out-of-distribution inputs in every real game, which silently degrades play
    while offline metrics keep improving.
    """
    seen: set[bytes] = set()
    for shard in list(shards)[:sample_shards]:
        with np.load(shard) as data:
            boards = data["boards"]
            if boards.ndim != 2 or boards.shape[1] < PACKED_BOARD_BYTES:
                return False
            if boards.shape[1] == FULL_BITPACKED_BOARD_BYTES:
                planes = np.unpackbits(boards[:, :], axis=1, count=PACKED_BOARD_BITS)
                aux = planes.reshape(-1, BOARD_CHANNELS, 64)[:, LEGACY_BOARD_CHANNELS:, 0]
            else:
                aux = boards[:, LEGACY_PACKED_BOARD_BYTES:PACKED_BOARD_BYTES]
            for row in np.unique(aux, axis=0):
                seen.add(row.tobytes())
                if len(seen) > 1:
                    return True
    return len(seen) > 1


def check_input_planes(shards: Sequence[Path], input_channels: int) -> None:
    """Refuse to train clock-aware models on data whose clock planes carry no information."""
    if input_channels <= LEGACY_BOARD_CHANNELS or not shards:
        return
    if not clock_planes_informative(shards):
        raise ValueError(
            f"model input_channels={input_channels} includes the halfmove/fullmove planes, but they are "
            "constant in this dataset (the Lichess eval dump has no move counters). Such a model plays "
            "increasingly badly as real games progress even though offline metrics look fine. "
            f"Use input_channels={LEGACY_BOARD_CHANNELS} (the default) for this data."
        )


def _select_planes(boards: torch.Tensor, input_channels: int) -> torch.Tensor:
    if boards.shape[1] == input_channels:
        return boards
    if boards.shape[1] < input_channels:
        raise ValueError(f"data provides {boards.shape[1]} planes but the model expects {input_channels}")
    return boards[:, :input_channels]


# ---------------------------------------------------------------------------
# Game-result shards (CCRL)
# ---------------------------------------------------------------------------


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
        board = _unpack_board_batch(shard["boards"][local_index : local_index + 1]).squeeze(0)
        return (
            board,
            torch.tensor(int(shard["policies"][local_index]), dtype=torch.long),
            torch.tensor(float(shard["values"][local_index]), dtype=torch.float32),
            torch.tensor(int(shard["plies"][local_index]), dtype=torch.long),
        )

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == shard_index and self._cache is not None:
            return self._cache
        with np.load(self.shards[shard_index]) as data:
            self._cache = _arrays_from_npz(data)
        self._cache_index = shard_index
        return self._cache


def _worker_shard_indices(num_shards: int) -> np.ndarray:
    worker = get_worker_info()
    worker_id = 0 if worker is None else worker.id
    num_workers = 1 if worker is None else worker.num_workers
    return np.arange(worker_id, num_shards, num_workers)


def _shard_groups(indices: np.ndarray, group_size: int, rng: np.random.Generator, shuffle: bool) -> Iterator[list[int]]:
    """Yield groups of shard indices; mixing several shards per group decorrelates batches."""
    order = rng.permutation(indices) if shuffle else np.asarray(indices)
    size = max(1, group_size)
    for start in range(0, len(order), size):
        yield [int(index) for index in order[start : start + size]]


def _epoch_rng(seed: int, epoch: int) -> np.random.Generator:
    worker = get_worker_info()
    worker_id = 0 if worker is None else worker.id
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, worker_id]))


class NPZShardBatchDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        *,
        shards: Sequence[Path] | None = None,
        shuffle: bool = True,
        mix_shards: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shards = sorted(data_dir.glob("shard-*.npz")) if shards is None else list(shards)
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mix_shards = max(1, mix_shards)
        self.seed = 0
        self.epoch = 0
        self.lengths: list[int] = []
        for shard in self.shards:
            with np.load(shard) as data:
                self.lengths.append(int(data["policies"].shape[0]))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = _epoch_rng(self.seed, self.epoch)
        self.epoch += 1
        indices = _worker_shard_indices(len(self.shards))
        for group in _shard_groups(indices, self.mix_shards if self.shuffle else 1, rng, self.shuffle):
            arrays = [_arrays_from_npz(np.load(self.shards[index])) for index in group]
            shard = {name: np.concatenate([item[name] for item in arrays]) for name in arrays[0]}
            order = np.arange(shard["policies"].shape[0])
            if self.shuffle:
                order = rng.permutation(order)
            for start in range(0, order.shape[0], self.batch_size):
                yield _batch_from_shard(shard, order[start : start + self.batch_size])

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


# ---------------------------------------------------------------------------
# Evaluation shards (Lichess Stockfish dump)
# ---------------------------------------------------------------------------

EvalBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""``(boards, scores, policy_indices, policy_scores)`` — scores are raw side-to-move integers."""


def detect_eval_shard_format(shard: Path) -> str:
    with np.load(shard) as data:
        files = set(data.files)
    if {"score", "policy_scores"} <= files:
        return EVAL_FORMAT_V2
    if {"wdl", "policy_probs"} <= files:
        return EVAL_FORMAT_LEGACY
    raise ValueError(f"{shard} is not an eval-distillation shard (keys: {sorted(files)})")


def _load_legacy_target_config(data_dir: Path) -> dict[str, Any]:
    metadata = data_dir / "metadata.json"
    if not metadata.exists():
        return {}
    try:
        config = json.loads(metadata.read_text(encoding="utf-8")).get("config", {})
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: config[key] for key in ("value_scale", "wdl_scale", "wdl_draw_margin", "policy_temperature") if key in config}


class EvalShardBatchDataset(IterableDataset[EvalBatch]):
    """Streams batches of raw-score targets from eval shards (v2 or legacy layout)."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        *,
        shards: Sequence[Path] | None = None,
        shuffle: bool = True,
        mix_shards: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shards = sorted(data_dir.glob("shard-*.npz")) if shards is None else list(shards)
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.npz files found under {data_dir}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mix_shards = max(1, mix_shards)
        self.seed = 0
        self.epoch = 0
        self.format = detect_eval_shard_format(self.shards[0])
        self.legacy_config = _load_legacy_target_config(data_dir) if self.format == EVAL_FORMAT_LEGACY else {}
        metadata = load_manifest(data_dir)
        lengths = {data_dir / entry["path"]: entry["rows"] for entry in metadata["shards"]} if metadata is not None else {}
        self.lengths: list[int] = []
        for shard in self.shards:
            if shard in lengths:
                self.lengths.append(int(lengths[shard]))
            else:
                with np.load(shard) as data:
                    self.lengths.append(int(data["boards"].shape[0]))

    def _load(self, shard: Path) -> dict[str, np.ndarray]:
        with np.load(shard) as data:
            if self.format == EVAL_FORMAT_V2:
                return {
                    "boards": data["boards"],
                    "scores": data["score"].astype(np.float32),
                    "policy_indices": data["policy_indices"].astype(np.int64),
                    "policy_scores": data["policy_scores"].astype(np.float32),
                }
            indices = data["policy_indices"].astype(np.int64)
            scores, policy_scores = reconstruct_legacy_scores(data["wdl"], indices, data["policy_probs"], self.legacy_config)
            return {"boards": data["boards"], "scores": scores, "policy_indices": indices, "policy_scores": policy_scores}

    def __iter__(self) -> Iterator[EvalBatch]:
        rng = _epoch_rng(self.seed, self.epoch)
        self.epoch += 1
        indices = _worker_shard_indices(len(self.shards))
        for group in _shard_groups(indices, self.mix_shards if self.shuffle else 1, rng, self.shuffle):
            arrays = [self._load(self.shards[index]) for index in group]
            shard = {name: np.concatenate([item[name] for item in arrays]) for name in arrays[0]}
            order = np.arange(shard["scores"].shape[0])
            if self.shuffle:
                order = rng.permutation(order)
            for start in range(0, order.shape[0], self.batch_size):
                yield _eval_batch_from_shard(shard, order[start : start + self.batch_size])

    def __len__(self) -> int:
        return sum((length + self.batch_size - 1) // self.batch_size for length in self.lengths)


def _eval_batch_from_shard(shard: dict[str, np.ndarray], indices: np.ndarray) -> EvalBatch:
    boards = _unpack_board_batch(shard["boards"][indices])
    scores = torch.from_numpy(np.ascontiguousarray(shard["scores"][indices], dtype=np.float32))
    policy_indices = torch.from_numpy(np.ascontiguousarray(shard["policy_indices"][indices], dtype=np.int64))
    policy_scores = torch.from_numpy(np.ascontiguousarray(shard["policy_scores"][indices], dtype=np.float32))
    return boards, scores, policy_indices, policy_scores


def _make_loader(dataset: IterableDataset, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),
    )


def _make_batch_loader(
    data_dir: Path,
    batch_size: int,
    shards: Sequence[Path],
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
    mix_shards: int = 1,
) -> DataLoader:
    dataset = NPZShardBatchDataset(data_dir, batch_size, shards=shards, shuffle=shuffle, mix_shards=mix_shards)
    return _make_loader(dataset, num_workers, device)


def _make_eval_batch_loader(
    data_dir: Path,
    batch_size: int,
    shards: Sequence[Path],
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
    mix_shards: int = 1,
) -> DataLoader:
    dataset = EvalShardBatchDataset(data_dir, batch_size, shards=shards, shuffle=shuffle, mix_shards=mix_shards)
    return _make_loader(dataset, num_workers, device)


def _split_train_validation_shards(
    data_dir: Path,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[list[Path], list[Path]]:
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0)")

    metadata = load_manifest(data_dir)
    if metadata is not None:
        splits = {split: manifest_shards(data_dir, metadata, split) for split in SPLITS}
        if not splits["train"]:
            raise ValueError("frozen dataset contains no training positions; build a larger dataset")
        return splits["train"], splits["validation"]
    if data_dir.name in SPLITS and load_manifest(data_dir.parent) is not None:
        raise ValueError("train with the dataset root, not an individual frozen split")
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


# ---------------------------------------------------------------------------
# Policy accuracy bookkeeping (shared by both formats)
# ---------------------------------------------------------------------------


def _new_policy_accuracy_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {"policy_samples": 0, "policy_top1_correct": 0, "policy_top5_correct": 0}
    for phase, _, _ in POLICY_PHASES:
        stats[f"policy_samples_{phase}"] = 0
        stats[f"policy_top1_correct_{phase}"] = 0
        stats[f"policy_top5_correct_{phase}"] = 0
    return stats


def _update_policy_accuracy_stats(
    stats: dict[str, Any],
    policy_logits: torch.Tensor,
    policies: torch.Tensor,
    plies: torch.Tensor | None,
) -> None:
    with torch.no_grad():
        topk = policy_logits.detach().topk(k=min(5, policy_logits.shape[1]), dim=1).indices
        top1_correct = topk[:, 0].eq(policies)
        top5_correct = topk.eq(policies.unsqueeze(1)).any(dim=1)

        _add_policy_accuracy_counts(stats, "", top1_correct, top5_correct)
        if plies is None:
            return
        for phase, start, stop in POLICY_PHASES:
            mask = plies >= start
            if stop is not None:
                mask = mask & (plies < stop)
            _add_policy_accuracy_counts(stats, f"_{phase}", top1_correct[mask], top5_correct[mask])


def _add_policy_accuracy_counts(
    stats: dict[str, Any],
    suffix: str,
    top1_correct: torch.Tensor,
    top5_correct: torch.Tensor,
) -> None:
    # Accumulate on-device tensors; converting per step would force a GPU sync.
    stats[f"policy_samples{suffix}"] += int(top1_correct.numel())
    stats[f"policy_top1_correct{suffix}"] = stats[f"policy_top1_correct{suffix}"] + top1_correct.sum()
    stats[f"policy_top5_correct{suffix}"] = stats[f"policy_top5_correct{suffix}"] + top5_correct.sum()


def _policy_accuracy_metrics(stats: dict[str, Any], *, phases: bool = True) -> dict[str, float]:
    metrics = {
        "policy_accuracy_top1": _accuracy(stats["policy_top1_correct"], stats["policy_samples"]),
        "policy_accuracy_top5": _accuracy(stats["policy_top5_correct"], stats["policy_samples"]),
    }
    if phases:
        for phase, _, _ in POLICY_PHASES:
            samples = stats[f"policy_samples_{phase}"]
            metrics[f"policy_accuracy_top1_{phase}"] = _accuracy(stats[f"policy_top1_correct_{phase}"], samples)
            metrics[f"policy_accuracy_top5_{phase}"] = _accuracy(stats[f"policy_top5_correct_{phase}"], samples)
    return metrics


def _accuracy(correct: Any, total: Any) -> float:
    total = int(total)
    return 0.0 if total == 0 else int(correct) / total


def _require_steps(steps: int, desc: str) -> None:
    if steps == 0:
        raise RuntimeError(f"{desc} did not process any batches")


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


# ---------------------------------------------------------------------------
# Objectives: turn a batch into a loss and running metrics
# ---------------------------------------------------------------------------


def _categorical_value_loss(value_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(value_logits.float(), dim=1)).sum(dim=1).mean()


class _Objective:
    """Base class: ``prepare`` moves a batch to the device, ``compute`` returns loss + metric tensors."""

    phases = False

    def __init__(self, target_config: TargetConfig, value_bins: int, value_weight: float) -> None:
        self.target_config = target_config
        self.value_bins = value_bins
        self.value_weight = value_weight

    def prepare(self, batch: tuple, device: torch.device, input_channels: int) -> tuple:
        tensors = []
        for index, item in enumerate(batch):
            tensor = item.to(device, non_blocking=True)
            if index == 0:
                tensor = _select_planes(tensor, input_channels)
                if device.type == "cuda":
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
            tensors.append(tensor)
        return tuple(tensors)

    def value_loss(self, outputs: dict[str, torch.Tensor], expected: torch.Tensor, bins_target: torch.Tensor) -> torch.Tensor:
        value_logits = outputs["value_logits"]
        if value_logits.shape[1] == self.value_bins:
            return _categorical_value_loss(value_logits, bins_target)
        # Legacy WDL heads (inference-only checkpoints): report a scalar regression loss instead.
        return F.mse_loss(outputs["value"].float(), 2.0 * expected - 1.0)

    def compute(self, outputs: dict[str, torch.Tensor], batch: tuple) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def new_stats(self) -> dict[str, Any]:
        return _new_policy_accuracy_stats()

    def update_stats(self, stats: dict[str, Any], outputs: dict[str, torch.Tensor], batch: tuple) -> None:
        raise NotImplementedError

    def finalize_stats(self, stats: dict[str, Any]) -> dict[str, float]:
        return _policy_accuracy_metrics(stats, phases=self.phases)


class GamesObjective(_Objective):
    """One-hot best move + HL-Gauss game-result value (CCRL shards)."""

    phases = True

    def compute(self, outputs: dict[str, torch.Tensor], batch: tuple) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, policies, values, _ = batch
        policy_loss = F.cross_entropy(outputs["policy"].float(), policies)
        expected = (values.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        bins_target = outcome_value_targets(values, self.value_bins, self.target_config)
        value_loss = self.value_loss(outputs, expected, bins_target)
        loss = policy_loss + self.value_weight * value_loss
        with torch.no_grad():
            value_mae = (outputs["value"].float() - values.float()).abs().mean()
        metrics = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "value_mae": value_mae,
        }
        return loss, metrics

    def update_stats(self, stats: dict[str, Any], outputs: dict[str, torch.Tensor], batch: tuple) -> None:
        _, policies, _, plies = batch
        _update_policy_accuracy_stats(stats, outputs["policy"], policies, plies)


class EvalsObjective(_Objective):
    """Soft multi-PV policy + HL-Gauss expected-score value (Lichess eval shards)."""

    phases = False

    def compute(self, outputs: dict[str, torch.Tensor], batch: tuple) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, scores, policy_indices, policy_scores = batch
        valid = policy_indices >= 0
        target = policy_targets(policy_scores, valid, self.target_config)
        log_probs = F.log_softmax(outputs["policy"].float(), dim=1)
        gathered = log_probs.gather(1, policy_indices.clamp(min=0))
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
        policy_loss = -(target * gathered).sum(dim=1).mean()

        expected = scores_to_expected(scores, self.target_config.cp_scale)
        bins_target = value_targets(scores, self.value_bins, self.target_config)
        value_loss = self.value_loss(outputs, expected, bins_target)
        loss = policy_loss + self.value_weight * value_loss

        with torch.no_grad():
            target_entropy = entropy(target).mean()
            policy_entropy = entropy(log_probs.exp()).mean()
            value_mae = (outputs["value"].float() - (2.0 * expected - 1.0)).abs().mean()
        metrics = {
            "policy_loss": policy_loss.detach(),
            "policy_kl": policy_loss.detach() - target_entropy,
            "value_loss": value_loss.detach(),
            "value_mae": value_mae,
            "target_entropy": target_entropy,
            "policy_entropy": policy_entropy,
        }
        return loss, metrics

    def update_stats(self, stats: dict[str, Any], outputs: dict[str, torch.Tensor], batch: tuple) -> None:
        _, _, policy_indices, policy_scores = batch
        _update_policy_accuracy_stats(stats, outputs["policy"], best_target_moves(policy_indices, policy_scores), None)


# ---------------------------------------------------------------------------
# Epoch loop shared by training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class _StepContext:
    optimizer: torch.optim.Optimizer | None = None
    scaler: torch.amp.GradScaler | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    ema: ModelEMA | None = None
    grad_clip: float = 1.0
    raw_model: nn.Module | None = None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    objective: _Objective,
    device: torch.device,
    *,
    input_channels: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_steps: int | None,
    desc: str,
    step: _StepContext | None = None,
) -> tuple[dict[str, float], int]:
    training = step is not None
    if training:
        model.train()
    else:
        model.eval()
    totals: dict[str, torch.Tensor | float] = {}
    stats = objective.new_stats()
    steps = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        progress = tqdm(loader, desc=desc, dynamic_ncols=True)
        for batch in progress:
            batch = objective.prepare(batch, device, input_channels)
            boards = batch[0]
            if training:
                step.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(boards)
            loss, metrics = objective.compute(outputs, batch)
            if training:
                step.scaler.scale(loss).backward()
                step.scaler.unscale_(step.optimizer)
                if step.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=step.grad_clip)
                step.scaler.step(step.optimizer)
                step.scaler.update()
                if step.scheduler is not None:
                    step.scheduler.step()
                if step.ema is not None:
                    step.ema.update(step.raw_model if step.raw_model is not None else model)

            steps += 1
            # Keep running sums on the device; a host sync per step would stall the pipeline.
            totals["loss"] = totals.get("loss", 0.0) + loss.detach()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            objective.update_stats(stats, outputs, batch)
            if steps % 20 == 1:
                accuracy = objective.finalize_stats(stats)
                progress.set_postfix(
                    loss=f"{float(totals['loss']) / steps:.4f}",
                    pol=f"{float(totals.get('policy_loss', 0.0)) / steps:.4f}",
                    val=f"{float(totals.get('value_loss', 0.0)) / steps:.4f}",
                    top1=f"{accuracy['policy_accuracy_top1']:.3f}",
                    top5=f"{accuracy['policy_accuracy_top5']:.3f}",
                )
            if max_steps is not None and steps >= max_steps:
                break

    _require_steps(steps, desc)
    metrics = {name: float(value) / steps for name, value in totals.items()}
    metrics.update(objective.finalize_stats(stats))
    return metrics, steps


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    epochs: int = 1
    batch_size: int = 1024
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    value_weight: float = 1.0
    num_workers: int = 8
    device_name: str | None = None
    compile_model: bool = False
    amp: bool = True
    max_steps: int | None = None
    validation_fraction: float = 0.02
    validation_seed: int = 0
    ema_decay: float = 0.9999
    warmup_steps: int | None = None
    grad_clip: float = 1.0
    mix_shards: int = 4
    resume: Path | None = None
    save_optimizer: bool = True
    seed: int = 0


def _build_objective(data_format: str, target_config: TargetConfig, model_config: ModelConfig, value_weight: float) -> _Objective:
    cls = EvalsObjective if data_format == "evals" else GamesObjective
    return cls(target_config, model_config.value_bins, value_weight)


def _fit(
    data_dir: Path,
    out_path: Path,
    *,
    data_format: str,
    model_config: ModelConfig,
    target_config: TargetConfig,
    options: TrainingOptions,
) -> list[dict[str, float]]:
    if data_format not in ("evals", "games"):
        raise ValueError("data_format must be 'evals' or 'games'")
    if options.seed < 0:
        raise ValueError("training seed must be nonnegative")
    torch.manual_seed(options.seed)
    device = _resolve_device(options.device_name)
    _configure_backend(device)
    train_shards, validation_shards = _split_train_validation_shards(
        data_dir, options.validation_fraction, options.validation_seed
    )
    dataset_manifest = load_manifest(data_dir)
    if dataset_manifest is not None:
        if data_format != "evals":
            raise ValueError("frozen evaluation dataset requires data_format='evals'")
        verify_shard_checksums(data_dir, dataset_manifest, train_shards + validation_shards)
    dataset_id = dataset_manifest["dataset_id"] if dataset_manifest is not None else None
    dataset_provenance = (
        {"dataset_id": dataset_id, "manifest_sha256": file_sha256(data_dir / "metadata.json"),
         "split_method": dataset_manifest["split_method"], "config": dataset_manifest["config"]}
        if dataset_manifest is not None else None
    )
    if data_format == "evals" and dataset_manifest is None and validation_shards:
        warnings.warn("legacy shard-level validation does not establish position-disjoint holdout; rebuild a frozen dataset for reliable validation", RuntimeWarning, stacklevel=2)
    check_input_planes(train_shards, model_config.input_channels)
    make_loader = _make_eval_batch_loader if data_format == "evals" else _make_batch_loader
    loader = make_loader(
        data_dir,
        options.batch_size,
        train_shards,
        options.num_workers,
        device,
        shuffle=True,
        mix_shards=options.mix_shards,
    )
    validation_loader = (
        make_loader(data_dir, options.batch_size, validation_shards, options.num_workers, device, shuffle=False)
        if validation_shards
        else None
    )
    eval_shard_format = getattr(loader.dataset, "format", None)
    if eval_shard_format == EVAL_FORMAT_LEGACY:
        warnings.warn(
            "training on legacy eval shards: approximate scores are reconstructed from the stored WDL/soft-policy "
            "arrays; mate distances and cross-depth policy anchors cannot be recovered. Re-run "
            "`superchess evals preprocess` to produce evals-v2 shards when convenient.",
            RuntimeWarning,
            stacklevel=2,
        )

    checkpoint_model = ChessCNNTransformer(model_config).to(device)
    if device.type == "cuda":
        checkpoint_model = checkpoint_model.to(memory_format=torch.channels_last)
    model: nn.Module = checkpoint_model
    if options.compile_model and hasattr(torch, "compile"):
        model = torch.compile(checkpoint_model)
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(checkpoint_model, options.weight_decay),
        lr=options.learning_rate,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    total_steps = _total_training_steps(loader, options.epochs, options.max_steps)
    scheduler = _make_lr_scheduler(optimizer, total_steps, warmup_steps=options.warmup_steps)
    ema = ModelEMA(checkpoint_model, options.ema_decay) if options.ema_decay > 0 else None
    use_amp = options.amp and device.type == "cuda"
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    objective = _build_objective(data_format, target_config, model_config, options.value_weight)
    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    global_step = 0

    if options.resume is not None:
        state = torch.load(options.resume, map_location=device, weights_only=False)
        training_state = state.get("training")
        if training_state is None:
            raise ValueError(f"{options.resume} has no optimizer state to resume from (was it saved with save_optimizer?)")
        saved_config = model_config_from_checkpoint(state["model_config"])
        if saved_config != model_config:
            raise ValueError(f"resume model config {saved_config} differs from requested {model_config}")
        if state.get("data_format", data_format) != data_format or TargetConfig.from_dict(state.get("target_config")) != target_config:
            raise ValueError("resume data format or target config differs from the checkpoint")
        if state.get("dataset_id") is not None and state["dataset_id"] != dataset_id:
            raise ValueError("resume dataset identity differs from the checkpoint; start a new experiment")
        if dataset_id is not None and state.get("dataset_id") is None:
            raise ValueError("resume checkpoint has no dataset identity; do not reuse optimizer/history across an unverified dataset change")
        if training_state.get("options", {}).get("seed", options.seed) != options.seed:
            raise ValueError("resume training seed differs from the checkpoint")
        checkpoint_model.load_state_dict(training_state["raw_model"])
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        if ema is not None and training_state.get("ema") is not None:
            ema.load_state_dict(training_state["ema"])
        if "scaler" in training_state:
            scaler.load_state_dict(training_state["scaler"])
        if "torch_rng" in training_state:
            torch.set_rng_state(training_state["torch_rng"].cpu())
        if device.type == "cuda" and training_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in training_state["cuda_rng"]])
        history = list(state.get("history", []))
        global_step = int(training_state.get("step", 0))
        best_val_loss = min((entry.get("val_loss", math.inf) for entry in history), default=math.inf)

    for active_loader in (loader, validation_loader):
        if active_loader is not None:
            active_loader.dataset.seed = options.seed
            active_loader.dataset.epoch = len(history)
            active_loader.generator.manual_seed(options.seed)

    best_path = out_path.with_name(f"{out_path.stem}-best{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = ema.module if ema is not None else checkpoint_model

    def save_checkpoint(path: Path, *, with_training_state: bool) -> None:
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": export_model.state_dict(),
            "model_config": asdict(model_config),
            "target_config": target_config.to_dict(),
            "policy_size": POLICY_SIZE,
            "policy_square_order": POLICY_SQUARE_ORDER,
            "data_format": data_format,
            "eval_shard_format": eval_shard_format,
            "value_weight": options.value_weight,
            "weights": "ema" if ema is not None else "raw",
            "history": history,
            "epochs_completed": len(history),
            "dataset_id": dataset_id,
            "dataset_provenance": dataset_provenance,
            "training_seed": options.seed,
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                        "amp": use_amp, "amp_dtype": str(amp_dtype), "compile_model": options.compile_model,
                        "num_workers": options.num_workers},
            "validation_fraction": dataset_manifest["config"]["validation_fraction"] if dataset_manifest is not None else options.validation_fraction,
            "validation_shards": [str(shard.relative_to(data_dir)) for shard in validation_shards],
        }
        if with_training_state:
            payload["training"] = {
                "raw_model": checkpoint_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "step": global_step,
                "options": {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(options).items()},
            }
        torch.save(payload, path)
        sidecar = {name: payload[name] for name in ("history", "model_config", "target_config", "dataset_id", "dataset_provenance", "validation_shards", "training_seed", "runtime")}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    step_context = _StepContext(
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        ema=ema,
        grad_clip=options.grad_clip,
        raw_model=checkpoint_model,
    )
    for epoch in range(len(history), options.epochs):
        desc = f"epoch {epoch + 1}/{options.epochs}"
        train_metrics, steps = _run_epoch(
            model,
            loader,
            objective,
            device,
            input_channels=model_config.input_channels,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_steps=options.max_steps,
            desc=desc,
            step=step_context,
        )
        global_step += steps
        epoch_metrics = {**train_metrics, "learning_rate": scheduler.get_last_lr()[0], "step": float(global_step)}
        if validation_loader is not None:
            validation_metrics, _ = _run_epoch(
                export_model,
                validation_loader,
                objective,
                device,
                input_channels=model_config.input_channels,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_steps=options.max_steps,
                desc=f"validation {epoch + 1}/{options.epochs}",
            )
            epoch_metrics.update(_prefix_metrics("val", validation_metrics))
        history.append(epoch_metrics)
        save_checkpoint(out_path, with_training_state=options.save_optimizer)
        if epoch_metrics.get("val_loss", math.inf) < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            save_checkpoint(best_path, with_training_state=False)

    return history


def train_supervised(
    data_dir: Path,
    out_path: Path,
    *,
    epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    value_weight: float = 1.0,
    num_workers: int = 2,
    device_name: str | None = None,
    model_config: ModelConfig = ModelConfig(input_channels=BOARD_CHANNELS),
    target_config: TargetConfig = TargetConfig(),
    compile_model: bool = False,
    amp: bool = True,
    max_steps: int | None = None,
    validation_fraction: float = 0.05,
    validation_seed: int = 0,
    **extra: Any,
) -> list[dict[str, float]]:
    """Train on CCRL game shards (one-hot move, game result)."""
    options = TrainingOptions(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
        num_workers=num_workers,
        device_name=device_name,
        compile_model=compile_model,
        amp=amp,
        max_steps=max_steps,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        **extra,
    )
    return _fit(
        data_dir, out_path, data_format="games", model_config=model_config, target_config=target_config, options=options
    )


def train_distillation(
    data_dir: Path,
    out_path: Path,
    *,
    epochs: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    value_weight: float = 1.0,
    num_workers: int = 8,
    device_name: str | None = None,
    model_config: ModelConfig = ModelConfig(),
    target_config: TargetConfig = TargetConfig(),
    compile_model: bool = False,
    amp: bool = True,
    max_steps: int | None = None,
    validation_fraction: float = 0.02,
    validation_seed: int = 0,
    **extra: Any,
) -> list[dict[str, float]]:
    """Distil Stockfish evaluations (soft multi-PV policy, categorical expected score)."""
    options = TrainingOptions(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
        num_workers=num_workers,
        device_name=device_name,
        compile_model=compile_model,
        amp=amp,
        max_steps=max_steps,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        **extra,
    )
    return _fit(
        data_dir, out_path, data_format="evals", model_config=model_config, target_config=target_config, options=options
    )


# ---------------------------------------------------------------------------
# Checkpoint loading and offline evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    model: ChessCNNTransformer
    model_config: ModelConfig
    target_config: TargetConfig
    data_format: str
    metadata: dict[str, Any]


def load_checkpoint_bundle(
    checkpoint_path: Path,
    device_name: str | None = None,
    *,
    allow_legacy_policy: bool = False,
) -> CheckpointBundle:
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("policy_square_order") != POLICY_SQUARE_ORDER:
        message = (
            "checkpoint policy square-order metadata is missing or incompatible; "
            "retrain it before using search, evaluate, or the GUI"
        )
        if not allow_legacy_policy:
            raise RuntimeError(f"{message}. Pass allow_legacy_policy=True only for deliberate legacy inspection.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    config = model_config_from_checkpoint(checkpoint["model_config"])
    model = ChessCNNTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key not in ("model", "training")}
    return CheckpointBundle(
        model=model,
        model_config=config,
        target_config=TargetConfig.from_dict(checkpoint.get("target_config")),
        data_format=str(checkpoint.get("data_format", "evals")),
        metadata=metadata,
    )


def load_model_checkpoint(
    checkpoint_path: Path,
    device_name: str | None = None,
    *,
    allow_legacy_policy: bool = False,
) -> tuple[ChessCNNTransformer, ModelConfig]:
    bundle = load_checkpoint_bundle(checkpoint_path, device_name, allow_legacy_policy=allow_legacy_policy)
    return bundle.model, bundle.model_config


def _evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    data_format: str,
    batch_size: int,
    value_weight: float,
    num_workers: int,
    device_name: str | None,
    amp: bool,
    max_steps: int | None,
    allow_legacy_policy: bool,
    target_config: TargetConfig | None,
) -> dict[str, float]:
    bundle = load_checkpoint_bundle(checkpoint_path, device_name=device_name, allow_legacy_policy=allow_legacy_policy)
    device = next(bundle.model.parameters()).device
    if load_manifest(data_dir) is not None:
        raise ValueError("evaluate an explicit validation or test subdirectory of the frozen dataset")
    parent_manifest = load_manifest(data_dir.parent) if data_dir.name in SPLITS else None
    shards = manifest_shards(data_dir.parent, parent_manifest, data_dir.name) if parent_manifest is not None else sorted(data_dir.glob("shard-*.npz"))
    if parent_manifest is not None:
        verify_shard_checksums(data_dir.parent, parent_manifest, shards)
    make_loader = _make_eval_batch_loader if data_format == "evals" else _make_batch_loader
    loader = make_loader(data_dir, batch_size, shards, num_workers, device, shuffle=False)
    objective = _build_objective(
        data_format, target_config or bundle.target_config, bundle.model_config, value_weight
    )
    metrics, _ = _run_epoch(
        bundle.model,
        loader,
        objective,
        device,
        input_channels=bundle.model_config.input_channels,
        use_amp=amp and device.type == "cuda",
        amp_dtype=_amp_dtype(device),
        max_steps=max_steps,
        desc="evaluate",
    )
    return metrics


def evaluate_supervised(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    batch_size: int = 512,
    value_weight: float = 1.0,
    num_workers: int = 2,
    device_name: str | None = None,
    amp: bool = True,
    max_steps: int | None = None,
    allow_legacy_policy: bool = False,
    target_config: TargetConfig | None = None,
) -> dict[str, float]:
    return _evaluate_checkpoint(
        checkpoint_path,
        data_dir,
        data_format="games",
        batch_size=batch_size,
        value_weight=value_weight,
        num_workers=num_workers,
        device_name=device_name,
        amp=amp,
        max_steps=max_steps,
        allow_legacy_policy=allow_legacy_policy,
        target_config=target_config,
    )


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
    allow_legacy_policy: bool = False,
    target_config: TargetConfig | None = None,
) -> dict[str, float]:
    return _evaluate_checkpoint(
        checkpoint_path,
        data_dir,
        data_format="evals",
        batch_size=batch_size,
        value_weight=value_weight,
        num_workers=num_workers,
        device_name=device_name,
        amp=amp,
        max_steps=max_steps,
        allow_legacy_policy=allow_legacy_policy,
        target_config=target_config,
    )
