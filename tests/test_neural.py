import json
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from superchess.encoding import BOARD_CHANNELS, POLICY_PLANES, move_to_policy, pack_board
from superchess.mcts import NeuralMCTS, SearchConfig
from superchess.model import ChessCNNTransformer, ModelConfig
from superchess.training import (
    NPZShardDataset,
    evaluate_supervised,
    train_supervised,
    train_distillation,
    evaluate_distillation,
    load_model_checkpoint,
    _new_policy_accuracy_stats,
    _policy_accuracy_metrics,
    _resolve_device,
    _update_policy_accuracy_stats,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(channels=16, cnn_blocks=1, transformer_layers=1, attention_heads=4)


def test_model_forward_shapes():
    model = ChessCNNTransformer(tiny_config())
    outputs = model(torch.zeros(2, BOARD_CHANNELS, 8, 8))
    assert outputs["policy"].shape == (2, 4672)
    assert outputs["value"].shape == (2,)
    assert torch.all(outputs["value"].abs() <= 1.0)


def test_model_emits_wdl_logits():
    model = ChessCNNTransformer(tiny_config())
    outputs = model(torch.zeros(2, BOARD_CHANNELS, 8, 8))
    assert outputs["wdl"].shape == (2, 3)
    probs = torch.softmax(outputs["wdl"], dim=1)
    expected_value = probs[:, 0] - probs[:, 2]
    assert torch.allclose(outputs["value"], expected_value, atol=1e-5)



def test_policy_logits_use_python_chess_square_order():
    model = ChessCNNTransformer(
        ModelConfig(input_channels=1, channels=1, cnn_blocks=1, transformer_layers=1, attention_heads=1)
    )
    model.stem = torch.nn.Identity()
    model.cnn = torch.nn.Identity()
    model.transformer = torch.nn.Identity()
    model.norm = torch.nn.Identity()
    model.square_embedding.data.zero_()
    model.policy_head = torch.nn.Linear(1, 73, bias=False)
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.weight[0, 0] = 1.0

    boards = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    outputs = model(boards)["policy"][0]

    assert outputs[chess.E2 * POLICY_PLANES] == boards[0, 0, 6, 4]
    assert outputs[chess.A8 * POLICY_PLANES] == boards[0, 0, 0, 0]


def test_shard_dataset_unpacks_packed_boards(tmp_path: Path):
    board = chess.Board()
    np.savez(
        tmp_path / "shard-00000.npz",
        boards=np.stack([pack_board(board)]),
        policies=np.asarray([move_to_policy(board, chess.Move.from_uci("e2e4")).index], dtype=np.uint16),
        values=np.asarray([1.0], dtype=np.float32),
        plies=np.asarray([12], dtype=np.uint16),
    )
    dataset = NPZShardDataset(tmp_path)
    planes, policy, value, ply = dataset[0]
    assert planes.shape == (BOARD_CHANNELS, 8, 8)
    assert planes.dtype == torch.float32
    assert int(policy) == move_to_policy(board, chess.Move.from_uci("e2e4")).index
    assert float(value) == 1.0
    assert int(ply) == 12


def test_policy_accuracy_metrics_include_topk_and_game_phase():
    stats = _new_policy_accuracy_stats()
    policy_logits = torch.tensor(
        [
            [1.0, 2.0, 9.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 4.0, 3.0, 2.0],
            [8.0, 2.0, 7.0, 6.0, 5.0, 9.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0],
        ]
    )
    policies = torch.tensor([2, 0, 1, 6], dtype=torch.long)
    plies = torch.tensor([0, 30, 65, 70], dtype=torch.long)

    _update_policy_accuracy_stats(stats, policy_logits, policies, plies)
    metrics = _policy_accuracy_metrics(stats)

    assert metrics["policy_accuracy_top1"] == 0.5
    assert metrics["policy_accuracy_top5"] == 0.75
    assert metrics["policy_accuracy_top1_early"] == 1.0
    assert metrics["policy_accuracy_top5_early"] == 1.0
    assert metrics["policy_accuracy_top1_mid"] == 0.0
    assert metrics["policy_accuracy_top5_mid"] == 1.0
    assert metrics["policy_accuracy_top1_endgame"] == 0.5
    assert metrics["policy_accuracy_top5_endgame"] == 0.5


def test_train_supervised_uses_batched_shard_loader(tmp_path: Path):
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    np.savez(
        tmp_path / "shard-00000.npz",
        boards=np.stack([pack_board(board), pack_board(board)]),
        policies=np.asarray(
            [move_to_policy(board, move).index, move_to_policy(board, move).index],
            dtype=np.uint16,
        ),
        values=np.asarray([1.0, -1.0], dtype=np.float32),
        plies=np.asarray([4, 64], dtype=np.uint16),
    )

    history = train_supervised(
        tmp_path,
        tmp_path / "checkpoint.pt",
        epochs=1,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
    )

    assert history[0]["policy_accuracy_top1"] >= 0.0
    assert "policy_accuracy_top5_early" in history[0]
    assert "policy_accuracy_top5_endgame" in history[0]
    assert (tmp_path / "checkpoint.pt").exists()


def test_train_supervised_reports_held_out_validation_metrics(tmp_path: Path):
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    policy = move_to_policy(board, move).index
    for shard_index, value in enumerate((1.0, -1.0)):
        np.savez(
            tmp_path / f"shard-{shard_index:05d}.npz",
            boards=np.stack([pack_board(board)]),
            policies=np.asarray([policy], dtype=np.uint16),
            values=np.asarray([value], dtype=np.float32),
            plies=np.asarray([shard_index], dtype=np.uint16),
        )

    checkpoint_path = tmp_path / "checkpoint.pt"
    history = train_supervised(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=1,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
        validation_fraction=0.5,
        validation_seed=7,
    )

    assert history[0]["val_loss"] >= 0.0
    assert "val_policy_accuracy_top5" in history[0]
    checkpoint_metadata = json.loads(checkpoint_path.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert checkpoint_metadata["history"][0]["val_loss"] == history[0]["val_loss"]
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert saved_checkpoint["policy_square_order"] == "python-chess"
    assert saved_checkpoint["data_format"] == "games"


def test_train_supervised_errors_when_no_batches_are_processed(tmp_path: Path):
    board = chess.Board()
    np.savez(
        tmp_path / "shard-00000.npz",
        boards=np.empty((0, pack_board(board).shape[0]), dtype=np.uint8),
        policies=np.asarray([], dtype=np.uint16),
        values=np.asarray([], dtype=np.float32),
        plies=np.asarray([], dtype=np.uint16),
    )

    with pytest.raises(RuntimeError, match="epoch 1/1 did not process any batches"):
        train_supervised(
            tmp_path,
            tmp_path / "checkpoint.pt",
            epochs=1,
            batch_size=2,
            num_workers=0,
            device_name="cpu",
            model_config=tiny_config(),
            validation_fraction=0.0,
        )


def test_evaluate_supervised_reports_checkpoint_metrics(tmp_path: Path):
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    np.savez(
        tmp_path / "shard-00000.npz",
        boards=np.stack([pack_board(board), pack_board(board)]),
        policies=np.asarray(
            [move_to_policy(board, move).index, move_to_policy(board, move).index],
            dtype=np.uint16,
        ),
        values=np.asarray([1.0, -1.0], dtype=np.float32),
        plies=np.asarray([4, 64], dtype=np.uint16),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_supervised(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
    )

    metrics = evaluate_supervised(
        checkpoint_path,
        tmp_path,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
        max_steps=1,
    )

    assert metrics["loss"] >= 0.0
    assert metrics["policy_loss"] >= 0.0
    assert metrics["value_loss"] >= 0.0
    assert "policy_accuracy_top5_endgame" in metrics


def _write_eval_shard(directory: Path, shard_index: int) -> None:
    board = chess.Board()
    e4 = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    d4 = move_to_policy(board, chess.Move.from_uci("d2d4")).index
    np.savez(
        directory / f"shard-{shard_index:05d}.npz",
        boards=np.stack([pack_board(board)]),
        wdl=np.asarray([[0.6, 0.3, 0.1]], dtype=np.float32),
        values=np.asarray([0.5], dtype=np.float32),
        policy_indices=np.asarray([[e4, d4, -1, -1]], dtype=np.int32),
        policy_probs=np.asarray([[0.7, 0.3, 0.0, 0.0]], dtype=np.float32),
    )


def test_train_distillation_learns_from_eval_shards(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    history = train_distillation(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=1,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
        validation_fraction=0.0,
    )

    assert history[0]["loss"] >= 0.0
    assert history[0]["policy_loss"] >= 0.0
    assert history[0]["value_loss"] >= 0.0
    assert "policy_accuracy_top1" in history[0]
    assert checkpoint_path.exists()
    saved = torch.load(checkpoint_path, map_location="cpu")
    assert saved["data_format"] == "evals"
    assert saved["policy_square_order"] == "python-chess"


def test_evaluate_distillation_reports_metrics(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_distillation(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=1,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
        validation_fraction=0.0,
    )

    metrics = evaluate_distillation(
        checkpoint_path,
        tmp_path,
        batch_size=1,
        num_workers=0,
        device_name="cpu",
        max_steps=1,
    )

    assert metrics["loss"] >= 0.0
    assert metrics["policy_loss"] >= 0.0
    assert metrics["value_loss"] >= 0.0
    assert "policy_accuracy_top5" in metrics


def test_load_model_checkpoint_rejects_legacy_policy_metadata(tmp_path: Path):
    config = tiny_config()
    model = ChessCNNTransformer(config)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(config),
            "policy_size": POLICY_PLANES * 64,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="policy square-order"):
        load_model_checkpoint(checkpoint_path, device_name="cpu")

    with pytest.warns(RuntimeWarning, match="policy square-order"):
        loaded, loaded_config = load_model_checkpoint(
            checkpoint_path,
            device_name="cpu",
            allow_legacy_policy=True,
        )
    assert isinstance(loaded, ChessCNNTransformer)
    assert loaded_config == config


def test_resolve_device_falls_back_when_cuda_kernels_are_unusable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def raise_unusable(_device):
        raise RuntimeError("unsupported GPU")

    monkeypatch.setattr("superchess.training._check_cuda_device", raise_unusable)

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        device = _resolve_device()

    assert device.type == "cpu"


def test_mcts_returns_legal_move_from_tiny_model():
    board = chess.Board()
    model = ChessCNNTransformer(tiny_config())
    result = NeuralMCTS(model, SearchConfig(simulations=5, evaluation_batch_size=3)).search(board)
    assert result.best_move in board.legal_moves
    assert set(result.visits).issubset(set(board.legal_moves))
    assert sum(result.visits.values()) == 5


def test_mcts_batched_evaluate_returns_each_board():
    boards = [chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")]
    model = ChessCNNTransformer(tiny_config())
    searcher = NeuralMCTS(model, SearchConfig(simulations=1, evaluation_batch_size=2))

    evaluations = searcher.evaluate_batch(boards)

    assert len(evaluations) == 2
    for board, (policy, value) in zip(boards, evaluations, strict=True):
        assert set(policy).issubset(set(board.legal_moves))
        assert sum(policy.values()) == pytest.approx(1.0)
        assert -1.0 <= value <= 1.0