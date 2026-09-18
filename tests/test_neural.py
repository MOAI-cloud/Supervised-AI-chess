import json
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from superchess.encoding import (
    BOARD_CHANNELS,
    LEGACY_BOARD_CHANNELS,
    POLICY_PLANES,
    encode_board,
    legal_policy_indices,
    move_to_policy,
    pack_board,
)
from superchess.mcts import MCTSNode, NeuralMCTS, SearchConfig
from superchess.model import MASKED_LOGIT, ChessCNNTransformer, ModelConfig, model_config_from_checkpoint
from superchess.targets import TargetConfig, mate_score
from superchess.training import (
    ModelEMA,
    NPZShardDataset,
    check_input_planes,
    clock_planes_informative,
    evaluate_supervised,
    train_supervised,
    train_distillation,
    evaluate_distillation,
    load_checkpoint_bundle,
    load_model_checkpoint,
    _new_policy_accuracy_stats,
    _policy_accuracy_metrics,
    _resolve_device,
    _update_policy_accuracy_stats,
)


def tiny_config(**overrides) -> ModelConfig:
    fields = dict(channels=16, cnn_blocks=1, transformer_layers=1, attention_heads=4, smolgen_gen=8, value_bins=16)
    fields.update(overrides)
    return ModelConfig(**fields)


def test_model_forward_shapes():
    model = ChessCNNTransformer(tiny_config())
    outputs = model(torch.zeros(2, LEGACY_BOARD_CHANNELS, 8, 8))
    assert outputs["policy"].shape == (2, 4672)
    assert outputs["value_logits"].shape == (2, 16)
    assert outputs["value"].shape == (2,)
    assert torch.all(outputs["value"].abs() <= 1.0)


def test_model_value_is_expected_bin_mean():
    model = ChessCNNTransformer(tiny_config())
    boards = torch.from_numpy(encode_board(chess.Board())[:LEGACY_BOARD_CHANNELS]).unsqueeze(0)
    outputs = model(boards)
    probs = torch.softmax(outputs["value_logits"].float(), dim=1)
    centers = (torch.arange(16) + 0.5) / 16
    assert torch.allclose(outputs["value"], 2 * (probs * centers).sum(dim=1) - 1, atol=1e-5)


def test_model_masks_impossible_moves_and_keeps_legal_ones():
    model = ChessCNNTransformer(tiny_config()).eval()
    for fen in (
        chess.STARTING_FEN,
        "r3k2r/pP2bppp/2n1pn2/q2p4/3P4/2NBBN2/PPQ2PpP/R3K2R b KQkq - 0 1",
    ):
        board = chess.Board(fen)
        planes = torch.from_numpy(encode_board(board)[:LEGACY_BOARD_CHANNELS]).unsqueeze(0)
        policy = model(planes)["policy"][0]
        legal = legal_policy_indices(board)
        assert all(policy[index] > MASKED_LOGIT for index in legal.values())
        assert int((policy > MASKED_LOGIT).sum()) < 200
        assert torch.isfinite(policy).all()


def test_attention_policy_head_uses_python_chess_square_order():
    """The from/to bilinear logit must land on the dataset index of the same move."""
    config = ModelConfig(channels=8, cnn_blocks=0, transformer_layers=0, attention_heads=1, smolgen_hidden=0, structural_mask=False)
    model = ChessCNNTransformer(config).eval()
    head = model.policy_head
    board = chess.Board()
    with torch.no_grad():
        # Query/key so that logit(from, to) = 100 * [from == e2] * [to == e4] (square order).
        for linear in (head.query, head.key):
            linear.weight.zero_()
            linear.bias.zero_()
    tokens = torch.zeros(1, 64, 8)
    tokens[0, chess.E2, 0] = 1.0
    tokens[0, chess.E4, 1] = 1.0
    with torch.no_grad():
        head.query.weight[0, 0] = 10.0
        head.key.weight[0, 1] = 10.0
    logits = head(tokens)[0]
    index = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    assert logits[index] > 30
    assert (logits > 30).sum() == 1
    # Underpromotions share the pair logit plus a learned offset for the promotion piece.
    promo_board = chess.Board("8/1P6/8/8/8/8/8/k6K w - - 0 1")
    queen = move_to_policy(promo_board, chess.Move.from_uci("b7b8q")).index
    knight = move_to_policy(promo_board, chess.Move.from_uci("b7b8n")).index
    assert int(head.pair_index[queen]) == int(head.pair_index[knight]) == chess.B7 * 64 + chess.B8
    assert bool(head.promo_mask[knight]) and not bool(head.promo_mask[queen])


def test_legacy_model_config_reconstructs_old_architecture():
    legacy = model_config_from_checkpoint({"input_channels": 18, "channels": 256, "cnn_blocks": 6, "transformer_layers": 10, "attention_heads": 8, "mlp_ratio": 4, "dropout": 0.0, "attention_bias": True})
    assert legacy.policy_head == "planes" and legacy.value_head == "wdl" and not legacy.uses_smolgen
    model = ChessCNNTransformer(ModelConfig(channels=16, cnn_blocks=1, transformer_layers=1, attention_heads=4, policy_head="planes", value_head="wdl", smolgen_hidden=0))
    outputs = model(torch.zeros(1, 18, 8, 8))
    assert outputs["value_logits"].shape == (1, 3)
    assert set(outputs) == {"policy", "value_logits", "value"}


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


def test_clock_plane_guard_rejects_constant_clocks(tmp_path: Path):
    board = chess.Board()
    shard = tmp_path / "shard-00000.npz"
    np.savez(shard, boards=np.stack([pack_board(board)] * 4), policies=np.zeros(4, np.uint16), values=np.zeros(4, np.float32))
    assert clock_planes_informative([shard]) is False
    with pytest.raises(ValueError, match="constant in this dataset"):
        check_input_planes([shard], BOARD_CHANNELS)
    check_input_planes([shard], LEGACY_BOARD_CHANNELS)  # clock-free models are fine

    varied = chess.Board()
    varied.push_san("Nf3")
    varied.push_san("Nf6")
    varied.push_san("Ng1")
    shard2 = tmp_path / "shard-00001.npz"
    np.savez(shard2, boards=np.stack([pack_board(board), pack_board(varied)]), policies=np.zeros(2, np.uint16), values=np.zeros(2, np.float32))
    assert clock_planes_informative([shard2]) is True
    check_input_planes([shard2], BOARD_CHANNELS)


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
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved_checkpoint["policy_square_order"] == "python-chess"
    assert saved_checkpoint["data_format"] == "games"
    assert saved_checkpoint["weights"] == "ema"
    assert saved_checkpoint["target_config"] == TargetConfig().to_dict()
    assert "training" in saved_checkpoint
    bundle = load_checkpoint_bundle(checkpoint_path, device_name="cpu")
    assert bundle.model_config == tiny_config()
    assert bundle.data_format == "games"


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


def _write_eval_shard(directory: Path, shard_index: int, *, legacy: bool = False) -> None:
    board = chess.Board()
    e4 = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    d4 = move_to_policy(board, chess.Move.from_uci("d2d4")).index
    if legacy:
        np.savez(
            directory / f"shard-{shard_index:05d}.npz",
            boards=np.stack([pack_board(board)]),
            wdl=np.asarray([[0.6, 0.3, 0.1]], dtype=np.float32),
            values=np.asarray([0.5], dtype=np.float32),
            policy_indices=np.asarray([[e4, d4, -1, -1]], dtype=np.int32),
            policy_probs=np.asarray([[0.7, 0.3, 0.0, 0.0]], dtype=np.float32),
        )
        return
    np.savez(
        directory / f"shard-{shard_index:05d}.npz",
        boards=np.stack([pack_board(board), pack_board(board)]),
        score=np.asarray([35, mate_score(2)], dtype=np.int16),
        depth=np.asarray([20, 30], dtype=np.uint8),
        policy_indices=np.asarray([[e4, d4, -1, -1], [d4, -1, -1, -1]], dtype=np.int16),
        policy_scores=np.asarray([[35, 10, 0, 0], [mate_score(2), 0, 0, 0]], dtype=np.int16),
        policy_depth=np.asarray([20, 30], dtype=np.uint8),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_train_distillation_learns_from_eval_shards(tmp_path: Path, legacy: bool):
    _write_eval_shard(tmp_path, 0, legacy=legacy)

    checkpoint_path = tmp_path / "checkpoint.pt"
    history = train_distillation(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
        validation_fraction=0.0,
    )

    assert history[0]["loss"] >= 0.0
    assert history[0]["policy_loss"] >= 0.0
    assert history[0]["value_loss"] >= 0.0
    assert history[0]["policy_kl"] >= -1e-4
    assert history[0]["target_entropy"] >= 0.0
    assert "policy_accuracy_top1" in history[0]
    assert checkpoint_path.exists()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["data_format"] == "evals"
    assert saved["eval_shard_format"] == ("evals-v1" if legacy else "evals-v2")
    assert saved["policy_square_order"] == "python-chess"


def test_train_distillation_rejects_clock_planes_on_eval_data(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    with pytest.raises(ValueError, match="constant in this dataset"):
        train_distillation(
            tmp_path,
            tmp_path / "checkpoint.pt",
            epochs=1,
            batch_size=2,
            num_workers=0,
            device_name="cpu",
            model_config=tiny_config(input_channels=BOARD_CHANNELS),
            max_steps=1,
            validation_fraction=0.0,
        )


def test_train_distillation_resumes_from_training_state(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    common = dict(batch_size=2, num_workers=0, device_name="cpu", model_config=tiny_config(), max_steps=1, validation_fraction=0.0)
    first = train_distillation(tmp_path, checkpoint_path, epochs=1, **common)
    assert len(first) == 1
    resumed_path = tmp_path / "resumed.pt"
    second = train_distillation(tmp_path, resumed_path, epochs=2, resume=checkpoint_path, **common)
    assert len(second) == 2
    assert second[0] == first[0]
    saved = torch.load(resumed_path, map_location="cpu", weights_only=False)
    assert saved["epochs_completed"] == 2
    assert saved["training"]["step"] == 2


def _write_frozen_eval_dataset(tmp_path: Path, *, seed: int = 0) -> Path:
    from superchess.evals import EvalConfig, preprocess_eval_files

    records = []
    for move in chess.Board().legal_moves:
        board = chess.Board()
        board.push(move)
        records.append({"fen": board.fen(), "evals": [{"depth": 20, "knodes": 100,
                        "pvs": [{"cp": 20, "line": next(iter(board.legal_moves)).uci()}]}]})
    source = tmp_path / "raw.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    output = tmp_path / f"frozen-{seed}"
    preprocess_eval_files([source], output, EvalConfig(shard_size=4, validation_fraction=0.2, test_fraction=0.2, split_seed=seed))
    return output


def test_frozen_dataset_training_preserves_split_and_provenance(tmp_path: Path):
    from superchess.data_quality import load_manifest, manifest_shards
    from superchess.training import _split_train_validation_shards

    data = _write_frozen_eval_dataset(tmp_path)
    manifest = load_manifest(data)
    train, validation = _split_train_validation_shards(data, 0.0, 999)
    assert train == manifest_shards(data, manifest, "train")
    assert validation == manifest_shards(data, manifest, "validation")
    assert validation and manifest_shards(data, manifest, "test")
    assert not set(train + validation) & set(manifest_shards(data, manifest, "test"))
    with pytest.raises(ValueError, match="dataset root"):
        _split_train_validation_shards(data / "test", 0.0, 0)

    checkpoint = tmp_path / "frozen.pt"
    train_distillation(data, checkpoint, batch_size=2, epochs=1, max_steps=1, num_workers=0,
                       device_name="cpu", model_config=tiny_config())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["dataset_id"] == manifest["dataset_id"]
    assert saved["validation_fraction"] == 0.2
    assert saved["validation_shards"] == [str(path.relative_to(data)) for path in validation]
    assert saved["dataset_provenance"]["split_method"] == manifest["split_method"]
    sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text())
    assert sidecar["dataset_id"] == manifest["dataset_id"]
    metrics = evaluate_distillation(checkpoint, data / "test", batch_size=2, num_workers=0, device_name="cpu", max_steps=1)
    assert metrics["loss"] >= 0
    with pytest.raises(ValueError, match="explicit validation or test"):
        evaluate_distillation(checkpoint, data, num_workers=0, device_name="cpu")

    other = _write_frozen_eval_dataset(tmp_path, seed=1)
    with pytest.raises(ValueError, match="dataset identity"):
        train_distillation(other, tmp_path / "wrong.pt", batch_size=2, epochs=2, max_steps=1, num_workers=0,
                           device_name="cpu", model_config=tiny_config(), resume=checkpoint)


def test_resume_rejects_changed_target_semantics(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    checkpoint = tmp_path / "original.pt"
    common = dict(batch_size=2, num_workers=0, device_name="cpu", model_config=tiny_config(), max_steps=1, validation_fraction=0.0)
    train_distillation(tmp_path, checkpoint, epochs=1, **common)
    with pytest.raises(ValueError, match="target config"):
        train_distillation(tmp_path, tmp_path / "changed.pt", epochs=2, resume=checkpoint,
                           target_config=TargetConfig(policy_temperature=0.2), **common)


def test_training_rejects_same_size_frozen_shard_corruption(tmp_path: Path):
    data = _write_frozen_eval_dataset(tmp_path)
    shard = next((data / "train").glob("*.npz"))
    size = shard.stat().st_size
    with shard.open("r+b") as handle:
        handle.seek(-10, 2)
        handle.write(b"corruption")
    assert shard.stat().st_size == size
    with pytest.raises(ValueError, match="checksum mismatch"):
        train_distillation(data, tmp_path / "bad.pt", epochs=1, max_steps=1, num_workers=0,
                           device_name="cpu", model_config=tiny_config())


def test_eval_batch_order_is_seeded_by_epoch(tmp_path: Path):
    from superchess.training import EvalShardBatchDataset

    board = chess.Board()
    policy = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    np.savez(tmp_path / "shard-00000.npz", boards=np.stack([pack_board(board)] * 32),
             score=np.arange(32, dtype=np.int16), policy_indices=np.full((32, 1), policy, np.int16),
             policy_scores=np.arange(32, dtype=np.int16).reshape(-1, 1))
    first = EvalShardBatchDataset(tmp_path, batch_size=4)
    first.seed = 17
    second = EvalShardBatchDataset(tmp_path, batch_size=4)
    second.seed = 17

    def order(dataset):
        return [int(score) for _, scores, _, _ in dataset for score in scores]

    initial = order(first)
    assert initial == order(second)
    next_epoch = order(first)
    assert next_epoch != initial
    resumed = EvalShardBatchDataset(tmp_path, batch_size=4)
    resumed.seed, resumed.epoch = 17, 1
    assert order(resumed) == next_epoch


def test_training_seed_reproduces_initialization_and_saves_rng(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    common = dict(batch_size=2, epochs=1, max_steps=1, num_workers=0, device_name="cpu",
                  model_config=tiny_config(), validation_fraction=0, seed=23)
    first = train_distillation(tmp_path, tmp_path / "first.pt", **common)
    second = train_distillation(tmp_path, tmp_path / "second.pt", **common)
    assert first == second
    saved = torch.load(tmp_path / "first.pt", map_location="cpu", weights_only=False)
    assert saved["training_seed"] == 23
    assert "torch_rng" in saved["training"] and "scaler" in saved["training"]


def test_model_ema_tracks_parameters_and_copies_buffers():
    model = ChessCNNTransformer(tiny_config())
    ema = ModelEMA(model, decay=0.5)
    initial = model.square_embedding.detach().clone()
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
        model.stem[1].running_mean.fill_(3.0)
    ema.update(model)
    # The first update uses the warm-up decay min(0.5, (1 + 1) / (10 + 1)).
    decay = 2.0 / 11.0
    expected = decay * initial + (1.0 - decay) * torch.ones_like(initial)
    assert torch.allclose(ema.module.square_embedding, expected, atol=1e-6)
    assert ema.module.stem[1].running_mean[0].item() == pytest.approx(3.0)
    assert not any(param.requires_grad for param in ema.module.parameters())
    state = ema.state_dict()
    restored = ModelEMA(ChessCNNTransformer(tiny_config()), decay=0.5)
    restored.load_state_dict(state)
    assert restored.updates == 1
    assert torch.equal(restored.module.square_embedding, ema.module.square_embedding)


def test_evaluate_distillation_reports_metrics(tmp_path: Path):
    _write_eval_shard(tmp_path, 0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_distillation(
        tmp_path,
        checkpoint_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
        model_config=tiny_config(),
        max_steps=1,
        validation_fraction=0.0,
    )

    metrics = evaluate_distillation(
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
    assert 0.0 <= metrics["value_mae"] <= 2.0
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


def test_load_model_checkpoint_reads_pre_v2_architecture(tmp_path: Path):
    """Checkpoints from before the attention/categorical heads keep loading unchanged."""
    legacy_config = ModelConfig(
        channels=16, cnn_blocks=1, transformer_layers=1, attention_heads=4, policy_head="planes", value_head="wdl", smolgen_hidden=0, structural_mask=False
    )
    model = ChessCNNTransformer(legacy_config)
    old_fields = {
        "input_channels": 18, "channels": 16, "cnn_blocks": 1, "transformer_layers": 1, "attention_heads": 4,
        "mlp_ratio": 4, "dropout": 0.0, "attention_bias": True,
    }
    checkpoint_path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "model_config": old_fields, "policy_square_order": "python-chess", "data_format": "evals"}, checkpoint_path)
    bundle = load_checkpoint_bundle(checkpoint_path, device_name="cpu")
    assert bundle.model_config == legacy_config
    assert bundle.target_config == TargetConfig()
    outputs = bundle.model(torch.zeros(1, 18, 8, 8))
    assert outputs["value_logits"].shape == (1, 3)


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
    assert result.stats is not None
    assert result.stats.simulations == 5
    assert result.stats.network_evaluations == 6
    assert 1 <= result.stats.network_batches <= 6
    assert result.stats.elapsed_seconds > 0


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("root_selection", ["puct", "sequential_halving", "value_rescue"])
def test_mcts_batch_does_not_spend_simulations_on_duplicate_pending_edges(batch_size, root_selection):
    board = chess.Board("7k/8/5K2/8/8/8/8/6R1 b - - 0 1")
    assert list(board.legal_moves) == [chess.Move.from_uci("h8h7")]
    model = ChessCNNTransformer(tiny_config())
    result = NeuralMCTS(model, SearchConfig(simulations=4, evaluation_batch_size=batch_size, root_selection=root_selection)).search(board)

    assert sum(result.visits.values()) == 4
    assert result.root is not None
    child = result.root.children[0]
    assert child is not None
    assert child.visit_count == 3


def test_mcts_failed_inference_releases_pending_visits(monkeypatch):
    board = chess.Board()
    moves = list(board.legal_moves)
    root = MCTSNode(moves, np.full(len(moves), 1.0 / len(moves), dtype=np.float32))
    root.visit_counts[0] = 2.0
    root.value_sums[0] = 0.5
    original_counts = root.visit_counts.copy()
    original_values = root.value_sums.copy()
    searcher = NeuralMCTS(ChessCNNTransformer(tiny_config()), SearchConfig(simulations=4, evaluation_batch_size=4))

    def fail_inference(_boards):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(searcher, "_run_model", fail_inference)
    with pytest.raises(RuntimeError, match="inference failed"):
        searcher.search(board, tree=root)

    np.testing.assert_array_equal(root.visit_counts, original_counts)
    np.testing.assert_array_equal(root.value_sums, original_values)
    assert all(child is None for child in root.children)


def test_mcts_halving_uses_searched_value_to_break_equal_visit_counts(monkeypatch):
    board = chess.Board()
    preferred = chess.Move.from_uci("e2e4")
    better = chess.Move.from_uci("d2d4")
    moves = list(board.legal_moves)
    priors = np.zeros(len(moves), dtype=np.float32)
    priors[moves.index(preferred)] = 0.8
    priors[moves.index(better)] = 0.2
    root = MCTSNode(moves, priors)
    searcher = NeuralMCTS(
        ChessCNNTransformer(tiny_config()),
        SearchConfig(simulations=2, evaluation_batch_size=2, root_selection="sequential_halving", root_candidates=2),
    )

    def evaluate(boards):
        results = []
        for position in boards:
            legal = list(position.legal_moves)
            value = -0.8 if position.peek() == better else 0.8
            results.append((legal, np.full(len(legal), 1.0 / len(legal), dtype=np.float32), value))
        return results

    monkeypatch.setattr(searcher, "_run_model", evaluate)
    result = searcher.search(board, tree=root)
    assert result.visits[preferred] == result.visits[better] == 1
    assert result.best_move == better
    assert root.child_q(root.move_index(better)) == pytest.approx(0.8)
    assert result.policy[better] == 1.0


@pytest.mark.parametrize("simulations", [0, 1, 2, 3, 7, 19])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("root_selection", ["sequential_halving", "value_rescue"])
def test_mcts_halving_honors_small_and_odd_budgets(simulations, batch_size, root_selection):
    board = chess.Board()
    searcher = NeuralMCTS(
        ChessCNNTransformer(tiny_config()),
        SearchConfig(simulations=simulations, evaluation_batch_size=batch_size, root_selection=root_selection, root_candidates=5),
    )
    result = searcher.search(board)
    assert sum(result.visits.values()) == simulations
    assert result.best_move in board.legal_moves
    assert sum(result.policy.values()) == pytest.approx(1.0)
    if simulations:
        assert result.visits[result.best_move] > 0


def test_mcts_halving_reuse_spends_a_new_budget_and_reports_cache_hits():
    board = chess.Board()
    searcher = NeuralMCTS(
        ChessCNNTransformer(tiny_config()),
        SearchConfig(simulations=7, evaluation_batch_size=4, root_selection="sequential_halving"),
    )
    first = searcher.search(board)
    second = searcher.search(board, tree=first.root)
    assert sum(second.visits.values()) == 14
    assert second.stats is not None and second.stats.simulations == 7
    assert second.stats.max_depth >= 2
    cached = searcher.search(board)
    assert cached.stats is not None and cached.stats.cache_hits >= 1
    assert cached.stats.network_evaluations == 0


@pytest.mark.parametrize("network_value,rescues", [(0.8, True), (-0.8, False)])
def test_mcts_value_rescue_reopens_policy_tail_only_on_value_disagreement(monkeypatch, network_value, rescues):
    board = chess.Board()
    preferred, second, overlooked = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "c2c4")]
    moves = [preferred, second, overlooked]
    root = MCTSNode(moves, np.asarray([0.55, 0.35, 0.1], dtype=np.float32), network_value=network_value)
    for move, value in zip(moves, [-0.8, -0.6, 0.8], strict=True):
        root.children[root.move_index(move)] = MCTSNode.terminal(-value)
    searcher = NeuralMCTS(
        ChessCNNTransformer(tiny_config()),
        SearchConfig(simulations=8, evaluation_batch_size=8, root_selection="value_rescue", root_candidates=2),
    )

    def unexpected_inference(_boards):
        pytest.fail("the constructed terminal tree must not request neural inference")

    monkeypatch.setattr(searcher, "_run_model", unexpected_inference)
    result = searcher.search(board, tree=root)
    assert sum(result.visits.values()) == 8
    assert result.stats is not None
    assert result.stats.root_rescues == int(rescues)
    assert result.stats.root_rescue_selected == int(rescues)
    assert (result.visits[overlooked] > 0) == rescues
    assert (result.best_move == overlooked) == rescues


@pytest.mark.parametrize("margin", [2.0, 3.0])
def test_mcts_value_rescue_control_disables_gate_even_at_extreme_values(margin):
    from superchess.mcts import _RootHalving

    root = MCTSNode(list(chess.Board().legal_moves), np.full(20, 0.05, dtype=np.float32), network_value=1.0)
    selector = _RootHalving(root, 16, SearchConfig(root_selection="value_rescue", root_candidates=2, root_rescue_margin=margin))
    selector.begin_round(16)
    assert selector.remaining == 2
    for index in selector.candidates:
        root.visit_counts[index] = 1
        root.value_sums[index] = -1
    selector.begin_round(14)
    assert selector.rescued == 0


@pytest.mark.parametrize("root_selection", ["puct", "sequential_halving", "value_rescue"])
def test_mcts_zero_budget_uses_policy_instead_of_move_generation_order(root_selection):
    board = chess.Board()
    moves = list(board.legal_moves)
    priors = np.zeros(len(moves), dtype=np.float32)
    best = chess.Move.from_uci("e2e4")
    priors[moves.index(best)] = 1.0
    root = MCTSNode(moves, priors)
    result = NeuralMCTS(
        ChessCNNTransformer(tiny_config()), SearchConfig(simulations=0, root_selection=root_selection)
    ).search(board, tree=root)
    assert result.best_move == best
    assert sum(result.visits.values()) == 0


def test_mcts_policy_softmax_temperature_flattens_priors():
    board = chess.Board()
    model = ChessCNNTransformer(tiny_config())
    with torch.no_grad():  # make the head opinionated so the temperature is measurable
        model.policy_head.query.weight.mul_(50.0)
    sharp, _ = NeuralMCTS(model, SearchConfig(policy_softmax_temperature=1.0)).evaluate(board)
    flat, _ = NeuralMCTS(model, SearchConfig(policy_softmax_temperature=4.0)).evaluate(board)
    assert max(sharp.values()) > max(flat.values())
    assert sum(flat.values()) == pytest.approx(1.0)


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