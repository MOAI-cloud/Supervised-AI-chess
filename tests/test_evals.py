from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from superchess.encoding import POLICY_SIZE, move_to_policy
from superchess.evals import (
    EVAL_SHARD_FORMAT,
    EvalConfig,
    eval_score,
    parse_first_move,
    preprocess_eval_files,
    sample_from_eval_record,
    white_cp_to_stm,
)
from superchess.targets import MATE_SCORE, SCORE_CLAMP, mate_score

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def _record(fen: str, pvs: list[dict], depth: int = 20) -> dict:
    return {"fen": fen, "evals": [{"depth": depth, "knodes": 1000, "pvs": pvs}]}


def test_white_cp_negated_for_black_to_move() -> None:
    assert white_cp_to_stm(120.0, chess.WHITE, white_relative=True) == 120.0
    assert white_cp_to_stm(120.0, chess.BLACK, white_relative=True) == -120.0


def test_eval_score_encodes_mates_above_every_cp() -> None:
    assert eval_score(None, 3, chess.WHITE) > SCORE_CLAMP
    assert eval_score(None, -3, chess.WHITE) < -SCORE_CLAMP
    assert eval_score(None, 3, chess.BLACK) == -eval_score(None, 3, chess.WHITE)
    assert eval_score(None, 1, chess.WHITE) > eval_score(None, 5, chess.WHITE) >= MATE_SCORE
    assert eval_score(50_000, None, chess.WHITE) == SCORE_CLAMP
    assert eval_score(-35, None, chess.BLACK) == 35
    assert mate_score(-2) == -mate_score(2)


def test_parse_chess960_castle_normalises() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    move = parse_first_move(board, "e1h1")
    assert move == chess.Move.from_uci("e1g1")


@pytest.mark.parametrize("uci", ["e1f3", "e1h8", "e1e3", "e1b1"])
def test_illegal_king_move_is_not_reinterpreted_as_castling(uci: str) -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    with pytest.raises(ValueError):
        parse_first_move(board, uci)


def test_sample_from_eval_record_keeps_raw_scores() -> None:
    record = _record(
        START_FEN,
        [
            {"cp": 30, "line": "e2e4 e7e5"},
            {"cp": 20, "line": "d2d4 d7d5"},
            {"cp": 20, "line": "d2d4 g8f6"},  # duplicate first move is dropped
        ],
    )
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None
    board = chess.Board(f"{START_FEN} 0 1")
    e4_index = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    d4_index = move_to_policy(board, chess.Move.from_uci("d2d4")).index
    assert sample.policy_indices == [e4_index, d4_index]
    assert sample.policy_scores == [30, 20]
    assert sample.score == 30
    assert sample.depth == 20


def test_black_to_move_score_flips_sign() -> None:
    black_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"
    white = sample_from_eval_record(_record(START_FEN, [{"cp": 200, "line": "e2e4"}]), EvalConfig())
    black = sample_from_eval_record(_record(black_fen, [{"cp": 200, "line": "e7e5"}]), EvalConfig())
    assert white is not None and black is not None
    # +200 cp (White-relative) is good for the side to move when White, bad when Black.
    assert white.score == 200
    assert black.score == -200
    assert black.policy_scores == [-200]


def test_policy_does_not_override_deep_evidence_with_wide_shallow_disagreement() -> None:
    record = {
        "fen": START_FEN,
        "evals": [
            {"depth": 30, "knodes": 1, "pvs": [{"cp": 25, "line": "e2e4"}]},
            {"depth": 18, "knodes": 1, "pvs": [{"cp": 31, "line": "d2d4"}, {"cp": 28, "line": "e2e4"}]},
        ],
    }
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None
    assert sample.score == 25 and sample.depth == 30
    assert sample.policy_scores == [25] and sample.policy_depth == 30


def test_policy_uses_wider_snapshot_only_with_depth_and_best_move_agreement() -> None:
    record = {"fen": START_FEN, "evals": [
        {"depth": 30, "knodes": 2000, "pvs": [{"cp": 25, "line": "e2e4"}]},
        {"depth": 28, "knodes": 1000, "pvs": [{"cp": 30, "line": "e2e4"}, {"cp": 20, "line": "d2d4"}]},
    ]}
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None
    assert sample.score == 25 and sample.policy_scores == [30, 20]
    assert sample.value_knodes == 2000 and sample.policy_knodes == 1000
    strict = sample_from_eval_record(record, EvalConfig(policy_max_depth_gap=0))
    assert strict is not None and strict.policy_scores == [25]


def test_policy_rejects_large_score_disagreement_despite_best_move_agreement() -> None:
    record = {"fen": START_FEN, "evals": [
        {"depth": 30, "pvs": [{"cp": 300, "line": "e2e4"}]},
        {"depth": 28, "pvs": [{"cp": -300, "line": "e2e4"}, {"cp": -400, "line": "d2d4"}]},
    ]}
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None and sample.policy_depth == 30 and sample.policy_scores == [300]


@pytest.mark.parametrize("pv", [
    {"cp": 20, "mate": 3, "line": "e2e4"},
    {"cp": True, "line": "e2e4"},
    {"cp": "30", "line": "e2e4"},
    {"cp": float("inf"), "line": "e2e4"},
    {"mate": 0, "line": "e2e4"},
    {"cp": 30, "lowerbound": True, "line": "e2e4"},
    {"cp": 30, "line": "e2e5"},
    {"cp": 30, "line": "e2e4 e7e4"},
    {"cp": 30, "line": None},
    None,
])
def test_invalid_best_pv_cannot_supply_value_target(pv) -> None:
    assert sample_from_eval_record(_record(START_FEN, [pv, {"cp": 20, "line": "d2d4"}]), EvalConfig()) is None


@pytest.mark.parametrize("record", [None, [], {"fen": 4, "evals": []}, {"fen": START_FEN, "evals": [None]},
    _record("8/8/8/8/8/8/8/8 w - -", [{"cp": 20, "line": "e2e4"}]),
    _record(START_FEN, [{"cp": 20, "line": "e2e4"}], depth="20"),
])
def test_malformed_record_is_skipped_without_crashing(record) -> None:
    assert sample_from_eval_record(record, EvalConfig()) is None


def test_duplicate_policy_slots_do_not_displace_distinct_moves() -> None:
    sample = sample_from_eval_record(_record(START_FEN, [
        {"cp": 30, "line": "e2e4"}, {"cp": 30, "line": "e2e4"}, {"cp": 20, "line": "d2d4"},
    ]), EvalConfig(max_policy_targets=2))
    assert sample is not None and sample.policy_scores == [30, 20]


def test_min_depth_filters_shallow_evals() -> None:
    record = _record(START_FEN, [{"cp": 30, "line": "e2e4"}], depth=5)
    assert sample_from_eval_record(record, EvalConfig(min_depth=12)) is None


@pytest.mark.parametrize("workers", [1, 2])
def test_preprocess_eval_files_writes_v2_shards(tmp_path: Path, workers: int) -> None:
    records = [
        _record(START_FEN, [{"cp": 30, "line": "e2e4 e7e5"}, {"cp": 10, "line": "d2d4 d7d5"}]),
        _record(START_FEN, [{"mate": 5, "line": "e2e4"}]),
        {"fen": START_FEN, "evals": []},
    ]
    source = tmp_path / "raw" / "evals.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    out_dir = tmp_path / f"processed{workers}"
    config = EvalConfig(shard_size=8, max_policy_targets=4, validation_fraction=0, test_fraction=0, deduplicate=False)
    stats = preprocess_eval_files([source], out_dir, config, workers=workers)

    assert stats.lines_seen == 3
    assert stats.positions_kept == 2
    assert stats.positions_skipped_empty == 1
    assert stats.shards_written == 1
    shard = next((out_dir / "train").glob("shard-*.npz"))
    with np.load(shard) as data:
        assert set(data.files) == {"boards", "score", "depth", "policy_indices", "policy_scores", "policy_depth", "value_knodes", "policy_knodes", "position_hash"}
        assert data["score"].dtype == np.int16 and data["policy_scores"].dtype == np.int16
        assert data["policy_indices"].shape == (2, 4)
        assert data["policy_indices"].max() < POLICY_SIZE
        assert data["policy_indices"][0].tolist()[2:] == [-1, -1]
        assert data["score"].tolist() == [30, mate_score(5)]
        assert data["policy_scores"][0].tolist()[:2] == [30, 10]
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == EVAL_SHARD_FORMAT
    assert metadata["clock_planes_informative"] is False
    assert metadata["build"]["elapsed_seconds"] > 0


def test_preprocessing_deduplicates_by_evidence_independent_of_record_order(tmp_path: Path) -> None:
    from superchess.data_quality import load_manifest

    records = [
        _record(START_FEN, [{"cp": 5, "line": "e2e4"}], depth=14),
        _record(START_FEN, [{"cp": 30, "line": "d2d4"}], depth=24),
    ]
    manifests = []
    for index, ordered in enumerate((records, records[::-1])):
        source = tmp_path / f"raw{index}.jsonl"
        source.write_text("\n".join(json.dumps(record) for record in ordered), encoding="utf-8")
        output = tmp_path / f"out{index}"
        stats = preprocess_eval_files([source], output, EvalConfig(validation_fraction=0, test_fraction=0), workers=index + 1)
        assert stats.positions_kept == 1 and stats.duplicate_records == 1 and stats.duplicate_score_conflicts == 1
        with np.load(next((output / "train").glob("*.npz"))) as shard:
            assert shard["score"].tolist() == [30]
        manifests.append(load_manifest(output))
    assert manifests[0]["dataset_id"] == manifests[1]["dataset_id"]


def test_preprocessing_refuses_to_overwrite_or_mix_existing_dataset(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    original = output / "shard-00000.npz"
    original.write_bytes(b"user data")
    with pytest.raises(FileExistsError, match="nonempty"):
        preprocess_eval_files([], output)
    assert original.read_bytes() == b"user data"


def test_invalid_input_does_not_publish_partial_dataset(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text('null\n{"fen": 42}\nnot-json\n', encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="no usable positions"):
        preprocess_eval_files([source], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output-build-*"))


def test_clock_variants_share_split_and_frozen_hash_ignores_shard_order(tmp_path: Path) -> None:
    from superchess.data_quality import position_key, split_for_key
    from superchess.encoding import pack_board

    board = chess.Board()
    other = chess.Board()
    other.halfmove_clock = 42
    other.fullmove_number = 60
    key = position_key(pack_board(board))
    assert key == position_key(pack_board(other))
    assert split_for_key(key, 0.2, 0.2, 17) == split_for_key(position_key(pack_board(other)), 0.2, 0.2, 17)


def test_en_passant_grouping_does_not_change_model_inputs() -> None:
    from superchess.data_quality import position_key
    from superchess.encoding import pack_board

    board = chess.Board()
    board.push_uci("e2e4")
    record = _record(board.fen(en_passant="fen"), [{"cp": 20, "line": "e7e5"}])
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None
    np.testing.assert_array_equal(sample.board_pack, pack_board(board))
    key = position_key(sample.board_pack)
    board.ep_square = None
    assert key == position_key(pack_board(board))


def test_record_limit_and_rejection_counters_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text("\n".join(["null", json.dumps(_record(START_FEN, [{"cp": 20, "line": "e2e4"}])), "broken"]), encoding="utf-8")
    stats = preprocess_eval_files([source], tmp_path / "out", max_records=2, workers=2)
    assert stats.lines_seen == 2 and stats.positions_valid == stats.positions_kept == 1
    assert stats.positions_skipped_schema == 1


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 13 30",
    "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 20",
    "8/1P6/8/8/8/8/8/k6K w - - 0 1",
])
def test_packed_board_audit_reconstruction_preserves_legal_moves(fen: str) -> None:
    from superchess.encoding import board_from_packed, pack_board

    board = chess.Board(fen)
    restored = board_from_packed(pack_board(board))
    assert restored.board_fen() == board.board_fen()
    assert restored.turn == board.turn
    assert set(restored.legal_moves) == set(board.legal_moves)


def test_frozen_dataset_audit_checks_hashes_and_legal_targets(tmp_path: Path) -> None:
    from superchess.data_quality import audit_eval_dataset

    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(_record(START_FEN, [{"cp": 20, "line": "e2e4"}])), encoding="utf-8")
    output = tmp_path / "frozen"
    preprocess_eval_files([source], output)
    report = audit_eval_dataset(output, legal_samples=0)
    assert report["valid"] and report["complete_legal_scan"]
    assert report["split_assignment_verified"] and report["duplicate_order_verified"]
    assert report["checksums_verified"] == report["shards_checked"] == 1
    assert report["positions_checked"] == report["legal_positions_checked"] == 1
    shard = next(output.rglob("*.npz"))
    with shard.open("r+b") as handle:
        handle.seek(-10, 2)
        handle.write(b"corruption")
    corrupted = audit_eval_dataset(output)
    assert not corrupted["valid"] and "checksum" in corrupted["errors"][0]


@pytest.mark.parametrize("policy", [[0, -1], [9000, -1], [100, 100], [-1, -1]])
def test_audit_rejects_illegal_and_malformed_policy_targets(tmp_path: Path, policy) -> None:
    from superchess.data_quality import audit_eval_dataset
    from superchess.encoding import pack_board

    np.savez(tmp_path / "shard-00000.npz", boards=np.stack([pack_board(chess.Board())]),
             score=np.asarray([20], np.int16), depth=np.asarray([20], np.uint16), policy_depth=np.asarray([20], np.uint16),
             policy_indices=np.asarray([policy], np.int16), policy_scores=np.zeros((1, 2), np.int16))
    report = audit_eval_dataset(tmp_path, legal_samples=0)
    assert not report["valid"] and report["error_count"] == 1


def test_audit_checks_board_invariants_even_outside_legal_sample(tmp_path: Path) -> None:
    from superchess.data_quality import audit_eval_dataset
    from superchess.encoding import pack_board

    board = chess.Board()
    packed = np.stack([pack_board(board)] * 3)
    packed[1, :8] = 255
    policy = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    np.savez(tmp_path / "shard-00000.npz", boards=packed, score=np.zeros(3, np.int16),
             depth=np.full(3, 20, np.uint16), policy_depth=np.full(3, 20, np.uint16),
             policy_indices=np.full((3, 1), policy, np.int16), policy_scores=np.zeros((3, 1), np.int16))
    report = audit_eval_dataset(tmp_path, legal_samples=1)
    assert not report["valid"] and report["invalid_positions"] == 1
    assert report["board_errors"]["piece_count"] == 1
    assert report["positions_checked"] == 3 and report["legal_positions_checked"] == 1


def test_cli_preprocess_and_audit_round_trip(tmp_path: Path, capsys) -> None:
    from superchess import cli

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "evals.jsonl").write_text(json.dumps(_record(START_FEN, [{"cp": 20, "line": "e2e4"}])), encoding="utf-8")
    output, report_path = tmp_path / "frozen", tmp_path / "audit.json"
    assert cli.main(["evals", "preprocess", "--raw", str(raw), "--out", str(output), "--max-records", "1",
                     "--policy-max-depth-gap", "0", "--validate-pv-plies", "0"]) == 0
    capsys.readouterr()
    assert cli.main(["evals", "audit", "--data", str(output), "--legal-samples", "0", "--out", str(report_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["complete_legal_scan"]
    assert json.loads(report_path.read_text())["dataset_id"] == report["dataset_id"]
