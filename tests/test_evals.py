from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from superchess.encoding import POLICY_SIZE, move_to_policy
from superchess.evals import (
    EvalConfig,
    cp_to_white_score,
    parse_first_move,
    preprocess_eval_files,
    sample_from_eval_record,
    stm_cp_to_wdl,
    white_cp_to_stm,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def _record(fen: str, pvs: list[dict], depth: int = 20) -> dict:
    return {"fen": fen, "evals": [{"depth": depth, "knodes": 1000, "pvs": pvs}]}


def test_wdl_symmetric_at_zero() -> None:
    win, draw, loss = stm_cp_to_wdl(0.0, EvalConfig())
    assert win == pytest.approx(loss)
    assert win + draw + loss == pytest.approx(1.0)


def test_wdl_monotonic_in_cp() -> None:
    config = EvalConfig()
    low = stm_cp_to_wdl(50.0, config)[0]
    high = stm_cp_to_wdl(400.0, config)[0]
    assert high > low


def test_white_cp_negated_for_black_to_move() -> None:
    assert white_cp_to_stm(120.0, chess.WHITE, white_relative=True) == 120.0
    assert white_cp_to_stm(120.0, chess.BLACK, white_relative=True) == -120.0


def test_mate_maps_to_large_score() -> None:
    assert cp_to_white_score(None, 3) > 1000
    assert cp_to_white_score(None, -3) < -1000


def test_parse_chess960_castle_normalises() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    move = parse_first_move(board, "e1h1")
    assert move == chess.Move.from_uci("e1g1")


def test_sample_from_eval_record_shapes_and_probs() -> None:
    record = _record(
        START_FEN,
        [
            {"cp": 30, "line": "e2e4 e7e5"},
            {"cp": 20, "line": "d2d4 d7d5"},
        ],
    )
    sample = sample_from_eval_record(record, EvalConfig())
    assert sample is not None
    assert len(sample.wdl) == 3
    assert sum(sample.wdl) == pytest.approx(1.0)
    assert sum(sample.policy_probs) == pytest.approx(1.0)
    assert sample.value == pytest.approx(sample.wdl[0] - sample.wdl[2])
    # the higher-cp move (e2e4) should carry more policy mass
    board = chess.Board(f"{START_FEN} 0 1")
    e4_index = move_to_policy(board, chess.Move.from_uci("e2e4")).index
    assert sample.policy_indices[0] == e4_index
    assert sample.policy_probs[0] >= sample.policy_probs[1]


def test_black_to_move_value_flips_sign() -> None:
    black_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"
    white = sample_from_eval_record(
        _record(START_FEN, [{"cp": 200, "line": "e2e4"}]), EvalConfig()
    )
    black = sample_from_eval_record(
        _record(black_fen, [{"cp": 200, "line": "e7e5"}]), EvalConfig()
    )
    assert white is not None and black is not None
    # +200 cp is good for the side to move when White, bad when Black.
    assert white.value > 0
    assert black.value < 0


def test_min_depth_filters_shallow_evals() -> None:
    record = _record(START_FEN, [{"cp": 30, "line": "e2e4"}], depth=5)
    assert sample_from_eval_record(record, EvalConfig(min_depth=12)) is None


def test_preprocess_eval_files_writes_shards(tmp_path: Path) -> None:
    records = [
        _record(START_FEN, [{"cp": 30, "line": "e2e4 e7e5"}, {"cp": 10, "line": "d2d4 d7d5"}]),
        _record(START_FEN, [{"mate": 5, "line": "e2e4"}]),
    ]
    source = tmp_path / "raw" / "evals.jsonl"
    source.parent.mkdir(parents=True)
    import json

    source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    out_dir = tmp_path / "processed"
    config = EvalConfig(shard_size=8, max_policy_targets=4)
    stats = preprocess_eval_files([source], out_dir, config)

    assert stats.positions_kept == 2
    assert stats.shards_written == 1
    shard = next(out_dir.glob("shard-*.npz"))
    with np.load(shard) as data:
        assert data["wdl"].shape == (2, 3)
        assert data["values"].shape == (2,)
        assert data["policy_indices"].shape == (2, 4)
        assert data["policy_probs"].shape == (2, 4)
        assert data["policy_indices"].max() < POLICY_SIZE
        assert data["policy_probs"][0].sum() == pytest.approx(1.0)
