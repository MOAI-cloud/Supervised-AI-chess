from __future__ import annotations

import math

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from superchess.encoding import (
    encode_board,
    legal_policy_indices,
    piece_plane_mask,
    policy_plane_geometry,
    structural_policy_mask,
)
from superchess.targets import (
    DEFAULT_CP_SCALE,
    MATE_SCORE,
    TargetConfig,
    best_target_moves,
    entropy,
    expected_to_score,
    hl_gauss_targets,
    mate_score,
    policy_targets,
    reconstruct_legacy_scores,
    score_to_expected,
    value_bin_centers,
    value_from_logits,
    value_targets,
    value_to_cp,
)


def test_expected_score_is_lichess_logistic_and_invertible() -> None:
    assert score_to_expected(0.0) == pytest.approx(0.5)
    assert score_to_expected(100.0) == pytest.approx(1.0 / (1.0 + math.exp(-0.00368208 * 100)))
    assert score_to_expected(mate_score(1)) == pytest.approx(1.0)
    for cp in (-900.0, -37.0, 0.0, 12.0, 450.0):
        assert expected_to_score(score_to_expected(cp)) == pytest.approx(cp, abs=1e-3)
    assert value_to_cp(0.0) == 0
    assert value_to_cp(2 * score_to_expected(300.0) - 1) == 300
    assert value_to_cp(-0.9) < -700


def test_hl_gauss_targets_are_normalised_and_mean_preserving() -> None:
    expected = torch.tensor([0.5, 0.73, 0.12])
    target = hl_gauss_targets(expected, bins=64, sigma_ratio=0.75)
    assert target.shape == (3, 64)
    assert torch.allclose(target.sum(dim=1), torch.ones(3), atol=1e-6)
    means = (target * value_bin_centers(64)).sum(dim=1)
    assert torch.allclose(means, expected, atol=2e-3)
    assert torch.all(target >= 0)


def test_value_from_logits_matches_bin_expectation() -> None:
    logits = torch.full((1, 8), -30.0)
    logits[0, 7] = 0.0  # all mass in the last bin (centre 0.9375)
    assert value_from_logits(logits).item() == pytest.approx(2 * 0.9375 - 1, abs=1e-6)
    uniform = value_from_logits(torch.zeros(2, 16))
    assert torch.allclose(uniform, torch.zeros(2), atol=1e-6)


def test_value_targets_saturate_for_mates() -> None:
    scores = torch.tensor([float(mate_score(3)), float(-mate_score(3)), 0.0])
    target = value_targets(scores, 32, TargetConfig())
    assert target[0].argmax() == 31 and target[1].argmax() == 0
    # sigma = 0.75 bin widths: the two central bins hold ~82% and +-3 bins ~all of the mass.
    assert target[2, 15:17].sum() > 0.8
    assert target[2, 13:19].sum() > 0.99
    assert torch.allclose(target[2, 15], target[2, 16])


def test_policy_targets_are_sharp_and_ignore_padding() -> None:
    config = TargetConfig()
    scores = torch.tensor([[30.0, 10.0, -20.0, -70.0, -170.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[True] * 5 + [False] * 3])
    target = policy_targets(scores, valid, config)
    assert target.shape == scores.shape
    assert target[0, 5:].sum() == 0
    assert target.sum() == pytest.approx(1.0)
    assert target[0, 0] > 0.45  # best move dominates a 20cp gap
    assert target[0, 4] < 0.01  # a 200cp blunder is nearly excluded
    assert torch.all(target[0, :-4].diff() <= 0)

    # A mate beats a huge material edge; mate distance breaks ties only slightly.
    mates = torch.tensor([[float(mate_score(1)), float(mate_score(4)), 1500.0, 0.0]])
    mate_target = policy_targets(mates, torch.tensor([[True, True, True, False]]), config)
    assert mate_target[0, 2] < 0.01
    assert mate_target[0, 0] >= mate_target[0, 1]

    # flat temperature -> flatter target
    flat = policy_targets(scores, valid, TargetConfig(policy_temperature=0.5))
    assert entropy(flat)[0] > entropy(target)[0]

    empty = policy_targets(scores, torch.zeros_like(valid), config)
    assert torch.all(empty == 0)


def test_best_target_moves_uses_highest_score_first_on_ties() -> None:
    indices = torch.tensor([[5, 7, 9, -1], [3, 4, -1, -1]])
    scores = torch.tensor([[10.0, 10.0, 3.0, 0.0], [-50.0, 20.0, 0.0, 0.0]])
    assert best_target_moves(indices, scores).tolist() == [5, 4]


def test_legacy_reconstruction_round_trips_scores() -> None:
    def legacy_arrays(cps: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = np.asarray(cps, dtype=np.float64)
        win = 1.0 / (1.0 + np.exp(-(cp[0] - 100.0) / 380.0))
        loss = 1.0 / (1.0 + np.exp(-(-cp[0] - 100.0) / 380.0))
        weights = np.exp((cp - cp.max()) / 400.0)
        probs = weights / weights.sum()
        pad = 8 - len(cps)
        wdl = np.asarray([[win, max(0.0, 1 - win - loss), loss]], dtype=np.float32)
        indices = np.asarray([list(range(len(cps))) + [-1] * pad], dtype=np.int32)
        probs = np.asarray([list(probs) + [0.0] * pad], dtype=np.float32)
        return wdl, indices, probs

    scores, policy_scores = reconstruct_legacy_scores(*legacy_arrays([35.0, 20.0, -15.0, -300.0]))
    assert scores[0] == pytest.approx(35.0, abs=0.05)
    assert policy_scores[0, :4].tolist() == pytest.approx([35.0, 20.0, -15.0, -300.0], abs=0.05)
    assert policy_scores[0, 4:].tolist() == [0.0] * 4

    scores, policy_scores = reconstruct_legacy_scores(*legacy_arrays([-250.0]))
    assert scores[0] == pytest.approx(-250.0, abs=0.05)

    scores, policy_scores = reconstruct_legacy_scores(*legacy_arrays([12000.0, 12000.0, 500.0]))
    assert scores[0] >= MATE_SCORE
    assert policy_scores[0, 0] >= MATE_SCORE and policy_scores[0, 1] >= MATE_SCORE
    assert policy_scores[0, 2] == pytest.approx(500.0, abs=0.5)


def test_target_config_validation_and_roundtrip() -> None:
    config = TargetConfig(policy_temperature=0.05)
    assert TargetConfig.from_dict(config.to_dict()) == config
    assert TargetConfig.from_dict(None) == TargetConfig()
    assert TargetConfig.from_dict({"cp_scale": DEFAULT_CP_SCALE, "unknown": 1}) == TargetConfig()
    with pytest.raises(ValueError):
        TargetConfig(policy_temperature=0.0)


def test_structural_mask_covers_every_legal_move() -> None:
    fens = [
        chess.STARTING_FEN,
        "r3k2r/pP2bppp/2n1pn2/q2p4/3P4/2NBBN2/PPQ2PpP/R3K2R w KQkq - 0 1",
        "r3k2r/pP2bppp/2n1pn2/q2p4/3P4/2NBBN2/PPQ2PpP/R3K2R b KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "8/8/8/8/8/8/1k6/K7 w - - 0 1",
    ]
    rng = np.random.default_rng(1)
    boards = [chess.Board(fen) for fen in fens]
    for _ in range(10):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 80))):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            board.push(legal[int(rng.integers(len(legal)))])
        boards.append(board)
    for board in boards:
        if board.is_game_over():
            continue
        mask = structural_policy_mask(encode_board(board))
        legal = legal_policy_indices(board)
        assert all(mask[index] for index in legal.values()), board.fen()
        assert mask.sum() <= 400  # a small superset of legality, not the whole action space
    assert piece_plane_mask().shape == (6, 64, 73)


def test_policy_plane_geometry_matches_move_encoding() -> None:
    from superchess.encoding import POLICY_PLANES, move_to_policy

    to_square, promotion = policy_plane_geometry()
    board = chess.Board("8/1P6/8/8/8/8/6p1/8 w - - 0 1")
    for uci in ("b7b8q", "b7b8n", "b7a8r", "b7c8b"):
        move = chess.Move.from_uci(uci)
        board.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))
        board.set_piece_at(chess.C8, chess.Piece(chess.ROOK, chess.BLACK))
        encoded = move_to_policy(board, move)
        assert to_square[encoded.from_square, encoded.plane] == move.to_square
        assert promotion[encoded.plane] == (move.promotion if move.promotion != chess.QUEEN else 0)
    assert (to_square >= 0).sum() == 2254
    assert to_square.shape == (64, POLICY_PLANES)
