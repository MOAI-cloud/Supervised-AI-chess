import chess
import numpy as np

from superchess.encoding import BOARD_CHANNELS, POLICY_SIZE, encode_board, legal_policy_indices, move_to_policy, pack_board, unpack_board


def test_start_position_encoding_shape_and_planes():
    board = chess.Board()
    encoded = encode_board(board)
    assert encoded.shape == (BOARD_CHANNELS, 8, 8)
    assert encoded.dtype == np.float32
    assert encoded[:12].sum() == 32


def test_start_position_legal_moves_have_unique_policy_indices():
    board = chess.Board()
    policies = legal_policy_indices(board)
    assert len(policies) == 20
    assert len(set(policies.values())) == 20
    assert all(0 <= policy < POLICY_SIZE for policy in policies.values())


def test_move_policy_is_perspective_invariant_for_pawn_pushes():
    white_board = chess.Board()
    black_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    white_move = chess.Move.from_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")
    assert move_to_policy(white_board, white_move).index == move_to_policy(black_board, black_move).index


def test_pack_unpack_preserves_binary_planes():
    board = chess.Board()
    packed = pack_board(board)
    unpacked = unpack_board(packed)
    assert unpacked.shape == (BOARD_CHANNELS, 8, 8)
    np.testing.assert_array_equal(unpacked[:18], encode_board(board)[:18])
