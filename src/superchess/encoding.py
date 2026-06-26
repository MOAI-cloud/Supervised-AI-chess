from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

BOARD_CHANNELS = 20
POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES

PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

SLIDING_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


@dataclass(frozen=True, slots=True)
class EncodedMove:
    index: int
    from_square: int
    plane: int


def orient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _square_to_row_col(square: chess.Square) -> tuple[int, int]:
    return 7 - chess.square_rank(square), chess.square_file(square)


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.float32)
    turn = board.turn

    for square, piece in board.piece_map().items():
        oriented = orient_square(square, turn)
        row, col = _square_to_row_col(oriented)
        owner_offset = 0 if piece.color == turn else 6
        planes[owner_offset + PIECE_PLANES[piece.piece_type], row, col] = 1.0

    planes[12, :, :] = 1.0 if turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(turn) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(turn) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(not turn) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(not turn) else 0.0

    if board.ep_square is not None:
        ep_square = orient_square(board.ep_square, turn)
        _, ep_file = _square_to_row_col(ep_square)
        planes[17, :, ep_file] = 1.0

    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[19, :, :] = min(board.fullmove_number, 200) / 200.0
    return planes


def pack_board(board: chess.Board) -> np.ndarray:
    binary_planes = encode_board(board)[:18].astype(np.uint8, copy=False)
    return np.packbits(binary_planes.reshape(-1))


def unpack_board(packed: np.ndarray) -> np.ndarray:
    flat = np.unpackbits(packed, count=18 * 8 * 8).astype(np.float32, copy=False)
    planes = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.float32)
    planes[:18] = flat.reshape(18, 8, 8)
    return planes


def move_to_policy(board: chess.Board, move: chess.Move) -> EncodedMove:
    turn = board.turn
    from_square = orient_square(move.from_square, turn)
    to_square = orient_square(move.to_square, turn)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    if move.promotion in UNDERPROMOTIONS:
        if delta_rank != 1 or delta_file not in (-1, 0, 1):
            raise ValueError(f"invalid underpromotion move: {move.uci()}")
        promotion_offset = UNDERPROMOTIONS.index(move.promotion) * 3
        plane = 64 + promotion_offset + delta_file + 1
    elif (delta_file, delta_rank) in KNIGHT_DIRECTIONS:
        plane = 56 + KNIGHT_DIRECTIONS.index((delta_file, delta_rank))
    else:
        plane = _sliding_plane(delta_file, delta_rank, move)

    return EncodedMove(index=from_square * POLICY_PLANES + plane, from_square=from_square, plane=plane)


def _sliding_plane(delta_file: int, delta_rank: int, move: chess.Move) -> int:
    distance = max(abs(delta_file), abs(delta_rank))
    if distance < 1 or distance > 7:
        raise ValueError(f"invalid sliding move: {move.uci()}")
    step_file = 0 if delta_file == 0 else delta_file // abs(delta_file)
    step_rank = 0 if delta_rank == 0 else delta_rank // abs(delta_rank)
    if (step_file, step_rank) not in SLIDING_DIRECTIONS:
        raise ValueError(f"invalid policy direction for move: {move.uci()}")
    if delta_file not in (0, step_file * distance) or delta_rank not in (0, step_rank * distance):
        raise ValueError(f"non-linear move cannot be encoded: {move.uci()}")
    return SLIDING_DIRECTIONS.index((step_file, step_rank)) * 7 + distance - 1


def legal_policy_indices(board: chess.Board) -> dict[chess.Move, int]:
    return {move: move_to_policy(board, move).index for move in board.legal_moves}
