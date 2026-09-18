from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

LEGACY_BOARD_CHANNELS = 18
BOARD_CHANNELS = 20
BOARD_SQUARES = 8 * 8
LEGACY_PACKED_BOARD_BITS = LEGACY_BOARD_CHANNELS * BOARD_SQUARES
LEGACY_PACKED_BOARD_BYTES = LEGACY_PACKED_BOARD_BITS // 8
PACKED_BOARD_BITS = BOARD_CHANNELS * BOARD_SQUARES
FULL_BITPACKED_BOARD_BYTES = PACKED_BOARD_BITS // 8
PACKED_BOARD_AUX_BYTES = BOARD_CHANNELS - LEGACY_BOARD_CHANNELS
PACKED_BOARD_BYTES = LEGACY_PACKED_BOARD_BYTES + PACKED_BOARD_AUX_BYTES
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
    planes = encode_board(board)
    binary = np.packbits(planes[:LEGACY_BOARD_CHANNELS].astype(np.uint8, copy=False).reshape(-1))
    aux = np.rint(planes[LEGACY_BOARD_CHANNELS:, 0, 0] * 255).clip(0, 255).astype(np.uint8)
    return np.concatenate([binary, aux])


def unpack_board(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed.size == FULL_BITPACKED_BOARD_BYTES:
        flat = np.unpackbits(packed, count=PACKED_BOARD_BITS).astype(np.float32, copy=False)
        return flat.reshape(BOARD_CHANNELS, 8, 8)
    if packed.size not in (LEGACY_PACKED_BOARD_BYTES, PACKED_BOARD_BYTES):
        raise ValueError(f"unexpected packed board byte count: {packed.size}")

    planes = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.float32)
    binary = np.unpackbits(
        packed[:LEGACY_PACKED_BOARD_BYTES],
        count=LEGACY_PACKED_BOARD_BITS,
    ).astype(np.float32, copy=False)
    planes[:LEGACY_BOARD_CHANNELS] = binary.reshape(LEGACY_BOARD_CHANNELS, 8, 8)
    if packed.size == PACKED_BOARD_BYTES:
        aux = packed[LEGACY_PACKED_BOARD_BYTES:PACKED_BOARD_BYTES].astype(np.float32) / 255.0
        planes[LEGACY_BOARD_CHANNELS:, :, :] = aux[:, None, None]
    return planes


def board_from_packed(packed: np.ndarray) -> chess.Board:
    """Reconstruct a standard-chess position for data auditing.

    Packed clocks are quantized and clipped; game history is unrecoverable.
    Reconstruction is for legality checks, not repetition adjudication.
    """
    planes = unpack_board(packed)
    if np.any(planes[:12].sum(axis=0) > 1):
        raise ValueError("overlapping pieces in packed board")
    for plane in planes[12:17]:
        if not np.all(plane == plane[0, 0]):
            raise ValueError("nonconstant side/castling plane in packed board")
    board = chess.Board(None)
    board.turn = bool(planes[12, 0, 0])
    for plane_index in range(12):
        color = board.turn if plane_index < 6 else not board.turn
        piece = chess.Piece(plane_index % 6 + 1, color)
        for row, column in np.argwhere(planes[plane_index] > 0.5):
            square = orient_square(chess.square(int(column), 7 - int(row)), board.turn)
            board.set_piece_at(square, piece)
    for offset, color in ((13, board.turn), (15, not board.turn)):
        rank = 0 if color == chess.WHITE else 7
        for plane_index, file_index in ((offset, 7), (offset + 1, 0)):
            if planes[plane_index, 0, 0]:
                board.castling_rights |= chess.BB_SQUARES[chess.square(file_index, rank)]
    ep_files = np.flatnonzero(planes[17, 0])
    if len(ep_files) > 1 or not np.all(planes[17] == planes[17, 0]):
        raise ValueError("invalid en passant plane in packed board")
    if len(ep_files):
        board.ep_square = chess.square(int(ep_files[0]), 5 if board.turn else 2)
    board.halfmove_clock = round(float(planes[18, 0, 0]) * 100)
    board.fullmove_number = max(1, round(float(planes[19, 0, 0]) * 200))
    if not board.is_valid():
        raise ValueError(f"invalid reconstructed board (status={int(board.status())})")
    return board


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


def plane_delta(plane: int) -> tuple[int, int, int]:
    """Return ``(delta_file, delta_rank, promotion_piece)`` for a policy plane.

    Deltas are in the oriented frame (side to move advances towards rank 8);
    ``promotion_piece`` is ``0`` except for the nine underpromotion planes.
    """
    if plane < 0 or plane >= POLICY_PLANES:
        raise ValueError(f"policy plane out of range: {plane}")
    if plane < 56:
        direction, distance = divmod(plane, 7)
        step_file, step_rank = SLIDING_DIRECTIONS[direction]
        return step_file * (distance + 1), step_rank * (distance + 1), 0
    if plane < 64:
        delta_file, delta_rank = KNIGHT_DIRECTIONS[plane - 56]
        return delta_file, delta_rank, 0
    promotion, delta_file = divmod(plane - 64, 3)
    return delta_file - 1, 1, UNDERPROMOTIONS[promotion]


def policy_plane_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Static tables describing the ``64 x 73`` policy layout.

    Returns ``to_square[64, 73]`` (oriented destination square, ``-1`` when the
    move leaves the board) and ``promotion[73]`` (``0`` or the underpromotion
    piece type of that plane). Squares use python-chess numbering (a1 = 0).
    """
    to_square = np.full((BOARD_SQUARES, POLICY_PLANES), -1, dtype=np.int64)
    promotion = np.zeros(POLICY_PLANES, dtype=np.int64)
    for plane in range(POLICY_PLANES):
        delta_file, delta_rank, promo = plane_delta(plane)
        promotion[plane] = promo
        for square in range(BOARD_SQUARES):
            to_file = chess.square_file(square) + delta_file
            to_rank = chess.square_rank(square) + delta_rank
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square[square, plane] = chess.square(to_file, to_rank)
    return to_square, promotion


def piece_plane_mask() -> np.ndarray:
    """Boolean ``[6, 64, 73]`` table: may piece type ``t`` on oriented square ``s`` use plane ``p``?

    This is the *structural* superset of legality (geometry only: piece
    movement pattern, board edges, pawn double-push and promotion ranks,
    castling from the king's home square). Blockers, checks, and pins are not
    considered, so every legal move is always allowed by this mask.
    """
    to_square, _ = policy_plane_geometry()
    mask = np.zeros((len(PIECE_PLANES), BOARD_SQUARES, POLICY_PLANES), dtype=bool)
    king_home = chess.E1
    for plane in range(POLICY_PLANES):
        delta_file, delta_rank, promo = plane_delta(plane)
        distance = max(abs(delta_file), abs(delta_rank))
        straight = delta_file == 0 or delta_rank == 0
        diagonal = abs(delta_file) == abs(delta_rank)
        knight = (delta_file, delta_rank) in KNIGHT_DIRECTIONS and plane >= 56
        sliding = plane < 56
        for square in range(BOARD_SQUARES):
            if to_square[square, plane] < 0:
                continue
            rank = chess.square_rank(square)
            if promo:
                mask[PIECE_PLANES[chess.PAWN], square, plane] = rank == 6
                continue
            if sliding and delta_rank == 1 and abs(delta_file) <= 1:
                mask[PIECE_PLANES[chess.PAWN], square, plane] = True
            if sliding and delta_rank == 2 and delta_file == 0:
                mask[PIECE_PLANES[chess.PAWN], square, plane] = rank == 1
            if knight:
                mask[PIECE_PLANES[chess.KNIGHT], square, plane] = True
            if sliding and diagonal:
                mask[PIECE_PLANES[chess.BISHOP], square, plane] = True
                mask[PIECE_PLANES[chess.QUEEN], square, plane] = True
            if sliding and straight:
                mask[PIECE_PLANES[chess.ROOK], square, plane] = True
                mask[PIECE_PLANES[chess.QUEEN], square, plane] = True
            if sliding and distance == 1:
                mask[PIECE_PLANES[chess.KING], square, plane] = True
            if sliding and delta_rank == 0 and distance == 2 and square == king_home:
                mask[PIECE_PLANES[chess.KING], square, plane] = True
    return mask


def structural_policy_mask(planes: np.ndarray) -> np.ndarray:
    """Boolean ``[POLICY_SIZE]`` mask of structurally possible moves for encoded ``planes``."""
    own = planes[:6, ::-1, :].reshape(6, BOARD_SQUARES) > 0.5
    table = piece_plane_mask()
    return np.einsum("ts,tsp->sp", own.astype(np.int64), table.astype(np.int64)).reshape(-1) > 0
