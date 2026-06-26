"""Build supervised training shards from the Lichess Stockfish evaluation dump.

Source: https://database.lichess.org/#evals (``lichess_db_eval.jsonl.zst``).

Each line is a JSON object::

    {"fen": "<pieces side castling ep>",
     "evals": [{"knodes": .., "depth": .., "pvs": [{"cp"|"mate": .., "line": "<uci...>"}, ...]}, ...]}

Important facts baked into this module:

* The FEN only has the first four fields (no move counters); we append ``"0 1"``.
* ``cp``/``mate`` are **from White's point of view** (verified against existing
  dump parsers). We convert them to the side-to-move perspective before turning
  them into win/draw/loss targets.
* ``line`` moves are in ``UCI_Chess960`` notation, so castling appears as
  king-takes-rook (e.g. ``e1h1``); we normalise that to standard ``e1g1`` form so
  the policy index matches what :func:`legal_policy_indices` produces at search
  time.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
import sys

import chess
import numpy as np

from superchess.encoding import POLICY_SIZE, move_to_policy, pack_board

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MATE_CP = 12_000  # centipawn magnitude assigned to a forced mate
PROGRESS_INTERVAL = 50_000


@dataclass(frozen=True, slots=True)
class EvalConfig:
    min_depth: int = 12
    max_policy_targets: int = 8
    cp_white_relative: bool = True
    value_scale: float = 400.0
    wdl_scale: float = 380.0
    wdl_draw_margin: float = 100.0
    policy_temperature: float = 1.0
    shard_size: int = 65_536
    compressed: bool = False


@dataclass(slots=True)
class EvalStats:
    lines_seen: int = 0
    positions_kept: int = 0
    positions_skipped_fen: int = 0
    positions_skipped_depth: int = 0
    positions_skipped_empty: int = 0
    shards_written: int = 0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cp_to_white_score(cp: int | None, mate: int | None) -> float:
    """Return a White-relative centipawn score, mapping mate to a large value."""

    if mate is not None:
        return math.copysign(MATE_CP, mate) if mate != 0 else MATE_CP
    if cp is None:
        raise ValueError("pv has neither cp nor mate")
    return float(cp)


def white_cp_to_stm(white_cp: float, turn: chess.Color, *, white_relative: bool) -> float:
    if not white_relative or turn == chess.WHITE:
        return white_cp
    return -white_cp


def stm_cp_to_wdl(stm_cp: float, config: EvalConfig) -> tuple[float, float, float]:
    """Convert a side-to-move centipawn score to a (win, draw, loss) distribution.

    Uses a symmetric two-threshold (ordered-logistic) model: a draw band of half
    width ``wdl_draw_margin`` centipawns with logistic softness ``wdl_scale``.
    """

    win = _sigmoid((stm_cp - config.wdl_draw_margin) / config.wdl_scale)
    loss = _sigmoid((-stm_cp - config.wdl_draw_margin) / config.wdl_scale)
    draw = max(0.0, 1.0 - win - loss)
    total = win + draw + loss
    return win / total, draw / total, loss / total


def board_from_eval_fen(fen: str) -> chess.Board:
    parts = fen.split()
    if len(parts) == 4:
        fen = f"{fen} 0 1"
    elif len(parts) != 6:
        raise ValueError(f"unexpected FEN field count: {fen!r}")
    return chess.Board(fen)


def parse_first_move(board: chess.Board, uci: str) -> chess.Move:
    """Parse the first move of a PV, normalising Chess960 castling to standard."""

    try:
        move = board.parse_uci(uci)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        move = _parse_chess960_castle(board, uci)
    if not board.is_legal(move):
        raise ValueError(f"illegal move {uci} for {board.fen()}")
    return move


def _parse_chess960_castle(board: chess.Board, uci: str) -> chess.Move:
    raw = chess.Move.from_uci(uci)
    king_square = board.king(board.turn)
    if king_square is None or raw.from_square != king_square:
        raise ValueError(f"cannot parse move {uci} for {board.fen()}")
    rank = chess.square_rank(raw.from_square)
    kingside = chess.square_file(raw.to_square) > chess.square_file(raw.from_square)
    target_file = 6 if kingside else 2
    return chess.Move(raw.from_square, chess.square(target_file, rank))


def _select_value_eval(evals: list[dict], config: EvalConfig) -> dict | None:
    candidates = [item for item in evals if int(item.get("depth", 0)) >= config.min_depth and item.get("pvs")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: int(item.get("depth", 0)))


def _select_policy_eval(evals: list[dict], config: EvalConfig) -> dict | None:
    candidates = [item for item in evals if int(item.get("depth", 0)) >= config.min_depth and item.get("pvs")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (len(item["pvs"]), int(item.get("depth", 0))))


def _soft_policy_targets(
    board: chess.Board,
    pvs: list[dict],
    config: EvalConfig,
) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    scores: list[float] = []
    for pv in pvs[: config.max_policy_targets]:
        line = pv.get("line", "").split()
        if not line:
            continue
        try:
            move = parse_first_move(board, line[0])
            index = move_to_policy(board, move).index
        except ValueError:
            continue
        white_cp = cp_to_white_score(pv.get("cp"), pv.get("mate"))
        stm_cp = white_cp_to_stm(white_cp, board.turn, white_relative=config.cp_white_relative)
        if index in indices:
            continue
        indices.append(index)
        scores.append(stm_cp / (config.value_scale * config.policy_temperature))
    if not indices:
        return [], []
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    total = sum(weights)
    return indices, [weight / total for weight in weights]


@dataclass(slots=True)
class EvalSample:
    board_pack: np.ndarray
    wdl: tuple[float, float, float]
    value: float
    policy_indices: list[int]
    policy_probs: list[float]


def sample_from_eval_record(record: dict, config: EvalConfig) -> EvalSample | None:
    fen = record.get("fen")
    evals = record.get("evals")
    if not fen or not evals:
        return None
    try:
        board = board_from_eval_fen(fen)
    except ValueError:
        return None

    value_eval = _select_value_eval(evals, config)
    if value_eval is None:
        return None
    best_pv = value_eval["pvs"][0]
    white_cp = cp_to_white_score(best_pv.get("cp"), best_pv.get("mate"))
    stm_cp = white_cp_to_stm(white_cp, board.turn, white_relative=config.cp_white_relative)
    win, draw, loss = stm_cp_to_wdl(stm_cp, config)

    policy_eval = _select_policy_eval(evals, config) or value_eval
    indices, probs = _soft_policy_targets(board, policy_eval["pvs"], config)
    if not indices:
        return None

    return EvalSample(
        board_pack=pack_board(board),
        wdl=(win, draw, loss),
        value=win - loss,
        policy_indices=indices,
        policy_probs=probs,
    )


def iter_eval_records(path: Path) -> Iterator[dict]:
    suffixes = path.suffixes
    if path.suffix == ".zst" or ".zst" in suffixes:
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised via error message
            raise RuntimeError('Install zstd support with: python -m pip install -e ".[evals]"') from exc
        with path.open("rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield from _iter_json_lines(reader)
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _iter_json_lines(reader) -> Iterator[dict]:
    buffer = b""
    while True:
        chunk = reader.read(1 << 20)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            if line:
                yield json.loads(line)
    if buffer.strip():
        yield json.loads(buffer)


def _eval_log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[evals] {message}", file=sys.stderr, flush=True)


def preprocess_eval_files(
    paths: Iterable[Path],
    out_dir: Path,
    config: EvalConfig = EvalConfig(),
    *,
    max_positions: int | None = None,
    verbose: bool = False,
) -> EvalStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = EvalStats()
    board_packs: list[np.ndarray] = []
    wdls: list[tuple[float, float, float]] = []
    values: list[float] = []
    policy_indices: list[list[int]] = []
    policy_probs: list[list[float]] = []
    shard_index = 0
    stop = False

    def flush() -> None:
        nonlocal shard_index
        if not board_packs:
            return
        path = out_dir / f"shard-{shard_index:05d}.npz"
        count = len(board_packs)
        indices = np.full((count, config.max_policy_targets), -1, dtype=np.int32)
        probs = np.zeros((count, config.max_policy_targets), dtype=np.float32)
        for row, (move_indices, move_probs) in enumerate(zip(policy_indices, policy_probs, strict=True)):
            width = len(move_indices)
            indices[row, :width] = move_indices
            probs[row, :width] = move_probs
        save = np.savez_compressed if config.compressed else np.savez
        save(
            path,
            boards=np.stack(board_packs).astype(np.uint8, copy=False),
            wdl=np.asarray(wdls, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
            policy_indices=indices,
            policy_probs=probs,
        )
        board_packs.clear()
        wdls.clear()
        values.clear()
        policy_indices.clear()
        policy_probs.clear()
        shard_index += 1
        stats.shards_written += 1
        _eval_log(verbose, f"Wrote {path} with {count:,} sample(s)")

    for path in paths:
        if stop:
            break
        _eval_log(verbose, f"Reading {path}")
        for record in iter_eval_records(path):
            stats.lines_seen += 1
            if verbose and stats.lines_seen % PROGRESS_INTERVAL == 0:
                _eval_log(verbose, f"seen={stats.lines_seen:,} kept={stats.positions_kept:,} shards={stats.shards_written:,}")
            sample = sample_from_eval_record(record, config)
            if sample is None:
                stats.positions_skipped_empty += 1
                continue
            board_packs.append(sample.board_pack)
            wdls.append(sample.wdl)
            values.append(sample.value)
            policy_indices.append(sample.policy_indices)
            policy_probs.append(sample.policy_probs)
            stats.positions_kept += 1
            if len(board_packs) >= config.shard_size:
                flush()
            if max_positions is not None and stats.positions_kept >= max_positions:
                stop = True
                break

    flush()
    metadata = {
        "source": LICHESS_EVAL_URL,
        "policy_size": POLICY_SIZE,
        "board_pack_bytes": int(pack_board(chess.Board()).size),
        "max_policy_targets": config.max_policy_targets,
        "config": asdict(config),
        "stats": asdict(stats),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _eval_log(verbose, f"Finished: kept={stats.positions_kept:,} shards={stats.shards_written:,}")
    return stats


def download_eval_dump(out_dir: Path, *, overwrite: bool = False) -> Path:
    from superchess.ccrl import download_url

    out_dir.mkdir(parents=True, exist_ok=True)
    return download_url(LICHESS_EVAL_URL, out_dir / "lichess_db_eval.jsonl.zst", overwrite=overwrite)


def preprocess_eval_directory(
    raw_dir: Path,
    out_dir: Path,
    config: EvalConfig = EvalConfig(),
    *,
    max_positions: int | None = None,
    verbose: bool = False,
) -> EvalStats:
    paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".jsonl") or ".jsonl." in path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no lichess eval jsonl(.zst/.gz) files found under {raw_dir}")
    return preprocess_eval_files(paths, out_dir, config, max_positions=max_positions, verbose=verbose)
