"""Build supervised training shards from the Lichess Stockfish evaluation dump.

Source: https://database.lichess.org/#evals (``lichess_db_eval.jsonl.zst``).

Each line is a JSON object::

    {"fen": "<pieces side castling ep>",
     "evals": [{"knodes": .., "depth": .., "pvs": [{"cp"|"mate": .., "line": "<uci...>"}, ...]}, ...]}

Important facts baked into this module:

* The FEN only has the first four fields (no move counters); we append ``"0 1"``.
  Consequently the halfmove/fullmove planes carry **no information** for this
  data and models trained on it must use the 18 clock-free input planes.
* ``cp``/``mate`` are **from White's point of view** (verified against existing
  dump parsers). We convert them to the side-to-move perspective.
* ``line`` moves are in ``UCI_Chess960`` notation, so castling appears as
  king-takes-rook (e.g. ``e1h1``); we normalise that to standard ``e1g1`` form so
  the policy index matches what :func:`legal_policy_indices` produces at search
  time.

Shards store *raw* side-to-move scores (see :mod:`superchess.targets`) rather
than pre-shaped probabilities, so the policy temperature and value mapping are
training-time hyper-parameters and can be changed without re-preprocessing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import gzip
from itertools import islice
import json
import math
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
import platform
import sqlite3
import struct
import sys
from tempfile import TemporaryDirectory
import time

import chess
import numpy as np

from superchess.data_quality import DATASET_SCHEMA, SPLITS, file_sha256, manifest_identity, position_key, split_for_key
from superchess.encoding import POLICY_SIZE, move_to_policy, pack_board
from superchess.targets import encode_score, score_to_expected

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
EVAL_SHARD_FORMAT = "evals-v2"
PROGRESS_INTERVAL = 50_000
PARALLEL_CHUNK_LINES = 2_048


@dataclass(frozen=True, slots=True)
class EvalConfig:
    min_depth: int = 12
    max_policy_targets: int = 8
    cp_white_relative: bool = True
    shard_size: int = 65_536
    compressed: bool = False
    min_knodes: int = 0
    policy_max_depth_gap: int = 4
    max_policy_value_gap: float = 0.1
    validate_pv_plies: int = 8
    validation_fraction: float = 0.02
    test_fraction: float = 0.02
    split_seed: int = 0
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if self.min_depth < 1 or self.shard_size < 1 or self.max_policy_targets < 1:
            raise ValueError("min_depth, shard_size, and max_policy_targets must be positive")
        if self.min_knodes < 0 or self.policy_max_depth_gap < 0 or self.validate_pv_plies < 0:
            raise ValueError("min_knodes, policy_max_depth_gap, and validate_pv_plies must be nonnegative")
        if not math.isfinite(self.max_policy_value_gap) or not 0 <= self.max_policy_value_gap <= 1:
            raise ValueError("max_policy_value_gap must be between zero and one")
        if not all(math.isfinite(value) and 0 <= value < 1 for value in (self.validation_fraction, self.test_fraction)) or self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation/test fractions must be nonnegative and sum to less than one")


@dataclass(slots=True)
class EvalStats:
    lines_seen: int = 0
    positions_kept: int = 0
    positions_skipped_fen: int = 0
    positions_skipped_depth: int = 0
    positions_skipped_empty: int = 0
    shards_written: int = 0
    positions_skipped_schema: int = 0
    positions_skipped_terminal: int = 0
    positions_skipped_quality: int = 0
    positions_missing_clocks: int = 0
    evaluations_rejected: int = 0
    pvs_rejected: int = 0
    duplicate_pvs: int = 0
    policy_depth_fallbacks: int = 0
    policy_disagreements: int = 0
    policy_value_disagreements: int = 0
    positions_valid: int = 0
    duplicate_records: int = 0
    duplicate_score_conflicts: int = 0
    duplicate_replacements: int = 0
    positions_unused_valid: int = 0


def eval_score(cp: int | None, mate: int | None, turn: chess.Color, *, white_relative: bool = True) -> int:
    """Side-to-move integer score for one PV evaluation (see :func:`encode_score`)."""

    score = encode_score(cp, mate)
    if white_relative and turn == chess.BLACK:
        return -score
    return score


def white_cp_to_stm(white_cp: float, turn: chess.Color, *, white_relative: bool) -> float:
    if not white_relative or turn == chess.WHITE:
        return white_cp
    return -white_cp


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
    if (
        king_square is None
        or raw.from_square != king_square
        or raw.promotion is not None
        or chess.square_rank(raw.to_square) != chess.square_rank(raw.from_square)
        or board.piece_at(raw.to_square) != chess.Piece(chess.ROOK, board.turn)
        or not board.clean_castling_rights() & chess.BB_SQUARES[raw.to_square]
    ):
        raise ValueError(f"cannot parse move {uci} for {board.fen()}")
    rank = chess.square_rank(raw.from_square)
    kingside = chess.square_file(raw.to_square) > chess.square_file(raw.from_square)
    target_file = 6 if kingside else 2
    return chess.Move(raw.from_square, chess.square(target_file, rank))


@dataclass(slots=True)
class _ValidatedEval:
    depth: int
    knodes: int
    indices: list[int]
    scores: list[int]


def _validated_pv(board: chess.Board, pv: dict, config: EvalConfig) -> tuple[int, int]:
    if not isinstance(pv, dict) or pv.get("lowerbound") or pv.get("upperbound"):
        raise ValueError("PV must carry an exact score")
    cp, mate = pv.get("cp"), pv.get("mate")
    if (cp is None) == (mate is None):
        raise ValueError("PV must carry exactly one of cp or mate")
    if type(cp if mate is None else mate) is not int or mate == 0:
        raise ValueError("PV score must be an integer; mate zero has no legal policy target")
    line = pv.get("line")
    if not isinstance(line, str) or not line.split():
        raise ValueError("PV line must contain moves")
    tokens = line.split()
    first = parse_first_move(board, tokens[0])
    if config.validate_pv_plies != 1:
        probe = board.copy(stack=False)
        probe.push(first)
        for token in tokens[1 : config.validate_pv_plies or None]:
            probe.push(parse_first_move(probe, token))
    return move_to_policy(board, first).index, eval_score(cp, mate, board.turn, white_relative=config.cp_white_relative)


def _validated_eval(board: chess.Board, item: dict, config: EvalConfig, stats: EvalStats) -> _ValidatedEval | None:
    if not isinstance(item, dict):
        stats.evaluations_rejected += 1
        return None
    depth, knodes, pvs = item.get("depth"), item.get("knodes", 0), item.get("pvs")
    if (
        type(depth) is not int or not config.min_depth <= depth <= 65_535
        or type(knodes) is not int or not config.min_knodes <= knodes <= np.iinfo(np.uint64).max
        or not isinstance(pvs, list) or not pvs
        or item.get("lowerbound") or item.get("upperbound")
    ):
        stats.evaluations_rejected += 1
        return None
    scores_by_move: dict[int, int] = {}
    for rank, pv in enumerate(pvs):
        try:
            index, score = _validated_pv(board, pv, config)
        except ValueError:
            stats.pvs_rejected += 1
            if rank == 0:
                stats.evaluations_rejected += 1
                return None
            continue
        if index in scores_by_move:
            stats.duplicate_pvs += 1
            if scores_by_move[index] != score:
                stats.evaluations_rejected += 1
                return None
        scores_by_move[index] = score
    ranked = sorted(scores_by_move, key=lambda index: (-scores_by_move[index], index))[: config.max_policy_targets]
    return _ValidatedEval(depth, knodes, ranked, [scores_by_move[index] for index in ranked])


@dataclass(slots=True)
class EvalSample:
    """One training position with raw side-to-move scores."""

    board_pack: np.ndarray
    score: int
    depth: int
    policy_indices: list[int]
    policy_scores: list[int]
    policy_depth: int
    value_knodes: int = 0
    policy_knodes: int = 0


def sample_from_eval_record(record: dict, config: EvalConfig) -> EvalSample | None:
    return _sample_from_eval_record(record, config, EvalStats())


def _sample_from_eval_record(record: dict, config: EvalConfig, stats: EvalStats) -> EvalSample | None:
    if not isinstance(record, dict):
        stats.positions_skipped_schema += 1
        return None
    fen = record.get("fen")
    evals = record.get("evals")
    if not isinstance(fen, str) or not isinstance(evals, list):
        stats.positions_skipped_schema += 1
        return None
    if not evals:
        stats.positions_skipped_empty += 1
        return None
    try:
        board = board_from_eval_fen(fen)
    except ValueError:
        stats.positions_skipped_fen += 1
        return None
    if not board.is_valid():
        stats.positions_skipped_fen += 1
        return None
    if board.is_game_over():
        stats.positions_skipped_terminal += 1
        return None
    if not any(isinstance(item, dict) and type(item.get("depth")) is int and item["depth"] >= config.min_depth for item in evals):
        stats.positions_skipped_depth += 1
        return None
    candidates = [candidate for item in evals if (candidate := _validated_eval(board, item, config, stats)) is not None]
    if not candidates:
        stats.positions_skipped_quality += 1
        return None
    value_eval = max(candidates, key=lambda item: (item.depth, item.knodes, len(item.indices)))
    eligible = [item for item in candidates if item.depth >= value_eval.depth - config.policy_max_depth_gap]
    stats.policy_depth_fallbacks += int(any(len(item.indices) > len(value_eval.indices) and item not in eligible for item in candidates))
    agreeing = [item for item in eligible if item.indices[0] == value_eval.indices[0]]
    stats.policy_disagreements += int(len(agreeing) < len(eligible))
    consistent = [item for item in agreeing if abs(score_to_expected(item.scores[0]) - score_to_expected(value_eval.scores[0])) <= config.max_policy_value_gap]
    stats.policy_value_disagreements += int(len(consistent) < len(agreeing))
    policy_eval = max(consistent, key=lambda item: (len(item.indices), item.depth, item.knodes))
    stats.positions_missing_clocks += int(len(fen.split()) == 4)

    return EvalSample(
        board_pack=pack_board(board),
        score=value_eval.scores[0],
        depth=value_eval.depth,
        policy_indices=policy_eval.indices,
        policy_scores=policy_eval.scores,
        policy_depth=policy_eval.depth,
        value_knodes=value_eval.knodes,
        policy_knodes=policy_eval.knodes,
    )


def iter_eval_lines(path: Path) -> Iterator[bytes]:
    """Yield raw non-empty JSON lines from a ``.jsonl``, ``.jsonl.gz`` or ``.jsonl.zst`` file."""

    if path.suffix == ".zst" or ".zst" in path.suffixes:
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised via error message
            raise RuntimeError('Install zstd support with: python -m pip install -e ".[evals]"') from exc
        with path.open("rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield from _iter_raw_lines(reader)
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            yield from _iter_raw_lines(handle)
    else:
        with path.open("rb") as handle:
            yield from _iter_raw_lines(handle)


def iter_eval_records(path: Path) -> Iterator[dict]:
    for line in iter_eval_lines(path):
        yield json.loads(line)


def _iter_raw_lines(reader) -> Iterator[bytes]:
    buffer = b""
    while True:
        chunk = reader.read(1 << 20)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            if line.strip():
                yield line
    if buffer.strip():
        yield buffer


def _chunked(lines: Iterator[bytes], size: int) -> Iterator[list[bytes]]:
    chunk: list[bytes] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


_WORKER_CONFIG: EvalConfig | None = None


def _init_worker(config: EvalConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _process_chunk(lines: list[bytes], config: EvalConfig | None = None) -> tuple[list[EvalSample], EvalStats]:
    """Parse a chunk with reason-specific rejection counters."""

    active = config if config is not None else _WORKER_CONFIG
    if active is None:
        raise RuntimeError("worker configuration is missing")
    samples: list[EvalSample] = []
    stats = EvalStats(lines_seen=len(lines))
    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats.positions_skipped_schema += 1
            continue
        sample = _sample_from_eval_record(record, active, stats)
        if sample is None:
            continue
        samples.append(sample)
        stats.positions_valid += 1
    return samples, stats


def _eval_log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[evals] {message}", file=sys.stderr, flush=True)


class _ShardWriter:
    """Accumulates samples and writes fixed-size ``evals-v2`` shards."""

    def __init__(self, out_dir: Path, config: EvalConfig, stats: EvalStats, verbose: bool) -> None:
        self.out_dir = out_dir
        self.config = config
        self.stats = stats
        self.verbose = verbose
        self.samples: list[EvalSample] = []
        self.shard_index = 0
        self.manifest: list[dict] = []

    def add(self, sample: EvalSample) -> None:
        self.samples.append(sample)
        self.stats.positions_kept += 1
        if len(self.samples) >= self.config.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.samples:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"shard-{self.shard_index:05d}.npz"
        count = len(self.samples)
        width = self.config.max_policy_targets
        indices = np.full((count, width), -1, dtype=np.int16)
        scores = np.zeros((count, width), dtype=np.int16)
        for row, sample in enumerate(self.samples):
            n = min(width, len(sample.policy_indices))
            indices[row, :n] = sample.policy_indices[:n]
            scores[row, :n] = sample.policy_scores[:n]
        save = np.savez_compressed if self.config.compressed else np.savez
        save(
            path,
            boards=np.stack([sample.board_pack for sample in self.samples]).astype(np.uint8, copy=False),
            score=np.asarray([sample.score for sample in self.samples], dtype=np.int16),
            depth=np.asarray([sample.depth for sample in self.samples], dtype=np.uint16),
            policy_indices=indices,
            policy_scores=scores,
            policy_depth=np.asarray([sample.policy_depth for sample in self.samples], dtype=np.uint16),
            value_knodes=np.asarray([sample.value_knodes for sample in self.samples], dtype=np.uint64),
            policy_knodes=np.asarray([sample.policy_knodes for sample in self.samples], dtype=np.uint64),
            position_hash=np.stack([np.frombuffer(position_key(sample.board_pack), dtype=np.uint8) for sample in self.samples]),
        )
        self.manifest.append({"path": f"{self.out_dir.name}/{path.name}", "split": self.out_dir.name,
                              "rows": count, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        self.samples.clear()
        self.shard_index += 1
        self.stats.shards_written += 1
        _eval_log(self.verbose, f"Wrote {path} with {count:,} sample(s)")


def _iter_samples(
    paths: Iterable[Path],
    config: EvalConfig,
    stats: EvalStats,
    *,
    workers: int,
    verbose: bool,
    max_records: int | None = None,
) -> Iterator[EvalSample]:
    """Stream samples from all files, optionally parsing chunks in a process pool."""

    def lines() -> Iterator[bytes]:
        for path in paths:
            _eval_log(verbose, f"Reading {path}")
            yield from iter_eval_lines(path)

    def account(chunk_stats: EvalStats) -> None:
        before = stats.lines_seen // PROGRESS_INTERVAL
        for name, count in asdict(chunk_stats).items():
            setattr(stats, name, getattr(stats, name) + count)
        if verbose and stats.lines_seen // PROGRESS_INTERVAL != before:
            _eval_log(
                verbose,
                f"seen={stats.lines_seen:,} kept={stats.positions_kept:,} shards={stats.shards_written:,}",
            )

    chunks = _chunked(islice(lines(), max_records), PARALLEL_CHUNK_LINES)
    if workers <= 1:
        for chunk in chunks:
            samples, chunk_stats = _process_chunk(chunk, config)
            account(chunk_stats)
            yield from samples
        return

    # Bounded window of in-flight chunks: Pool.imap has no backpressure and would
    # otherwise buffer the whole (multi-GB) dump ahead of the workers.
    with Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
        window: deque[tuple[AsyncResult, int]] = deque()
        limit = workers * 4
        for chunk in chunks:
            window.append((pool.apply_async(_process_chunk, (chunk,)), len(chunk)))
            if len(window) >= limit:
                result, _ = window.popleft()
                samples, chunk_stats = result.get()
                account(chunk_stats)
                yield from samples
        while window:
            result, _ = window.popleft()
            samples, chunk_stats = result.get()
            account(chunk_stats)
            yield from samples


_SAMPLE_HEADER = struct.Struct("<hHQHQH")


class _SampleStore:
    """Exact, bounded-RAM deduplication; retain the strongest whole sample.

    The composite key checks packed bytes as well as their digest. Clock/EP
    variants remain distinct samples but share a split. Hash-key ordering
    mixes positions deterministically before sharding.
    """

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size = -65536")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("""CREATE TABLE samples (
            position_id BLOB, board BLOB, quality BLOB NOT NULL, payload BLOB NOT NULL,
            duplicates INTEGER NOT NULL DEFAULT 0, conflicts INTEGER NOT NULL DEFAULT 0,
            replacements INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (position_id, board)
        ) WITHOUT ROWID""")
        self.pending = 0

    def add(self, sample: EvalSample) -> None:
        header = _SAMPLE_HEADER.pack(sample.score, sample.depth, sample.value_knodes,
                                     sample.policy_depth, sample.policy_knodes, len(sample.policy_indices))
        payload = header + np.asarray(sample.policy_indices, dtype="<i2").tobytes() + np.asarray(sample.policy_scores, dtype="<i2").tobytes()
        quality = struct.pack(">HQHHQ", sample.depth, sample.value_knodes, sample.policy_depth,
                              len(sample.policy_indices), sample.policy_knodes) + payload
        self.connection.execute("""INSERT INTO samples (position_id, board, quality, payload) VALUES (?, ?, ?, ?)
            ON CONFLICT(position_id, board) DO UPDATE SET
                duplicates = samples.duplicates + 1,
                conflicts = samples.conflicts + (substr(samples.payload, 1, 2) != substr(excluded.payload, 1, 2)),
                replacements = samples.replacements + (excluded.quality > samples.quality),
                payload = CASE WHEN excluded.quality > samples.quality THEN excluded.payload ELSE samples.payload END,
                quality = max(samples.quality, excluded.quality)
            """, (position_key(sample.board_pack), sample.board_pack.tobytes(), quality, payload))
        self.pending += 1
        if self.pending >= 4096:
            self.connection.commit()
            self.pending = 0

    def samples(self, stats: EvalStats) -> Iterator[EvalSample]:
        self.connection.commit()
        counts = self.connection.execute("SELECT coalesce(sum(duplicates), 0), coalesce(sum(conflicts), 0), coalesce(sum(replacements), 0) FROM samples").fetchone()
        stats.duplicate_records, stats.duplicate_score_conflicts, stats.duplicate_replacements = counts
        for board, payload in self.connection.execute("SELECT board, payload FROM samples ORDER BY position_id, board"):
            score, depth, value_knodes, policy_depth, policy_knodes, width = _SAMPLE_HEADER.unpack_from(payload)
            indices = np.frombuffer(payload, dtype="<i2", count=width, offset=_SAMPLE_HEADER.size).tolist()
            scores = np.frombuffer(payload, dtype="<i2", count=width, offset=_SAMPLE_HEADER.size + 2 * width).tolist()
            yield EvalSample(np.frombuffer(board, dtype=np.uint8), score, depth, indices, scores,
                             policy_depth, value_knodes, policy_knodes)

    def close(self) -> None:
        self.connection.close()


def preprocess_eval_files(
    paths: Iterable[Path],
    out_dir: Path,
    config: EvalConfig = EvalConfig(),
    *,
    max_positions: int | None = None,
    verbose: bool = False,
    workers: int = 1,
    max_records: int | None = None,
) -> EvalStats:
    """Publish an immutable, filtered ``evals-v2`` dataset from ``paths``.

    Workers parse in bounded, ordered chunks. The parent deduplicates and writes
    hash-ordered frozen splits, or preserves source order without deduplication.
    """

    started = time.perf_counter()
    if max_positions is not None and max_positions <= 0 or max_records is not None and max_records <= 0:
        raise ValueError("max_positions and max_records must be positive when supplied")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"refusing to mix datasets in nonempty output directory: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stats = EvalStats()
    paths = sorted(Path(path) for path in paths)
    sources = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths]
    with TemporaryDirectory(prefix=f".{out_dir.name}-build-", dir=out_dir.parent) as staging:
        stage = Path(staging)
        writers = {split: _ShardWriter(stage / split, config, stats, verbose) for split in SPLITS}
        store = _SampleStore(stage / "dedup.sqlite") if config.deduplicate else None
        ingested = 0
        dedup_bytes = 0

        def write(sample: EvalSample) -> None:
            split = split_for_key(position_key(sample.board_pack), config.validation_fraction, config.test_fraction, config.split_seed)
            writers[split].add(sample)

        samples = _iter_samples(paths, config, stats, workers=max(1, workers), verbose=verbose, max_records=max_records)
        try:
            for sample in samples:
                if store is not None:
                    store.add(sample)
                else:
                    write(sample)
                ingested += 1
                if max_positions is not None and ingested >= max_positions:
                    break
            if store is not None:
                _eval_log(verbose, f"Deduplicating {ingested:,} accepted records and writing frozen splits")
                for sample in store.samples(stats):
                    write(sample)
                dedup_bytes = (stage / "dedup.sqlite").stat().st_size
        finally:
            samples.close()
            if store is not None:
                store.close()
                (stage / "dedup.sqlite").unlink(missing_ok=True)
        stats.positions_unused_valid = stats.positions_valid - ingested
        for writer in writers.values():
            writer.flush()
        if stats.positions_kept == 0:
            raise ValueError("no usable positions; no dataset was published")
        for path, source in zip(paths, sources, strict=True):
            current = path.stat()
            if current.st_size != source["bytes"] or current.st_mtime_ns != source["mtime_ns"]:
                raise RuntimeError(f"source changed during preprocessing: {path}")
        metadata = {
            "dataset_schema": DATASET_SCHEMA,
            "status": "complete",
            "format": EVAL_SHARD_FORMAT,
            "source": LICHESS_EVAL_URL,
            "sources": sources,
            "limits": {"max_positions": max_positions, "max_records": max_records},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "chess": chess.__version__},
            "build": {"workers": max(1, workers), "elapsed_seconds": time.perf_counter() - started,
                      "dedup_database_bytes": dedup_bytes},
            "policy_size": POLICY_SIZE,
            "policy_square_order": "python-chess",
            "board_pack_bytes": int(pack_board(chess.Board()).size),
            "clock_planes_informative": False if stats.positions_missing_clocks == stats.positions_valid else None,
            "max_policy_targets": config.max_policy_targets,
            "config": asdict(config),
            "stats": asdict(stats),
            "split_method": "blake2b-clock-ep-group-v1",
            "shards": [entry for writer in writers.values() for entry in writer.manifest],
        }
        metadata["dataset_id"] = manifest_identity(metadata)
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(out_dir)
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
    workers: int = 1,
    max_records: int | None = None,
) -> EvalStats:
    paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".jsonl") or ".jsonl." in path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no lichess eval jsonl(.zst/.gz) files found under {raw_dir}")
    return preprocess_eval_files(
        paths, out_dir, config, max_positions=max_positions, verbose=verbose, workers=workers, max_records=max_records
    )
