from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
import chess
import chess.pgn
import numpy as np
import requests
from tqdm import tqdm

from superchess.encoding import POLICY_SIZE, move_to_policy, pack_board

CCRL_BASE_URL = "https://computerchess.org.uk/4040/"
CCRL_RATING_LIST_URL = urljoin(CCRL_BASE_URL, "rating_list_all.html")
CCRL_GAMES_URL = urljoin(CCRL_BASE_URL, "games.html")
CCRL_COMMENTED_ARCHIVE_URL = urljoin(CCRL_BASE_URL, "CCRL-4040-commented.[2349311].pgn.7z")
USER_AGENT = "superchess/0.1 (+https://computerchess.org.uk/ccrl/)"
RESULTS = {"1-0", "0-1", "1/2-1/2"}
PREPROCESS_PROGRESS_INTERVAL = 1_000
COMMENTED_ENGINE_ARCHIVE_RE = re.compile(r"(?P<name>.+?)\.commented\.\[(?P<games>\d+)\]\.pgn\.7z$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EngineRating:
    name: str
    elo: int


@dataclass(frozen=True, slots=True)
class EngineArchive:
    name: str
    games: int
    commented_url: str
    size_mb: float | None = None


@dataclass(slots=True)
class PreprocessStats:
    games_seen: int = 0
    games_kept: int = 0
    games_skipped_rating: int = 0
    games_skipped_result: int = 0
    games_skipped_duplicate: int = 0
    games_skipped_error: int = 0
    samples: int = 0
    shards_written: int = 0


def _preprocess_log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[preprocess] {message}", file=sys.stderr, flush=True)


def _preprocess_progress(stats: PreprocessStats) -> str:
    skipped = (
        stats.games_skipped_rating
        + stats.games_skipped_result
        + stats.games_skipped_duplicate
        + stats.games_skipped_error
    )
    return (
        f"seen={stats.games_seen:,} kept={stats.games_kept:,} skipped={skipped:,} "
        f"(rating={stats.games_skipped_rating:,} result={stats.games_skipped_result:,} "
        f"duplicate={stats.games_skipped_duplicate:,} error={stats.games_skipped_error:,}) "
        f"samples={stats.samples:,} shards={stats.shards_written:,}"
    )


def clean_text(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def engine_key(name: str) -> str:
    return clean_text(name).casefold()


def engine_match_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", engine_key(name)).strip()


def parse_int(text: str) -> int | None:
    match = re.search(r"\d[\d']*", clean_text(text))
    if match is None:
        return None
    return int(match.group(0).replace("'", ""))


def parse_float(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", clean_text(text))
    return None if match is None else float(match.group(0))


def parse_size_mb(text: str) -> float | None:
    size = parse_float(text)
    if size is None:
        return None
    normalized = clean_text(text).casefold()
    if "gb" in normalized:
        return size * 1024
    if "kb" in normalized:
        return size / 1024
    return size


def parse_rating_list(html_text: str) -> dict[str, EngineRating]:
    soup = BeautifulSoup(html_text, "html.parser")
    ratings: dict[str, EngineRating] = {}

    for row in soup.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if len(cells) < 3 or cells[0].casefold() in {"rank", "#"}:
            continue
        rank = parse_int(cells[0])
        elo = parse_int(cells[2])
        if rank is None or elo is None or not 1000 <= elo <= 5000:
            continue
        name = cells[1]
        if name:
            ratings[engine_key(name)] = EngineRating(name=name, elo=elo)

    if ratings:
        return ratings

    line_pattern = re.compile(r"^\s*\d+\s+(.+?)\s+([1-4]\d{3})\b")
    for line in soup.get_text("\n").splitlines():
        match = line_pattern.match(clean_text(line))
        if not match:
            continue
        name, elo_text = match.groups()
        ratings[engine_key(name)] = EngineRating(name=name, elo=int(elo_text))
    return ratings


def parse_engine_archives(html_text: str, base_url: str = CCRL_GAMES_URL) -> dict[str, EngineArchive]:
    soup = BeautifulSoup(html_text, "html.parser")
    archives: dict[str, EngineArchive] = {}

    for table in soup.find_all("table"):
        headers = [clean_text(header.get_text(" ", strip=True)).casefold() for header in table.find_all("th")]
        if not any(header.startswith("engine") for header in headers):
            continue
        if not any("commented" in header for header in headers):
            continue

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            name = clean_text(cells[0].get_text(" ", strip=True))
            games = parse_int(cells[1].get_text(" ", strip=True))
            link = cells[-1].find("a", href=True)
            if not name or games is None or link is None:
                continue
            archives[engine_key(name)] = EngineArchive(
                name=name,
                games=games,
                commented_url=urljoin(base_url, link["href"]),
                size_mb=parse_size_mb(cells[-1].get_text(" ", strip=True)),
            )

    for link in soup.find_all("a", href=True):
        archive = parse_engine_archive_link(link["href"], link.get_text(" ", strip=True), base_url)
        if archive is not None:
            archives.setdefault(engine_key(archive.name), archive)
    return archives


def parse_engine_archive_link(href: str, label: str, base_url: str = CCRL_GAMES_URL) -> EngineArchive | None:
    filename = unquote(Path(urlparse(href).path).name)
    match = COMMENTED_ENGINE_ARCHIVE_RE.match(filename)
    if match is None:
        return None
    name = clean_text(match.group("name").replace("_", " "))
    return EngineArchive(
        name=name,
        games=int(match.group("games")),
        commented_url=urljoin(base_url, href),
        size_mb=parse_size_mb(label),
    )


def high_elo_archives(
    ratings: dict[str, EngineRating],
    archives: dict[str, EngineArchive],
    min_elo: int = 3500,
    max_archives: int | None = None,
) -> list[EngineArchive]:
    archive_by_match_key = {engine_match_key(archive.name): archive for archive in archives.values()}
    selected: list[tuple[EngineRating, EngineArchive]] = []
    for rating in ratings.values():
        if rating.elo < min_elo:
            continue
        archive = archive_by_match_key.get(engine_match_key(rating.name))
        if archive is None:
            continue
        selected.append(
            (
                rating,
                EngineArchive(
                    name=rating.name,
                    games=archive.games,
                    commented_url=archive.commented_url,
                    size_mb=archive.size_mb,
                ),
            )
        )
    selected.sort(key=lambda pair: (-pair[0].elo, pair[0].name.casefold()))
    archives_by_rating = [archive for _, archive in selected]
    return archives_by_rating if max_archives is None else archives_by_rating[:max_archives]


def fetch_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(10, 60))
    response.raise_for_status()
    return response.text


def save_ratings_json(ratings: dict[str, EngineRating], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((asdict(rating) for rating in ratings.values()), key=lambda row: (-row["elo"], row["name"].casefold()))
    path.write_text(json.dumps({"source": CCRL_RATING_LIST_URL, "ratings": rows}, indent=2), encoding="utf-8")


def load_ratings_json(path: Path) -> dict[str, EngineRating]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["ratings"] if isinstance(payload, dict) else payload
    return {engine_key(row["name"]): EngineRating(name=row["name"], elo=int(row["elo"])) for row in rows}


def archive_filename(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


def download_url(url: str, destination: Path, *, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with temporary.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)
    return destination


def download_ccrl(
    out_dir: Path,
    *,
    min_elo: int = 3500,
    prefer_engine_archives: bool = True,
    include_all_games: bool = False,
    max_archives: int | None = None,
    overwrite: bool = False,
    polite_delay_seconds: float = 0.5,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings = parse_rating_list(fetch_text(CCRL_RATING_LIST_URL))
    save_ratings_json(ratings, out_dir / "ccrl_ratings.json")

    downloads: list[dict[str, object]] = []
    if prefer_engine_archives:
        archives = parse_engine_archives(fetch_text(CCRL_GAMES_URL))
        for archive in high_elo_archives(ratings, archives, min_elo=min_elo, max_archives=max_archives):
            path = download_url(archive.commented_url, out_dir / archive_filename(archive.commented_url), overwrite=overwrite)
            downloads.append({"engine": archive.name, "url": archive.commented_url, "path": str(path), "games": archive.games})
            if polite_delay_seconds > 0:
                time.sleep(polite_delay_seconds)

    if include_all_games or not downloads:
        path = download_url(CCRL_COMMENTED_ARCHIVE_URL, out_dir / archive_filename(CCRL_COMMENTED_ARCHIVE_URL), overwrite=overwrite)
        downloads.append({"engine": None, "url": CCRL_COMMENTED_ARCHIVE_URL, "path": str(path), "games": None})

    manifest = {
        "min_elo": min_elo,
        "ratings_url": CCRL_RATING_LIST_URL,
        "games_url": CCRL_GAMES_URL,
        "downloads": downloads,
    }
    (out_dir / "ccrl_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def extract_7z_archive(archive_path: Path, out_dir: Path, *, verbose: bool = False) -> list[Path]:
    extract_dir = out_dir / "_extracted" / archive_path.name.removesuffix(".7z")
    existing = sorted(extract_dir.rglob("*.pgn")) if extract_dir.exists() else []
    if existing:
        _preprocess_log(verbose, f"Using {len(existing):,} extracted PGN(s) from {archive_path.name}")
        return existing

    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError('Install extraction support with: python -m pip install -e ".[extract]"') from exc

    extract_dir.mkdir(parents=True, exist_ok=True)
    _preprocess_log(verbose, f"Extracting {archive_path.name} to {extract_dir}")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)
    return sorted(extract_dir.rglob("*.pgn"))


def collect_pgn_paths(raw_dir: Path, *, extract_archives: bool = True, verbose: bool = False) -> list[Path]:
    pgn_paths = sorted(path for path in raw_dir.rglob("*.pgn") if "_extracted" not in path.parts)
    _preprocess_log(verbose, f"Found {len(pgn_paths):,} PGN file(s) directly under {raw_dir}")
    if extract_archives:
        archive_paths = sorted(raw_dir.rglob("*.7z"))
        _preprocess_log(verbose, f"Found {len(archive_paths):,} 7z archive(s) under {raw_dir}")
        for archive_path in archive_paths:
            pgn_paths.extend(extract_7z_archive(archive_path, raw_dir, verbose=verbose))
    unique: dict[Path, Path] = {}
    for path in pgn_paths:
        unique[path.resolve()] = path
    return sorted(unique.values())


def preprocess_raw_directory(
    raw_dir: Path,
    out_dir: Path,
    *,
    ratings_path: Path | None = None,
    min_elo: int = 3500,
    shard_size: int = 65_536,
    compressed: bool = False,
    extract_archives: bool = True,
    max_games: int | None = None,
    max_positions: int | None = None,
    verbose: bool = False,
) -> PreprocessStats:
    ratings_path = ratings_path or raw_dir / "ccrl_ratings.json"
    if ratings_path.exists():
        _preprocess_log(verbose, f"Loading ratings from {ratings_path}")
        ratings = load_ratings_json(ratings_path)
    else:
        _preprocess_log(verbose, f"Fetching CCRL ratings because {ratings_path} is missing")
        ratings = parse_rating_list(fetch_text(CCRL_RATING_LIST_URL))
        save_ratings_json(ratings, ratings_path)
    _preprocess_log(verbose, f"Loaded {len(ratings):,} engine rating(s)")

    _preprocess_log(verbose, f"Collecting PGN files under {raw_dir}")
    pgn_paths = collect_pgn_paths(raw_dir, extract_archives=extract_archives, verbose=verbose)
    if not pgn_paths:
        raise FileNotFoundError(f"no PGN files found under {raw_dir}")
    _preprocess_log(verbose, f"Preprocessing {len(pgn_paths):,} PGN file(s)")

    stats = preprocess_pgn_files(
        pgn_paths,
        ratings,
        out_dir,
        min_elo=min_elo,
        shard_size=shard_size,
        compressed=compressed,
        max_games=max_games,
        max_positions=max_positions,
        verbose=verbose,
    )
    metadata = {
        "min_elo": min_elo,
        "policy_size": POLICY_SIZE,
        "board_pack_bytes": int(pack_board(chess.Board()).size),
        "compressed": compressed,
        "stats": asdict(stats),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _preprocess_log(verbose, f"Wrote metadata to {metadata_path}")
    return stats


def preprocess_pgn_files(
    pgn_paths: Iterable[Path],
    ratings: dict[str, EngineRating],
    out_dir: Path,
    *,
    min_elo: int = 3500,
    shard_size: int = 65_536,
    compressed: bool = False,
    max_games: int | None = None,
    max_positions: int | None = None,
    verbose: bool = False,
) -> PreprocessStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = PreprocessStats()
    seen_games: set[tuple[str, ...]] = set()
    boards: list[np.ndarray] = []
    policies: list[int] = []
    values: list[float] = []
    plies: list[int] = []
    shard_index = 0
    stop = False

    _preprocess_log(verbose, f"Writing shards to {out_dir} (shard_size={shard_size:,}, compressed={compressed})")

    def flush() -> None:
        nonlocal shard_index
        if not boards:
            return
        path = out_dir / f"shard-{shard_index:05d}.npz"
        shard_samples = len(boards)
        save = np.savez_compressed if compressed else np.savez
        save(
            path,
            boards=np.stack(boards).astype(np.uint8, copy=False),
            policies=np.asarray(policies, dtype=np.uint16),
            values=np.asarray(values, dtype=np.float32),
            plies=np.asarray(plies, dtype=np.uint16),
        )
        boards.clear()
        policies.clear()
        values.clear()
        plies.clear()
        shard_index += 1
        stats.shards_written += 1
        _preprocess_log(verbose, f"Wrote {path} with {shard_samples:,} sample(s)")

    for pgn_path in pgn_paths:
        if stop:
            break
        _preprocess_log(verbose, f"Processing {pgn_path}")
        with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                if max_games is not None and stats.games_seen >= max_games:
                    stop = True
                    break
                game = chess.pgn.read_game(handle)
                if game is None:
                    break

                stats.games_seen += 1
                if stats.games_seen % PREPROCESS_PROGRESS_INTERVAL == 0:
                    _preprocess_log(verbose, _preprocess_progress(stats))
                result = game.headers.get("Result", "*")
                if result not in RESULTS:
                    stats.games_skipped_result += 1
                    continue
                if not is_high_elo_game(game.headers, ratings, min_elo=min_elo):
                    stats.games_skipped_rating += 1
                    continue

                key = game_key(game)
                if key in seen_games:
                    stats.games_skipped_duplicate += 1
                    continue
                seen_games.add(key)

                try:
                    game_samples = samples_from_game(game, result)
                except ValueError:
                    stats.games_skipped_error += 1
                    continue

                stats.games_kept += 1
                for board_pack, policy, value, ply in game_samples:
                    boards.append(board_pack)
                    policies.append(policy)
                    values.append(value)
                    plies.append(ply)
                    stats.samples += 1
                    if len(boards) >= shard_size:
                        flush()
                    if max_positions is not None and stats.samples >= max_positions:
                        stop = True
                        break
                if stop:
                    break

    flush()
    _preprocess_log(verbose, f"Finished: {_preprocess_progress(stats)}")
    return stats


def is_high_elo_game(headers: chess.pgn.Headers, ratings: dict[str, EngineRating], *, min_elo: int) -> bool:
    white_elo = rating_for(headers.get("White", ""), "White", headers, ratings)
    black_elo = rating_for(headers.get("Black", ""), "Black", headers, ratings)
    return white_elo is not None and black_elo is not None and white_elo >= min_elo and black_elo >= min_elo


def rating_for(name: str, side: str, headers: chess.pgn.Headers, ratings: dict[str, EngineRating]) -> int | None:
    rating = ratings.get(engine_key(name))
    if rating is not None:
        return rating.elo
    for header_name in (f"{side}Elo", f"{side}Rating"):
        value = headers.get(header_name)
        if value:
            parsed = parse_int(value)
            if parsed is not None:
                return parsed
    return None


def game_key(game: chess.pgn.Game) -> tuple[str, ...]:
    headers = game.headers
    moves = " ".join(move.uci() for move in game.mainline_moves())
    return tuple(headers.get(name, "") for name in ("Date", "White", "Black", "Result")) + (game.board().fen(), moves)


def samples_from_game(game: chess.pgn.Game, result: str) -> list[tuple[np.ndarray, int, float, int]]:
    board = game.board()
    samples: list[tuple[np.ndarray, int, float, int]] = []
    for ply, move in enumerate(game.mainline_moves()):
        if not board.is_legal(move):
            raise ValueError(f"illegal move {move.uci()} in {game.headers.get('White', '?')} - {game.headers.get('Black', '?')}")
        policy = move_to_policy(board, move).index
        samples.append((pack_board(board), policy, result_value(result, board.turn), ply))
        board.push(move)
    return samples


def result_value(result: str, turn: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    side_won = white_won if turn == chess.WHITE else not white_won
    return 1.0 if side_won else -1.0