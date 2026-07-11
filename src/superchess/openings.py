"""Stable opening classification using the pinned Lichess CC0 opening database."""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn as chess_pgn
import requests

# Pin the same curated database used by Lichess so names do not drift between
# runs. The data is CC0: https://github.com/lichess-org/chess-openings
LICHESS_OPENINGS_REVISION = "17ee660257de02870636f36248e919f2e01d8e85"
_LICHESS_OPENINGS_URL = (
    "https://raw.githubusercontent.com/lichess-org/chess-openings/"
    f"{LICHESS_OPENINGS_REVISION}/{{volume}}.tsv"
)


@dataclass(frozen=True, slots=True)
class OpeningInfo:
    eco: str
    name: str
    matched_ply: int

    @property
    def display_name(self) -> str:
        return f"{self.eco} · {self.name}" if self.eco else self.name


class OpeningBook:
    """Position-indexed ECO database with a persistent user cache.

    Classification follows the Lichess dataset recommendation: walk the game
    backwards and return the most recent named position. Once a game leaves
    theory, the last real opening therefore remains displayed throughout the
    middlegame instead of disappearing or being guessed from later moves.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path or _default_cache_path()
        self._by_epd: dict[str, tuple[str, str]] = {}
        self._loaded = False
        self._lock = threading.Lock()
        self.source = "fallback"

    @property
    def count(self) -> int:
        self._load_cache()
        return len(self._by_epd)

    def _load_cache(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if payload.get("revision") == LICHESS_OPENINGS_REVISION:
                    entries = payload.get("entries", {})
                    self._by_epd = {
                        epd: (str(value[0]), str(value[1]))
                        for epd, value in entries.items()
                        if isinstance(value, list) and len(value) == 2
                    }
                    self.source = "lichess-cache"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._by_epd = {}
            self._loaded = True

    def refresh(self, *, timeout: float = 15.0) -> int:
        """Download and cache the pinned Lichess opening database."""

        entries: dict[str, tuple[str, str]] = {}
        headers = {"User-Agent": "superchess/0.1 opening-cache"}
        for volume in "abcde":
            response = requests.get(
                _LICHESS_OPENINGS_URL.format(volume=volume),
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            reader = csv.DictReader(io.StringIO(response.text), delimiter="\t")
            for row in reader:
                eco = (row.get("eco") or "").strip()
                name = (row.get("name") or "").strip()
                pgn = (row.get("pgn") or "").strip()
                if not eco or not name or not pgn:
                    continue
                game = chess_pgn.read_game(io.StringIO(pgn))
                if game is None:
                    continue
                board = game.end().board()
                entries[board.epd()] = (eco, name)

        if not entries:
            raise RuntimeError("Lichess opening download produced no entries")

        payload: dict[str, Any] = {
            "revision": LICHESS_OPENINGS_REVISION,
            "license": "CC0-1.0",
            "source": "https://github.com/lichess-org/chess-openings",
            "entries": {epd: list(value) for epd, value in entries.items()},
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.cache_path)

        with self._lock:
            self._by_epd = entries
            self._loaded = True
            self.source = "lichess-cache"
        return len(entries)

    def classify(self, board: chess.Board) -> OpeningInfo | None:
        """Return the last named position in ``board``'s actual move history."""

        self._load_cache()
        if not self._by_epd:
            return None

        probe = board.copy(stack=True)
        while True:
            entry = self._by_epd.get(probe.epd())
            if entry is not None:
                eco, name = entry
                return OpeningInfo(eco=eco, name=name, matched_ply=len(probe.move_stack))
            if not probe.move_stack:
                return None
            probe.pop()

    def status(self) -> dict[str, Any]:
        self._load_cache()
        return {
            "source": self.source,
            "revision": LICHESS_OPENINGS_REVISION,
            "entries": len(self._by_epd),
            "cache": str(self.cache_path),
        }


def _default_cache_path() -> Path:
    override = os.environ.get("SUPERCHESS_OPENINGS_CACHE")
    if override:
        return Path(override).expanduser()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "superchess" / f"lichess-openings-{LICHESS_OPENINGS_REVISION}.json"


OPENING_BOOK = OpeningBook()


def ensure_opening_database() -> dict[str, Any]:
    """Ensure the full Lichess database is cached, retaining offline fallback."""

    if OPENING_BOOK.count == 0:
        try:
            OPENING_BOOK.refresh()
        except (OSError, requests.RequestException, RuntimeError, ValueError):
            pass
    return OPENING_BOOK.status()
