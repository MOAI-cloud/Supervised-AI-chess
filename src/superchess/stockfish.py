"""Local Stockfish integration using the strength presets from Lichess fishnet."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import chess
import chess.engine as chess_engine


@dataclass(frozen=True, slots=True)
class LichessStockfishLevel:
    """The UCI limits used by Lichess fishnet for one AI level."""

    level: int
    skill: int
    move_time_ms: int
    depth: int


# Source: lichess-org/fishnet, src/api.rs, SkillLevel (checked 2026-07-11).
# Fishnet sends both movetime and depth, so Stockfish stops at whichever limit
# is reached first. It also uses one PV. Some official Stockfish builds
# advertise Skill Level 0..20; those builds cannot accept Lichess's negative
# skills for levels 1-3, and the handle reports the effective value explicitly.
LICHESS_STOCKFISH_LEVELS: dict[int, LichessStockfishLevel] = {
    1: LichessStockfishLevel(1, -9, 50, 5),
    2: LichessStockfishLevel(2, -5, 100, 5),
    3: LichessStockfishLevel(3, -1, 150, 5),
    4: LichessStockfishLevel(4, 3, 200, 5),
    5: LichessStockfishLevel(5, 7, 300, 5),
    6: LichessStockfishLevel(6, 11, 400, 8),
    7: LichessStockfishLevel(7, 16, 500, 13),
    8: LichessStockfishLevel(8, 20, 1000, 22),
}


class StockfishUnavailableError(RuntimeError):
    """Raised when the configured Stockfish process cannot be used."""


@dataclass(frozen=True, slots=True)
class StockfishPlayResult:
    move: chess.Move
    info: dict[str, Any]
    preset: LichessStockfishLevel
    effective_skill: int
    exact_skill: bool
    elapsed_ms: int


@dataclass
class StockfishHandle:
    """Own one lazy, serialized UCI process for GUI requests."""

    command: str | Path = "stockfish"
    _engine: chess_engine.SimpleEngine | None = field(default=None, init=False, repr=False)
    _resolved_path: Path | None = field(default=None, init=False, repr=False)
    _error: str | None = field(default=None, init=False, repr=False)
    _name: str | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _resolve_command(self) -> Path | None:
        if self._resolved_path is not None:
            return self._resolved_path

        raw = os.fspath(self.command)
        expanded = Path(raw).expanduser()
        has_directory = expanded.is_absolute() or expanded.parent != Path(".")
        if has_directory:
            candidate = expanded.resolve()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                self._resolved_path = candidate
                return candidate
            self._error = f"Stockfish executable is not runnable: {candidate}"
            return None

        found = shutil.which(raw)
        if found:
            self._resolved_path = Path(found).resolve()
            return self._resolved_path

        # Debian/Ubuntu installs games outside PATH on some configurations.
        if raw == "stockfish":
            for candidate in (Path("/usr/games/stockfish"), Path("/usr/local/games/stockfish")):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    self._resolved_path = candidate
                    return candidate

        self._error = (
            "Stockfish was not found. Install it or launch the GUI with "
            "--stockfish /path/to/stockfish."
        )
        return None

    def status(self, *, probe: bool = True) -> dict[str, Any]:
        """Return availability, UCI capabilities, and exact Lichess presets."""

        path = self._resolve_command()
        engine = self._engine
        if probe and path is not None and engine is None:
            try:
                with self._lock:
                    engine = self._ensure_loaded()
            except StockfishUnavailableError:
                engine = None

        skill_option = engine.options.get("Skill Level") if engine is not None else None
        skill_min = int(skill_option.min) if skill_option is not None and skill_option.min is not None else None
        skill_max = int(skill_option.max) if skill_option is not None and skill_option.max is not None else None
        levels = []
        for level in range(1, 9):
            preset = LICHESS_STOCKFISH_LEVELS[level]
            effective = preset.skill
            if skill_min is not None:
                effective = max(skill_min, effective)
            if skill_max is not None:
                effective = min(skill_max, effective)
            levels.append(
                {
                    **asdict(preset),
                    "effective_skill": effective,
                    "exact_skill": effective == preset.skill,
                }
            )
        return {
            "configured": os.fspath(self.command),
            "path": str(path) if path else None,
            "available": path is not None and self._error is None,
            "ready": engine is not None,
            "name": self._name,
            "error": self._error,
            "skill_min": skill_min,
            "skill_max": skill_max,
            "levels": levels,
            "lichess_defaults": {
                "threads_per_process": 1,
                "hash_mb": 16,
                "multipv": 1,
                "uci_limit_strength": False,
                "stop_condition": "movetime or depth, whichever comes first",
            },
        }

    def _ensure_loaded(self) -> chess_engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        path = self._resolve_command()
        if path is None:
            raise StockfishUnavailableError(self._error or "Stockfish is unavailable")
        try:
            self._engine = chess_engine.SimpleEngine.popen_uci(str(path), timeout=10.0)
        except (OSError, TimeoutError, chess_engine.EngineError) as exc:
            self._error = f"Could not start Stockfish at {path}: {exc}"
            raise StockfishUnavailableError(self._error) from exc
        self._name = self._engine.id.get("name") or "Stockfish"
        startup_options: dict[str, Any] = {}
        if "Threads" in self._engine.options:
            startup_options["Threads"] = 1
        if "Hash" in self._engine.options:
            startup_options["Hash"] = 16
        if startup_options:
            self._engine.configure(startup_options)
        self._error = None
        return self._engine

    @staticmethod
    def _effective_skill(engine: chess_engine.SimpleEngine, requested: int) -> tuple[int, bool]:
        option = engine.options.get("Skill Level")
        if option is None:
            raise StockfishUnavailableError("The configured UCI engine has no 'Skill Level' option")
        minimum = requested if option.min is None else int(option.min)
        maximum = requested if option.max is None else int(option.max)
        effective = max(minimum, min(maximum, requested))
        return effective, effective == requested

    def play(self, board: chess.Board, level: int, *, game_id: object | None = None) -> StockfishPlayResult:
        """Choose a move with Lichess's skill, time, and depth limits."""

        try:
            preset = LICHESS_STOCKFISH_LEVELS[int(level)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stockfish level must be an integer from 1 to 8") from exc
        if board.is_game_over(claim_draw=True):
            raise ValueError("game is already over")

        with self._lock:
            engine = self._ensure_loaded()
            effective_skill, exact_skill = self._effective_skill(engine, preset.skill)
            options: dict[str, Any] = {"Skill Level": effective_skill}
            if "UCI_LimitStrength" in engine.options:
                options["UCI_LimitStrength"] = False

            started = time.perf_counter()
            try:
                result = engine.play(
                    board,
                    chess_engine.Limit(time=preset.move_time_ms / 1000.0, depth=preset.depth),
                    game=game_id,
                    info=chess_engine.INFO_ALL,
                    options=options,
                )
            except chess_engine.EngineTerminatedError as exc:
                self._engine = None
                self._error = f"Stockfish terminated during search: {exc}"
                raise StockfishUnavailableError(self._error) from exc
            elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))

        if result.move is None or result.move not in board.legal_moves:
            raise StockfishUnavailableError("Stockfish did not return a legal move")
        return StockfishPlayResult(
            move=result.move,
            info=result.info,
            preset=preset,
            effective_skill=effective_skill,
            exact_skill=exact_skill,
            elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Stop the child process, if it was started."""

        with self._lock:
            engine, self._engine = self._engine, None
            if engine is None:
                return
            try:
                engine.quit()
            except (OSError, TimeoutError, chess_engine.EngineError):
                try:
                    engine.close()
                except Exception:
                    pass
