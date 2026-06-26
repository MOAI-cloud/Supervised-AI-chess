"""Web GUI server for playing against the Superchess neural MCTS engine.

Dependency-free backend built on the Python standard library. Chess rules are
handled by python-chess (already a project dependency) and moves are produced by
the neural MCTS engine. Serves a polished single-page frontend from ``web/``.
"""

from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chess

WEB_ROOT = Path(__file__).resolve().parent / "web"

# Lc0-style mapping from a [-1, 1] value head to centipawns.
_CP_SCALE = 111.714640912
_CP_SLOPE = 1.5620688421


def value_to_cp(value: float) -> int:
    """Convert a side-to-move value in [-1, 1] to an integer centipawn score."""

    value = max(-0.9999, min(0.9999, float(value)))
    cp = _CP_SCALE * math.tan(_CP_SLOPE * value)
    return int(max(-12000, min(12000, round(cp))))


# Minimal opening recognition keyed by the opening move sequence (SAN).
_OPENINGS: dict[str, str] = {
    "e4": "King's Pawn",
    "e4 c5": "Sicilian Defence",
    "e4 c5 Nf3 d6": "Sicilian, Najdorf-style",
    "e4 e5": "Open Game",
    "e4 e5 Nf3 Nc6 Bb5": "Ruy López",
    "e4 e5 Nf3 Nc6 Bc4": "Italian Game",
    "e4 e6": "French Defence",
    "e4 c6": "Caro-Kann Defence",
    "e4 d5": "Scandinavian Defence",
    "e4 g6": "Modern Defence",
    "e4 d6": "Pirc Defence",
    "d4": "Queen's Pawn",
    "d4 d5": "Closed Game",
    "d4 d5 c4": "Queen's Gambit",
    "d4 d5 c4 e6": "Queen's Gambit Declined",
    "d4 d5 c4 dxc4": "Queen's Gambit Accepted",
    "d4 Nf6": "Indian Defence",
    "d4 Nf6 c4 g6": "King's Indian / Grünfeld",
    "d4 Nf6 c4 e6": "Nimzo/Queen's Indian",
    "d4 f5": "Dutch Defence",
    "c4": "English Opening",
    "Nf3": "Réti Opening",
    "g3": "King's Fianchetto",
    "b3": "Larsen's Opening",
    "f4": "Bird's Opening",
}


def detect_opening(board: chess.Board) -> str | None:
    """Best-effort opening name from the SAN move list."""

    try:
        sans: list[str] = []
        probe = chess.Board()
        for move in board.move_stack:
            sans.append(probe.san(move))
            probe.push(move)
    except Exception:  # pragma: no cover - defensive
        return None

    best: str | None = None
    for length in range(1, min(len(sans), 8) + 1):
        key = " ".join(sans[:length])
        if key in _OPENINGS:
            best = _OPENINGS[key]
    return best

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".woff2": "font/woff2",
}


@dataclass
class EngineHandle:
    """Lazily loaded engine state shared across requests."""

    checkpoint: Path
    device: str | None = None
    _model: Any = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def ensure_loaded(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from superchess.training import load_model_checkpoint

                    model, _ = load_model_checkpoint(self.checkpoint, device_name=self.device)
                    self._model = model
        return self._model

    def search(self, board: chess.Board, simulations: int, c_puct: float, temperature: float):
        from superchess.mcts import NeuralMCTS, SearchConfig

        model = self.ensure_loaded()
        config = SearchConfig(simulations=simulations, c_puct=c_puct, temperature=temperature)
        # NeuralMCTS keeps no per-search state, build per request to stay thread-safe.
        engine = NeuralMCTS(model, config)
        return engine

    def evaluate(self, board: chess.Board):
        from superchess.mcts import NeuralMCTS

        model = self.ensure_loaded()
        return NeuralMCTS(model).evaluate(board)


def _board_state(board: chess.Board) -> dict[str, Any]:
    """Serialize legality and status info the frontend needs."""

    legal: dict[str, list[str]] = {}
    for move in board.legal_moves:
        legal.setdefault(chess.square_name(move.from_square), []).append(move.uci())

    outcome = board.outcome(claim_draw=True)
    is_over = outcome is not None
    result = outcome.result() if outcome else "*"
    termination = outcome.termination.name.lower() if outcome else None

    last = board.peek().uci() if board.move_stack else None
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal": legal,
        "in_check": board.is_check(),
        "check_square": chess.square_name(board.king(board.turn)) if board.is_check() else None,
        "is_over": is_over,
        "result": result,
        "termination": termination,
        "last_move": last,
        "fullmove": board.fullmove_number,
        "halfmove": board.halfmove_clock,
        "opening": detect_opening(board),
        "ply": len(board.move_stack),
    }


def _line_from_pv(board: chess.Board, pv: list[chess.Move]) -> dict[str, Any]:
    """Build SAN + UCI strings and detect a forced mate along ``pv``."""

    probe = board.copy(stack=False)
    sans: list[str] = []
    ucis: list[str] = []
    mate: int | None = None
    for idx, move in enumerate(pv):
        if move not in probe.legal_moves:
            break
        sans.append(probe.san(move))
        ucis.append(move.uci())
        probe.push(move)
        if probe.is_checkmate():
            # Plies until mate -> moves, signed from the searching side's view.
            plies = idx + 1
            moves_to_mate = (plies + 1) // 2
            mate = moves_to_mate if plies % 2 == 1 else -moves_to_mate
            break
    return {"san": sans, "uci": ucis, "mate": mate}


def build_analysis(
    engine: "EngineHandle",
    board: chess.Board,
    simulations: int,
    c_puct: float,
    temperature: float,
    multipv: int,
    pv_length: int = 12,
) -> dict[str, Any]:
    """Run a search and return Stockfish-style multi-PV analysis for ``board``."""

    started = time.perf_counter()
    searcher = engine.search(board, simulations, c_puct, temperature)
    result = searcher.search(board)
    elapsed = max(1e-6, time.perf_counter() - started)

    total_visits = sum(result.visits.values()) or 1
    ranked = sorted(
        result.visits.items(),
        key=lambda kv: (kv[1], result.policy.get(kv[0], 0.0)),
        reverse=True,
    )

    root = result.root
    lines: list[dict[str, Any]] = []
    for move, visits in ranked[: max(1, multipv)]:
        child = root.children.get(move) if root else None
        # Evaluate the child value (negate: child value is from opponent's view).
        cp = None
        if child is not None and child.visit_count > 0:
            cp = value_to_cp(-child.value)
        pv_moves = [move]
        if child is not None:
            after = board.copy(stack=False)
            after.push(move)
            pv_moves += searcher.principal_variation(child, after, max_len=pv_length - 1)
        info = _line_from_pv(board, pv_moves)
        if cp is None:
            cp = value_to_cp(0.0)
        lines.append(
            {
                "cp": cp,
                "mate": info["mate"],
                "san": info["san"],
                "uci": info["uci"],
                "visits": visits,
                "share": visits / total_visits,
            }
        )

    nps = int(total_visits / elapsed)
    # A coarse "depth" proxy: average PV length weighted toward the main line.
    depth = max((len(line["uci"]) for line in lines), default=1)
    return {
        "lines": lines,
        "nodes": total_visits,
        "nps": nps,
        "depth": depth,
        "time_ms": int(elapsed * 1000),
        "best_move": result.best_move.uci(),
        "multipv": len(lines),
    }


def _board_from_payload(payload: dict[str, Any]) -> chess.Board:
    fen = payload.get("fen", "startpos")
    if not fen or fen == "startpos":
        return chess.Board()
    return chess.Board(fen)


class _Handler(BaseHTTPRequestHandler):
    server_version = "Superchess/1.0"
    engine: EngineHandle  # injected on the server instance

    # Silence the default noisy logging; keep errors only.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    # ---- helpers -------------------------------------------------------
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _serve_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        target = (WEB_ROOT / rel).resolve()
        # Prevent path traversal outside the web root.
        if WEB_ROOT not in target.parents and target != WEB_ROOT:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- routing -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/info":
            self._send_json(self._engine_info())
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/state":
                self._handle_state()
            elif path == "/api/move":
                self._handle_move()
            elif path == "/api/engine":
                self._handle_engine()
            elif path == "/api/analyze":
                self._handle_analyze()
            elif path == "/api/eval":
                self._handle_eval()
            elif path == "/api/import":
                self._handle_import()
            else:
                self.send_error(404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json({"error": f"internal error: {exc}"}, status=500)

    # ---- endpoints -----------------------------------------------------
    def _engine_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.engine.checkpoint),
            "device": self.engine.device or "auto",
            "ready": self.engine._model is not None,
        }

    def _handle_state(self) -> None:
        board = _board_from_payload(self._read_json())
        self._send_json(_board_state(board))

    def _handle_move(self) -> None:
        payload = self._read_json()
        board = _board_from_payload(payload)
        uci = payload.get("uci")
        if not uci:
            raise ValueError("missing move")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"invalid move: {uci}") from exc
        if move not in board.legal_moves:
            raise ValueError(f"illegal move: {uci}")
        san = board.san(move)
        board.push(move)
        state = _board_state(board)
        state["san"] = san
        self._send_json(state)

    def _handle_engine(self) -> None:
        payload = self._read_json()
        board = _board_from_payload(payload)
        if board.is_game_over(claim_draw=True):
            raise ValueError("game is already over")
        simulations = max(1, min(int(payload.get("simulations", 128)), 4096))
        c_puct = float(payload.get("c_puct", 1.5))
        temperature = float(payload.get("temperature", 0.0))
        multipv = max(1, min(int(payload.get("multipv", 1)), 6))

        analysis = build_analysis(
            self.engine, board, simulations, c_puct, temperature, multipv
        )

        best = chess.Move.from_uci(analysis["best_move"])
        san = board.san(best)
        # Eval of the position before moving, from White's perspective.
        before_lines = analysis["lines"]
        board.push(best)
        state = _board_state(board)
        _, value = self.engine.evaluate(board)
        state.update(
            {
                "san": san,
                "engine_move": analysis["best_move"],
                "value": value,
                "analysis": before_lines,
                "nodes": analysis["nodes"],
                "nps": analysis["nps"],
                "depth": analysis["depth"],
                "time_ms": analysis["time_ms"],
            }
        )
        self._send_json(state)

    def _handle_analyze(self) -> None:
        """Analyze a position without making a move (Stockfish-style infinite/fixed)."""

        payload = self._read_json()
        board = _board_from_payload(payload)
        turn = "white" if board.turn == chess.WHITE else "black"
        if board.is_game_over(claim_draw=True):
            self._send_json(
                {"turn": turn, "value": 0.0, "lines": [], "is_over": True, "nodes": 0}
            )
            return
        simulations = max(1, min(int(payload.get("simulations", 256)), 4096))
        c_puct = float(payload.get("c_puct", 1.5))
        multipv = max(1, min(int(payload.get("multipv", 3)), 6))

        analysis = build_analysis(self.engine, board, simulations, c_puct, 0.0, multipv)
        _, value = self.engine.evaluate(board)
        self._send_json(
            {
                "turn": turn,
                "value": value,
                "lines": analysis["lines"],
                "best_move": analysis["best_move"],
                "nodes": analysis["nodes"],
                "nps": analysis["nps"],
                "depth": analysis["depth"],
                "time_ms": analysis["time_ms"],
                "is_over": False,
            }
        )

    def _handle_import(self) -> None:
        """Load a position from a FEN string or a PGN game."""

        payload = self._read_json()
        text = (payload.get("text") or "").strip()
        if not text:
            raise ValueError("nothing to import")

        board: chess.Board | None = None
        history: list[dict[str, Any]] = []
        kind = "fen"

        # Try PGN first when it looks like a game.
        looks_like_pgn = "[" in text or "1." in text or len(text.split()) > 6
        if looks_like_pgn:
            try:
                import io
                import chess.pgn as chess_pgn

                game = chess_pgn.read_game(io.StringIO(text))
                if game is not None:
                    probe = game.board()
                    for move in game.mainline_moves():
                        san = probe.san(move)
                        probe.push(move)
                        history.append(
                            {
                                "fen": probe.fen(),
                                "san": san,
                                "uci": move.uci(),
                                "color": "white" if not probe.turn else "black",
                            }
                        )
                    board = probe
                    kind = "pgn"
            except Exception:
                board = None

        if board is None:
            try:
                board = chess.Board(text)
            except ValueError as exc:
                raise ValueError(f"invalid FEN or PGN: {exc}") from exc

        state = _board_state(board)
        state["import_kind"] = kind
        state["history"] = history
        self._send_json(state)

    def _handle_eval(self) -> None:
        payload = self._read_json()
        board = _board_from_payload(payload)
        if board.is_game_over(claim_draw=True):
            self._send_json({"value": 0.0, "turn": "white" if board.turn else "black"})
            return
        _, value = self.engine.evaluate(board)
        self._send_json(
            {
                "value": value,
                "cp": value_to_cp(value),
                "turn": "white" if board.turn == chess.WHITE else "black",
            }
        )


@lru_cache(maxsize=1)
def _check_web_root() -> None:
    if not (WEB_ROOT / "index.html").is_file():
        raise FileNotFoundError(f"web assets missing at {WEB_ROOT}")


def serve(
    checkpoint: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str | None = None,
    open_browser: bool = True,
) -> None:
    """Start the GUI server (blocking)."""

    _check_web_root()
    engine = EngineHandle(checkpoint=Path(checkpoint), device=device)

    handler = type("BoundHandler", (_Handler,), {"engine": engine})
    httpd = ThreadingHTTPServer((host, port), handler)

    url = f"http://{host}:{port}/"
    print(f"Superchess GUI serving at {url}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  device    : {device or 'auto'}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
