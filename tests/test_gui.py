from __future__ import annotations

import json
from io import BytesIO
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import threading
import urllib.error
import urllib.request

import chess
import chess.engine
import numpy as np
import pytest
from PIL import Image

from superchess.gif import render_replay_gif
from superchess.gui import _Handler, _board_from_payload, build_analysis, value_to_cp
from superchess.mcts import MCTSNode
from superchess.openings import OpeningBook
from superchess.stockfish import LICHESS_STOCKFISH_LEVELS, StockfishHandle, StockfishPlayResult


class DummyEngine:
    checkpoint = Path("dummy.pt")
    device = None
    allow_legacy_checkpoint = False
    _model = None

    def evaluate(self, board: chess.Board):
        return {}, -0.25


class DummyStockfish:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, object | None]] = []

    def status(self) -> dict:
        return {
            "configured": "dummy-stockfish",
            "path": "/test/dummy-stockfish",
            "available": True,
            "ready": True,
            "name": "DummyFish",
            "error": None,
            "levels": [],
        }

    def play(self, board: chess.Board, level: int, *, game_id=None) -> StockfishPlayResult:
        self.calls.append((board.fen(), level, game_id))
        move = chess.Move.from_uci("e2e4") if chess.Move.from_uci("e2e4") in board.legal_moves else next(iter(board.legal_moves))
        return StockfishPlayResult(
            move=move,
            info={
                "score": chess.engine.PovScore(chess.engine.Cp(32), chess.WHITE),
                "pv": [move],
                "depth": 5,
                "nodes": 123,
                "nps": 1230,
                "time": 0.1,
            },
            preset=LICHESS_STOCKFISH_LEVELS[level],
            effective_skill=LICHESS_STOCKFISH_LEVELS[level].skill,
            exact_skill=True,
            elapsed_ms=100,
        )


@pytest.fixture
def gui_server():
    stockfish = DummyStockfish()
    handler = type("TestHandler", (_Handler,), {"engine": DummyEngine(), "stockfish": stockfish})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_bytes(base_url: str, path: str, payload: dict) -> tuple[bytes, str]:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read(), response.headers.get_content_type()


def test_gui_info_route_reports_engine_metadata(gui_server):
    payload = get_json(gui_server, "/api/info")

    assert {key: payload[key] for key in ("checkpoint", "device", "allow_legacy_checkpoint", "ready", "stockfish")} == {
        "checkpoint": "dummy.pt",
        "device": "auto",
        "allow_legacy_checkpoint": False,
        "ready": False,
        "stockfish": {
            "configured": "dummy-stockfish",
            "path": "/test/dummy-stockfish",
            "available": True,
            "ready": True,
            "name": "DummyFish",
            "error": None,
            "levels": [],
        },
    }
    assert payload["openings"]["revision"]
    assert payload["openings"]["entries"] >= 0


def test_gui_piece_route_serves_svg_artwork(gui_server):
    with urllib.request.urlopen(gui_server + "/piece/wN.svg", timeout=5) as response:
        data = response.read()
        content_type = response.headers.get_content_type()

    assert content_type == "image/svg+xml"
    assert b"<svg" in data
    assert b"white-knight" in data


def test_gui_import_route_loads_pgn_history(gui_server):
    payload = post_json(gui_server, "/api/import", {"text": "1. e4 e5 2. Nf3 Nc6"})

    assert payload["import_kind"] == "pgn"
    assert payload["ply"] == 4
    assert [move["uci"] for move in payload["history"]] == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_gui_import_route_rejects_invalid_text(gui_server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(gui_server, "/api/import", {"text": "not a fen"})

    assert error.value.code == 400
    payload = json.loads(error.value.read().decode("utf-8"))
    assert "invalid FEN or PGN" in payload["error"]


def test_gui_stockfish_route_plays_selected_lichess_level(gui_server):
    payload = post_json(
        gui_server,
        "/api/stockfish",
        {
            "fen": chess.Board().fen(),
            "start_fen": chess.Board().fen(),
            "moves": [],
            "level": 6,
            "game_id": "arena-test",
        },
    )

    assert payload["stockfish_move"] == "e2e4"
    assert payload["san"] == "e4"
    assert payload["actor"] == "stockfish"
    assert payload["stockfish"] == {
        "level": 6,
        "requested_skill": 11,
        "effective_skill": 11,
        "exact_skill": True,
        "move_time_ms": 400,
        "depth_limit": 8,
    }
    assert payload["cp_white"] == value_to_cp(0.25)
    assert payload["superchess_eval"] == {
        "value": -0.25,
        "cp_white": value_to_cp(0.25),
        "mate_white": None,
        "source": "superchess",
    }
    assert payload["analysis"][0]["cp"] == 32
    assert payload["ply"] == 1


def test_opening_book_keeps_last_named_position_in_middlegame(tmp_path: Path):
    named = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"):
        named.push_san(san)
    book = OpeningBook(tmp_path / "openings.json")
    book._by_epd = {named.epd(): ("C50", "Italian Game")}  # type: ignore[attr-defined]
    book._loaded = True  # type: ignore[attr-defined]

    middlegame = named.copy(stack=True)
    for san in ("c3", "Nf6", "d3", "d6", "O-O", "O-O", "Re1", "a6"):
        middlegame.push_san(san)

    opening = book.classify(middlegame)

    assert opening is not None
    assert opening.display_name == "C50 · Italian Game"
    assert opening.matched_ply == 6


def test_gif_renderer_writes_each_replay_frame(gui_server):
    board = chess.Board()
    frames = [{"ply": 0, "fen": board.fen(), "display_ms": 100}]
    board.push_uci("e2e4")
    frames.append(
        {
            "ply": 1,
            "fen": board.fen(),
            "uci": "e2e4",
            "san": "e4",
            "actor": "superchess",
            "evaluation_cp_white": 24,
            "display_ms": 100,
        }
    )
    replay = {"event": "Test game", "white": "Superchess", "black": "Stockfish", "frames": frames}

    direct = render_replay_gif(replay)
    endpoint, content_type = post_bytes(gui_server, "/api/gif", replay)

    assert content_type == "image/gif"
    assert endpoint[:6] == b"GIF89a"
    assert direct[:6] == b"GIF89a"
    with Image.open(BytesIO(endpoint)) as image:
        assert image.n_frames == 2


def test_board_payload_reconstructs_history_for_draw_claims():
    board = chess.Board()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in moves:
        board.push_uci(uci)

    rebuilt = _board_from_payload(
        {"fen": board.fen(), "start_fen": chess.Board().fen(), "moves": moves}
    )

    assert rebuilt.can_claim_threefold_repetition()


def test_lichess_stockfish_presets_match_fishnet():
    assert [
        (preset.skill, preset.move_time_ms, preset.depth)
        for preset in LICHESS_STOCKFISH_LEVELS.values()
    ] == [
        (-9, 50, 5),
        (-5, 100, 5),
        (-1, 150, 5),
        (3, 200, 5),
        (7, 300, 5),
        (11, 400, 8),
        (16, 500, 13),
        (20, 1000, 22),
    ]


def test_stockfish_handle_applies_level_limits_and_clamps_unsupported_negative_skill():
    class FakeOption:
        min = 0
        max = 20

    class FakeEngine:
        options = {"Skill Level": FakeOption(), "UCI_LimitStrength": object()}

        def __init__(self) -> None:
            self.call = None

        def play(self, board, limit, **kwargs):
            self.call = (board, limit, kwargs)
            return SimpleNamespace(move=chess.Move.from_uci("e2e4"), info={})

    fake = FakeEngine()
    handle = StockfishHandle("unused")
    handle._engine = fake  # type: ignore[assignment]

    result = handle.play(chess.Board(), 1, game_id="game-one")

    assert result.preset.skill == -9
    assert result.effective_skill == 0
    assert result.exact_skill is False
    _, limit, kwargs = fake.call
    assert limit.time == pytest.approx(0.05)
    assert limit.depth == 5
    assert kwargs["game"] == "game-one"
    assert kwargs["options"] == {"Skill Level": 0, "UCI_LimitStrength": False}


class FakeSearcher:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result

    def search(self, board: chess.Board) -> SimpleNamespace:
        return self.result

    def principal_variation(self, node: MCTSNode, board: chess.Board, max_len: int = 12) -> list[chess.Move]:
        line: list[chess.Move] = []
        current: MCTSNode | None = node
        while current is not None and current.moves and len(line) < max_len:
            line.append(current.moves[0])
            current = current.children[0]
        return line


class FakeAnalysisEngine:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result

    def search(
        self,
        board: chess.Board,
        simulations: int,
        c_puct: float,
        temperature: float,
        evaluation_batch_size: int,
    ) -> FakeSearcher:
        return FakeSearcher(self.result)


def _linear_result(board: chess.Board, length: int) -> SimpleNamespace:
    probe = board.copy(stack=False)
    moves: list[chess.Move] = []
    for _ in range(length):
        move = next(iter(probe.legal_moves))
        moves.append(move)
        probe.push(move)

    root = MCTSNode([moves[0]], np.ones(1, dtype=np.float32))
    current = root
    for move in moves[1:]:
        current.visit_counts[0] = 1.0
        child = MCTSNode([move], np.ones(1, dtype=np.float32))
        current.children[0] = child
        current = child
    current.visit_counts[0] = 1.0
    current.children[0] = MCTSNode([], np.empty(0, dtype=np.float32))

    return SimpleNamespace(
        best_move=moves[0],
        visits={moves[0]: 64},
        policy={moves[0]: 1.0},
        root=root,
    )


def test_build_analysis_reports_tree_depth_separately_from_pv_length():
    board = chess.Board()
    result = _linear_result(board, 20)

    analysis = build_analysis(
        FakeAnalysisEngine(result),
        board,
        simulations=64,
        c_puct=1.5,
        temperature=0.0,
        multipv=1,
        pv_length=8,
    )

    assert analysis["depth"] == 20
    assert analysis["pv_depth"] == 8
    assert analysis["pv_length"] == 8
    assert len(analysis["lines"][0]["uci"]) == 8