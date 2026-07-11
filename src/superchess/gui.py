"""Web GUI server for analysis, play, and Superchess-vs-Stockfish games.

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
import chess.svg as chess_svg

from superchess.openings import OPENING_BOOK, OpeningInfo, ensure_opening_database
from superchess.stockfish import StockfishHandle, StockfishPlayResult, StockfishUnavailableError

WEB_ROOT = Path(__file__).resolve().parent / "web"

# Lc0-style mapping from a [-1, 1] value head to centipawns.
_CP_SCALE = 111.714640912
_CP_SLOPE = 1.5620688421
_MAX_GUI_SIMULATIONS = 65_536
_MAX_PV_LENGTH = 128


# Opening recognition keyed by the opening move sequence (SAN). Longest prefix wins.
_OPENINGS: dict[str, str] = {
    "e4": "King's Pawn",
    "e4 a6": "St. George Defence",
    "e4 b6": "Owen's Defence",
    "e4 c5": "Sicilian Defence",
    "e4 c5 b3": "Sicilian, Snyder Variation",
    "e4 c5 c3": "Sicilian, Alapin Variation",
    "e4 c5 d4": "Sicilian, Smith-Morra Gambit",
    "e4 c5 f4": "Sicilian, Grand Prix Attack",
    "e4 c5 Nf3": "Sicilian Defence",
    "e4 c5 Nf3 Nc6": "Sicilian, Open",
    "e4 c5 Nf3 Nc6 Bb5": "Sicilian, Rossolimo Variation",
    "e4 c5 Nf3 Nc6 d4": "Sicilian, Open",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4": "Sicilian, Open",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5": "Sicilian, Kalashnikov Variation",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5 Nb5 d6": "Sicilian, Sveshnikov Variation",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6": "Sicilian, Accelerated Dragon",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6": "Sicilian, Four Knights Variation",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5": "Sicilian, Sveshnikov Variation",
    "e4 c5 Nf3 d6": "Sicilian, Najdorf-style",
    "e4 c5 Nf3 d6 Bb5+": "Sicilian, Moscow Variation",
    "e4 c5 Nf3 d6 d4": "Sicilian, Open",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4": "Sicilian, Open",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6": "Sicilian, Open",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3": "Sicilian, Open",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6": "Sicilian, Najdorf Variation",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6": "Sicilian, Dragon Variation",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6": "Sicilian, Classical Variation",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6": "Sicilian, Scheveningen Variation",
    "e4 c5 Nf3 e6": "Sicilian, Paulsen/Kan",
    "e4 c5 Nf3 e6 d4": "Sicilian, Open",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4": "Sicilian, Open",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6": "Sicilian, Kan Variation",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6": "Sicilian, Taimanov Variation",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6": "Sicilian, Four Knights Variation",
    "e4 c5 Nf3 g6": "Sicilian, Hyperaccelerated Dragon",
    "e4 e5": "Open Game",
    "e4 e5 Bc4": "Bishop's Opening",
    "e4 e5 Bc4 Nf6": "Bishop's Opening, Berlin Defence",
    "e4 e5 Bc4 Bc5": "Bishop's Opening, Classical Defence",
    "e4 e5 d4": "Center Game",
    "e4 e5 d4 exd4 Qxd4": "Center Game",
    "e4 e5 f4": "King's Gambit",
    "e4 e5 f4 exf4": "King's Gambit Accepted",
    "e4 e5 f4 Bc5": "King's Gambit Declined, Classical Defence",
    "e4 e5 Nc3": "Vienna Game",
    "e4 e5 Nc3 Nf6": "Vienna Game, Falkbeer Defence",
    "e4 e5 Nc3 Nf6 f4": "Vienna Gambit",
    "e4 e5 Nf3": "King's Knight Opening",
    "e4 e5 Nf3 d6": "Philidor Defence",
    "e4 e5 Nf3 d6 d4": "Philidor Defence",
    "e4 e5 Nf3 f5": "Latvian Gambit",
    "e4 e5 Nf3 Nc6": "Open Game",
    "e4 e5 Nf3 Nc6 Bb5": "Ruy López",
    "e4 e5 Nf3 Nc6 Bb5 a6": "Ruy López, Morphy Defence",
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4": "Ruy López, Morphy Defence",
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6": "Ruy López, Closed/Open Systems",
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7": "Ruy López, Closed Defence",
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4": "Ruy López, Open Defence",
    "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6": "Ruy López, Exchange Variation",
    "e4 e5 Nf3 Nc6 Bb5 Bc5": "Ruy López, Classical Defence",
    "e4 e5 Nf3 Nc6 Bb5 d6": "Ruy López, Steinitz Defence",
    "e4 e5 Nf3 Nc6 Bb5 f5": "Ruy López, Schliemann Defence",
    "e4 e5 Nf3 Nc6 Bb5 Nf6": "Ruy López, Berlin Defence",
    "e4 e5 Nf3 Nc6 Bc4": "Italian Game",
    "e4 e5 Nf3 Nc6 Bc4 Bc5": "Italian Game, Giuoco Piano",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 b4": "Evans Gambit",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3": "Italian Game, Giuoco Piano",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6": "Italian Game, Giuoco Piano",
    "e4 e5 Nf3 Nc6 Bc4 Nf6": "Italian Game, Two Knights Defence",
    "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5": "Two Knights Defence, Fried Liver Attack",
    "e4 e5 Nf3 Nc6 d4": "Scotch Game",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4": "Scotch Game",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5": "Scotch Game, Classical Variation",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6": "Scotch Game, Schmidt Variation",
    "e4 e5 Nf3 Nc6 Nc3": "Three Knights Game",
    "e4 e5 Nf3 Nc6 Nc3 Nf6": "Four Knights Game",
    "e4 e5 Nf3 Nf6": "Petrov's Defence",
    "e4 e5 Nf3 Nf6 Nxe5": "Petrov's Defence",
    "e4 e5 Nf3 Nf6 Nxe5 d6": "Petrov's Defence, Classical Attack",
    "e4 e6": "French Defence",
    "e4 e6 d3": "King's Indian Attack",
    "e4 e6 d4": "French Defence",
    "e4 e6 d4 d5": "French Defence",
    "e4 e6 d4 d5 e5": "French Defence, Advance Variation",
    "e4 e6 d4 d5 exd5": "French Defence, Exchange Variation",
    "e4 e6 d4 d5 Nd2": "French Defence, Tarrasch Variation",
    "e4 e6 d4 d5 Nc3": "French Defence",
    "e4 e6 d4 d5 Nc3 Bb4": "French Defence, Winawer Variation",
    "e4 e6 d4 d5 Nc3 Nf6": "French Defence, Classical Variation",
    "e4 e6 d4 d5 Nc3 Nf6 Bg5": "French Defence, Classical Variation",
    "e4 c6": "Caro-Kann Defence",
    "e4 c6 d4": "Caro-Kann Defence",
    "e4 c6 d4 d5": "Caro-Kann Defence",
    "e4 c6 d4 d5 e5": "Caro-Kann Defence, Advance Variation",
    "e4 c6 d4 d5 exd5": "Caro-Kann Defence, Exchange Variation",
    "e4 c6 d4 d5 exd5 cxd5 c4": "Caro-Kann Defence, Panov-Botvinnik Attack",
    "e4 c6 d4 d5 Nc3": "Caro-Kann Defence",
    "e4 c6 d4 d5 Nc3 dxe4": "Caro-Kann Defence, Classical Variation",
    "e4 c6 d4 d5 Nd2": "Caro-Kann Defence",
    "e4 c6 d4 d5 Nd2 dxe4": "Caro-Kann Defence, Classical Variation",
    "e4 c6 d4 d5 f3": "Caro-Kann Defence, Fantasy Variation",
    "e4 d5": "Scandinavian Defence",
    "e4 d5 exd5": "Scandinavian Defence",
    "e4 d5 exd5 Nf6": "Scandinavian Defence, Modern Variation",
    "e4 d5 exd5 Qxd5": "Scandinavian Defence, Main Line",
    "e4 d5 exd5 Qxd5 Nc3": "Scandinavian Defence, Main Line",
    "e4 d5 exd5 Qxd5 Nc3 Qa5": "Scandinavian Defence, Main Line",
    "e4 g6": "Modern Defence",
    "e4 g6 d4": "Modern Defence",
    "e4 g6 d4 Bg7": "Modern Defence",
    "e4 g6 d4 Bg7 Nc3": "Modern Defence",
    "e4 d6": "Pirc Defence",
    "e4 d6 d4": "Pirc Defence",
    "e4 d6 d4 Nf6": "Pirc Defence",
    "e4 d6 d4 Nf6 Nc3": "Pirc Defence",
    "e4 d6 d4 Nf6 Nc3 g6": "Pirc Defence",
    "e4 d6 d4 Nf6 Nc3 g6 f4": "Pirc Defence, Austrian Attack",
    "e4 Nf6": "Alekhine Defence",
    "e4 Nf6 e5": "Alekhine Defence",
    "e4 Nf6 e5 Nd5": "Alekhine Defence",
    "e4 Nf6 e5 Nd5 d4": "Alekhine Defence",
    "e4 Nf6 e5 Nd5 d4 d6": "Alekhine Defence",
    "e4 Nf6 e5 Nd5 d4 d6 c4": "Alekhine Defence, Four Pawns/Chase",
    "e4 Nf6 e5 Nd5 d4 d6 Nf3": "Alekhine Defence, Modern Variation",
    "e4 Nc6": "Nimzowitsch Defence",
    "e4 Nc6 d4": "Nimzowitsch Defence",
    "e4 b5": "Polish Defence",
    "e4 f5": "Fred Defence",
    "d4": "Queen's Pawn",
    "d4 b6": "English Defence",
    "d4 c5": "Old Benoni Defence",
    "d4 c5 d5": "Benoni Defence",
    "d4 c5 d5 e5": "Old Benoni Defence",
    "d4 d6": "Queen's Pawn Game",
    "d4 d5": "Closed Game",
    "d4 d5 Bf4": "London System",
    "d4 d5 Bg5": "Levitsky Attack",
    "d4 d5 c4": "Queen's Gambit",
    "d4 d5 c4 c6": "Slav Defence",
    "d4 d5 c4 c6 Nf3 Nf6": "Slav Defence",
    "d4 d5 c4 c6 Nf3 Nf6 e3": "Slav Defence",
    "d4 d5 c4 c6 Nc3 Nf6": "Slav Defence",
    "d4 d5 c4 c6 Nc3 Nf6 Nf3": "Slav Defence",
    "d4 d5 c4 c6 Nc3 Nf6 Nf3 dxc4": "Slav Defence, Main Line",
    "d4 d5 c4 e6": "Queen's Gambit Declined",
    "d4 d5 c4 e6 Nc3": "Queen's Gambit Declined",
    "d4 d5 c4 e6 Nc3 Be7": "Queen's Gambit Declined, Alatortsev Variation",
    "d4 d5 c4 e6 Nc3 c5": "Queen's Gambit Declined, Tarrasch Defence",
    "d4 d5 c4 e6 Nc3 Nf6": "Queen's Gambit Declined",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5": "Queen's Gambit Declined, Orthodox",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7": "Queen's Gambit Declined, Orthodox Defence",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Nbd7": "Queen's Gambit Declined, Cambridge Springs",
    "d4 d5 c4 e6 Nf3": "Queen's Gambit Declined",
    "d4 d5 c4 e6 Nf3 Nf6": "Queen's Gambit Declined",
    "d4 d5 c4 dxc4": "Queen's Gambit Accepted",
    "d4 d5 c4 dxc4 e3": "Queen's Gambit Accepted",
    "d4 d5 c4 dxc4 e4": "Queen's Gambit Accepted, Central Variation",
    "d4 d5 c4 e5": "Albin Countergambit",
    "d4 d5 c4 Nc6": "Chigorin Defence",
    "d4 d5 c4 Nf6": "Marshall Defence",
    "d4 d5 e3": "Queen's Pawn Game",
    "d4 d5 e4": "Blackmar-Diemer Gambit",
    "d4 d5 Nf3": "Queen's Pawn Game",
    "d4 d5 Nf3 Nf6": "Queen's Pawn Game",
    "d4 d5 Nf3 Nf6 Bf4": "London System",
    "d4 d5 Nf3 Nf6 c4": "Queen's Gambit",
    "d4 d5 Nc3": "Veresov Opening",
    "d4 d5 Nc3 Nf6 Bg5": "Richter-Veresov Attack",
    "d4 Nf6": "Indian Defence",
    "d4 Nf6 Bf4": "London System",
    "d4 Nf6 Bg5": "Trompowsky Attack",
    "d4 Nf6 c4": "Indian Defence",
    "d4 Nf6 c4 c5": "Benoni Defence",
    "d4 Nf6 c4 c5 d5": "Benoni Defence",
    "d4 Nf6 c4 c5 d5 e6": "Modern Benoni Defence",
    "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6": "Modern Benoni Defence",
    "d4 Nf6 c4 c6": "Slav Indian Defence",
    "d4 Nf6 c4 d6": "Indian Defence",
    "d4 Nf6 c4 d6 Nc3 e5": "Old Indian Defence",
    "d4 Nf6 c4 d6 Nc3 g6": "King's Indian Defence",
    "d4 Nf6 c4 e5": "Budapest Gambit",
    "d4 Nf6 c4 e5 dxe5": "Budapest Gambit",
    "d4 Nf6 c4 e5 dxe5 Ng4": "Budapest Gambit",
    "d4 Nf6 c4 g6": "King's Indian / Grünfeld",
    "d4 Nf6 c4 g6 Nc3": "King's Indian / Grünfeld",
    "d4 Nf6 c4 g6 Nc3 Bg7": "King's Indian Defence",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4": "King's Indian Defence",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6": "King's Indian Defence",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3": "King's Indian, Saemisch Variation",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3": "King's Indian, Classical Variation",
    "d4 Nf6 c4 g6 Nc3 d5": "Grünfeld Defence",
    "d4 Nf6 c4 g6 Nc3 d5 cxd5": "Grünfeld Defence, Exchange Variation",
    "d4 Nf6 c4 g6 g3": "King's Indian, Fianchetto Variation",
    "d4 Nf6 c4 e6": "Nimzo/Queen's Indian",
    "d4 Nf6 c4 e6 Nf3": "Queen's Indian / Bogo-Indian",
    "d4 Nf6 c4 e6 Nf3 b6": "Queen's Indian Defence",
    "d4 Nf6 c4 e6 Nf3 Bb4+": "Bogo-Indian Defence",
    "d4 Nf6 c4 e6 Nc3": "Nimzo-Indian Defence",
    "d4 Nf6 c4 e6 Nc3 Bb4": "Nimzo-Indian Defence",
    "d4 Nf6 c4 e6 Nc3 Bb4 e3": "Nimzo-Indian, Rubinstein Variation",
    "d4 Nf6 c4 e6 Nc3 Bb4 Qc2": "Nimzo-Indian, Classical Variation",
    "d4 Nf6 c4 e6 Nc3 Bb4 f3": "Nimzo-Indian, Saemisch Variation",
    "d4 f5": "Dutch Defence",
    "d4 f5 c4": "Dutch Defence",
    "d4 f5 c4 Nf6": "Dutch Defence",
    "d4 f5 c4 Nf6 g3": "Dutch Defence, Fianchetto Variation",
    "d4 f5 g3": "Dutch Defence",
    "d4 f5 g3 Nf6 Bg2": "Dutch Defence, Leningrad/Stonewall Systems",
    "d4 Nf6 Nf3": "Queen's Pawn Game",
    "d4 Nf6 Nf3 e6": "Queen's Pawn Game",
    "d4 Nf6 Nf3 e6 Bf4": "London System",
    "d4 Nf6 Nf3 d5": "Queen's Pawn Game",
    "d4 Nf6 Nf3 d5 Bf4": "London System",
    "d4 Nf6 Nf3 g6": "King's Indian Attack",
    "c4": "English Opening",
    "c4 c5": "English Opening, Symmetrical Variation",
    "c4 c5 Nc3": "English Opening, Symmetrical Variation",
    "c4 c5 Nf3": "English Opening, Symmetrical Variation",
    "c4 e5": "English Opening, King's English",
    "c4 e5 Nc3": "English Opening, King's English",
    "c4 e5 Nc3 Nf6": "English Opening, Four Knights",
    "c4 e5 Nc3 Nf6 Nf3": "English Opening, Four Knights",
    "c4 e5 Nc3 Nf6 g3": "English Opening, Bremen System",
    "c4 e5 g3": "English Opening, King's English",
    "c4 e6": "English Opening",
    "c4 e6 Nc3": "English Opening",
    "c4 e6 Nc3 d5": "Queen's Gambit Declined Transposition",
    "c4 Nf6": "English Opening, Anglo-Indian Defence",
    "c4 Nf6 Nc3": "English Opening, Anglo-Indian Defence",
    "c4 Nf6 Nc3 e5": "English Opening, Mikenas-Carls Variation",
    "c4 Nf6 Nc3 e6": "English Opening, Anglo-Indian Defence",
    "c4 Nf6 Nc3 g6": "English Opening, King's Indian Defence",
    "c4 g6": "English Opening, Great Snake Variation",
    "Nf3": "Réti Opening",
    "Nf3 c5": "Réti Opening",
    "Nf3 d5": "Réti Opening",
    "Nf3 d5 c4": "Réti Opening",
    "Nf3 d5 c4 d4": "Réti Opening, Advance Variation",
    "Nf3 d5 c4 e6": "Réti Opening",
    "Nf3 d5 g3": "King's Indian Attack",
    "Nf3 Nf6": "Réti Opening",
    "Nf3 Nf6 c4": "English Opening",
    "Nf3 Nf6 g3": "King's Indian Attack",
    "Nf3 Nf6 g3 g6": "King's Indian Attack",
    "Nf3 g6": "King's Indian Attack",
    "g3": "King's Fianchetto",
    "g3 d5": "King's Fianchetto",
    "g3 e5": "King's Fianchetto",
    "g3 Nf6": "King's Fianchetto",
    "g3 Nf6 Bg2": "King's Fianchetto",
    "g3 d5 Bg2": "King's Fianchetto",
    "b3": "Larsen's Opening",
    "b3 d5": "Larsen's Opening",
    "b3 e5": "Larsen's Opening",
    "b3 Nf6": "Larsen's Opening",
    "b3 d5 Bb2": "Larsen's Opening",
    "f4": "Bird's Opening",
    "f4 d5": "Bird's Opening",
    "f4 d5 Nf3": "Bird's Opening",
    "f4 e5": "From's Gambit",
    "f4 Nf6": "Bird's Opening",
    "Nc3": "Dunst Opening",
    "Nc3 d5": "Dunst Opening",
    "Nc3 e5": "Vienna Game",
    "b4": "Polish Opening",
    "b4 e5": "Polish Opening",
    "b4 Nf6": "Polish Opening",
    "e3": "Van't Kruijs Opening",
    "d3": "Mieses Opening",
    "c3": "Saragossa Opening",
    "g4": "Grob Opening",
    "a3": "Anderssen's Opening",
    "a4": "Ware Opening",
    "h3": "Clemenz Opening",
    "h4": "Kadas Opening",
}


def value_to_cp(value: float) -> int:
    """Convert a side-to-move value in [-1, 1] to an integer centipawn score."""

    value = max(-0.9999, min(0.9999, float(value)))
    cp = _CP_SCALE * math.tan(_CP_SLOPE * value)
    return int(max(-12000, min(12000, round(cp))))


def _superchess_evaluation(engine: "EngineHandle", board: chess.Board) -> dict[str, Any]:
    """Evaluate ``board`` exclusively with Superchess, White-relative for UI use."""

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            value = 0.0
            cp_white = 0
        else:
            value = 1.0 if outcome.winner == board.turn else -1.0
            cp_white = 12000 if outcome.winner == chess.WHITE else -12000
        return {
            "value": value,
            "cp_white": cp_white,
            "mate_white": None,
            "source": "superchess",
        }

    _, value = engine.evaluate(board)
    white_value = value if board.turn == chess.WHITE else -value
    return {
        "value": value,
        "cp_white": value_to_cp(white_value),
        "mate_white": None,
        "source": "superchess",
    }


def detect_opening(board: chess.Board) -> OpeningInfo | None:
    """Return the last real opening position in the game history.

    The full cached Lichess ECO database is position-based and handles common
    transpositions. The compact SAN table remains an offline fallback.
    """

    opening = OPENING_BOOK.classify(board)
    if opening is not None:
        return opening

    try:
        sans: list[str] = []
        probe = chess.Board()
        for move in board.move_stack:
            sans.append(probe.san(move))
            probe.push(move)
    except Exception:  # pragma: no cover - defensive
        return None

    best: str | None = None
    best_ply = 0
    for length in range(1, len(sans) + 1):
        key = " ".join(sans[:length])
        if key in _OPENINGS:
            best = _OPENINGS[key]
            best_ply = length
    return OpeningInfo(eco="", name=best, matched_ply=best_ply) if best else None

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
    allow_legacy_checkpoint: bool = False
    _model: Any = None
    _lock: threading.Lock = None  # type: ignore[assignment]
    _search_lock: threading.Lock = None  # type: ignore[assignment]
    _eval_cache: dict = None  # type: ignore[assignment]
    _tree_state: Any = None  # (board, root) of the most recent search, for reuse

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._eval_cache = {}

    def ensure_loaded(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from superchess.training import load_model_checkpoint

                    model, _ = load_model_checkpoint(
                        self.checkpoint,
                        device_name=self.device,
                        allow_legacy_policy=self.allow_legacy_checkpoint,
                    )
                    self._model = model
        return self._model

    def search(
        self,
        board: chess.Board,
        simulations: int,
        c_puct: float,
        temperature: float,
        evaluation_batch_size: int,
    ):
        from superchess.mcts import NeuralMCTS, SearchConfig

        model = self.ensure_loaded()
        config = SearchConfig(
            simulations=simulations,
            c_puct=c_puct,
            temperature=temperature,
            evaluation_batch_size=evaluation_batch_size,
        )
        # Share the evaluation cache across requests; the wrapper adds tree reuse
        # and serializes searches so the shared tree is never mutated concurrently.
        searcher = NeuralMCTS(model, config, eval_cache=self._eval_cache)
        return _ReusableSearcher(self, searcher)

    def evaluate(self, board: chess.Board):
        from superchess.mcts import NeuralMCTS

        model = self.ensure_loaded()
        return NeuralMCTS(model, eval_cache=self._eval_cache).evaluate(board)


class _ReusableSearcher:
    """Delegates to ``NeuralMCTS`` while reusing the previous request's tree."""

    def __init__(self, handle: EngineHandle, searcher: Any) -> None:
        self._handle = handle
        self._searcher = searcher

    def search(self, board: chess.Board):
        from superchess.mcts import find_subtree

        handle = self._handle
        with handle._search_lock:
            tree = None
            if handle._tree_state is not None:
                previous_board, previous_root = handle._tree_state
                tree = find_subtree(previous_root, previous_board, board)
            result = self._searcher.search(board, tree=tree)
            handle._tree_state = (board.copy(stack=False), result.root)
            return result

    def principal_variation(self, node: Any, board: chess.Board, max_len: int = 12):
        return self._searcher.principal_variation(node, board, max_len=max_len)


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
    opening = detect_opening(board)
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
        "opening": opening.display_name if opening else None,
        "opening_name": opening.name if opening else None,
        "opening_eco": opening.eco if opening else None,
        "opening_ply": opening.matched_ply if opening else None,
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


def _visited_tree_depth(root: Any | None) -> int:
    """Return the deepest path in the MCTS tree that actually received visits."""

    if root is None:
        return 0
    max_depth = 0
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        counts = getattr(node, "visit_counts", None)
        if counts is None:
            continue
        for index, child in enumerate(node.children):
            if child is not None and counts[index] > 0:
                stack.append((child, depth + 1))
    return max_depth


def build_analysis(
    engine: "EngineHandle",
    board: chess.Board,
    simulations: int,
    c_puct: float,
    temperature: float,
    multipv: int,
    evaluation_batch_size: int = 8,
    pv_length: int = 32,
) -> dict[str, Any]:
    """Run a search and return Stockfish-style multi-PV analysis for ``board``."""

    pv_length = max(1, min(int(pv_length), _MAX_PV_LENGTH))
    started = time.perf_counter()
    searcher = engine.search(board, simulations, c_puct, temperature, evaluation_batch_size)
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
        index = root.move_index(move) if root is not None else -1
        child = root.children[index] if root is not None and index >= 0 else None
        # Edge Q-values are already stored from the root player's perspective.
        cp = None
        if index >= 0 and root.child_visits(index) > 0:
            cp = value_to_cp(root.child_q(index))
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
    pv_depth = max((len(line["uci"]) for line in lines), default=0)
    depth = max(1, _visited_tree_depth(root), pv_depth)
    return {
        "lines": lines,
        "nodes": total_visits,
        "nps": nps,
        "depth": depth,
        "pv_depth": pv_depth,
        "pv_length": pv_length,
        "time_ms": int(elapsed * 1000),
        "best_move": result.best_move.uci(),
        "multipv": len(lines),
    }


def _board_from_payload(payload: dict[str, Any]) -> chess.Board:
    moves = payload.get("moves")
    start_fen = payload.get("start_fen")
    if moves is not None or start_fen is not None:
        if not isinstance(moves, list) or not all(isinstance(uci, str) for uci in moves):
            raise ValueError("moves must be a list of UCI strings")
        board = chess.Board() if not start_fen or start_fen == "startpos" else chess.Board(start_fen)
        for uci in moves:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as exc:
                raise ValueError(f"invalid move in game history: {uci}") from exc
            if move not in board.legal_moves:
                raise ValueError(f"illegal move in game history: {uci}")
            board.push(move)

        fen = payload.get("fen")
        if fen and fen != "startpos" and board.fen() != chess.Board(fen).fen():
            raise ValueError("game history does not match the supplied FEN")
        return board

    fen = payload.get("fen", "startpos")
    if not fen or fen == "startpos":
        return chess.Board()
    return chess.Board(fen)


def _score_parts(score: Any, color: chess.Color) -> tuple[int, int | None]:
    """Return a python-chess score as centipawns and mate, from ``color``'s view."""

    if score is None:
        return 0, None
    pov = score.pov(color) if hasattr(score, "pov") else score
    mate = pov.mate() if hasattr(pov, "mate") else None
    cp = pov.score(mate_score=12000) if hasattr(pov, "score") else None
    return int(cp or 0), int(mate) if mate is not None else None


def _stockfish_analysis(board: chess.Board, result: StockfishPlayResult) -> dict[str, Any]:
    """Convert python-chess UCI info into the analysis shape used by the web app."""

    info = result.info
    pv = list(info.get("pv") or [result.move])
    if not pv or pv[0] != result.move:
        pv.insert(0, result.move)
    line = _line_from_pv(board, pv[:_MAX_PV_LENGTH])
    cp, mate = _score_parts(info.get("score"), board.turn)
    nodes = int(info.get("nodes") or 0)
    time_seconds = float(info.get("time") or (result.elapsed_ms / 1000.0))
    nps = int(info.get("nps") or (nodes / time_seconds if time_seconds > 0 else 0))
    return {
        "line": {
            "cp": cp,
            "mate": mate,
            "san": line["san"],
            "uci": line["uci"],
            "visits": nodes,
            "share": 1.0,
        },
        "nodes": nodes,
        "nps": nps,
        "depth": int(info.get("depth") or result.preset.depth),
        "time_ms": int(round(time_seconds * 1000)),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "Superchess/1.0"
    engine: EngineHandle  # injected on the server instance
    stockfish: StockfishHandle  # injected on the server instance

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

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        filename: str | None = None,
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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

    def _serve_piece(self, path: str) -> None:
        """Render a bundled python-chess piece as a cacheable SVG asset."""
        name = path.removeprefix("/piece/")
        if "/" in name or not name.endswith(".svg"):
            self.send_error(404)
            return
        code = name[:-4]
        if len(code) != 2 or code[0] not in "wb" or code[1] not in "KQRBNP":
            self.send_error(404)
            return
        symbol = code[1] if code[0] == "w" else code[1].lower()
        data = str(chess_svg.piece(chess.Piece.from_symbol(symbol))).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    # ---- routing -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/info":
            self._send_json(self._engine_info())
            return
        if path.startswith("/piece/"):
            self._serve_piece(path)
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
            elif path == "/api/stockfish":
                self._handle_stockfish()
            elif path == "/api/analyze":
                self._handle_analyze()
            elif path == "/api/eval":
                self._handle_eval()
            elif path == "/api/import":
                self._handle_import()
            elif path == "/api/gif":
                self._handle_gif()
            else:
                self.send_error(404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except StockfishUnavailableError as exc:
            self._send_json({"error": str(exc)}, status=503)
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json({"error": f"internal error: {exc}"}, status=500)

    # ---- endpoints -----------------------------------------------------
    def _engine_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.engine.checkpoint),
            "device": self.engine.device or "auto",
            "allow_legacy_checkpoint": self.engine.allow_legacy_checkpoint,
            "ready": self.engine._model is not None,
            "stockfish": self.stockfish.status(),
            "openings": OPENING_BOOK.status(),
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
        simulations = max(1, min(int(payload.get("simulations", 128)), _MAX_GUI_SIMULATIONS))
        c_puct = float(payload.get("c_puct", 1.5))
        temperature = float(payload.get("temperature", 0.0))
        multipv = max(1, min(int(payload.get("multipv", 1)), 6))
        evaluation_batch_size = max(1, min(int(payload.get("eval_batch_size", 8)), 128))
        pv_length = max(1, min(int(payload.get("pv_length", 32)), _MAX_PV_LENGTH))

        analysis = build_analysis(
            self.engine, board, simulations, c_puct, temperature, multipv, evaluation_batch_size, pv_length
        )

        best = chess.Move.from_uci(analysis["best_move"])
        san = board.san(best)
        # Eval of the position before moving, from White's perspective.
        before_lines = analysis["lines"]
        board.push(best)
        state = _board_state(board)
        evaluation = _superchess_evaluation(self.engine, board)
        state.update(
            {
                "san": san,
                "engine_move": analysis["best_move"],
                "actor": "superchess",
                "value": evaluation["value"],
                "cp_white": evaluation["cp_white"],
                "mate_white": evaluation["mate_white"],
                "superchess_eval": evaluation,
                "analysis": before_lines,
                "nodes": analysis["nodes"],
                "nps": analysis["nps"],
                "depth": analysis["depth"],
                "time_ms": analysis["time_ms"],
            }
        )
        self._send_json(state)

    def _handle_stockfish(self) -> None:
        """Make one move with a local Stockfish using a Lichess level preset."""

        payload = self._read_json()
        board = _board_from_payload(payload)
        if board.is_game_over(claim_draw=True):
            raise ValueError("game is already over")
        try:
            level = int(payload.get("level", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Stockfish level must be an integer from 1 to 8") from exc

        turn = board.turn
        result = self.stockfish.play(board, level, game_id=payload.get("game_id"))
        analysis = _stockfish_analysis(board, result)
        san = board.san(result.move)
        board.push(result.move)
        evaluation = _superchess_evaluation(self.engine, board)
        state = _board_state(board)
        state.update(
            {
                "san": san,
                "engine_move": result.move.uci(),
                "stockfish_move": result.move.uci(),
                "actor": "stockfish",
                "analysis_turn": "white" if turn == chess.WHITE else "black",
                "analysis": [analysis["line"]],
                "value": evaluation["value"],
                "cp_white": evaluation["cp_white"],
                "mate_white": evaluation["mate_white"],
                "superchess_eval": evaluation,
                "nodes": analysis["nodes"],
                "nps": analysis["nps"],
                "depth": analysis["depth"],
                "time_ms": analysis["time_ms"],
                "stockfish": {
                    "level": result.preset.level,
                    "requested_skill": result.preset.skill,
                    "effective_skill": result.effective_skill,
                    "exact_skill": result.exact_skill,
                    "move_time_ms": result.preset.move_time_ms,
                    "depth_limit": result.preset.depth,
                },
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
        simulations = max(1, min(int(payload.get("simulations", 256)), _MAX_GUI_SIMULATIONS))
        c_puct = float(payload.get("c_puct", 1.5))
        multipv = max(1, min(int(payload.get("multipv", 3)), 6))
        evaluation_batch_size = max(1, min(int(payload.get("eval_batch_size", 8)), 128))
        pv_length = max(1, min(int(payload.get("pv_length", 32)), _MAX_PV_LENGTH))

        analysis = build_analysis(self.engine, board, simulations, c_puct, 0.0, multipv, evaluation_batch_size, pv_length)
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
        initial_fen: str | None = None
        headers: dict[str, str] = {}

        # Try PGN first when it looks like a game.
        looks_like_pgn = "[" in text or "1." in text or len(text.split()) > 6
        if looks_like_pgn:
            try:
                import io
                import chess.pgn as chess_pgn

                game = chess_pgn.read_game(io.StringIO(text))
                if game is not None:
                    probe = game.board()
                    initial_fen = probe.fen()
                    headers = dict(game.headers)
                    for move in game.mainline_moves():
                        san = probe.san(move)
                        move_number = probe.fullmove_number
                        probe.push(move)
                        opening = detect_opening(probe)
                        history.append(
                            {
                                "fen": probe.fen(),
                                "san": san,
                                "uci": move.uci(),
                                "color": "white" if not probe.turn else "black",
                                "move_number": move_number,
                                "opening": opening.display_name if opening else None,
                                "opening_name": opening.name if opening else None,
                                "opening_eco": opening.eco if opening else None,
                                "opening_ply": opening.matched_ply if opening else None,
                            }
                        )
                    board = probe
                    kind = "pgn"
            except Exception:
                board = None

        if board is None:
            try:
                board = chess.Board(text)
                initial_fen = board.fen()
            except ValueError as exc:
                raise ValueError(f"invalid FEN or PGN: {exc}") from exc

        state = _board_state(board)
        state["import_kind"] = kind
        state["history"] = history
        state["start_fen"] = initial_fen or chess.Board().fen()
        state["headers"] = headers
        self._send_json(state)

    def _handle_eval(self) -> None:
        payload = self._read_json()
        board = _board_from_payload(payload)
        evaluation = _superchess_evaluation(self.engine, board)
        self._send_json(
            {
                **evaluation,
                "cp": value_to_cp(evaluation["value"]),
                "turn": "white" if board.turn == chess.WHITE else "black",
            }
        )

    def _handle_gif(self) -> None:
        """Render a replay payload to an animated GIF without rerunning engines."""

        from superchess.gif import render_replay_gif

        payload = self._read_json()
        data = render_replay_gif(payload)
        self._send_bytes(data, "image/gif", filename="superchess-replay.gif")


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
    allow_legacy_checkpoint: bool = False,
    stockfish_path: str | Path = "stockfish",
) -> None:
    """Start the GUI server (blocking)."""

    _check_web_root()
    engine = EngineHandle(
        checkpoint=Path(checkpoint),
        device=device,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )
    stockfish = StockfishHandle(stockfish_path)
    openings = ensure_opening_database()

    handler = type("BoundHandler", (_Handler,), {"engine": engine, "stockfish": stockfish})
    httpd = ThreadingHTTPServer((host, port), handler)

    url = f"http://{host}:{port}/"
    print(f"Superchess GUI serving at {url}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  device    : {device or 'auto'}")
    status = stockfish.status()
    print(f"  stockfish : {status['path'] or status['error']}")
    print(f"  openings  : {openings['entries']} Lichess ECO positions ({openings['source']})")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
        stockfish.close()
