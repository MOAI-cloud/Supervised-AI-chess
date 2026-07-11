"""Animated GIF rendering for Superchess replay payloads."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import chess
import chess.svg as chess_svg
import cairosvg
from PIL import Image, ImageDraw, ImageFont

_FILES = "abcdefgh"
_GUI_THEMES = {
    "brown": {"light": "#f0d9b5", "dark": "#b58863"},
    "green": {"light": "#eeeed2", "dark": "#769656"},
    "blue": {"light": "#dee3e6", "dark": "#788aab"},
    "slate": {"light": "#dfe6ef", "dark": "#5a6b80"},
    "purple": {"light": "#e7e2f5", "dark": "#8476c4"},
}
_MAX_FRAMES = 512


@dataclass(frozen=True, slots=True)
class GifRenderOptions:
    board_size: int = 560
    orientation: str = "white"
    theme: str = "brown"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GifRenderOptions":
        raw_size = int(payload.get("board_size", 560))
        size = max(320, min(800, raw_size))
        size -= size % 8
        orientation = str(payload.get("orientation", "white")).lower()
        if orientation not in {"white", "black"}:
            raise ValueError("GIF orientation must be 'white' or 'black'")
        theme = str(payload.get("theme", "brown")).lower()
        if theme not in _GUI_THEMES:
            theme = "brown"
        return cls(board_size=size, orientation=orientation, theme=theme)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _frame_duration(frame: dict[str, Any], final: bool) -> int:
    default = 1800 if final else 700
    try:
        duration = int(frame.get("display_ms", default))
    except (TypeError, ValueError):
        duration = default
    return max(100, min(5000, duration))


def _validate_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("replay must contain at least one frame")
    if len(frames) > _MAX_FRAMES:
        raise ValueError(f"replay has too many frames (maximum {_MAX_FRAMES})")
    validated: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"replay frame {index} must be an object")
        fen = frame.get("fen")
        if not isinstance(fen, str):
            raise ValueError(f"replay frame {index} is missing a FEN")
        try:
            chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"invalid FEN in replay frame {index}: {exc}") from exc
        validated.append(frame)
    return validated


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _blend(base: str, overlay: tuple[int, int, int], alpha: float) -> str:
    base_rgb = tuple(int(base[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(overlay[channel] * alpha + base_rgb[channel] * (1.0 - alpha)) for channel in range(3))
    return "#" + "".join(f"{value:02x}" for value in mixed)


def _render_gui_board(board: chess.Board, frame: dict[str, Any], options: GifRenderOptions) -> Image.Image:
    """Rasterize the GUI's Cburnett SVG pieces and board theme for one frame."""

    theme = _GUI_THEMES[options.theme]
    colors = {
        "square light": theme["light"],
        "square dark": theme["dark"],
        # Match the translucent last-move layers in web/style.css.
        "square light lastmove": _blend(theme["light"], (205, 210, 106), 0.78),
        "square dark lastmove": _blend(theme["dark"], (170, 162, 58), 0.78),
    }
    last_move = None
    raw_uci = frame.get("uci")
    if isinstance(raw_uci, str):
        try:
            last_move = chess.Move.from_uci(raw_uci)
        except ValueError:
            pass
    check = board.king(board.turn) if board.is_check() else None
    svg = chess_svg.board(
        board,
        orientation=chess.WHITE if options.orientation == "white" else chess.BLACK,
        lastmove=last_move,
        check=check,
        size=options.board_size,
        coordinates=False,
        colors=colors,
    )
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=options.board_size,
        output_height=options.board_size,
    )
    return Image.open(BytesIO(png)).convert("RGBA")


def _draw_frame(
    frame: dict[str, Any],
    payload: dict[str, Any],
    options: GifRenderOptions,
    index: int,
    total: int,
) -> Image.Image:
    board_size = options.board_size
    square = board_size // 8
    margin = max(20, board_size // 24)
    header = max(70, board_size // 8)
    footer = max(92, board_size // 6)
    gauge_width = max(14, square // 5)
    canvas_width = margin * 2 + board_size + gauge_width
    canvas_height = header + board_size + footer
    board_x = margin
    board_y = header

    image = Image.new("RGB", (canvas_width, canvas_height), "#161512")
    draw = ImageDraw.Draw(image)
    title_font = _font(max(17, board_size // 30))
    small_font = _font(max(12, board_size // 45))
    coord_font = _font(max(10, square // 7))

    event = _shorten(str(payload.get("event") or "Superchess replay"), 58)
    white = _shorten(str(payload.get("white") or "White"), 28)
    black = _shorten(str(payload.get("black") or "Black"), 28)
    draw.text((margin, 12), event, fill="#f0f0f0", font=title_font)
    draw.text((margin, 42), f"White  {white}", fill="#dedede", font=small_font)
    right_text = f"Black  {black}"
    right_box = draw.textbbox((0, 0), right_text, font=small_font)
    draw.text((canvas_width - margin - (right_box[2] - right_box[0]), 42), right_text, fill="#aaa", font=small_font)

    board = chess.Board(str(frame["fen"]))
    board_image = _render_gui_board(board, frame, options)
    image.paste(board_image, (board_x, board_y), board_image)

    display_files = _FILES if options.orientation == "white" else _FILES[::-1]
    display_ranks = "87654321" if options.orientation == "white" else "12345678"
    theme = _GUI_THEMES[options.theme]
    for i, file_name in enumerate(display_files):
        draw.text(
            (board_x + i * square + 3, board_y + board_size - coord_font.size - 1),
            file_name,
            fill=theme["light"] if i % 2 == 0 else theme["dark"],
            font=coord_font,
        )
    for i, rank_name in enumerate(display_ranks):
        draw.text(
            (board_x + board_size - coord_font.size, board_y + i * square + 2),
            rank_name,
            fill=theme["light"] if i % 2 == 0 else theme["dark"],
            font=coord_font,
        )

    cp = frame.get("evaluation_cp_white")
    try:
        cp_value = float(cp) if cp is not None else 0.0
    except (TypeError, ValueError):
        cp_value = 0.0
    probability = 1.0 / (1.0 + 10.0 ** (-max(-12000.0, min(12000.0, cp_value)) / 400.0))
    gauge_x = board_x + board_size
    draw.rectangle((gauge_x, board_y, gauge_x + gauge_width, board_y + board_size), fill="#403e3c")
    white_height = int(board_size * probability)
    draw.rectangle(
        (gauge_x, board_y + board_size - white_height, gauge_x + gauge_width, board_y + board_size),
        fill="#f0f0f0",
    )

    san = str(frame.get("san") or "Starting position")
    actor = str(frame.get("actor") or "")
    ply = int(frame.get("ply", index))
    move_label = f"Ply {ply}/{max(0, total - 1)} · {san}"
    if actor:
        move_label += f" · {actor.capitalize()}"
    draw.text((margin, header + board_size + 15), move_label, fill="#f0f0f0", font=title_font)

    opening = str(frame.get("opening") or payload.get("opening") or "")
    eval_text = f"Superchess eval {cp_value / 100:+.2f}"
    detail = eval_text + (f" · {opening}" if opening else "")
    draw.text((margin, header + board_size + 51), _shorten(detail, 75), fill="#999591", font=small_font)
    if index == total - 1:
        result = str(payload.get("result") or "*")
        result_box = draw.textbbox((0, 0), result, font=title_font)
        draw.text(
            (canvas_width - margin - (result_box[2] - result_box[0]), header + board_size + 15),
            result,
            fill="#9ac766",
            font=title_font,
        )
    return image


def render_replay_gif(payload: dict[str, Any]) -> bytes:
    """Render a ``superchess-replay-v1`` payload as an animated GIF."""

    if not isinstance(payload, dict):
        raise ValueError("GIF payload must be an object")
    frames = _validate_frames(payload)
    options = GifRenderOptions.from_payload(payload)
    images = [
        _draw_frame(frame, payload, options, index, len(frames))
        for index, frame in enumerate(frames)
    ]
    durations = [_frame_duration(frame, index == len(frames) - 1) for index, frame in enumerate(frames)]

    # Quantize all frames against one palette to avoid color flicker.
    palette = images[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    quantized = [palette]
    quantized.extend(
        image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for image in images[1:]
    )
    output = BytesIO()
    quantized[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output.getvalue()
