"use strict";

/* ----------------------------------------------------------------------
 * Superchess web client — Stockfish-style neural analysis board.
 *
 * The Python server (python-chess) is authoritative for legality and the
 * neural MCTS engine. This client renders state, handles input, navigation,
 * live multi-PV analysis, arrows, an eval bar, an eval graph, and PGN/FEN
 * import / export.
 * -------------------------------------------------------------------- */

const PIECE_GLYPHS = {
  K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};
const PIECE_URLS = {
  K: "/piece/wK.svg", Q: "/piece/wQ.svg", R: "/piece/wR.svg",
  B: "/piece/wB.svg", N: "/piece/wN.svg", P: "/piece/wP.svg",
  k: "/piece/bK.svg", q: "/piece/bQ.svg", r: "/piece/bR.svg",
  b: "/piece/bB.svg", n: "/piece/bN.svg", p: "/piece/bP.svg",
};
const PIECE_NAMES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const FILES = "abcdefgh";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const LICHESS_STOCKFISH_PRESETS = [
  { level: 1, skill: -9, move_time_ms: 50, depth: 5 },
  { level: 2, skill: -5, move_time_ms: 100, depth: 5 },
  { level: 3, skill: -1, move_time_ms: 150, depth: 5 },
  { level: 4, skill: 3, move_time_ms: 200, depth: 5 },
  { level: 5, skill: 7, move_time_ms: 300, depth: 5 },
  { level: 6, skill: 11, move_time_ms: 400, depth: 8 },
  { level: 7, skill: 16, move_time_ms: 500, depth: 13 },
  { level: 8, skill: 20, move_time_ms: 1000, depth: 22 },
];

const state = {
  startFen: START_FEN,
  moves: [],            // [{fen, san, uci, color, actor, cpWhite, timeMs}]
  cursor: -1,           // -1 = start position; i = after moves[i]
  board: null,          // server state for the displayed position
  orientation: "white",
  mode: "analysis",     // analysis | play | arena
  playAs: "white",      // white | black | both
  arenaLevel: 4,
  arenaSuperchessColor: "white",
  arenaRunning: false,
  arenaPaused: false,
  arenaToken: 0,
  stockfishInfo: null,
  game: null,
  selected: null,
  thinking: false,      // engine is making a move (play mode)
  analyzeOn: true,
  analysis: null,       // latest analysis for the displayed position
  simulations: 256,
  multipv: 3,
  pvLength: 32,
  temperature: 0.0,
  showArrows: true,
  cpWhite: 0,
  mateWhite: null,
  evaluationSource: "superchess",
  opening: "Starting position",
};

const el = (id) => document.getElementById(id);
const boardEl = el("board");
const arrowsEl = el("arrows");
const SVG_NS = "http://www.w3.org/2000/svg";

/* ---------------- API ---------------- */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "request failed");
  return data;
}

async function apiBlob(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let message = "request failed";
    try { message = (await res.json()).error || message; } catch (_) { /* binary/empty error */ }
    throw new Error(message);
  }
  return res.blob();
}

/* ---------------- position helpers ---------------- */
function currentFen() {
  return state.cursor < 0 ? state.startFen : state.moves[state.cursor].fen;
}
function atLatest() {
  return state.cursor === state.moves.length - 1;
}

function positionPayload(extra) {
  return {
    fen: currentFen(),
    start_fen: state.startFen,
    moves: state.moves.slice(0, state.cursor + 1).map((move) => move.uci),
    ...(extra || {}),
  };
}

function makeGameId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `game-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/* ---------------- FEN rendering ---------------- */
function parsePlacement(fen) {
  const board = {};
  const placement = fen.split(" ")[0];
  const rows = placement.split("/");
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (/\d/.test(ch)) file += parseInt(ch, 10);
      else { board[FILES[file] + (8 - r)] = ch; file++; }
    }
  }
  return board;
}

function buildBoard() {
  boardEl.innerHTML = "";
  const ranks = state.orientation === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = state.orientation === "white" ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
  for (const rank of ranks) {
    for (const f of files) {
      const name = FILES[f] + rank;
      const sq = document.createElement("div");
      sq.className = "sq " + (((f + rank) % 2 === 0) ? "light" : "dark");
      sq.dataset.square = name;
      sq.addEventListener("pointerdown", onPointerDown);
      sq.addEventListener("click", onSquareClick);
      boardEl.appendChild(sq);
    }
  }
  buildLabels(ranks, files);
}

function buildLabels(ranks, files) {
  const rankBox = el("rankLabels");
  const fileBox = el("fileLabels");
  rankBox.innerHTML = "";
  fileBox.innerHTML = "";
  for (const rank of ranks) {
    const s = document.createElement("span");
    s.textContent = rank;
    rankBox.appendChild(s);
  }
  for (const f of files) {
    const s = document.createElement("span");
    s.textContent = FILES[f];
    fileBox.appendChild(s);
  }
}

const squareEl = (name) => boardEl.querySelector(`[data-square="${name}"]`);

function render() {
  const s = state.board;
  if (!s) return;
  const placement = parsePlacement(s.fen);
  const last = s.last_move ? [s.last_move.slice(0, 2), s.last_move.slice(2, 4)] : [];

  for (const sq of boardEl.children) {
    const name = sq.dataset.square;
    sq.className = "sq " + (sq.classList.contains("dark") ? "dark" : "light");
    sq.innerHTML = "";

    const piece = placement[name];
    if (piece) {
      const image = document.createElement("img");
      image.className = "piece";
      image.src = PIECE_URLS[piece];
      image.alt = `${piece === piece.toUpperCase() ? "White" : "Black"} ${PIECE_NAMES[piece.toLowerCase()]}`;
      image.draggable = false;
      sq.appendChild(image);
      sq.classList.add(piece === piece.toUpperCase() ? "white-piece" : "black-piece");
    }
    if (last.includes(name)) sq.classList.add("lastmove");
    if (s.check_square === name) sq.classList.add("check");
  }

  if (state.selected) showSelection(state.selected);
  drawArrows();
  updateMaterial();
  renderMoves();
  updateNavButtons();
  updateOpening();
  updatePositionFields();
  setEvalBar(state.cpWhite, state.mateWhite);
}

function showSelection(from) {
  clearHints();
  const sqEl = squareEl(from);
  if (sqEl) sqEl.classList.add("selected");
  const targets = (state.board.legal[from] || []).map((uci) => uci.slice(2, 4));
  const placement = parsePlacement(state.board.fen);
  for (const t of new Set(targets)) {
    const te = squareEl(t);
    if (!te) continue;
    if (placement[t]) te.classList.add("capture");
    const dot = document.createElement("span");
    dot.className = "dot";
    te.appendChild(dot);
  }
}

function clearHints() {
  for (const sq of boardEl.children) {
    sq.classList.remove("selected", "capture", "hover-target");
    const dot = sq.querySelector(".dot");
    if (dot) dot.remove();
  }
}

/* ---------------- arrows ---------------- */
function squareCenter(name) {
  const file = FILES.indexOf(name[0]);
  const rank = parseInt(name[1], 10);
  let x, y;
  if (state.orientation === "white") { x = file; y = 8 - rank; }
  else { x = 7 - file; y = rank - 1; }
  return { x: x + 0.5, y: y + 0.5 };
}

function drawArrows() {
  arrowsEl.innerHTML = "";
  if (!state.showArrows || !state.analysis || !state.analysis.lines) return;
  const lines = state.analysis.lines;
  const palette = ["#a7a7a7", "#3692e7", "#629924", "#d59120", "#bf811d", "#c94b4b"];
  // Draw weaker lines first so the best move sits on top.
  for (let i = Math.min(lines.length, palette.length) - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line.uci || !line.uci.length) continue;
    const uci = line.uci[0];
    drawArrow(uci.slice(0, 2), uci.slice(2, 4), palette[i], i === 0);
  }
}

function drawArrow(from, to, color, primary) {
  const a = squareCenter(from);
  const b = squareCenter(to);
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const head = primary ? 0.34 : 0.28;
  const width = primary ? 0.14 : 0.10;
  // Shorten the shaft so the head sits at the target centre.
  const ex = b.x - ux * head, ey = b.y - uy * head;

  const shaft = document.createElementNS(SVG_NS, "line");
  shaft.setAttribute("x1", a.x); shaft.setAttribute("y1", a.y);
  shaft.setAttribute("x2", ex); shaft.setAttribute("y2", ey);
  shaft.setAttribute("stroke", color);
  shaft.setAttribute("stroke-width", width);
  shaft.setAttribute("stroke-linecap", "round");
  shaft.setAttribute("opacity", primary ? "0.92" : "0.6");
  arrowsEl.appendChild(shaft);

  const px = -uy, py = ux; // perpendicular
  const hw = head * 0.6;
  const p1 = `${b.x},${b.y}`;
  const p2 = `${ex + px * hw},${ey + py * hw}`;
  const p3 = `${ex - px * hw},${ey - py * hw}`;
  const tri = document.createElementNS(SVG_NS, "polygon");
  tri.setAttribute("points", `${p1} ${p2} ${p3}`);
  tri.setAttribute("fill", color);
  tri.setAttribute("opacity", primary ? "0.92" : "0.6");
  arrowsEl.appendChild(tri);
}

/* ---------------- interaction ---------------- */
function humanCanMove() {
  if (state.thinking || !state.board || state.board.is_over) return false;
  if (!atLatest()) return false; // only move from the live position
  if (state.mode === "analysis") return true;
  if (state.mode === "arena") return false;
  if (state.playAs === "both") return true;
  return state.board.turn === state.playAs;
}

function onSquareClick(e) {
  if (!humanCanMove()) return;
  const name = e.currentTarget.dataset.square;
  const legal = state.board.legal;
  if (state.selected && state.selected !== name) {
    const moves = legal[state.selected] || [];
    const match = moves.filter((m) => m.slice(2, 4) === name);
    if (match.length) { choosePromotion(match, playUserMove); return; }
  }
  if (legal[name]) { state.selected = name; showSelection(name); }
  else { state.selected = null; clearHints(); }
}

function hidePromotionChooser() {
  const chooser = el("promotionChooser");
  chooser.hidden = true;
  chooser.innerHTML = "";
}

function choosePromotion(matches, onChoice) {
  if (matches.length === 1) { onChoice(matches[0]); return; }
  hidePromotionChooser();
  const chooser = el("promotionChooser");
  const target = matches[0].slice(2, 4);
  const center = squareCenter(target);
  const fromBottom = center.y > 3.5;
  chooser.style.left = `${(center.x - 0.5) * 12.5}%`;
  chooser.style.top = fromBottom ? "auto" : "0";
  chooser.style.bottom = fromBottom ? "0" : "auto";
  chooser.classList.toggle("from-bottom", fromBottom);

  const isWhite = state.board && state.board.turn === "white";
  for (const role of ["q", "r", "b", "n"]) {
    const move = matches.find((candidate) => candidate.endsWith(role));
    if (!move) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.title = `Promote to ${PIECE_NAMES[role]}`;
    const image = document.createElement("img");
    const symbol = isWhite ? role.toUpperCase() : role;
    image.src = PIECE_URLS[symbol];
    image.alt = PIECE_NAMES[role];
    image.draggable = false;
    button.appendChild(image);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      hidePromotionChooser();
      onChoice(move);
    });
    chooser.appendChild(button);
  }
  chooser.hidden = false;
}

/* drag and drop */
let drag = null;
function onPointerDown(e) {
  if (!humanCanMove()) return;
  const name = e.currentTarget.dataset.square;
  if (!state.board.legal[name]) return;
  e.preventDefault();
  state.selected = name;
  showSelection(name);

  const pieceEl = e.currentTarget.querySelector(".piece");
  if (!pieceEl) return;
  const ghost = pieceEl.cloneNode(true);
  ghost.className = "drag-ghost";
  ghost.alt = "";
  ghost.setAttribute("aria-hidden", "true");
  document.body.appendChild(ghost);

  drag = { from: name, ghost, target: null };
  e.currentTarget.classList.add("dragging");
  moveGhost(e);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp, { once: true });
}

function moveGhost(e) {
  if (!drag) return;
  drag.ghost.style.left = e.clientX + "px";
  drag.ghost.style.top = e.clientY + "px";
}

function onPointerMove(e) {
  if (!drag) return;
  moveGhost(e);
  const target = document.elementFromPoint(e.clientX, e.clientY);
  const sq = target ? target.closest(".sq") : null;
  if (drag.target && drag.target !== sq) drag.target.classList.remove("hover-target");
  if (sq && sq !== squareEl(drag.from)) { sq.classList.add("hover-target"); drag.target = sq; }
  else drag.target = null;
}

function onPointerUp(e) {
  window.removeEventListener("pointermove", onPointerMove);
  if (!drag) return;
  const fromEl = squareEl(drag.from);
  if (fromEl) fromEl.classList.remove("dragging");
  drag.ghost.remove();
  if (drag.target) drag.target.classList.remove("hover-target");

  const target = document.elementFromPoint(e.clientX, e.clientY);
  const sq = target ? target.closest(".sq") : null;
  const to = sq ? sq.dataset.square : null;
  const from = drag.from;
  drag = null;
  if (to && to !== from) {
    const matches = (state.board.legal[from] || []).filter((m) => m.slice(2, 4) === to);
    if (matches.length) { choosePromotion(matches, playUserMove); return; }
  }
  render();
}

/* ---------------- game flow ---------------- */
function pushMove(result, uci, actor, timeMs) {
  // Truncate any future when branching from a past position.
  if (!atLatest()) state.moves.length = state.cursor + 1;
  const color = result.turn === "white" ? "black" : "white";
  state.moves.push({
    fen: result.fen,
    san: result.san,
    uci: uci,
    color,
    moveNumber: color === "white" ? result.fullmove : Math.max(1, result.fullmove - 1),
    actor: actor || "human",
    engine: actor !== "human" && actor !== "imported",
    timeMs: timeMs ?? result.time_ms ?? null,
    isOver: !!result.is_over,
    result: result.result || "*",
    termination: result.termination || null,
    cpWhite: null,
    opening: result.opening || state.opening,
    openingEco: result.opening_eco || null,
    openingName: result.opening_name || null,
    openingPly: result.opening_ply ?? null,
  });
  state.cursor = state.moves.length - 1;
  analyzeToken++;
  state.analysis = null;
  state.mateWhite = null;
}

async function playUserMove(uci) {
  state.selected = null;
  clearHints();
  try {
    const result = await api("/api/move", positionPayload({ uci }));
    pushMove(result, uci, "human");
    state.board = result;
    render();
    if (result.is_over) { showOverlay(); return; }
    if (state.mode === "play") await maybeEngineMove();
    else scheduleAnalysis();
  } catch (err) { toast(err.message); }
}

async function maybeEngineMove() {
  if (state.mode !== "play") return;
  const engineShouldPlay = state.playAs !== "both" && state.board.turn !== state.playAs;
  if (!engineShouldPlay) { scheduleAnalysis(); return; }
  await engineMove();
}

function applyAutomatedMove(result, actor) {
  pushMove(result, result.engine_move, actor, result.time_ms);
  state.board = result;
  const mover = state.moves[state.cursor].color;
  state.analysis = {
    lines: result.analysis || [],
    depth: result.depth,
    nodes: result.nodes,
    nps: result.nps,
    time_ms: result.time_ms,
    turn: result.analysis_turn || mover,
  };
  const gauge = result.superchess_eval;
  if (gauge && gauge.source === "superchess" && Number.isFinite(gauge.cp_white)) {
    recordWhiteEval(gauge.cp_white, gauge.mate_white ?? null, gauge.source);
  }
  setActiveEngine(actor, result);
  renderAnalysis();
  render();
  if (result.is_over) showOverlay();
}

function setActiveEngine(actor, result) {
  if (actor === "stockfish") {
    const level = result.stockfish ? result.stockfish.level : state.arenaLevel;
    el("activeEngineName").textContent = `STOCKFISH · LEVEL ${level}`;
    el("activeEngineDetail").textContent = "Lichess strength preset";
  } else {
    el("activeEngineName").textContent = "SUPERCHESS NN";
    el("activeEngineDetail").textContent = "local neural engine";
  }
}

async function engineMove(arenaToken = null) {
  if (!state.board || state.board.is_over) return;
  state.thinking = true;
  setBusy(true);
  try {
    const result = await api("/api/engine", positionPayload({
      simulations: state.simulations,
      temperature: state.temperature,
      multipv: state.multipv,
      pv_length: state.pvLength,
      game_id: state.game ? state.game.id : null,
    }));
    if (arenaToken != null && arenaToken !== state.arenaToken) return false;
    applyAutomatedMove(result, "superchess");
    return true;
  } catch (err) {
    toast(err.message);
    return false;
  }
  finally {
    state.thinking = false;
    setBusy(false);
    updateNavButtons();
    updateArenaControls();
  }
}

async function stockfishMove(arenaToken) {
  if (!state.board || state.board.is_over) return false;
  state.thinking = true;
  setBusy(true);
  try {
    const result = await api("/api/stockfish", positionPayload({
      level: state.arenaLevel,
      game_id: state.game ? state.game.id : null,
    }));
    if (arenaToken !== state.arenaToken) return false;
    applyAutomatedMove(result, "stockfish");
    if (result.stockfish && !result.stockfish.exact_skill) {
      el("stockfishAvailability").textContent =
        `This Stockfish accepts skill ${result.stockfish.effective_skill}, not Lichess skill ${result.stockfish.requested_skill}; time and depth still match.`;
    }
    return true;
  } catch (err) {
    toast(err.message);
    return false;
  } finally {
    state.thinking = false;
    setBusy(false);
    updateNavButtons();
    updateArenaControls();
  }
}

const arenaDelay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function runArena(token) {
  while (state.mode === "arena" && state.arenaRunning && token === state.arenaToken) {
    if (!state.board || state.board.is_over) {
      finishArena();
      return;
    }
    const actor = state.board.turn === state.arenaSuperchessColor ? "superchess" : "stockfish";
    el("arenaStatus").textContent = actor === "superchess" ? "Superchess thinking…" : "Stockfish thinking…";
    updateArenaControls();
    const moved = actor === "superchess" ? await engineMove(token) : await stockfishMove(token);
    if (!moved) {
      if (token === state.arenaToken && state.arenaRunning) {
        state.arenaRunning = false;
        state.arenaPaused = false;
        el("arenaStatus").textContent = "Stopped after engine error";
      }
      break;
    }
    if (token !== state.arenaToken || !state.arenaRunning) break;
    if (state.board.is_over) {
      finishArena();
      return;
    }
    await arenaDelay(220);
  }
  updateArenaControls();
}

function finishArena() {
  state.arenaRunning = false;
  state.arenaPaused = false;
  el("arenaStatus").textContent = state.board && state.board.is_over
    ? `Finished · ${state.board.result}`
    : "Stopped";
  updateArenaControls();
  if (state.board && state.board.is_over) showOverlay();
}

function stopArena({ paused = false } = {}) {
  if (state.arenaRunning || state.arenaPaused || state.thinking) state.arenaToken++;
  state.arenaRunning = false;
  state.arenaPaused = paused;
  updateArenaControls();
}

/* ---------------- analysis ---------------- */
let analyzeToken = 0;
let analyzeTimer = null;

function scheduleAnalysis(force) {
  if (state.mode === "arena" && !force) { renderAnalysis(); return; }
  if (state.mode === "arena" && (state.arenaRunning || state.thinking)) return;
  if (!state.analyzeOn && !force) { renderAnalysis(); return; }
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(() => runAnalysis(), 120);
}

async function runAnalysis() {
  if (!state.board || state.board.is_over) { state.analysis = null; renderAnalysis(); return; }
  const token = ++analyzeToken;
  el("engineStats").classList.add("busy");
  try {
    const data = await api("/api/analyze", positionPayload({
      simulations: state.simulations,
      multipv: state.multipv,
      pv_length: state.pvLength,
    }));
    if (token !== analyzeToken) return; // stale
    state.analysis = data;
    setActiveEngine("superchess", data);
    recordEval(data.value, data.turn);
    renderAnalysis();
    drawArrows();
  } catch (err) {
    if (token === analyzeToken) toast(err.message);
  } finally {
    if (token === analyzeToken) el("engineStats").classList.remove("busy");
  }
}

function recordEval(value, turn) {
  recordWhiteEval(valueToCpWhite(value, turn), bestMateWhite(), "superchess");
}

function recordWhiteEval(cpWhite, mateWhite, source) {
  if (source !== "superchess") return;
  state.cpWhite = cpWhite;
  state.mateWhite = mateWhite;
  state.evaluationSource = source;
  setEvalBar(state.cpWhite, state.mateWhite);
  if (state.cursor >= 0) state.moves[state.cursor].cpWhite = cpWhite;
  drawGraph();
}

function valueToCpWhite(value, turn) {
  const white = turn === "white" ? value : -value;
  const v = Math.max(-0.9999, Math.min(0.9999, white));
  return 111.714640912 * Math.tan(1.5620688421 * v);
}

function bestMateWhite() {
  if (!state.analysis || !state.analysis.lines || !state.analysis.lines.length) return null;
  const line = state.analysis.lines[0];
  if (line.mate == null) return null;
  const turn = state.analysis.turn || (state.board ? state.board.turn : "white");
  return turn === "white" ? line.mate : -line.mate;
}

/* ---------------- rendering: analysis panel ---------------- */
function fmtScore(cpWhite, mateWhite) {
  if (mateWhite != null) return (mateWhite >= 0 ? "#" : "#-") + Math.abs(mateWhite);
  const pawns = cpWhite / 100;
  const sign = pawns > 0 ? "+" : pawns < 0 ? "" : "";
  return sign + pawns.toFixed(2);
}

function scoreClass(cpWhite, mateWhite) {
  const v = mateWhite != null ? mateWhite : cpWhite;
  if (v > 0.2) return "pos";
  if (v < -0.2) return "neg";
  return "even";
}

function renderAnalysis() {
  const list = el("engineLines");
  const a = state.analysis;
  if (!a || !a.lines || !a.lines.length) {
    list.innerHTML = `<li class="lines-empty">${state.analyzeOn ? "No engine lines." : "Analysis is off."}</li>`;
    el("engineScore").textContent = state.analyzeOn ? "—" : "Off";
    el("statDepth").textContent = "—";
    el("statNodes").textContent = "—";
    el("statNps").textContent = "—";
    el("statTime").textContent = "—";
    return;
  }
  el("statDepth").textContent = a.depth ?? "—";
  el("statNodes").textContent = fmtCount(a.nodes);
  el("statNps").textContent = fmtCount(a.nps);
  el("statTime").textContent = a.time_ms != null ? (a.time_ms / 1000).toFixed(1) + "s" : "—";

  const turn = a.turn || (state.board ? state.board.turn : "white");
  const best = a.lines[0];
  const bestCpWhite = turn === "white" ? best.cp : -best.cp;
  const bestMateWhite = best.mate == null ? null : (turn === "white" ? best.mate : -best.mate);
  el("engineScore").textContent = fmtScore(bestCpWhite, bestMateWhite);
  list.innerHTML = "";
  a.lines.forEach((line, idx) => {
    const cpWhite = turn === "white" ? line.cp : -line.cp;
    const mateWhite = line.mate == null ? null : (turn === "white" ? line.mate : -line.mate);
    const li = document.createElement("li");
    li.className = "line" + (idx === 0 ? " best" : "");
    const sans = line.san || [];
    const pv = sans
      .map((s, i) => i === 0 ? `<span class="pv-first">${s}</span>` : s)
      .join(" ");
    li.innerHTML = `
      <span class="line-score ${scoreClass(cpWhite, mateWhite)}">${fmtScore(cpWhite, mateWhite)}</span>
      <span class="line-pv">${pv}</span>`;
    li.addEventListener("click", () => { if (line.uci && line.uci[0]) tryPlayUci(line.uci[0]); });
    list.appendChild(li);
  });
}

async function tryPlayUci(uci) {
  if (!humanCanMove()) return;
  const matches = Object.values(state.board.legal).flat().filter((m) => m === uci || m.slice(0, 4) === uci.slice(0, 4));
  if (matches.length) await playUserMove(matches[0]);
}

function fmtCount(n) {
  if (n == null) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

/* ---------------- eval bar ---------------- */
function setEvalBar(cpWhite, mateWhite) {
  let prob;
  if (mateWhite != null) prob = mateWhite >= 0 ? 1 : 0;
  else prob = 1 / (1 + Math.pow(10, -cpWhite / 400));
  const flipped = state.orientation === "black";
  el("evalBar").classList.toggle("flipped", flipped);
  el("evalBarFill").style.height = (prob * 100).toFixed(1) + "%";
  const label = fmtScore(cpWhite, mateWhite);
  const whiteAtBottom = !flipped;
  const labelAtBottom = prob >= 0.5 ? whiteAtBottom : !whiteAtBottom;
  el("evalBarLabelBottom").textContent = labelAtBottom ? label : "";
  el("evalBarLabelTop").textContent = labelAtBottom ? "" : label;
  el("evalBar").title = `${label} · ${prob >= 0.5 ? "White" : "Black"} advantage · Superchess evaluation`;
}

/* ---------------- eval graph ---------------- */
function drawGraph() {
  const W = 300, H = 90, mid = 45;
  const pts = [{ ply: -1, cp: 0 }];
  state.moves.forEach((m, i) => pts.push({ ply: i, cp: m.cpWhite == null ? null : m.cpWhite }));
  // Forward-fill nulls so the line stays continuous.
  let lastCp = 0;
  const coords = pts.map((p, i) => {
    const cp = p.cp == null ? lastCp : p.cp;
    lastCp = cp;
    const x = pts.length <= 1 ? 0 : (i / (pts.length - 1)) * W;
    const y = mid - Math.tanh(cp / 350) * (mid - 4);
    return [x, y];
  });
  const line = coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  el("graphLine").setAttribute("points", line);
  const area = `0,${mid} ${line} ${W},${mid}`;
  el("graphArea").setAttribute("points", area);
  const idx = state.cursor + 1;
  const cx = pts.length <= 1 ? 0 : (idx / (pts.length - 1)) * W;
  el("graphCursor").setAttribute("x1", cx);
  el("graphCursor").setAttribute("x2", cx);
}

/* ---------------- moves ---------------- */
function renderMoves() {
  const list = el("moveList");
  list.innerHTML = "";
  if (!state.moves.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "move-placeholder";
    placeholder.textContent = "Make a move or load a game to begin.";
    list.appendChild(placeholder);
    return;
  }
  for (let i = 0; i < state.moves.length; i += 2) {
    const no = document.createElement("span");
    no.className = "move-no";
    no.textContent = (i / 2 + 1) + ".";
    list.appendChild(no);
    list.appendChild(moveCell(i));
    if (i + 1 < state.moves.length) list.appendChild(moveCell(i + 1));
    else list.appendChild(document.createElement("span"));
  }
  const cur = list.querySelector(".move-san.current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function moveCell(idx) {
  const entry = state.moves[idx];
  const span = document.createElement("span");
  const actorClass = entry.actor === "stockfish" ? " stockfish"
    : entry.actor === "superchess" ? " superchess" : "";
  span.className = "move-san" + (entry.engine ? " engine" : "") + actorClass + (idx === state.cursor ? " current" : "");
  span.textContent = entry.san;
  span.title = entry.actor === "stockfish" ? `Stockfish level ${(state.game && state.game.lichessLevel) || state.arenaLevel}`
    : entry.actor === "superchess" ? "Superchess" : "";
  span.addEventListener("click", () => goTo(idx));
  return span;
}

function updateMaterial() {
  const placement = parsePlacement(state.board.fen);
  const counts = {};
  for (const p of Object.values(placement)) counts[p] = (counts[p] || 0) + 1;
  const start = { P: 8, N: 2, B: 2, R: 2, Q: 1, p: 8, n: 2, b: 2, r: 2, q: 1 };
  const capturedBlack = [], capturedWhite = [];
  let scoreW = 0, scoreB = 0;
  for (const t of ["q", "r", "b", "n", "p"]) {
    const wMissing = start[t.toUpperCase()] - (counts[t.toUpperCase()] || 0);
    const bMissing = start[t] - (counts[t] || 0);
    for (let i = 0; i < bMissing; i++) capturedBlack.push(PIECE_GLYPHS[t]);
    for (let i = 0; i < wMissing; i++) capturedWhite.push(PIECE_GLYPHS[t.toUpperCase()]);
    scoreW += bMissing * VALUES[t];
    scoreB += wMissing * VALUES[t];
  }
  el("capturedByWhite").textContent = capturedBlack.join("");
  el("capturedByBlack").textContent = capturedWhite.join("");
  const diff = scoreW - scoreB;
  el("matWhite").textContent = diff > 0 ? `+${diff}` : "";
  el("matBlack").textContent = diff < 0 ? `+${-diff}` : "";
  el("materialBadge").textContent = diff === 0 ? "even" : (diff > 0 ? `White +${diff}` : `Black +${-diff}`);
}

function updateOpening() {
  if (state.cursor < 0) {
    state.opening = state.startFen === START_FEN ? "Starting position" : "Custom position";
  } else {
    const moveOpening = state.moves[state.cursor] ? state.moves[state.cursor].opening : null;
    state.opening = (state.board && state.board.opening) || moveOpening || "Unclassified opening";
  }
  if (atLatest() && state.game && state.opening && state.opening !== "Starting position") {
    state.game.opening = state.opening;
    state.game.openingEco = state.board ? state.board.opening_eco : null;
    state.game.openingName = state.board ? state.board.opening_name : null;
  }
  const name = state.opening || "Unclassified position";
  el("openingName").textContent = name;
  el("topStatus").textContent = name;
}

function updatePositionFields() {
  const fenField = el("fenField");
  const pgnField = el("pgnField");
  if (document.activeElement !== fenField) fenField.value = currentFen();
  if (document.activeElement !== pgnField) pgnField.value = buildPgn();
}

/* ---------------- navigation ---------------- */
async function goTo(index) {
  index = Math.max(-1, Math.min(index, state.moves.length - 1));
  if (index === state.cursor && state.board) return;
  if (state.mode === "arena" && state.arenaRunning) stopArena({ paused: true });
  state.cursor = index;
  state.selected = null;
  analyzeToken++;
  state.cpWhite = index < 0 ? 0 : (state.moves[index].cpWhite ?? 0);
  state.mateWhite = null;
  hidePromotionChooser();
  hideOverlay();
  try {
    state.board = await api("/api/state", positionPayload());
  } catch (err) { toast(err.message); return; }
  if (state.board.is_over && atLatest()) showOverlay();
  render();
  scheduleAnalysis();
}

function updateNavButtons() {
  el("navStart").disabled = state.cursor < 0 || state.thinking;
  el("navPrev").disabled = state.cursor < 0 || state.thinking;
  el("navNext").disabled = atLatest() || state.thinking;
  el("navEnd").disabled = atLatest() || state.thinking;
  el("undoBtn").disabled = state.moves.length === 0 || state.thinking || state.mode === "arena";
}

/* ---------------- game lifecycle ---------------- */
function newGameMetadata(headers) {
  const source = headers || {};
  let white = source.White || "White";
  let black = source.Black || "Black";
  let event = source.Event || "Superchess analysis";
  if (!headers && state.mode === "play" && state.playAs !== "both") {
    white = state.playAs === "white" ? "Human" : "Superchess";
    black = state.playAs === "black" ? "Human" : "Superchess";
    event = "Human vs Superchess";
  } else if (!headers && state.mode === "arena") {
    white = state.arenaSuperchessColor === "white" ? "Superchess" : `Stockfish level ${state.arenaLevel}`;
    black = state.arenaSuperchessColor === "black" ? "Superchess" : `Stockfish level ${state.arenaLevel}`;
    event = `Superchess vs Stockfish level ${state.arenaLevel}`;
  }
  return {
    id: makeGameId(),
    createdAt: new Date().toISOString(),
    event,
    site: source.Site || "Superchess GUI",
    round: source.Round || "-",
    white,
    black,
    lichessLevel: state.mode === "arena" ? state.arenaLevel : null,
    superchessColor: state.mode === "arena" ? state.arenaSuperchessColor : null,
    result: source.Result || "*",
    openingEco: source.ECO || null,
    openingName: source.Opening || null,
  };
}

async function newGame() {
  stopArena();
  hidePromotionChooser();
  hideOverlay();
  analyzeToken++;
  state.moves = [];
  state.cursor = -1;
  state.selected = null;
  state.thinking = false;
  state.analysis = null;
  state.cpWhite = 0;
  state.mateWhite = null;
  state.evaluationSource = "superchess";
  state.opening = "Starting position";
  state.startFen = START_FEN;
  state.game = newGameMetadata();
  setActiveEngine("superchess", {});
  state.orientation = state.mode === "arena" ? state.arenaSuperchessColor
    : (state.mode === "play" && state.playAs === "black") ? "black" : "white";
  try {
    state.board = await api("/api/state", positionPayload());
  } catch (err) { toast(err.message); return false; }
  buildBoard();
  render();
  setEvalBar(0, null);
  drawGraph();
  if (state.mode === "play" && state.playAs === "black") engineMove();
  else if (state.mode === "arena") {
    el("arenaStatus").textContent = "Ready";
    state.analysis = null;
    renderAnalysis();
    updateArenaControls();
  } else scheduleAnalysis();
  return true;
}

async function undo() {
  if (state.thinking || state.moves.length === 0 || state.mode === "arena") return;
  const toPop = (state.mode === "play" && state.playAs !== "both") ? Math.min(2, state.moves.length) : 1;
  state.moves.length = Math.max(0, state.moves.length - toPop);
  await goTo(state.moves.length - 1);
}

async function hint() {
  if (!state.board || state.board.is_over) return;
  await runAnalysis();
  if (state.analysis && state.analysis.lines && state.analysis.lines[0]) {
    const uci = state.analysis.lines[0].uci[0];
    state.selected = uci.slice(0, 2);
    render();
    const te = squareEl(uci.slice(2, 4));
    if (te) te.classList.add("hover-target");
    toast(`Best: ${state.analysis.lines[0].san[0]}`);
  }
}

/* ---------------- overlay / status ---------------- */
function describeOutcome(s) {
  if (s.termination === "checkmate") {
    const winner = s.turn === "white" ? "Black" : "White";
    return `Checkmate — ${winner} wins.`;
  }
  if (s.result === "1/2-1/2") return `Draw (${(s.termination || "").replace(/_/g, " ")}).`;
  return `Game over: ${s.result}`;
}

function showOverlay() {
  const s = state.board;
  const title = s.termination === "checkmate" ? "Checkmate"
    : s.result === "1/2-1/2" ? "Draw" : "Game over";
  el("overlayTitle").textContent = title;
  el("overlaySub").textContent = describeOutcome(s);
  el("boardOverlay").hidden = false;
}
function hideOverlay() { el("boardOverlay").hidden = true; }

function setBusy(on) {
  el("hintBtn").disabled = on;
  el("newBtn").disabled = on;
  el("analyzeNowBtn").disabled = on;
  document.querySelectorAll("#sideSelect button").forEach((button) => { button.disabled = on; });
  document.querySelectorAll("#modeSwitch button").forEach((button) => { button.disabled = on; });
  updateNavButtons();
  updateArenaControls();
}

let toastTimer = null;
function toast(msg) {
  const t = el("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}

/* ---------------- PGN / FEN ---------------- */
function pgnEscape(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function gameResult() {
  const last = state.moves[state.moves.length - 1];
  if (last && last.isOver) return last.result || "*";
  if (atLatest() && state.board && state.board.is_over) return state.board.result || "*";
  return state.game && state.game.result ? state.game.result : "*";
}

function buildMovetext() {
  const tokens = [];
  let previous = null;
  state.moves.forEach((move, index) => {
    const number = move.moveNumber || Math.floor(index / 2) + 1;
    if (move.color === "white") tokens.push(`${number}.`, move.san);
    else if (previous && previous.color === "white" && (previous.moveNumber || number) === number) tokens.push(move.san);
    else tokens.push(`${number}...`, move.san);
    previous = move;
  });
  tokens.push(gameResult());
  return tokens.join(" ").trim();
}

function buildPgn() {
  const game = state.game || newGameMetadata();
  const date = (game.createdAt || new Date().toISOString()).slice(0, 10).replace(/-/g, ".");
  const headers = [
    ["Event", game.event || "Superchess game"],
    ["Site", game.site || "Superchess GUI"],
    ["Date", date],
    ["Round", game.round || "-"],
    ["White", game.white || "White"],
    ["Black", game.black || "Black"],
    ["Result", gameResult()],
  ];
  if (state.startFen !== START_FEN) headers.push(["SetUp", "1"], ["FEN", state.startFen]);
  if (game.lichessLevel) headers.push(["LichessLevel", String(game.lichessLevel)]);
  if (game.superchessColor) headers.push(["SuperchessColor", game.superchessColor]);
  if (game.openingEco) headers.push(["ECO", game.openingEco]);
  if (game.openingName) headers.push(["Opening", game.openingName]);
  headers.push(["ReplayFormat", "superchess-replay-v1"]);
  return `${headers.map(([key, value]) => `[${key} "${pgnEscape(value)}"]`).join("\n")}\n\n${buildMovetext()}`;
}

function replayPayload() {
  const lastIndex = state.moves.length - 1;
  return {
    format: "superchess-replay-v1",
    created_at: state.game ? state.game.createdAt : new Date().toISOString(),
    game_id: state.game ? state.game.id : null,
    event: state.game ? state.game.event : "Superchess game",
    white: state.game ? state.game.white : "White",
    black: state.game ? state.game.black : "Black",
    result: gameResult(),
    start_fen: state.startFen,
    orientation: state.orientation,
    theme: document.body.getAttribute("data-theme") || "brown",
    piece_set: "cburnett",
    opening: state.game && state.game.opening ? state.game.opening : state.opening,
    pgn: buildPgn(),
    frames: [
      {
        ply: 0,
        fen: state.startFen,
        opening: state.startFen === START_FEN ? "Starting position" : "Custom position",
        evaluation_cp_white: 0,
        display_ms: state.moves.length ? 900 : 1800,
      },
      ...state.moves.map((move, index) => ({
        ply: index + 1,
        fen: move.fen,
        san: move.san,
        uci: move.uci,
        color: move.color,
        actor: move.actor || "unknown",
        opening: move.opening || null,
        opening_eco: move.openingEco || null,
        opening_name: move.openingName || null,
        evaluation_cp_white: move.cpWhite,
        think_time_ms: move.timeMs,
        display_ms: index === lastIndex ? 1800 : 700,
      })),
    ],
  };
}

function downloadText(text, filename, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportStem() {
  const date = new Date().toISOString().replace(/[:.]/g, "-");
  return state.game && state.game.lichessLevel
    ? `superchess-vs-stockfish-l${state.game.lichessLevel}-${date}`
    : `superchess-game-${date}`;
}

function downloadPgn() {
  downloadText(buildPgn(), `${exportStem()}.pgn`, "application/x-chess-pgn;charset=utf-8");
  toast("PGN downloaded");
}

function downloadReplay() {
  downloadText(JSON.stringify(replayPayload(), null, 2), `${exportStem()}.replay.json`, "application/json;charset=utf-8");
  toast("Replay downloaded — FEN frames are ready for a GIF renderer");
}

async function downloadGif() {
  const button = el("downloadGifBtn");
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Rendering GIF…";
  try {
    const blob = await apiBlob("/api/gif", replayPayload());
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${exportStem()}.gif`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    toast("Animated GIF downloaded");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label} copied`);
  } catch (_) {
    // Fallback for non-secure contexts.
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); toast(`${label} copied`); }
    catch (e) { toast("copy failed"); }
    ta.remove();
  }
}

function hideImportModal() {
  el("importModal").hidden = true;
}

async function loadPositionText(text) {
  text = text.trim();
  if (!text) { toast("Nothing to load"); return false; }
  try {
    stopArena();
    hidePromotionChooser();
    analyzeToken++;
    const data = await api("/api/import", { text });
    const history = data.history || [];
    state.moves = history.map((h, index) => ({
      fen: h.fen,
      san: h.san,
      uci: h.uci,
      color: h.color,
      moveNumber: h.move_number || Math.floor(index / 2) + 1,
      actor: "imported",
      engine: false,
      timeMs: null,
      isOver: index === history.length - 1 && !!data.is_over,
      result: index === history.length - 1 ? data.result : "*",
      termination: index === history.length - 1 ? data.termination : null,
      cpWhite: null,
      opening: h.opening || null,
      openingEco: h.opening_eco || null,
      openingName: h.opening_name || null,
      openingPly: h.opening_ply ?? null,
    }));
    state.startFen = data.start_fen || (state.moves.length ? START_FEN : data.fen);
    state.game = newGameMetadata(data.headers || {});
    state.cursor = state.moves.length - 1;
    state.selected = null;
    state.analysis = null;
    state.cpWhite = 0;
    state.mateWhite = null;
    state.opening = data.opening || (state.moves.length ? "Imported game" : "Custom position");
    state.orientation = "white";
    buildBoard();
    state.board = await api("/api/state", positionPayload());
    render();
    setEvalBar(0, null);
    drawGraph();
    scheduleAnalysis();
    toast(`Loaded ${data.import_kind === "pgn" ? "PGN" : "FEN"}`);
    return true;
  } catch (err) {
    toast(err.message);
    return false;
  }
}

async function applyImport() {
  if (await loadPositionText(el("importText").value)) hideImportModal();
}

/* ---------------- mode switching ---------------- */
function setMode(mode) {
  if (state.mode === "arena" && mode !== "arena") stopArena();
  state.mode = mode;
  document.querySelectorAll("#modeSwitch button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  const playSetup = el("playAsGroup");
  const arenaSetup = el("arenaSetup");
  playSetup.hidden = mode !== "play";
  arenaSetup.hidden = mode !== "arena";
  updatePlaySideControls();
  if (mode === "analysis") {
    state.analyzeOn = true;
    el("analyzeToggle").checked = true;
  }
  el("analyzeToggle").disabled = mode === "arena";
  if (mode !== "arena") scheduleAnalysis();
  else updateArenaControls();
  const visibleSetup = mode === "play" ? playSetup : mode === "arena" ? arenaSetup : null;
  if (visibleSetup) {
    requestAnimationFrame(() => visibleSetup.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  }
}

function updatePlaySideControls() {
  document.querySelectorAll("#sideSelect button").forEach((button) => {
    const active = button.dataset.side === state.playAs;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  el("playSideHint").textContent = state.playAs === "both"
    ? "Control both sides"
    : `You play ${state.playAs === "white" ? "White" : "Black"}`;
}

async function choosePlaySide(side) {
  if (state.thinking) return;
  state.playAs = side;
  if (!state.moves.length) state.game = newGameMetadata();
  state.selected = null;
  analyzeToken++;
  hidePromotionChooser();
  updatePlaySideControls();

  if (side !== "both" && state.orientation !== side) {
    state.orientation = side;
    buildBoard();
  }
  render();

  if (state.mode === "play" && state.board && !state.board.is_over) {
    await maybeEngineMove();
  }
}

function updateArenaLabels() {
  const stockfishColor = state.arenaSuperchessColor === "white" ? "Black" : "White";
  el("arenaSuperchessColor").textContent = state.arenaSuperchessColor === "white" ? "White" : "Black";
  el("arenaStockfishLabel").textContent = `Level ${state.arenaLevel} · ${stockfishColor}`;
  el("arenaLevelHint").textContent = String(state.arenaLevel);
  document.querySelectorAll("#arenaLevelSelect button").forEach((button) =>
    button.classList.toggle("active", Number(button.dataset.level) === state.arenaLevel));
  document.querySelectorAll("#arenaColorSelect button").forEach((button) =>
    button.classList.toggle("active", button.dataset.color === state.arenaSuperchessColor));

  const reported = state.stockfishInfo && Array.isArray(state.stockfishInfo.levels)
    ? state.stockfishInfo.levels.find((item) => Number(item.level) === state.arenaLevel)
    : null;
  const preset = reported || LICHESS_STOCKFISH_PRESETS[state.arenaLevel - 1];
  const effective = Number.isFinite(Number(preset.effective_skill))
    ? Number(preset.effective_skill) : preset.skill;
  const limited = effective !== preset.skill;
  el("arenaPresetDetail").innerHTML = limited
    ? `Lichess: skill ${preset.skill} · ${preset.move_time_ms} ms · depth ${preset.depth}<br><span class="limited">Installed Stockfish uses skill ${effective}</span>`
    : `Skill ${preset.skill} · ${preset.move_time_ms} ms · depth ${preset.depth}`;
  document.querySelectorAll("#arenaLevelSelect button").forEach((button) => {
    const item = LICHESS_STOCKFISH_PRESETS[Number(button.dataset.level) - 1];
    button.title = `Lichess level ${item.level}: skill ${item.skill}, ${item.move_time_ms} ms, depth ${item.depth}`;
  });
}

function updateArenaControls() {
  const available = !!(state.stockfishInfo && state.stockfishInfo.available);
  const locked = state.arenaRunning || state.arenaPaused || state.thinking;
  updateArenaLabels();
  document.querySelectorAll("#arenaLevelSelect button, #arenaColorSelect button").forEach((button) => {
    button.disabled = locked;
  });
  el("arenaStartBtn").disabled = !available || locked;
  el("arenaPauseBtn").disabled = !state.arenaRunning && (!state.arenaPaused || state.thinking);
  el("arenaPauseBtn").textContent = state.arenaPaused ? "Resume" : "Pause";
}

function chooseArenaLevel(level) {
  if (state.arenaRunning || state.arenaPaused || state.thinking) return;
  state.arenaLevel = Math.max(1, Math.min(8, Number(level)));
  if (!state.moves.length) state.game = newGameMetadata();
  updateArenaControls();
}

function chooseArenaColor(color) {
  if (state.arenaRunning || state.arenaPaused || state.thinking || !["white", "black"].includes(color)) return;
  state.arenaSuperchessColor = color;
  state.orientation = color;
  if (!state.moves.length) state.game = newGameMetadata();
  buildBoard();
  render();
  updateArenaControls();
}

async function startArenaMatch() {
  if (state.thinking) return;
  if (!state.stockfishInfo || !state.stockfishInfo.available) {
    toast((state.stockfishInfo && state.stockfishInfo.error) || "Stockfish is unavailable");
    return;
  }
  if (!await newGame()) return;
  state.game = newGameMetadata();
  state.arenaPaused = false;
  state.arenaRunning = true;
  const token = ++state.arenaToken;
  el("arenaStatus").textContent = "Starting…";
  updateArenaControls();
  void runArena(token);
}

async function toggleArenaPause() {
  if (state.arenaRunning) {
    stopArena({ paused: true });
    el("arenaStatus").textContent = "Paused";
    return;
  }
  if (!state.arenaPaused || state.thinking || !state.board || state.board.is_over) return;
  if (!atLatest()) await goTo(state.moves.length - 1);
  state.arenaPaused = false;
  state.arenaRunning = true;
  const token = ++state.arenaToken;
  updateArenaControls();
  void runArena(token);
}

function flipBoard() {
  state.orientation = state.orientation === "white" ? "black" : "white";
  hidePromotionChooser();
  buildBoard();
  render();
}

function showSettings() {
  const settings = el("settingsPanel");
  settings.open = true;
  settings.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------------- controls wiring ---------------- */
function wireControls() {
  el("newBtn").addEventListener("click", newGame);
  el("overlayNew").addEventListener("click", newGame);
  el("hintBtn").addEventListener("click", hint);
  el("undoBtn").addEventListener("click", undo);

  el("flipBtn").addEventListener("click", flipBoard);
  el("preferencesBtn").addEventListener("click", showSettings);
  el("panelSettingsBtn").addEventListener("click", showSettings);

  el("navStart").addEventListener("click", () => goTo(-1));
  el("navPrev").addEventListener("click", () => goTo(state.cursor - 1));
  el("navNext").addEventListener("click", () => goTo(state.cursor + 1));
  el("navEnd").addEventListener("click", () => goTo(state.moves.length - 1));

  document.querySelectorAll("#modeSwitch button").forEach((btn) =>
    btn.addEventListener("click", () => setMode(btn.dataset.mode)));

  document.querySelectorAll("#sideSelect button").forEach((btn) =>
    btn.addEventListener("click", () => choosePlaySide(btn.dataset.side)));

  document.querySelectorAll("#arenaLevelSelect button").forEach((btn) =>
    btn.addEventListener("click", () => chooseArenaLevel(btn.dataset.level)));
  document.querySelectorAll("#arenaColorSelect button").forEach((btn) =>
    btn.addEventListener("click", () => chooseArenaColor(btn.dataset.color)));
  el("arenaStartBtn").addEventListener("click", startArenaMatch);
  el("arenaPauseBtn").addEventListener("click", toggleArenaPause);

  document.querySelectorAll("#themeRow button").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll("#themeRow button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.body.setAttribute("data-theme", btn.dataset.theme);
      try { localStorage.setItem("superchess-theme", btn.dataset.theme); } catch (_) { /* unavailable */ }
    }));

  el("analyzeToggle").addEventListener("change", (e) => {
    state.analyzeOn = e.target.checked;
    if (state.analyzeOn) scheduleAnalysis();
    else {
      analyzeToken++;
      state.analysis = null;
      renderAnalysis();
      drawArrows();
    }
  });

  el("coordToggle").addEventListener("change", (e) =>
    document.body.classList.toggle("no-coords", !e.target.checked));
  el("arrowToggle").addEventListener("change", (e) => {
    state.showArrows = e.target.checked;
    drawArrows();
  });

  const simSlider = el("simSlider");
  simSlider.addEventListener("input", () => {
    state.simulations = parseInt(simSlider.value, 10);
    el("simValue").textContent = `${state.simulations} sims`;
  });
  simSlider.addEventListener("change", () => scheduleAnalysis());
  const mpvSlider = el("mpvSlider");
  mpvSlider.addEventListener("input", () => {
    state.multipv = parseInt(mpvSlider.value, 10);
    el("mpvValue").textContent = `${state.multipv} line${state.multipv > 1 ? "s" : ""}`;
  });
  mpvSlider.addEventListener("change", () => scheduleAnalysis());
  const pvSlider = el("pvSlider");
  pvSlider.addEventListener("input", () => {
    state.pvLength = parseInt(pvSlider.value, 10);
    el("pvValue").textContent = `${state.pvLength} plies`;
  });
  pvSlider.addEventListener("change", () => scheduleAnalysis());
  const tempSlider = el("tempSlider");
  tempSlider.addEventListener("input", () => {
    state.temperature = parseInt(tempSlider.value, 10) / 100;
    el("tempValue").textContent = state.temperature.toFixed(2);
  });

  el("importBtn").addEventListener("click", () => {
    el("importText").value = "";
    el("importModal").hidden = false;
    el("importText").focus();
  });
  el("importCancel").addEventListener("click", hideImportModal);
  el("importClose").addEventListener("click", hideImportModal);
  el("importModal").addEventListener("pointerdown", (event) => {
    if (event.target === el("importModal")) hideImportModal();
  });
  el("importApply").addEventListener("click", applyImport);
  el("copyFenBtn").addEventListener("click", () => copyText(currentFen(), "FEN"));
  el("copyPgnBtn").addEventListener("click", () => copyText(buildPgn() || "*", "PGN"));
  el("downloadPgnBtn").addEventListener("click", downloadPgn);
  el("downloadReplayBtn").addEventListener("click", downloadReplay);
  el("downloadGifBtn").addEventListener("click", downloadGif);
  el("analyzeNowBtn").addEventListener("click", () => scheduleAnalysis(true));

  el("fenField").addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    await loadPositionText(event.currentTarget.value);
    event.currentTarget.blur();
  });
  el("pgnField").addEventListener("keydown", async (event) => {
    if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    await loadPositionText(event.currentTarget.value);
    event.currentTarget.blur();
  });
  el("evalGraph").addEventListener("click", (event) => {
    if (!state.moves.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    goTo(Math.round(ratio * state.moves.length) - 1);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hidePromotionChooser();
      hideImportModal();
      return;
    }
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") { goTo(state.cursor - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { goTo(state.cursor + 1); e.preventDefault(); }
    else if (e.key === "Home") { goTo(-1); e.preventDefault(); }
    else if (e.key === "End") { goTo(state.moves.length - 1); e.preventDefault(); }
    else if (e.key.toLowerCase() === "f") flipBoard();
  });
}

function restorePreferences() {
  let theme = "brown";
  try { theme = localStorage.getItem("superchess-theme") || theme; } catch (_) { /* unavailable */ }
  if (!["brown", "green", "blue", "slate", "purple"].includes(theme)) theme = "brown";
  document.body.setAttribute("data-theme", theme);
  document.querySelectorAll("#themeRow button").forEach((button) =>
    button.classList.toggle("active", button.dataset.theme === theme));
}

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    const dev = info.device === "auto" ? "auto-device" : info.device;
    el("engineInfo").textContent = `${info.checkpoint.split("/").pop()} · ${dev}`;
    state.stockfishInfo = info.stockfish || { available: false, error: "Stockfish status unavailable" };
    const stockfishStatus = el("stockfishAvailability");
    stockfishStatus.classList.toggle("unavailable", !state.stockfishInfo.available);
    const skillRange = state.stockfishInfo.skill_min != null && Number.isFinite(Number(state.stockfishInfo.skill_min))
      ? ` · UCI skill ${state.stockfishInfo.skill_min}–${state.stockfishInfo.skill_max}` : "";
    stockfishStatus.textContent = state.stockfishInfo.available
      ? `${state.stockfishInfo.name || "Stockfish"}${skillRange} · ${state.stockfishInfo.path}`
      : state.stockfishInfo.error;
    updateArenaControls();
  } catch (_) {
    el("engineInfo").textContent = "engine offline";
    state.stockfishInfo = { available: false, error: "Could not reach the GUI backend" };
    el("stockfishAvailability").textContent = state.stockfishInfo.error;
    el("stockfishAvailability").classList.add("unavailable");
    updateArenaControls();
  }
}

async function main() {
  restorePreferences();
  wireControls();
  loadInfo();
  await newGame();
}

main();
