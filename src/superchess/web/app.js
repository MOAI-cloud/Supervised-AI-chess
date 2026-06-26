"use strict";

/* ----------------------------------------------------------------------
 * Superchess web client — Stockfish-style neural analysis board.
 *
 * The Python server (python-chess) is authoritative for legality and the
 * neural MCTS engine. This client renders state, handles input, navigation,
 * live multi-PV analysis, arrows, an eval bar, an eval graph, and PGN/FEN
 * import / export.
 * -------------------------------------------------------------------- */

const PIECES = {
  K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};
const VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const FILES = "abcdefgh";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const state = {
  startFen: START_FEN,
  moves: [],            // [{fen, san, uci, color, cpWhite}]
  cursor: -1,           // -1 = start position; i = after moves[i]
  board: null,          // server state for the displayed position
  orientation: "white",
  mode: "analysis",     // analysis | play
  playAs: "white",      // white | black | both
  selected: null,
  thinking: false,      // engine is making a move (play mode)
  analyzeOn: true,
  analysis: null,       // latest analysis for the displayed position
  simulations: 256,
  multipv: 3,
  temperature: 0.0,
  showArrows: true,
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

/* ---------------- position helpers ---------------- */
function currentFen() {
  return state.cursor < 0 ? state.startFen : state.moves[state.cursor].fen;
}
function atLatest() {
  return state.cursor === state.moves.length - 1;
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
      sq.className = "sq " + (((f + rank) % 2 === 0) ? "dark" : "light");
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
      const span = document.createElement("span");
      span.className = "piece";
      span.textContent = PIECES[piece];
      sq.appendChild(span);
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
  const palette = ["#6ea8fe", "#8b7bff", "#57d28a", "#ffcc66", "#ff9f6b", "#ff6b81"];
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
    if (match.length) { playUserMove(pickPromotion(match)); return; }
  }
  if (legal[name]) { state.selected = name; showSelection(name); }
  else { state.selected = null; clearHints(); }
}

function pickPromotion(matches) {
  if (matches.length === 1) return matches[0];
  const choice = (prompt("Promote to: q, r, b, n", "q") || "q").toLowerCase();
  return matches.find((m) => m.endsWith(choice)) || matches.find((m) => m.endsWith("q")) || matches[0];
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
  const ghost = document.createElement("div");
  ghost.className = "drag-ghost";
  ghost.textContent = pieceEl.textContent;
  ghost.style.color = getComputedStyle(pieceEl).color;
  ghost.style.textShadow = getComputedStyle(pieceEl).textShadow;
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
    if (matches.length) { playUserMove(pickPromotion(matches)); return; }
  }
  render();
}

/* ---------------- game flow ---------------- */
function pushMove(result, uci, isEngine) {
  // Truncate any future when branching from a past position.
  if (!atLatest()) state.moves.length = state.cursor + 1;
  state.moves.push({
    fen: result.fen,
    san: result.san,
    uci: uci,
    color: result.turn === "white" ? "black" : "white",
    engine: !!isEngine,
    cpWhite: null,
  });
  state.cursor = state.moves.length - 1;
}

async function playUserMove(uci) {
  state.selected = null;
  clearHints();
  try {
    const result = await api("/api/move", { fen: currentFen(), uci });
    pushMove(result, uci, false);
    state.board = result;
    render();
    if (result.is_over) { showOverlay(); return; }
    if (state.mode === "play") await maybeEngineMove();
    else scheduleAnalysis();
  } catch (err) { toast(err.message); }
}

async function maybeEngineMove() {
  const engineShouldPlay = state.playAs !== "both" && state.board.turn !== state.playAs;
  if (!engineShouldPlay) { scheduleAnalysis(); return; }
  await engineMove();
}

async function engineMove() {
  if (!state.board || state.board.is_over) return;
  state.thinking = true;
  setBusy(true);
  try {
    const result = await api("/api/engine", {
      fen: currentFen(),
      simulations: state.simulations,
      temperature: state.temperature,
      multipv: state.multipv,
    });
    pushMove(result, result.engine_move, true);
    state.board = result;
    state.analysis = { lines: result.analysis, depth: result.depth, nodes: result.nodes, nps: result.nps, time_ms: result.time_ms, turn: state.moves[state.cursor].color };
    recordEval(result.value, state.board.turn);
    renderAnalysis();
    render();
    if (result.is_over) showOverlay();
  } catch (err) { toast(err.message); }
  finally { state.thinking = false; setBusy(false); }
}

/* ---------------- analysis ---------------- */
let analyzeToken = 0;
let analyzeTimer = null;

function scheduleAnalysis(force) {
  if (!state.analyzeOn && !force) { renderAnalysis(); return; }
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(() => runAnalysis(), 120);
}

async function runAnalysis() {
  if (!state.board || state.board.is_over) { state.analysis = null; renderAnalysis(); return; }
  const token = ++analyzeToken;
  const fen = currentFen();
  el("engineStats").classList.add("busy");
  try {
    const data = await api("/api/analyze", {
      fen,
      simulations: state.simulations,
      multipv: state.multipv,
    });
    if (token !== analyzeToken) return; // stale
    state.analysis = data;
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
  const cpWhite = valueToCpWhite(value, turn);
  setEvalBar(cpWhite, bestMateWhite());
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
  el("evalBarFill").style.height = (prob * 100).toFixed(1) + "%";
  const label = fmtScore(cpWhite, mateWhite);
  // Show the score on whichever side is ahead.
  if (prob >= 0.5) {
    el("evalBarLabelBottom").textContent = label;
    el("evalBarLabelTop").textContent = "";
  } else {
    el("evalBarLabelTop").textContent = label;
    el("evalBarLabelBottom").textContent = "";
  }
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
  span.className = "move-san" + (entry.engine ? " engine" : "") + (idx === state.cursor ? " current" : "");
  span.textContent = entry.san;
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
    for (let i = 0; i < bMissing; i++) capturedBlack.push(PIECES[t]);
    for (let i = 0; i < wMissing; i++) capturedWhite.push(PIECES[t.toUpperCase()]);
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
  el("openingName").textContent = state.board && state.board.opening ? state.board.opening : "";
}

/* ---------------- navigation ---------------- */
async function goTo(index) {
  index = Math.max(-1, Math.min(index, state.moves.length - 1));
  if (index === state.cursor && state.board) return;
  state.cursor = index;
  state.selected = null;
  hideOverlay();
  try {
    state.board = await api("/api/state", { fen: currentFen() });
  } catch (err) { toast(err.message); return; }
  if (state.board.is_over && atLatest()) showOverlay();
  render();
  scheduleAnalysis();
}

function updateNavButtons() {
  el("navStart").disabled = state.cursor < 0;
  el("navPrev").disabled = state.cursor < 0;
  el("navNext").disabled = atLatest();
  el("navEnd").disabled = atLatest();
  el("undoBtn").disabled = state.moves.length === 0 || state.thinking;
}

/* ---------------- game lifecycle ---------------- */
async function newGame() {
  hideOverlay();
  state.moves = [];
  state.cursor = -1;
  state.selected = null;
  state.thinking = false;
  state.analysis = null;
  state.startFen = START_FEN;
  state.orientation = (state.mode === "play" && state.playAs === "black") ? "black" : "white";
  try {
    state.board = await api("/api/state", { fen: "startpos" });
  } catch (err) { toast(err.message); return; }
  buildBoard();
  render();
  setEvalBar(0, null);
  drawGraph();
  if (state.mode === "play" && state.playAs === "black") engineMove();
  else scheduleAnalysis();
}

async function undo() {
  if (state.thinking || state.moves.length === 0) return;
  const toPop = (state.mode === "play" && state.playAs !== "both") ? Math.min(2, state.moves.length) : 1;
  state.moves.length = Math.max(0, state.moves.length - toPop);
  state.cursor = state.moves.length - 1;
  await goTo(state.cursor);
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
function buildPgn() {
  let out = "";
  for (let i = 0; i < state.moves.length; i += 2) {
    out += (i / 2 + 1) + ". " + state.moves[i].san + " ";
    if (i + 1 < state.moves.length) out += state.moves[i + 1].san + " ";
  }
  return out.trim();
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

async function applyImport() {
  const text = el("importText").value.trim();
  if (!text) { toast("nothing to load"); return; }
  try {
    const data = await api("/api/import", { text });
    el("importModal").hidden = true;
    state.moves = (data.history || []).map((h) => ({
      fen: h.fen, san: h.san, uci: h.uci, color: h.color, engine: false, cpWhite: null,
    }));
    state.startFen = state.moves.length ? START_FEN : data.fen;
    state.cursor = state.moves.length - 1;
    state.selected = null;
    state.orientation = "white";
    buildBoard();
    state.board = await api("/api/state", { fen: currentFen() });
    render();
    setEvalBar(0, null);
    drawGraph();
    scheduleAnalysis();
    toast(`Loaded ${data.import_kind === "pgn" ? "PGN" : "FEN"}`);
  } catch (err) { toast(err.message); }
}

/* ---------------- mode switching ---------------- */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#modeSwitch button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  el("playAsGroup").style.display = mode === "play" ? "" : "none";
  if (mode === "analysis") { state.analyzeOn = true; el("analyzeToggle").checked = true; }
  scheduleAnalysis();
}

/* ---------------- controls wiring ---------------- */
function wireControls() {
  el("newBtn").addEventListener("click", newGame);
  el("overlayNew").addEventListener("click", newGame);
  el("hintBtn").addEventListener("click", hint);
  el("undoBtn").addEventListener("click", undo);

  el("flipBtn").addEventListener("click", () => {
    state.orientation = state.orientation === "white" ? "black" : "white";
    buildBoard();
    render();
  });

  el("navStart").addEventListener("click", () => goTo(-1));
  el("navPrev").addEventListener("click", () => goTo(state.cursor - 1));
  el("navNext").addEventListener("click", () => goTo(state.cursor + 1));
  el("navEnd").addEventListener("click", () => goTo(state.moves.length - 1));

  document.querySelectorAll("#modeSwitch button").forEach((btn) =>
    btn.addEventListener("click", () => setMode(btn.dataset.mode)));

  document.querySelectorAll("#sideSelect button").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll("#sideSelect button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.playAs = btn.dataset.side;
      newGame();
    }));

  document.querySelectorAll("#themeRow button").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll("#themeRow button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (btn.dataset.theme === "blue") document.body.removeAttribute("data-theme");
      else document.body.setAttribute("data-theme", btn.dataset.theme);
    }));

  el("analyzeToggle").addEventListener("change", (e) => {
    state.analyzeOn = e.target.checked;
    if (state.analyzeOn) scheduleAnalysis();
    else { analyzeToken++; renderAnalysis(); }
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
  const mpvSlider = el("mpvSlider");
  mpvSlider.addEventListener("input", () => {
    state.multipv = parseInt(mpvSlider.value, 10);
    el("mpvValue").textContent = `${state.multipv} line${state.multipv > 1 ? "s" : ""}`;
  });
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
  el("importCancel").addEventListener("click", () => { el("importModal").hidden = true; });
  el("importApply").addEventListener("click", applyImport);
  el("copyFenBtn").addEventListener("click", () => copyText(currentFen(), "FEN"));
  el("copyPgnBtn").addEventListener("click", () => copyText(buildPgn() || "*", "PGN"));
  el("analyzeNowBtn").addEventListener("click", () => scheduleAnalysis(true));

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") { goTo(state.cursor - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { goTo(state.cursor + 1); e.preventDefault(); }
    else if (e.key === "Home") { goTo(-1); e.preventDefault(); }
    else if (e.key === "End") { goTo(state.moves.length - 1); e.preventDefault(); }
    else if (e.key.toLowerCase() === "f") {
      state.orientation = state.orientation === "white" ? "black" : "white";
      buildBoard(); render();
    }
  });
}

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    const dev = info.device === "auto" ? "auto-device" : info.device;
    el("engineInfo").textContent = `${info.checkpoint.split("/").pop()} · ${dev}`;
  } catch (_) {
    el("engineInfo").textContent = "engine offline";
  }
}

async function main() {
  wireControls();
  loadInfo();
  await newGame();
}

main();
