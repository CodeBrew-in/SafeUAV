'use strict';
/* SafeUAV console. All text from the server (place names, AI replies, log lines) is inserted with
   textContent / canvas fillText only, never innerHTML. */

const TOKEN = document.querySelector('meta[name=token]').content;
const $ = id => document.getElementById(id);
const CRUISE_MPS = 10, WH_PER_M = 0.005;

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e[k] = v;
    else if (v === true) e.setAttribute(k, ''); else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

// one drone per browser tab: this id lives in sessionStorage, so a new tab gets its own drone
function rand32() { return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join(''); }
let DRONE = '';
try { DRONE = sessionStorage.getItem('drone') || ''; if (!/^[0-9a-f]{32}$/.test(DRONE)) { DRONE = rand32(); sessionStorage.setItem('drone', DRONE); } } catch (e) { DRONE = DRONE || rand32(); }

async function api(path, body) {
  const opt = { headers: { 'X-Session-Token': TOKEN, 'X-Drone-Id': DRONE } };
  if (body !== undefined) { opt.method = 'POST'; opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 8000); opt.signal = ctl.signal;
  let r;
  try { r = await fetch(path, opt); } catch (e) { throw new Error(e.name === 'AbortError' ? 'The server did not answer in time.' : 'Cannot reach the server.'); } finally { clearTimeout(to); }
  let js = null; try { js = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error(r.status === 403 ? 'Session expired: reload this page.' : ((js && js.error) || ('HTTP ' + r.status)));
  return js;
}

// ---------------------------------------------------------------- state
let MAP = null, S = null;
let chat = [], audit = [], trail = [], route = [], zones = [];
let drawMode = false, circlePrev = null, circleSig = '', ghost = null, rejects = 0, violations = 0, firstPoll = true;
let since = { epoch: -1, chat: 0, audit: 0, track: 0, route_rev: -1, zones_sig: null };
let lastOk = 0, busySince = 0, initialChat = true, logShown = -1;
let userMoved = false, view = { cx: 0, cy: 0, scale: 1 }, follow = false, dirtyBase = true, hover = null, selected = null;
const disp = { x: 0, y: 0, h: 0, init: false };
let BASE = null, drawOrder = [], places = [], placeByName = {}, hatchNoFly = null, hatchAvoid = null, baseCanvas = null, baseKey = '';

// ---------------------------------------------------------------- geometry
function pip(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function distSeg(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
  let t = l2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}
function distPoly(p, poly) {
  if (pip(p, poly)) return 0;
  let d = Infinity;
  for (let i = 0; i < poly.length; i++) d = Math.min(d, distSeg(p, poly[i], poly[(i + 1) % poly.length]));
  return d;
}
function polyArea(poly) { let a = 0; for (let i = 0; i < poly.length; i++) { const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length]; a += x1 * y2 - x2 * y1; } return Math.abs(a) / 2; }
function remainLen(rt, idx, pos) {
  if (!rt || rt.length < 2 || !pos) return 0;
  let d = 0, prev = pos;
  for (let i = Math.max(1, idx); i < rt.length; i++) { d += Math.hypot(rt[i][0] - prev[0], rt[i][1] - prev[1]); prev = rt[i]; }
  return d;
}
const mmss = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

// ---------------------------------------------------------------- view transform
const cv = $('map'), ctx = cv.getContext('2d');
let W = 1, H = 1, DPR = 1;
const P = (x, y) => [W / 2 + (x - view.cx) * view.scale, H / 2 - (y - view.cy) * view.scale];
const toWorld = (px, py) => [view.cx + (px - W / 2) / view.scale, view.cy - (py - H / 2) / view.scale];

function fitView() {
  const [x0, y0, x1, y1] = BASE.focus;      // look at the Thapar campus, not the whole download area
  view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
  view.scale = Math.min(W / (x1 - x0), H / (y1 - y0)) * 0.97;
  clampView(); dirtyBase = true;
}
function clampView() {
  const [bx0, by0, bx1, by1] = BASE.bounds;  // the map never shows the empty area outside the data
  const cover = Math.max(W / (bx1 - bx0), H / (by1 - by0));
  view.scale = Math.max(cover, Math.min(7, view.scale));
  const hw = W / (2 * view.scale), hh = H / (2 * view.scale);
  view.cx = bx1 - bx0 <= 2 * hw ? (bx0 + bx1) / 2 : Math.max(bx0 + hw, Math.min(bx1 - hw, view.cx));
  view.cy = by1 - by0 <= 2 * hh ? (by0 + by1) / 2 : Math.max(by0 + hh, Math.min(by1 - hh, view.cy));
}
function zoomAt(px, py, f) {
  userMoved = true;
  const [wx, wy] = toWorld(px, py);
  view.scale *= f; clampView();
  view.cx = wx - (px - W / 2) / view.scale; view.cy = wy + (py - H / 2) / view.scale;
  clampView(); dirtyBase = true;
}
function resize() {
  const r = cv.getBoundingClientRect(); DPR = window.devicePixelRatio || 1;
  W = Math.max(1, r.width); H = Math.max(1, r.height);
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  if (MAP) { if (!userMoved && W > 50 && H > 50) fitView(); dirtyBase = true; }
}
new ResizeObserver(resize).observe(cv);

// input: drag to pan, wheel to zoom, click to select
let drag = null;
cv.addEventListener('pointerdown', e => {
  if (e.button !== 0) return; cv.setPointerCapture(e.pointerId);
  if (drawMode) { const r = cv.getBoundingClientRect(); circlePrev = { c: toWorld(e.clientX - r.left, e.clientY - r.top), r: 0 }; return; }
  drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false }; });
cv.addEventListener('pointermove', e => {
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (circlePrev) { const w = toWorld(px, py); circlePrev.r = Math.hypot(w[0] - circlePrev.c[0], w[1] - circlePrev.c[1]); return; }
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; userMoved = true; cv.classList.add('dragging'); setFollow(false); hideCard(); }
    if (drag.moved) { view.cx = drag.cx - dx / view.scale; view.cy = drag.cy + dy / view.scale; clampView(); dirtyBase = true; }
  } else {
    const hv = placeAt(toWorld(px, py));
    if (hv !== hover) { hover = hv; dirtyBase = true; cv.title = hv ? hv.name : ''; }
  }
});
const endDrag = () => { drag = null; circlePrev = null; cv.classList.remove('dragging'); };
cv.addEventListener('pointercancel', endDrag);
cv.addEventListener('lostpointercapture', endDrag);
cv.addEventListener('pointerup', async e => {
  if (circlePrev) {
    const c = circlePrev; circlePrev = null; setDraw(false);
    if (c.r < 20) return toast('AREA', 'Drag further out: the radius must be 20 to 300 m.', 'warn');
    try {
      const res = await api('/api/circle', { x: c.c[0], y: c.c[1], r: Math.min(300, c.r) }); await poll();
      const pl = placeByName[res.name]; if (pl) { const q = P(pl.center[0], pl.center[1]); onMapClick(q[0], q[1], pl); }
    } catch (err) { showReject(err.message); }
    return;
  }
  if (e.button !== 0) return endDrag();
  cv.classList.remove('dragging');
  const wasClick = drag && !drag.moved; drag = null;
  if (wasClick) { const r = cv.getBoundingClientRect(); onMapClick(e.clientX - r.left, e.clientY - r.top); }
});
cv.addEventListener('wheel', e => { e.preventDefault(); if (!MAP) return; const r = cv.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015))); }, { passive: false });
cv.addEventListener('dblclick', e => { if (!MAP) return; const r = cv.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, 1.8); });
$('zin').onclick = () => zoomAt(W / 2, H / 2, 1.4);
$('zout').onclick = () => zoomAt(W / 2, H / 2, 1 / 1.4);
$('fit').onclick = () => { setFollow(false); userMoved = false; fitView(); };
$('follow').onclick = () => setFollow(!follow);
function setDraw(v) { drawMode = v; circlePrev = null; $('draw').setAttribute('aria-pressed', String(v)); cv.style.cursor = v ? 'crosshair' : ''; if (v) toast('DRAW AN AREA', 'Press on the map and drag outward to set the radius (20 to 300 m).'); }
$('draw').onclick = () => setDraw(!drawMode);
function setFollow(v) { follow = v; $('follow').setAttribute('aria-pressed', String(v)); }
document.addEventListener('click', e => { const b = e.target.closest && e.target.closest('#controls button, #mapctl button'); if (b) b.blur(); });  // so Space keeps pausing
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { hideCard(); const d = $('dlg'); if (!d.classList.contains('hidden')) d.querySelectorAll('button')[1].click(); }
  const a = document.activeElement;
  if (e.key === ' ' && S && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && (a === document.body || a === cv)) { e.preventDefault(); $('pause').click(); }
});

// ---------------------------------------------------------------- map data
const KIND_FILL = { pitch: '#d3ebc9', park: '#c3e4b6', residential: '#ead9c0', university: '#dce4f3', college: '#dce4f3',
  school: '#dce4f3', commercial: '#f4dcc6', fountain: '#cfe3f0', service: '#e6e1d6', yes: '#e6e1d6' };
function hatchPattern(color) {
  const c = document.createElement('canvas'); c.width = c.height = 10; const g = c.getContext('2d');
  g.fillStyle = '#ffffff99'; g.fillRect(0, 0, 10, 10); g.strokeStyle = color; g.lineWidth = 2;
  g.beginPath(); g.moveTo(-2, 12); g.lineTo(12, -2); g.moveTo(-2, 2); g.lineTo(2, -2); g.moveTo(8, 12); g.lineTo(12, 8); g.stroke();
  return ctx.createPattern(c, 'repeat');
}
function loadMap(m, basemap) {
  MAP = m; BASE = basemap;
  places = m.places.map(p => ({ ...p, area: polyArea(p.poly), region: polyArea(p.poly) > 150000 })).sort((a, b) => a.area - b.area);
  drawOrder = places.filter(p => !p.region).slice().reverse();   // big shapes first, small ones on top
  placeByName = Object.fromEntries(places.map(p => [p.name, p]));
  hatchNoFly = hatchPattern('#d92d20'); hatchAvoid = hatchPattern('#ee7a00');
  $('ver').textContent = 'v' + m.version;
  if (m.public) $('pubnote').classList.remove('hidden');
  resize(); fitView();
}
function syncCircles(list) {
  const sig = JSON.stringify(list || []);
  if (sig === circleSig) return; circleSig = sig;
  places = places.filter(p => p.kind !== 'area');
  for (const c of list || []) places.push({ name: c.name, kind: 'area', poly: c.poly, center: c.center, area: Math.PI * c.r * c.r, region: false });
  placeByName = Object.fromEntries(places.map(p => [p.name, p])); dirtyBase = true;
}
function drawCircles() {
  ctx.save(); ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
  const ring = (c, r, col) => { const q = P(c[0], c[1]); ctx.beginPath(); ctx.arc(q[0], q[1], r * view.scale, 0, 7); ctx.fillStyle = col + '1c'; ctx.fill(); ctx.strokeStyle = col; ctx.stroke(); };
  for (const c of (S.circles || [])) ring(c.center, c.r, '#6b7280');
  if (circlePrev && circlePrev.r > 0) { ring(circlePrev.c, Math.min(300, circlePrev.r), '#ee7a00'); const q = P(circlePrev.c[0], circlePrev.c[1]); ctx.setLineDash([]); ctx.fillStyle = '#14213d'; ctx.font = '700 13px system-ui'; ctx.textAlign = 'center'; ctx.fillText(Math.round(Math.min(300, circlePrev.r)) + ' m', q[0], q[1]); }
  ctx.restore();
}
function placeAt(w) {
  for (const z of MAP.no_fly) if (pip(w, z.poly)) return { name: z.name, kind: 'no-fly', nofly: true };
  for (const [n, c] of Object.entries(MAP.pads)) if (Math.hypot(w[0] - c[0], w[1] - c[1]) < 14 / view.scale) return placeByName[n] || { name: n, kind: 'pad' };
  for (const p of places) if (!p.region && pip(w, p.poly)) return p;
  // thin or ragged footprints are easy to miss: accept a click within ~10 px of a named building
  const tol = 10 / view.scale; let best = null, bd = tol;
  for (const p of places) {
    if (p.region) continue;
    if (!p.bb) { const xs = p.poly.map(q => q[0]), ys = p.poly.map(q => q[1]); p.bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; }
    if (w[0] < p.bb[0] - tol || w[0] > p.bb[2] + tol || w[1] < p.bb[1] - tol || w[1] > p.bb[3] + tol) continue;
    const d = distPoly(w, p.poly); if (d < bd) { bd = d; best = p; }
  }
  return best;
}

// ---------------------------------------------------------------- drawing: static layer (cached)
function pathPoly(g, poly) { poly.forEach((p, i) => { const q = P(p[0], p[1]); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.closePath(); }

function drawBase() {
  const key = [view.cx.toFixed(1), view.cy.toFixed(1), view.scale.toFixed(4), W, H, S ? S.zones_sig : '', S ? S.config.base_pad : '',
    hover ? hover.name : '', S ? JSON.stringify(S.desired.targets) + S.desired.avoid.map(a => a.zone).join() : ''].join('|');
  if (!dirtyBase && key === baseKey && baseCanvas) return;
  baseKey = key; dirtyBase = false;
  if (!baseCanvas) baseCanvas = document.createElement('canvas');
  baseCanvas.width = Math.round(W * DPR); baseCanvas.height = Math.round(H * DPR);
  const g = baseCanvas.getContext('2d'); g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.lineJoin = g.lineCap = 'round';
  const poly = (pts, fill, stroke, lw) => {
    g.beginPath(); pathPoly(g, pts);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1; g.stroke(); }
  };
  const line = (pts, color, w, dash) => {
    g.beginPath(); pts.forEach((p, i) => { const q = P(p[0], p[1]); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
    g.setLineDash(dash || []); g.strokeStyle = color; g.lineWidth = w; g.stroke(); g.setLineDash([]);
  };
  const sc = view.scale;
  // land, then areas from the ground up
  g.fillStyle = '#f2efe9'; g.fillRect(0, 0, W, H);
  for (const a of BASE.landuse) poly(a, '#ebe6dc');
  for (const a of BASE.green) poly(a, '#cde7bd');
  for (const a of BASE.water) poly(a, '#a9d3e0', '#8cbccb', 1);
  for (const w of BASE.waterways) line(w, '#8cbccb', Math.max(1.5, 3 * sc));
  // faint 100 m grid when zoomed in
  const [bx0, by0, bx1, by1] = BASE.bounds;
  if (sc * 100 > 30) {
    g.strokeStyle = '#0000000d'; g.lineWidth = 1; g.beginPath();
    for (let x = Math.ceil(bx0 / 100) * 100; x <= bx1; x += 100) { const a = P(x, by0), b = P(x, by1); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); }
    for (let y = Math.ceil(by0 / 100) * 100; y <= by1; y += 100) { const a = P(bx0, y), b = P(bx1, y); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); }
    g.stroke();
  }
  // campus boundary and walls
  if (BASE.campus.length) poly(BASE.campus, 'rgba(255,255,255,.35)', '#8a7fb0', 2);
  for (const w of BASE.walls) line(w, '#a39d90', Math.max(1, 1.2 * sc), null);
  // roads: casing first so junctions merge, then the surface
  const width = { major: [10, 3.2], minor: [6, 2.2], service: [3.5, 1.5] };
  for (const pass of [0, 1]) for (const cls of ['service', 'minor', 'major']) {
    const [wm, minpx] = width[cls];
    const lw = Math.max(minpx, wm * sc);
    for (const r of BASE.roads) if (r.c === cls) line(r.pts, pass ? (cls === 'major' ? '#ffe9b3' : '#ffffff') : '#c8c1b2', pass ? lw : lw + 1.8);
  }
  for (const r of BASE.paths) line(r.pts, '#c9a98f', Math.max(1, 1.6 * sc), [4, 3]);
  // buildings (all of them), then the named places in colour on top
  for (const b of BASE.buildings) poly(b, '#e1d8cb', '#b9ad9a', 0.8);
  g.lineJoin = 'round';
  for (const p of drawOrder) {
    const hov = hover && hover.name === p.name;
    poly(p.poly, KIND_FILL[p.kind] || '#e1d8cb', hov ? '#1d5fd1' : '#9a9486', hov ? 2.5 : 1);
  }
  // zones: wide translucent stroke = the safety buffer, then hatched fill on top of the inside half
  const drawZone = (pts, buf, color, pat) => {
    g.beginPath(); pathPoly(g, pts); g.strokeStyle = color + '30'; g.lineWidth = Math.max(3, 2 * buf * sc); g.stroke();
    g.beginPath(); pathPoly(g, pts); g.fillStyle = '#fff'; g.fill(); g.fillStyle = pat; g.fill();
    g.strokeStyle = color; g.lineWidth = 2.5; g.stroke();
  };
  for (const z of MAP.no_fly) drawZone(z.poly, z.buffer, '#d92d20', hatchNoFly);
  for (const z of zones) drawZone(z.poly, z.buffer, '#ee7a00', hatchAvoid);
  // flight area edge (thin; nothing is dimmed or hidden outside it)
  const [x0, y0, x1, y1] = MAP.fence, fa = P(x0, y1), fb = P(x1, y0);
  g.strokeStyle = '#5b6b8a99'; g.lineWidth = 1.5; g.setLineDash([10, 7]); g.strokeRect(fa[0], fa[1], fb[0] - fa[0], fb[1] - fa[1]); g.setLineDash([]);
  // pads
  const basePad = S ? S.config.base_pad : MAP.base_default;
  for (const [n, c] of Object.entries(MAP.pads)) {
    const q = P(c[0], c[1]), isBase = n === basePad, sz = isBase ? 20 : 15;
    g.fillStyle = '#15a05a'; g.strokeStyle = '#fff'; g.lineWidth = 2.5;
    g.beginPath(); g.roundRect(q[0] - sz / 2, q[1] - sz / 2, sz, sz, 4); g.fill(); g.stroke();
    g.fillStyle = '#fff'; g.font = '700 ' + (isBase ? 13 : 10) + 'px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(isBase ? 'H' : 'P', q[0], q[1] + 0.5);
    if (isBase) { g.strokeStyle = '#15a05a'; g.lineWidth = 2; g.beginPath(); g.arc(q[0], q[1], 16, 0, 7); g.stroke(); }
  }
  drawLabels(g, basePad);
}

function drawLabels(g, basePad) {
  const placed = [], want = [];
  const tgt = new Set(S ? S.desired.targets.map(t => t) : []), av = new Set(S ? S.desired.avoid.map(a => a.zone) : []);
  for (const p of places) {
    if (p.region) continue;
    const prio = (p.name === basePad || MAP.pads[p.name] || tgt.has(p.name) || av.has(p.name)) ? 100 : 0;
    want.push({ name: p.name, c: p.center, prio, size: Math.sqrt(p.area), hover: hover && hover.name === p.name });
  }
  for (const z of MAP.no_fly) want.push({ name: 'NO-FLY ZONE', c: z.poly.reduce((a, q) => [a[0] + q[0] / z.poly.length, a[1] + q[1] / z.poly.length], [0, 0]), prio: 200, size: 1e4, color: '#b42318' });
  want.sort((a, b) => (b.prio + b.hover * 300) - (a.prio + a.hover * 300) || b.size - a.size);
  g.textAlign = 'center'; g.textBaseline = 'middle';
  for (const w of want) {
    const fs = 12.5; g.font = `600 ${fs}px system-ui`;
    let txt = w.name; if (view.scale < 0.45 && txt.length > 16) txt = txt.slice(0, 15) + '…';
    const tw = g.measureText(txt).width;
    if (!w.prio && !w.hover && w.size * view.scale < tw * 0.55) continue;
    const q = P(w.c[0], w.c[1]); const r = [q[0] - tw / 2 - 2, q[1] - fs / 2 - 2, tw + 4, fs + 4];
    if (r[0] < 0 || r[1] < 0 || r[0] + r[2] > W || r[1] + r[3] > H) continue;
    if (!w.hover && placed.some(o => r[0] < o[0] + o[2] && r[0] + r[2] > o[0] && r[1] < o[1] + o[3] && r[1] + r[3] > o[1])) continue;
    placed.push(r);
    g.lineWidth = 3.5; g.strokeStyle = 'rgba(255,255,255,.92)'; g.strokeText(txt, q[0], q[1]);
    g.fillStyle = w.color || '#14213d'; g.fillText(txt, q[0], q[1]);
  }
}

// ---------------------------------------------------------------- drawing: dynamic layer
const STATE_COLOR = { CRUISE: '#1d5fd1', ESCAPE: '#ee7a00', TO_PAD: '#ee7a00', LANDED_WAIT: '#15a05a', EMERGENCY: '#d92d20', DONE: '#15a05a', PREFLIGHT: '#667085', CRASHED: '#d92d20', GROUNDED_NEEDS_PICKUP: '#d92d20' };

function strokeLine(pts, color, w, dash, alpha) {
  if (!pts || pts.length < 2) return;
  ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha; ctx.lineJoin = ctx.lineCap = 'round';
  ctx.beginPath(); pts.forEach((p, i) => { const q = P(p[0], p[1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
  ctx.setLineDash([]); ctx.strokeStyle = '#fff'; ctx.lineWidth = w + 3; ctx.stroke();
  ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.stroke(); ctx.restore();
}

function pin(x, y, label, color, done) {
  const q = P(x, y);
  ctx.save(); ctx.shadowColor = '#0006'; ctx.shadowBlur = 6; ctx.shadowOffsetY = 2;
  ctx.fillStyle = done ? '#15803d' : color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(q[0], q[1] - 14, 13, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
  ctx.fillStyle = '#fff'; ctx.font = '700 14px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(done ? '✓' : label, q[0], q[1] - 13.5);
}

function drawDrone(now) {
  const q = P(disp.x, disp.y), st = S ? S.state : 'PREFLIGHT', col = STATE_COLOR[st] || '#667085';
  const moving = S && ['CRUISE', 'ESCAPE', 'TO_PAD', 'EMERGENCY'].includes(st) && !S.paused;
  ctx.save(); ctx.translate(q[0], q[1]);
  ctx.fillStyle = 'rgba(0,0,0,.22)'; ctx.beginPath(); ctx.ellipse(5, 9, 15, 7, 0, 0, 7); ctx.fill();
  const pulse = moving ? 3 * Math.sin(now / 250) : 0;
  ctx.fillStyle = col + '33'; ctx.strokeStyle = col + 'aa'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(0, 0, 24 + pulse, 0, 7); ctx.fill(); ctx.stroke();
  ctx.rotate(-disp.h);
  ctx.beginPath(); ctx.moveTo(17, 0); ctx.lineTo(-12, 12); ctx.lineTo(-6, 0); ctx.lineTo(-12, -12); ctx.closePath();
  ctx.fillStyle = '#0f1d3a'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.stroke(); ctx.fill();
  ctx.fillStyle = col; ctx.beginPath(); ctx.arc(0, 0, 3.5, 0, 7); ctx.fill();
  ctx.restore();
}

function render(now) {
  if (!MAP) return;
  drawBase();
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, W, H);
  if (baseCanvas) ctx.drawImage(baseCanvas, 0, 0, W, H);
  if (S) {
    const live = S.config.mode !== 'direct';
    drawCircles();
    if (ghost) { const age = (now - ghost.t0) / 2500; if (age >= 1) ghost = null; else strokeLine(ghost.route, '#7a869a', 3, [4, 7], 0.8 * (1 - age)); }
    if (S.direct && S.direct.route) strokeLine(S.direct.route, '#d92d20', 3, [10, 6]);
    if (live && route.length > 1) {
      const rest = [[disp.x, disp.y], ...route.slice(Math.max(1, S.route_index))];
      strokeLine(S.phase === 'DRAFT' ? route : rest, '#1d5fd1', 3.5, [10, 7]);
    }
    if (live && trail.length > 1) {
      const n = trail.length, split = Math.max(0, n - 60);
      strokeLine(trail.slice(0, split + 1).map(t => t), '#101828', 3, null, 0.4);
      strokeLine(trail.slice(split), '#101828', 3.5, null, 0.95);
    }
    if (live) S.targets.forEach((t, i) => pin(t.center[0], t.center[1], String(i + 1), '#1d5fd1', t.done));
    if (disp.init && live) drawDrone(now);
  }
  // north arrow tracks nothing (map is north-up); scale bar:
  const meters = [10, 20, 50, 100, 200, 500, 1000].find(m => m * view.scale >= 70) || 1000;
  $('scalebar-line').style.width = Math.round(meters * view.scale) + 'px'; $('scalebar-txt').textContent = meters + ' m';
}

// ---------------------------------------------------------------- map click card
function hideCard() { $('card').classList.add('hidden'); selected = null; }
function onMapClick(px, py, forced) {
  const wp = toWorld(px, py);
  const pinHit = S && S.targets.find(t => Math.hypot(wp[0] - t.center[0], wp[1] - t.center[1]) < 14 / view.scale);
  const p = forced || (pinHit && placeByName[pinHit.name]) || placeAt(wp);
  if (!p) return hideCard();
  selected = p;
  const card = $('card'); card.replaceChildren();
  const kindText = k => (!k || k === 'yes') ? 'Building' : k;
  card.append(h('h4', {}, p.name), h('small', {}, p.nofly ? 'Permanent no-fly zone. It cannot be changed.' : (MAP.pads[p.name] ? 'Landing pad · ' + kindText(p.kind) : kindText(p.kind))));
  if (!p.nofly && S) {
    const isT = S.desired.targets.includes(p.name), isA = S.desired.avoid.some(a => a.zone === p.name), ended = ['GROUNDED', 'CRASHED'].includes(S.phase);
    const isBase = S.config.base_pad === p.name, row = h('div', { class: 'row' });
    row.append(h('button', { class: 'primary small', disabled: ended, onclick: () => edit(isT ? 'target_remove' : 'target_add', p.name) }, isT ? 'Remove target' : 'Make target'));
    row.append(h('button', { class: 'warn small', disabled: ended || (!isA && (isBase || isT)), title: (!isA && (isBase || isT)) ? 'The base and the targets cannot be restricted' : '', onclick: () => edit(isA ? 'avoid_remove' : 'avoid_add', p.name) }, isA ? 'Allow again' : 'Restrict'));
    if (MAP.pads[p.name]) row.append(h('button', { class: 'small', disabled: S.phase !== 'DRAFT', title: S.phase !== 'DRAFT' ? 'The base can only change before Start' : '', onclick: () => config({ base_pad: p.name }) }, 'Set as base'));
    if (p.kind === 'area') row.append(h('button', { class: 'small', title: 'Delete this drawn area', onclick: () => edit('circle_remove', p.name) }, 'Delete area'));
    if (MAP.pads[p.name]) row.append(h('button', { class: 'warn small', disabled: !['FLYING', 'PAUSED'].includes(S.phase), title: 'Fly here and land (checked by the shield)', onclick: () => { hideCard(); post('/api/emergency', { kind: 'land', pad: p.name }); } }, 'Land here'));
    card.append(row);
  }
  card.classList.remove('hidden');
  card.style.left = Math.min(W - 290, Math.max(8, px + 12)) + 'px'; card.style.top = Math.min(H - 190, Math.max(8, py + 12)) + 'px';
}
async function edit(op, place, value) { hideCard(); try { const r = await api('/api/edit', { op, place, value }); toast('EDIT', r.message, 'ok'); poll(); } catch (e) { missionSig = ''; showReject(e.message, true); poll(); } }
async function config(kv) {
  hideCard();
  for (const v of Object.values(kv)) if (typeof v === 'number' && !Number.isFinite(v)) { missionSig = ''; showReject('Enter a number.'); return poll(); }
  try { await api('/api/config', kv); poll(); } catch (e) { missionSig = ''; showReject(e.message); poll(); }
}
// in-page confirmation (native confirm() is blocked in embedded browsers)
function ask(text, okLabel) {
  return new Promise(res => {
    const dlg = $('dlg'), done = v => { dlg.classList.add('hidden'); res(v); };
    dlg.replaceChildren(h('div', { class: 'box', role: 'alertdialog' }, h('p', {}, text),
      h('div', { class: 'row' }, h('button', { class: 'primary', onclick: () => done(true) }, okLabel || 'OK'), h('button', { onclick: () => done(false) }, 'Cancel'))));
    dlg.classList.remove('hidden'); dlg.querySelector('button').focus();
  });
}

// ---------------------------------------------------------------- feedback widgets
function toast(title, text, kind) {
  const t = h('div', { class: 'toast ' + (kind || '') }, h('b', {}, title), text);
  $('toasts').append(t); while ($('toasts').children.length > 3) $('toasts').firstChild.remove();
  setTimeout(() => t.remove(), kind === 'bad' ? 6000 : 4500);
}
let rejectTimer = 0;
function showReject(msg, count) {
  msg = String(msg).replace(/^Not applied:\s*/, '');
  if (count) rejects++;
  const c = $('reject'); c.replaceChildren(h('button', { class: 'x', onclick: () => c.classList.add('hidden'), 'aria-label': 'Close' }, '×'),
    h('h4', {}, '⛔ Not applied'), h('p', {}, msg), count ? h('p', { class: 'hint' }, 'Your current plan was not changed.') : null);
  c.classList.remove('hidden'); clearTimeout(rejectTimer); rejectTimer = setTimeout(() => c.classList.add('hidden'), 12000);
}
function showReplan(text) {
  const c = $('replan'); c.textContent = text; c.classList.remove('hidden');
  c.classList.toggle('right', $('reject').classList.contains('hidden'));
  clearTimeout(c._t); c._t = setTimeout(() => c.classList.add('hidden'), 7000);
}
function auditToast(e, prevState) {
  const m = e.msg || '';
  if (e.kind === 'takeoff') return toast('TAKE-OFF', 'The drone is airborne.', 'ok');
  if (e.kind === 'replan') return /^replanning/.test(m) ? toast('REPLANNING', 'Checking the new route with the shield…') : null;
  if (e.kind === 'monitor') return toast('BATTERY CHECK', m, 'warn');
  if (e.kind === 'warning') return toast('NOTE', m, 'warn');
  if (e.kind === 'state') {
    const to = (m.split('->')[1] || '').trim().split(':')[0].trim(), why = (m.split(':').slice(1).join(':') || '').trim();
    const map = { ESCAPE: ['ESCAPE', 'warn'], TO_PAD: ['LANDING ON A PAD', 'warn'], LANDED_WAIT: ['ON THE GROUND', 'warn'], DONE: ['MISSION COMPLETE', 'ok'],
      EMERGENCY: ['EMERGENCY', 'bad'], GROUNDED_NEEDS_PICKUP: ['GROUNDED', 'bad'], CRASHED: ['CRASHED', 'bad'] };
    if (map[to]) return toast(map[to][0], why, map[to][1]);
    if (to === 'CRUISE' && /^(LANDED_WAIT|ESCAPE) ->/.test(m)) return toast('RESUMING', why, 'ok');
  }
}

// ---------------------------------------------------------------- panels
function specText(sp) {
  if (!sp) return '';
  const av = sp.avoid.map(a => a.zone + (a.from || a.to ? ` (${a.from || 'now'}–${a.to || '∞'})` : ''));
  return `Targets: ${sp.targets.join(' → ') || '—'}${av.length ? ' · Restricted: ' + av.join(', ') : ''} · ${sp.stay_at_end ? 'stays at the last place' : 'returns to base'}`;
}
function updateControls() {
  const ph = S.phase, flying = ph === 'FLYING' || ph === 'PAUSED', waiting = ph === 'WAITING', ended = ['GROUNDED', 'CRASHED'].includes(ph);
  $('phase').textContent = ph; $('phase').className = 'badge ' + ph;
  const noTarget = !S.desired.targets.length;
  $('start').disabled = ph !== 'DRAFT' || S.busy || noTarget;
  $('start').title = ph !== 'DRAFT' ? 'Already started' : S.busy ? 'The AI is still reading your message' : noTarget ? 'Add a target first' : 'Start the flight';
  $('pause').textContent = ph === 'PAUSED' ? 'Resume' : 'Pause'; $('pause').disabled = !flying;
  $('rtb').title = waiting ? 'The drone is already on the ground' : 'Fly back to base (works without the AI)';
  $('reset').disabled = false;
  $('rtb').disabled = !flying; $('land').disabled = !flying;
  $('endm').disabled = !flying && !waiting;
  [...$('speeds').children].forEach(b => b.classList.toggle('on', +b.dataset.s === S.speed));
  $('batt').style.width = Math.max(0, S.battery_pct) + '%'; $('batt').style.background = S.battery_pct < 20 ? '#b91c1c' : '#15a05a';
  $('battv').textContent = S.battery_wh.toFixed(1) + ' Wh';
  $('clk').textContent = S.clock + ' · t=' + Math.round(S.t) + ' s';
  const bn = $('banner-main'), bs = $('banner-sub'), cls = $('banner');
  const rest = remainLen(route, S.route_index, S.pos), eta = rest / CRUISE_MPS, nxt = S.targets.find(t => !t.done);
  let main = '', sub = '', kind = '';
  const lastWhy = (() => { for (let i = audit.length - 1; i >= 0; i--) if (audit[i].kind === 'state') return (audit[i].msg.split(':').slice(1).join(':') || '').trim(); return ''; })();
  if (ph === 'DRAFT') { main = S.busy ? 'Understanding your message… ' + Math.round((performance.now() - busySince) / 1000) + ' s' : 'Draft mission'; sub = S.busy ? 'Nothing flies until you press Start' : noTarget ? 'Click a building to add a target, or type an order in Chat' : 'Check the plan, then press Start'; }
  else if (ph === 'WAITING') { main = 'On the ground, waiting'; sub = S.busy ? 'Understanding your message…' : 'Send a new order (click or chat) to fly again. Idle battery use only.'; kind = 'warn'; }
  else if (ph === 'PAUSED') { main = 'Paused'; sub = 'Simulation clock paused (battery use is paused too)'; kind = 'warn'; }
  else if (ph === 'DONE') { main = 'Mission complete'; sub = 'Landed and waiting. Send a new order to fly again, or press Reset.'; kind = 'ok'; }
  else if (ph === 'GROUNDED' || ph === 'CRASHED') { main = ph === 'CRASHED' ? 'Crashed' : 'Grounded: pick-up needed'; sub = lastWhy; kind = 'bad'; }
  else switch (S.state) {
    case 'CRUISE': main = nxt ? 'Flying to ' + nxt.name : 'Returning to base'; sub = 'ETA ≈ ' + mmss(eta) + ' sim-time · ' + Math.round(rest) + ' m of route left'; break;
    case 'ESCAPE': main = 'Leaving a restricted zone'; sub = lastWhy || 'Taking the nearest exit'; kind = 'warn'; break;
    case 'TO_PAD': main = 'Landing on a pad to wait'; sub = lastWhy || 'The old route was dropped'; kind = 'warn'; break;
    case 'LANDED_WAIT': main = 'Waiting on the pad'; sub = 'Send a new order to resume. Idle battery use only.'; kind = 'warn'; break;
    case 'EMERGENCY': main = 'Emergency landing'; sub = lastWhy; kind = 'bad'; break;
    default: main = S.state; sub = lastWhy;
  }
  if (S.busy && ['FLYING', 'PAUSED'].includes(ph)) sub = (S.holding ? 'Holding while the AI reads your message · ' : 'AI reading your message · ') + sub;
  if (S.queued) sub = S.queued + ' change(s) queued' + (ph === 'PAUSED' ? ', applied on Resume · ' : ' · ') + sub;
  bn.textContent = main; bs.textContent = sub; cls.className = kind;
  $('shield').replaceChildren('Shield: ', h('b', { class: rejects ? 'warn' : '' }, String(rejects)), ' blocked · no-fly entries seen (estimate): ', h('b', { class: violations ? 'bad' : 'ok' }, String(violations)));
  $('directnote').classList.toggle('hidden', S.config.mode !== 'direct');
}

let missionSig = '';
function renderMission() {
  const sig = JSON.stringify([S.desired, S.visited_names, S.config, S.phase, S.busy]);
  if (sig === missionSig) return; missionSig = sig;
  const root = $('tab-mission'), c = S.config, locked = S.phase !== 'DRAFT', ended = ['GROUNDED', 'CRASHED'].includes(S.phase);
  const active = document.activeElement && root.contains(document.activeElement) ? document.activeElement.id : null;
  root.replaceChildren();
  root.append(h('h3', {}, 'Base'),
    h('select', { id: 'f-base', disabled: locked, onchange: e => config({ base_pad: e.target.value }) }, Object.keys(MAP.pads).map(n => h('option', { value: n, selected: n === c.base_pad }, n))),
    h('h3', {}, 'Targets (in order)'));
  if (!S.desired.targets.length) root.append(h('div', { class: 'hint' }, 'Click a building on the map, or type an order in Chat.'));
  S.targets.forEach((t, i) => root.append(h('div', { class: 'item' }, h('span', { class: 'n' + (t.done ? ' done' : '') }, t.done ? '✓' : i + 1), t.name,
    h('button', { class: 'x small', disabled: ended || S.phase === 'DONE', onclick: () => edit('target_remove', t.name), title: 'Remove' }, '✕'))));
  root.append(h('h3', {}, 'Restricted areas'));
  if (!S.desired.avoid.length) root.append(h('div', { class: 'hint' }, 'None. Click a building and choose Restrict, or say “stay away from …”.'));
  S.desired.avoid.forEach(a => root.append(h('div', { class: 'item' }, h('span', { class: 'n avoid' }, '!'), a.zone + (a.from || a.to ? ` (${a.from || 'now'}–${a.to || '∞'})` : ''),
    h('button', { class: 'x small', disabled: ended, onclick: () => edit('avoid_remove', a.zone), title: 'Allow again' }, '✕'))));
  root.append(h('div', { class: 'row' }, h('label', {}, h('input', { type: 'checkbox', id: 'f-stay', checked: S.desired.stay_at_end, disabled: ended, onchange: e => edit('stay', null, e.target.checked) }), ' Stay at the last place (don’t return to base)')));
  if (!locked) root.append(h('button', { class: 'small', onclick: () => edit('clear') }, 'Clear mission'));
  root.append(h('h3', {}, 'Settings'));
  if (locked) root.append(h('div', { class: 'locked' }, '🔒 Locked after Start. Press Reset to change.'));
  const fld = (id, label, el) => h('label', { class: 'f' }, label, el);
  const num = (id, key, min, max, step) => h('input', { id, type: 'number', min, max, step, value: c[key], disabled: locked, onchange: e => config({ [key]: e.target.value.trim() === '' ? NaN : +e.target.value }) });
  const sel = (id, key, opts) => h('select', { id, disabled: locked, onchange: e => config({ [key]: e.target.value }) }, opts.map(o => h('option', { value: o[0], selected: String(c[key]) === String(o[0]) }, o[1])));
  root.append(h('div', { class: 'grid2' },
    fld('f-cap', 'Battery (Wh)', num('f-cap', 'capacity', 5, 80, 1)),
    fld('f-clock', 'Start clock', h('input', { id: 'f-clock', value: c.clock, disabled: locked, onchange: e => config({ clock: e.target.value }) })),
    fld('f-wind', 'Real wind (m/s)', num('f-wind', 'wind_speed', 0, 9, 0.5)),
    fld('f-wdir', 'Wind from', sel('f-wdir', 'wind_from', [[0, 'North'], [90, 'East'], [180, 'South'], [270, 'West']])),
    fld('f-model', 'AI model', sel('f-model', 'model', MAP.models.map(m => [m, m]))),
    fld('f-mode', 'Mode', sel('f-mode', 'mode', [['ours', 'Ours: AI → shield'], ['classic', 'Classic: no AI'], ['direct', 'AI-direct: no shield']]))));
  root.append(h('div', { class: 'row' }, h('label', {}, h('input', { type: 'checkbox', id: 'f-wait', checked: c.wait_for_ai, disabled: locked, onchange: e => config({ wait_for_ai: e.target.checked }) }), ' Hold the drone while the AI reads my message')),
    h('div', { class: 'row' }, h('label', {}, h('input', { type: 'checkbox', id: 'f-mon', checked: c.monitor, disabled: locked, onchange: e => config({ monitor: e.target.checked }) }), ' Runtime battery monitor')));
  if (active) { const el = $(active); if (el && el.focus) el.focus(); }
  root.scrollTop = 0;
}

function renderChat() {
  const box = $('chat');
  for (const m of chat) {
    const meta = [m.who, m.clock, m.status, m.latency_ms ? (m.latency_ms / 1000).toFixed(1) + ' s' : ''].filter(Boolean).join(' · ');
    box.append(h('div', { class: 'msg ' + m.who + (m.level ? ' ' + m.level : '') }, h('small', {}, meta), m.text, m.spec ? h('div', { class: 'spec' }, specText(m.spec)) : null));
    if (!initialChat && ((m.who === 'system' && m.level === 'bad') || (m.who === 'ai' && m.status === 'refused'))) showReject(m.text, true);
  }
  initialChat = false;
  while (box.children.length > 300) box.firstChild.remove();
  chat = []; box.scrollTop = box.scrollHeight;
  const th = $('thinking');
  th.classList.toggle('hidden', !S.busy);
  if (S.busy) th.textContent = 'Understanding your message… ' + Math.round((performance.now() - busySince) / 1000) + ' s';
}
async function download(kind) {
  try {
    const r = await fetch('/api/logs/' + kind, { headers: { 'X-Session-Token': TOKEN, 'X-Drone-Id': DRONE } });
    if (!r.ok) throw new Error('Download failed (HTTP ' + r.status + ').');
    const url = URL.createObjectURL(await r.blob()), a = h('a', { href: url, download: 'safeuav-' + new Date().toISOString().slice(0, 16).replace(/[:T]/g, '') + '-' + kind });
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) { showReject(e.message); }
}
function renderLog() {
  if (logShown === audit.length) return; logShown = audit.length;
  const el = $('log'); el.replaceChildren(...audit.slice(-200).reverse().map(e => h('div', {}, `${e.clock} [${e.state}] ${e.kind}: ${e.msg}`)));
}

// ---------------------------------------------------------------- polling
let pollChain = Promise.resolve();
function poll() { pollChain = pollChain.then(doPoll); return pollChain; }   // one at a time, in order
async function doPoll() {
  try {
    const q = new URLSearchParams({ epoch: since.epoch, chat: since.chat, audit: since.audit, track: since.track, route_rev: since.route_rev });
    if (since.zones_sig != null) q.set('zones_sig', since.zones_sig);
    const d = await api('/api/state?' + q);
    lastOk = performance.now();
    if (d.reset) { chat = []; audit = []; trail = []; route = []; zones = []; ghost = null; rejects = 0; violations = 0; disp.init = false; $('chat').replaceChildren(); missionSig = ''; hideCard(); $('reject').classList.add('hidden'); $('replan').classList.add('hidden'); $('toasts').replaceChildren(); initialChat = true; }
    if (d.busy && !(S && S.busy)) busySince = performance.now();
    const newAudit = d.audit;
    chat.push(...d.chat); audit.push(...d.audit);
    for (const t of d.track) { trail.push([t[0], t[1]]); if (MAP.no_fly.some(z => distPoly([t[0], t[1]], z.poly) < z.buffer - 0.5)) violations++; }
    const prev = S; S = d; syncCircles(d.circles);
    if (d.route !== undefined) {
      if (prev && prev.phase === 'FLYING' && route.length > 1 && JSON.stringify(route) !== JSON.stringify(d.route) && d.route.length > 1
          && Math.abs(remainLen(d.route, d.route_index, d.pos) - remainLen(route, prev.route_index, d.pos)) >= 1) {
        const oldRest = remainLen(route, prev.route_index, d.pos), newRest = remainLen(d.route, d.route_index, d.pos), dm = newRest - oldRest;
        ghost = { route: [d.pos, ...route.slice(Math.max(1, prev.route_index))], t0: performance.now() };
        const sg = dm >= 0 ? '+' : '−', ab = Math.abs(dm);
        showReplan(`Replanned: ${sg}${Math.round(ab)} m · ≈${sg}${Math.round(ab / CRUISE_MPS)} s · ≈${sg}${(ab * WH_PER_M).toFixed(1)} Wh`);
      }
      route = d.route;
    }
    if (d.zones !== undefined) { if (JSON.stringify(zones) !== JSON.stringify(d.zones)) dirtyBase = true; zones = d.zones; }
    if (trail.length > 6000) trail = trail.slice(-4000);
    if (audit.length > 800) audit = audit.slice(-500);
    since = { epoch: d.epoch, chat: d.chat_total, audit: d.audit_total, track: d.track_total, route_rev: d.route_rev, zones_sig: d.zones_sig };
    if (!firstPoll) newAudit.forEach(e => auditToast(e)); firstPoll = false;
    if (!disp.init) { disp.x = d.pos[0]; disp.y = d.pos[1]; disp.init = true; }
    $('stale').classList.add('hidden');
    updateControls(); renderMission(); renderChat(); if (!$('tab-log').classList.contains('hidden')) renderLog();
  } catch (e) {
    $('stale').textContent = (e instanceof TypeError ? 'Display error: ' : 'Connection lost: view frozen. ') + e.message; $('stale').classList.remove('hidden');
  }
}
async function pollHealth() {
  try {
    const hl = await api('/api/health'), c = $('aichip');
    let t, k;
    if (S && S.config.mode === 'classic') { t = 'Basic parser (no AI)'; k = 'warn'; }
    else if (!hl.gemini_key_present && !(hl.ollama_models || []).length) { t = 'AI not configured: basic parser'; k = 'warn'; }
    else if (hl.public && hl.ai_budget_left === 0) { t = 'AI limit for today reached: basic parser'; k = 'warn'; }
    else { t = hl.public ? 'AI ready · ' + hl.ai_budget_left + ' calls left today' : 'AI ready'; k = 'ok'; }
    c.textContent = t; c.className = 'chip ' + k;
  } catch (e) { /* the stale banner already reports a lost connection */ }
  setTimeout(pollHealth, 30000);
}
async function pollLoop() { await poll(); setTimeout(pollLoop, 250); }

// ---------------------------------------------------------------- animation loop
let lastT = performance.now(), lastBase = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (S && disp.init) {
    const dx = S.pos[0] - disp.x, dy = S.pos[1] - disp.y, d = Math.hypot(dx, dy);
    if (d > 80) { disp.x = S.pos[0]; disp.y = S.pos[1]; }
    else if (d > 0.01) {
      const k = 1 - Math.exp(-dt / 0.2); disp.x += dx * k; disp.y += dy * k;
      if (d > 0.4) { let a = Math.atan2(dy, dx), da = a - disp.h; da = Math.atan2(Math.sin(da), Math.cos(da)); disp.h += da * Math.min(1, dt * 8); }
    }
    if (follow) { view.cx += (disp.x - view.cx) * Math.min(1, dt * 4); view.cy += (disp.y - view.cy) * Math.min(1, dt * 4); clampView(); if (now - lastBase > 100) { dirtyBase = true; lastBase = now; } }
    if (S.busy) { const t = Math.round((performance.now() - busySince) / 1000); $('thinking').textContent = 'Understanding your message… ' + t + ' s'; }
  }
  render(now); requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- wiring
function post(path, body) { return api(path, body || {}).then(() => poll()).catch(e => showReject(e.message)); }
async function init() {
  loadMap(await api('/api/map'), await api('/api/basemap'));
  (MAP.speeds || [1, 2, 5]).forEach(s => $('speeds').append(h('button', { 'data-s': s, onclick: () => post('/api/speed', { speed: s }) }, s + '×')));
  $('start').onclick = () => post('/api/start');
  $('pause').onclick = () => { if (!$('pause').disabled) post(S.phase === 'PAUSED' ? '/api/resume' : '/api/pause'); };
  $('reset').onclick = async () => { if (S && S.phase !== 'DRAFT' && !(await ask('Reset ends this flight and clears the chat and log.', 'Reset'))) return; post('/api/reset'); };
  $('rtb').onclick = () => post('/api/emergency', { kind: 'return' });
  $('land').onclick = () => post('/api/emergency', { kind: 'land' });
  $('endm').onclick = async () => { if (await ask('End this mission? The drone lands on the nearest pad, then waits for a new order.', 'End mission')) post('/api/end'); };
  document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => {
    document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b));
    for (const t of ['mission', 'chat', 'log']) $('tab-' + t).classList.toggle('hidden', t !== b.dataset.tab);
    if (b.dataset.tab === 'log') renderLog();
  });
  $('downloads').append(h('span', { class: 'lbl' }, 'Download:'), ...[['report.html', 'Flight report'], ['audit.jsonl', 'Audit log'], ['track.csv', 'Track CSV']].map(([k, t]) => h('button', { class: 'small', onclick: () => download(k) }, t)));
  MAP.presets.forEach(p => $('presets').append(h('button', { type: 'button', onclick: () => { $('chatin').value = p.text; $('chatin').focus(); } }, p.label)));
  $('chatform').onsubmit = async e => {
    e.preventDefault(); const t = $('chatin').value.trim(); if (!t) return;
    try { await api('/api/chat', { text: t }); $('chatin').value = ''; busySince = performance.now(); poll(); } catch (err) { showReject(err.message); }
  };
  setInterval(() => { if (!document.hidden && lastOk && performance.now() - lastOk > 5000) { $('stale').textContent = 'Connection lost: view frozen.'; $('stale').classList.remove('hidden'); } }, 1000);
  pollLoop(); pollHealth(); requestAnimationFrame(frame);
}
init().catch(e => { document.body.prepend(h('div', { class: 'tag' }, 'Could not start: ' + e.message)); });
