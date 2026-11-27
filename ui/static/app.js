/* PrivacyGuard AI — dashboard logic (vanilla ES2020, no build step) */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const PII_COLORS = {
  CNP: '#fb7185', PERSON_NAME: '#fbbf24', ADDRESS: '#a78bfa', ID_SERIES: '#60a5fa', IBAN: '#34d399',
  PHONE: '#22d3ee', EMAIL: '#f472b6', BIRTH_DATE: '#fdba74', MRZ: '#a3e635',
};
const CLUSTER_COLORS = ['#34d399', '#22d3ee', '#a78bfa', '#fbbf24', '#fb7185', '#60a5fa', '#f472b6', '#a3e635'];
const TYPE_SHORT = { PERSON_NAME: 'NAME', ID_SERIES: 'ID', BIRTH_DATE: 'DOB' };
const STATUS_CLASS = {
  'QUEUED': 'b-queued', 'OCR DONE': 'b-ocr', 'PII DETECTED': 'b-pii', 'REDACTED': 'b-red',
  'VERIFIED PASS': 'b-pass', 'VERIFIED FAIL': 'b-fail', 'ERROR': 'b-err',
};
const PII_TYPES = Object.keys(PII_COLORS);

const S = {
  files: new Map(), selected: null, links: null, mode: 'direct', running: false,
  providers: {}, leaks: 0, filter: 'all', highlight: null, view: 'curtain',
  tf: { scale: 1, x: 0, y: 0 }, curtain: 0.3, imgSize: null, cy: null,
};

// ------------------------------------------------------------------ utils
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast' + (err ? ' err' : '');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('hidden'), 3800);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return r.json();
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const shortId = (id) => (id ? id.replace(/^(doc|det|res)_/, '$1·').slice(0, 12) : '—');
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });

function highlightJson(obj) {
  const json = esc(JSON.stringify(obj, null, 2));
  return json.replace(/(&quot;(?:\\.|[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/g,
    (m, str, colon, bool) => {
      if (str) {
        if (colon) return `<span class="j-k">${str}</span>${colon}`;
        if (/&quot;(doc|det|res)_[0-9a-f]+&quot;|&quot;cluster_\d+&quot;|&quot;FILE_\d+&quot;/.test(str)) return `<span class="j-id">${str}</span>`;
        return `<span class="j-s">${str}</span>`;
      }
      if (bool) return `<span class="j-b">${m}</span>`;
      return `<span class="j-n">${m}</span>`;
    });
}

// ------------------------------------------------------------------ files
function upsertFile(f) {
  S.files.set(f.file_id, f);
  if (!S.selected) S.selected = f.file_id;
}
function badge(status) {
  const running = S.running && !['VERIFIED PASS', 'VERIFIED FAIL', 'ERROR'].includes(status) && status !== 'QUEUED';
  return `<span class="badge ${STATUS_CLASS[status] || 'b-queued'} ${running ? 'b-run' : ''}">${esc(status)}</span>`;
}
function renderFiles() {
  const list = $('#fileList');
  const files = [...S.files.values()];
  $('#queueCount').textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;
  if (!files.length) {
    list.innerHTML = `<li class="text-[12px] text-slate-500 text-center py-8 px-4">Queue is empty.<br>Drop files above or load the synthetic dataset.</li>`;
    return;
  }
  const hlDocs = S.highlight ? new Set(S.highlight.docs) : null;
  list.innerHTML = files.map((f) => {
    const thumb = f.has_before ? `style="background-image:url('/api/image/${f.file_id}/before?v=${f.doc_id || 'src'}')"` : '';
    const meta = [f.doc_id ? shortId(f.doc_id) : (f.source === 'demo' ? 'synthetic' : 'uploaded'),
      f.page_count ? `${f.page_count}p` : null, f.word_count != null ? `${f.word_count}w` : null].filter(Boolean).join(' · ');
    const hl = hlDocs && f.doc_id && hlDocs.has(f.doc_id) ? `hl" style="--hl:${S.highlight.color}` : '';
    return `<li class="file ${f.file_id === S.selected ? 'selected' : ''} ${hl}" data-id="${f.file_id}">
      <div class="thumb" ${thumb}></div>
      <div class="min-w-0"><div class="fname" title="${esc(f.name)}">${esc(f.name)}</div><div class="fmeta">${esc(meta)}</div></div>
      <div class="flex items-center gap-1">${badge(f.status)}<button class="rm" data-rm="${f.file_id}" title="Remove">✕</button></div>
    </li>`;
  }).join('');
}

// ------------------------------------------------------------------ KPIs
function renderKpis() {
  const files = [...S.files.values()];
  $('#kDocs').textContent = files.filter((f) => f.doc_id).length;
  $('#kBoxes').textContent = files.reduce((a, f) => a + (f.boxes_drawn || 0), 0);
  $('#kPass').textContent = files.filter((f) => f.status === 'VERIFIED PASS').length;
  $('#kFail').textContent = files.filter((f) => f.status === 'VERIFIED FAIL').length;
  $('#kClusters').textContent = S.links ? S.links.clusters.length : 0;
  $('#kLeaks').textContent = S.leaks;
  $('.kpi-shield').classList.toggle('leak', S.leaks > 0);
  $('#dlAll').classList.toggle('disabled', !files.some((f) => f.has_after));
}

// ------------------------------------------------------------------ inspector
function renderInspector() {
  const f = S.files.get(S.selected);
  $('#emptyState').classList.toggle('hidden', !!f);
  $('#curtain').classList.toggle('hidden', !f || S.view !== 'curtain');
  $('#split').classList.toggle('hidden', !f || S.view !== 'split');
  $('#pendingAfter').classList.toggle('hidden', !f || f.has_after);
  $('#inspTitle').textContent = f ? f.name : '';

  const v = $('#verifyBadge');
  if (!f || !f.status.startsWith('VERIFIED')) { v.className = 'verify verify-none'; v.textContent = f && f.status === 'ERROR' ? 'ERROR' : 'NOT VERIFIED'; }
  else if (f.status === 'VERIFIED PASS') { v.className = 'verify verify-pass'; v.textContent = '✓ VERIFIED PASS'; }
  else { v.className = 'verify verify-fail'; v.textContent = '✕ VERIFIED FAIL'; }
  $('#sConf').textContent = f && f.avg_confidence != null ? `${f.avg_confidence.toFixed(1)}%` : '—';
  $('#sWords').textContent = f && f.word_count != null ? f.word_count : '—';
  $('#sBoxes').textContent = f && f.boxes_drawn != null ? f.boxes_drawn : '—';
  $('#sResidual').textContent = f && f.residual_pii_count != null ? f.residual_pii_count : '—';
  $('#sDoc').textContent = f && f.doc_id ? f.doc_id : '—';
  const dl = $('#dlOne');
  dl.classList.toggle('disabled', !(f && f.has_after));
  dl.href = f && f.has_after ? `/api/download/${f.file_id}` : '#';

  const summary = (f && f.summary) || {};
  $('#piiBadges').innerHTML = PII_TYPES.map((t) => {
    const n = summary[t] || 0;
    return `<span class="pii ${n ? 'on' : ''}" style="--c:${PII_COLORS[t]}"><i></i>${t}<b>${n}</b></span>`;
  }).join('');

  if (!f) return;
  const key = `${f.file_id}|${f.doc_id}|${f.has_after}|${f.has_before}`;
  if (renderInspector._key === key) return;
  const sameDoc = renderInspector._file === f.file_id && S.imgSize;
  renderInspector._key = key; renderInspector._file = f.file_id;
  const v2 = f.doc_id || 'src';
  const before = f.has_before ? `/api/image/${f.file_id}/before?v=${v2}` : '';
  const after = f.has_after ? `/api/image/${f.file_id}/after?v=${v2}` : before;
  for (const id of ['#imgBefore', '#imgBefore2']) $(id).src = before;
  for (const id of ['#imgAfter', '#imgAfter2']) $(id).src = after;
  const img = $('#imgBefore');
  img.onload = () => {
    const changed = !S.imgSize || S.imgSize.w !== img.naturalWidth || S.imgSize.h !== img.naturalHeight;
    S.imgSize = { w: img.naturalWidth, h: img.naturalHeight };
    if (!sameDoc || changed) fit(); else applyTransform();
  };
}

function paneSize() {
  const vw = $('#viewer').clientWidth, vh = $('#viewer').clientHeight;
  return S.view === 'split' ? { w: vw / 2, h: vh } : { w: vw, h: vh };
}
function fit() {
  if (!S.imgSize) return;
  const p = paneSize();
  const scale = Math.min((p.w - 32) / S.imgSize.w, (p.h - 32) / S.imgSize.h, 2);
  S.tf = { scale, x: (p.w - S.imgSize.w * scale) / 2, y: (p.h - S.imgSize.h * scale) / 2 };
  applyTransform();
}
function applyTransform() {
  const { scale, x, y } = S.tf;
  for (const st of $$('.stage')) st.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  $('#zoomVal').textContent = `${Math.round(scale * 100)}%`;
  const pct = S.curtain * 100;
  $('#layerAfter').style.clipPath = `inset(0 0 0 ${pct}%)`;
  $('#divider').style.left = `${pct}%`;
}
function zoomAt(factor, cx, cy) {
  const t = S.tf;
  const ns = Math.min(8, Math.max(0.1, t.scale * factor));
  const k = ns / t.scale;
  S.tf = { scale: ns, x: cx - (cx - t.x) * k, y: cy - (cy - t.y) * k };
  applyTransform();
}
function setupViewer() {
  const viewer = $('#viewer');
  let drag = null;
  viewer.addEventListener('wheel', (e) => {
    if (!S.files.get(S.selected)) return;
    e.preventDefault();
    const r = viewer.getBoundingClientRect();
    let cx = e.clientX - r.left;
    if (S.view === 'split' && cx > r.width / 2) cx -= r.width / 2;
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, cx, e.clientY - r.top);
  }, { passive: false });
  viewer.addEventListener('pointerdown', (e) => {
    if (!S.files.get(S.selected)) return;
    const onDivider = e.target.closest('#divider');
    drag = { mode: onDivider ? 'curtain' : 'pan', x: e.clientX, y: e.clientY, tx: S.tf.x, ty: S.tf.y };
    viewer.setPointerCapture(e.pointerId);
    if (!onDivider) viewer.classList.add('panning');
  });
  viewer.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (drag.mode === 'curtain') {
      const r = viewer.getBoundingClientRect();
      S.curtain = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    } else {
      S.tf.x = drag.tx + (e.clientX - drag.x); S.tf.y = drag.ty + (e.clientY - drag.y);
    }
    applyTransform();
  });
  const end = () => { drag = null; viewer.classList.remove('panning'); };
  viewer.addEventListener('pointerup', end); viewer.addEventListener('pointercancel', end);
  viewer.addEventListener('dblclick', fit);
  $('#zoomIn').onclick = () => { const p = paneSize(); zoomAt(1.25, p.w / 2, p.h / 2); };
  $('#zoomOut').onclick = () => { const p = paneSize(); zoomAt(0.8, p.w / 2, p.h / 2); };
  $('#zoomFit').onclick = fit;
  window.addEventListener('resize', () => { fit(); if (S.cy) S.cy.resize(); });
  $('#privacyScreen').onchange = (e) => viewer.classList.toggle('privacy', e.target.checked);
  $$('#viewSeg .seg-btn').forEach((b) => b.onclick = () => {
    $$('#viewSeg .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
    S.view = b.dataset.view; renderInspector(); fit();
  });
}

// ------------------------------------------------------------------ graph
function renderGraph() {
  const docs = [...S.files.values()].filter((f) => f.doc_id);
  const has = !!(S.links && docs.length);
  $('#graphEmpty').classList.toggle('hidden', has);
  const chips = $('#clusterChips');
  if (!has) { chips.innerHTML = ''; if (S.cy) S.cy.elements().remove(); $('#graphMeta').textContent = 'HMAC-SHA256 · ids only'; return; }
  if (typeof cytoscape === 'undefined') { $('#graphEmpty').classList.remove('hidden'); $('#graphEmpty').innerHTML = '<div class="text-amber-300 text-center">Graph library could not load (offline?). Cluster list below.</div>'; }

  const byDoc = Object.fromEntries(docs.map((f) => [f.doc_id, f]));
  const clusterOf = {};
  S.links.clusters.forEach((c, i) => c.doc_ids.forEach((d) => { clusterOf[d] = { id: c.cluster_id, color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }; }));
  $('#graphMeta').textContent = `${S.links.clusters.length} cluster${S.links.clusters.length === 1 ? '' : 's'} · ${S.links.unlinked_doc_ids.length} unlinked`;

  chips.innerHTML = S.links.clusters.map((c, i) => {
    const col = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const active = S.highlight && S.highlight.id === c.cluster_id ? 'active' : '';
    return `<span class="chip ${active}" data-cluster="${c.cluster_id}" style="color:${col};border-color:${col}55;background:${col}14">
      ● ${c.cluster_id} · ${c.doc_ids.length} docs · ${c.shared_entity_types.map((t) => TYPE_SHORT[t] || t).join('+')}</span>`;
  }).join('') + (S.links.unlinked_doc_ids.length ? `<span class="chip" style="color:#94a3b8;border-color:#94a3b833">○ ${S.links.unlinked_doc_ids.length} unlinked</span>` : '');

  if (typeof cytoscape === 'undefined') return;
  const els = [];
  S.links.clusters.forEach((c, i) => els.push({ data: { id: c.cluster_id, label: `${c.cluster_id}  ·  strength ${c.link_strength}`, color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }, classes: 'cluster' }));
  docs.forEach((f) => {
    const c = clusterOf[f.doc_id];
    const ok = f.status === 'VERIFIED PASS';
    els.push({ data: { id: f.doc_id, parent: c ? c.id : undefined, label: `${f.name.replace(/\.(png|jpe?g|pdf)$/i, '')}\n${shortId(f.doc_id)}`,
      color: c ? c.color : '#64748b', ring: ok ? '#34d399' : (f.status === 'VERIFIED FAIL' ? '#fb7185' : '#334155'), file: f.file_id }, classes: 'doc' });
  });
  S.links.clusters.forEach((c) => c.links.forEach((l) => {
    if (!byDoc[l.doc_a] || !byDoc[l.doc_b]) return;
    els.push({ data: { id: `${l.doc_a}-${l.doc_b}`, source: l.doc_a, target: l.doc_b, conf: l.confidence,
      label: `${l.shared_entity_types.map((t) => TYPE_SHORT[t] || t).join('+')} · ${l.confidence}`,
      color: clusterOf[l.doc_a] ? clusterOf[l.doc_a].color : '#64748b' } });
  }));

  if (!S.cy) {
    S.cy = cytoscape({
      container: $('#graph'), wheelSensitivity: 0.25, minZoom: 0.2, maxZoom: 3,
      style: [
        { selector: 'node.doc', style: {
          'background-color': '#0e1520', 'border-width': 2.5, 'border-color': 'data(ring)', width: 30, height: 30,
          label: 'data(label)', color: '#cbd5e1', 'font-size': 8.5, 'text-wrap': 'wrap', 'text-valign': 'bottom', 'text-margin-y': 5,
          'font-family': 'JetBrains Mono, Consolas, monospace', 'overlay-opacity': 0,
          'background-image': 'data(img)', 'background-fit': 'cover', 'background-image-opacity': 0.9 } },
        { selector: 'node.cluster', style: {
          'background-color': 'data(color)', 'background-opacity': 0.06, 'border-color': 'data(color)', 'border-width': 1.2,
          'border-opacity': 0.55, shape: 'round-rectangle', padding: 18, label: 'data(label)', color: 'data(color)',
          'font-size': 9, 'text-valign': 'top', 'text-halign': 'center', 'text-margin-y': -4, 'font-weight': 700, 'overlay-opacity': 0 } },
        { selector: 'edge', style: {
          width: 'mapData(conf, 0.5, 1, 1.2, 4.2)', 'line-color': 'data(color)', opacity: 0.75, 'curve-style': 'bezier',
          label: 'data(label)', 'font-size': 7.5, color: '#e2e8f0', 'text-background-color': '#0a0f17', 'text-background-opacity': 0.92,
          'text-background-padding': 2, 'text-background-shape': 'round-rectangle', 'font-family': 'JetBrains Mono, Consolas, monospace',
          'text-rotation': 'autorotate', 'overlay-opacity': 0 } },
        { selector: '.faded', style: { opacity: 0.15 } },
        { selector: 'node.doc.picked', style: { 'border-color': '#22d3ee', 'border-width': 4 } },
      ],
    });
    S.cy.on('tap', 'node.doc', (e) => { selectFile(e.target.data('file')); });
    S.cy.on('tap', 'node.cluster', (e) => toggleCluster(e.target.id()));
    S.cy.on('tap', (e) => { if (e.target === S.cy) toggleCluster(null); });
  }
  S.cy.elements().remove();
  S.cy.add(els);
  S.cy.nodes('node.doc').forEach((n) => { const f = S.files.get(n.data('file')); if (f && f.has_after) n.data('img', `/api/image/${f.file_id}/after?v=${f.doc_id}`); });
  // Deterministic layout: one grid cell per cluster (unlinked docs share the last cells), members on a ring
  const groups = S.links.clusters.map((c) => c.doc_ids.filter((d) => byDoc[d]));
  docs.filter((f) => !clusterOf[f.doc_id]).forEach((f) => groups.push([f.doc_id]));
  const box = $('#graph');
  const cols = Math.max(1, Math.round(Math.sqrt(groups.length * (box.clientWidth / Math.max(box.clientHeight, 1)))));
  const cell = 190;
  const pos = {};
  groups.forEach((g, gi) => {
    const cx = (gi % cols) * cell, cyy = Math.floor(gi / cols) * cell;
    const r = g.length === 1 ? 0 : 42 + 6 * g.length;
    g.forEach((d, k) => {
      const a = -Math.PI / 2 + (2 * Math.PI * k) / g.length;
      pos[d] = { x: cx + r * Math.cos(a), y: cyy + r * Math.sin(a) };
    });
  });
  S.cy.layout({ name: 'preset', positions: (n) => pos[n.id()], fit: true, padding: 22, animate: false }).run();
  if (S.cy.zoom() > 1.25) { S.cy.zoom(1.25); S.cy.center(); }     // small graphs: keep labels readable
  applyGraphHighlight();
}
function toggleCluster(id) {
  if (!id || (S.highlight && S.highlight.id === id)) S.highlight = null;
  else {
    const i = S.links.clusters.findIndex((c) => c.cluster_id === id);
    S.highlight = { id, docs: S.links.clusters[i].doc_ids, color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] };
  }
  renderFiles(); applyGraphHighlight();
  $$('#clusterChips .chip').forEach((c) => c.classList.toggle('active', !!S.highlight && c.dataset.cluster === S.highlight.id));
}
function applyGraphHighlight() {
  if (!S.cy) return;
  S.cy.elements().removeClass('faded picked');
  const sel = S.files.get(S.selected);
  if (sel && sel.doc_id) S.cy.$id(sel.doc_id).addClass('picked');
  if (!S.highlight) return;
  const keep = new Set([S.highlight.id, ...S.highlight.docs]);
  S.cy.nodes().forEach((n) => { if (!keep.has(n.id())) n.addClass('faded'); });
  S.cy.edges().forEach((e) => { if (!keep.has(e.source().id())) e.addClass('faded'); });
}

// ------------------------------------------------------------------ console
function consoleAppend(ev) {
  const box = $('#console');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const el = document.createElement('div');
  const ts = `<span class="ev-ts">${fmtTime(ev.ts)}</span>`;
  let cls = 'ev', head = '', body = null, kind = 'info';

  if (ev.type === 'tool') {
    kind = ev.sent_to_llm ? 'llm' : 'tool';
    cls += ev.result && ev.result.error ? ' ev-err' : (ev.sent_to_llm ? ' ev-llm' : ' ev-tool');
    const args = Object.entries(ev.args || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ');
    const scan = ev.pii_pattern_hits ? `<span class="dirty">⚠ ${ev.pii_pattern_hits} PII pattern(s)</span>` : '<span class="clean">✓ 0 raw-PII patterns</span>';
    const res = ev.result && ev.result.error ? `<span class="text-rose-300">error</span>` : resultSummary(ev.tool, ev.result);
    head = `${ts}<span class="text-slate-500">#${String(ev.step).padStart(2, '0')}</span>
      <span class="dir d-call">AGENT→MCP</span><span class="tool">${esc(ev.tool)}</span><span class="args">(${esc(args)})</span>
      <span class="dir d-res">MCP→${ev.sent_to_llm ? 'LLM' : 'AGENT'}</span><span class="text-slate-300">${res}</span>${scan}`;
    body = `<div class="text-slate-500 mb-1">${ev.sent_to_llm ? 'Exact tool message appended to the LLM conversation:' : 'Boundary payload (what an LLM would receive — direct mode sends nothing to any LLM):'}</div>${highlightJson(safeParse(ev.llm_content))}`;
  } else if (ev.type === 'agent') {
    kind = 'llm';
    if (ev.event === 'start') {
      cls += ' ev-llm';
      const scan = ev.pii_pattern_hits ? `<span class="dirty">⚠ PII</span>` : '<span class="clean">✓ clean</span>';
      head = `${ts}<span class="dir d-llm">AGENT→LLM</span><span class="text-slate-200">task prompt · ${esc(ev.provider)} / ${esc(ev.model)} · tools: ${esc((ev.tools || []).join(', '))}</span>${scan}`;
      body = `<div class="text-slate-500 mb-1">User message sent to the LLM (file paths replaced by aliases):</div>${esc(ev.llm_user_message)}`;
    } else if (ev.event === 'nudge') {
      cls += ' ev-err'; kind = 'llm';
      head = `${ts}<span class="dir d-warn">GUARDRAIL</span><span class="text-amber-200">model stopped early — nudged with ${ev.pending.length} pending step(s)</span>`;
      body = `<div class="text-slate-500 mb-1">Model said:</div>${esc(ev.model_said)}\n\n<div class="text-slate-500 mb-1">Pending:</div>${esc(ev.pending.join('\n'))}`;
    } else if (ev.event === 'report') {
      cls += ' ev-llm';
      head = `${ts}<span class="dir d-llm">LLM→AGENT</span><span class="text-slate-200">final report</span>`;
      body = esc(ev.content); el.classList.add('open');
    } else if (ev.event === 'error') {
      cls += ' ev-err';
      head = `${ts}<span class="dir d-err">ERROR</span><span class="text-rose-200">${esc(ev.message)}</span>`;
    } else return;
  } else if (ev.type === 'job') {
    cls += ' ev-job';
    if (ev.state === 'started') head = `${ts}<span class="dir d-info">JOB</span><span class="text-cyan-200">started · mode=${esc(ev.mode)}</span>`;
    else {
      const s = ev.stats || {};
      head = `${ts}<span class="dir d-info">JOB</span><span class="text-emerald-200">finished in ${ev.seconds}s · ${s.documents_processed} docs · ${s.boxes_redacted} PII redacted · ${s.verified_pass} pass / ${s.verified_fail} fail · ${s.clusters} clusters</span>`;
    }
  } else if (ev.type === 'log') {
    cls += ev.level === 'error' ? ' ev-err' : '';
    head = `${ts}<span class="dir ${ev.level === 'error' ? 'd-err' : 'd-info'}">${ev.level === 'error' ? 'ERROR' : 'INFO'}</span><span class="text-slate-300">${esc(ev.message)}</span>`;
  } else return;

  el.className = cls; el.dataset.kind = kind;
  el.innerHTML = `<div class="ev-head">${head}</div>${body ? `<div class="ev-body">${body}</div>` : ''}`;
  if (body) el.querySelector('.ev-head').onclick = () => el.classList.toggle('open');
  if (!matchesFilter(el)) el.style.display = 'none';
  box.appendChild(el);
  while (box.children.length > 800) box.firstChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
}
function safeParse(s) { try { return JSON.parse(s); } catch (_) { return s; } }
function resultSummary(tool, r) {
  if (!r) return '';
  switch (tool) {
    case 'ocr_document': return `${esc(shortId(r.doc_id))} · ${r.word_count}w · ${r.avg_confidence}%`;
    case 'detect_pii': return `${Object.values(r.summary || {}).reduce((a, b) => a + b, 0)} entities · ${r.total_boxes} boxes`;
    case 'redact_image': return `${r.boxes_drawn} boxes drawn`;
    case 'verify_redaction': return r.status === 'PASS' ? '<span class="text-emerald-300">PASS</span>' : `<span class="text-rose-300">FAIL (${r.residual_pii_count})</span>`;
    case 'link_documents': return `${(r.clusters || []).length} clusters`;
    default: return '';
  }
}
function matchesFilter(el) {
  if (S.filter === 'all') return true;
  if (S.filter === 'tool') return el.classList.contains('ev-tool') || el.classList.contains('ev-llm') && el.querySelector('.d-call');
  return el.dataset.kind === 'llm';
}
function applyFilter() { $$('#console .ev').forEach((el) => { el.style.display = matchesFilter(el) ? '' : 'none'; }); }

// ------------------------------------------------------------------ events
function handleEvent(ev, live = true) {
  switch (ev.type) {
    case 'file': upsertFile(ev.file); break;
    case 'file_removed': S.files.delete(ev.file_id); if (S.selected === ev.file_id) S.selected = S.files.keys().next().value || null; break;
    case 'links': S.links = ev.links; S.highlight = null; break;
    case 'reset': S.files.clear(); S.links = null; S.selected = null; S.leaks = 0; S.highlight = null; $('#console').innerHTML = ''; break;
    case 'tool': S.leaks += ev.pii_pattern_hits || 0; break;
    case 'agent': if (ev.event === 'start') S.leaks += ev.pii_pattern_hits || 0; break;
    case 'job':
      S.running = ev.state === 'started';
      setJob(ev.state === 'started' ? 'run' : 'done', ev.state === 'started' ? `Running · ${ev.mode}` : `Done · ${ev.seconds}s`);
      if (live && ev.state === 'finished') toast(`Job finished — ${ev.stats.verified_pass} pass, ${ev.stats.verified_fail} fail, ${ev.stats.clusters} clusters`);
      break;
  }
  consoleAppend(ev);
}
let renderQueued = false;
function scheduleRender(graph = false) {
  scheduleRender.graph = scheduleRender.graph || graph;
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderFiles(); renderKpis(); renderInspector(); applyGraphHighlight();
    if (scheduleRender.graph) { scheduleRender.graph = false; renderGraph(); }
    $('#runBtn').disabled = S.running;
  });
}
function setJob(kind, text) {
  $('#jobPill').className = `pill pill-${kind}`;
  $('#jobText').textContent = text;
}
function connect() {
  const es = new EventSource('/api/events');
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    handleEvent(ev);
    scheduleRender(ev.type === 'links' || ev.type === 'reset' || (ev.type === 'file' && S.links));
  };
  es.onerror = () => setJob('idle', 'Reconnecting…');
  es.onopen = () => { if (!S.running) setJob('idle', 'Idle'); };
}

// ------------------------------------------------------------------ actions
function selectFile(id) {
  if (!S.files.has(id)) return;
  S.selected = id; scheduleRender();
}
async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) => /\.(png|jpe?g|pdf)$/i.test(f.name));
  if (!files.length) return toast('Only PNG, JPG or PDF files are accepted', true);
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  try { const r = await api('/api/upload', { method: 'POST', body: fd }); toast(`Added ${r.added.length} document(s)`); }
  catch (e) { toast(e.message, true); }
}
function setMode(mode) {
  S.mode = mode;
  $$('#modeSeg .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  const p = S.providers[mode] || {};
  $('#providerLabel').textContent = p.label || mode;
  const note = $('#providerNote');
  if (p.available === false) { note.textContent = `Unavailable: ${p.detail || ''}`; note.classList.remove('hidden'); }
  else if (mode !== 'direct') { note.textContent = 'Agent mode: the LLM drives the MCP tools. Local CPU models take several minutes for 12 documents.'; note.classList.remove('hidden'); note.classList.replace('text-amber-300/90', 'text-sky-300/90'); }
  else note.classList.add('hidden');
  if (p.available !== false) note.classList.replace('text-amber-300/90', 'text-sky-300/90'); else note.classList.replace('text-sky-300/90', 'text-amber-300/90');
}
async function run() {
  const p = S.providers[S.mode];
  if (p && p.available === false) return toast(`${p.label} is not available: ${p.detail}`, true);
  try {
    await api('/api/process', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: S.mode, padding_px: +$('#padding').value, use_local_slm: $('#slm').checked, tts: $('#tts').checked }) });
  } catch (e) { toast(e.message, true); }
}

function bindUi() {
  const dz = $('#dropZone');
  ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', (e) => uploadFiles(e.dataTransfer.files));
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => { e.preventDefault(); if (!dz.contains(e.target)) uploadFiles(e.dataTransfer.files); });
  $('#fileInput').onchange = (e) => { uploadFiles(e.target.files); e.target.value = ''; };
  $('#demoBtn').onclick = async () => {
    try { const r = await api('/api/load-demo', { method: 'POST' }); toast(r.added.length ? `Loaded ${r.added.length} synthetic documents` : 'Synthetic documents already in the queue'); }
    catch (e) { toast(e.message, true); }
  };
  $('#resetBtn').onclick = async () => {
    if (!confirm('Clear the queue? Uploaded files are deleted from ui/uploads/.')) return;
    try { await api('/api/reset', { method: 'POST' }); } catch (e) { toast(e.message, true); }
  };
  $('#fileList').onclick = async (e) => {
    const rm = e.target.closest('[data-rm]');
    if (rm) { e.stopPropagation(); try { await api(`/api/files/${rm.dataset.rm}`, { method: 'DELETE' }); } catch (err) { toast(err.message, true); } return; }
    const li = e.target.closest('.file'); if (li) selectFile(li.dataset.id);
  };
  $('#clusterChips').onclick = (e) => { const c = e.target.closest('[data-cluster]'); if (c) toggleCluster(c.dataset.cluster); };
  $$('#modeSeg .seg-btn').forEach((b) => b.onclick = () => setMode(b.dataset.mode));
  $('#padding').oninput = (e) => { $('#padVal').textContent = `${e.target.value} px`; };
  $('#runBtn').onclick = run;
  $('#graphFit').onclick = () => S.cy && S.cy.fit(undefined, 24);
  $$('#logSeg .seg-btn').forEach((b) => b.onclick = () => {
    $$('#logSeg .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
    S.filter = b.dataset.filter; applyFilter();
  });
  $('#clearLog').onclick = () => { $('#console').innerHTML = ''; };
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    const ids = [...S.files.keys()]; const i = ids.indexOf(S.selected);
    if (e.key === 'ArrowDown' || e.key === 'j') { selectFile(ids[Math.min(ids.length - 1, i + 1)]); e.preventDefault(); }
    if (e.key === 'ArrowUp' || e.key === 'k') { selectFile(ids[Math.max(0, i - 1)]); e.preventDefault(); }
    if (e.key === 'f') fit();
  });
}

async function init() {
  bindUi(); setupViewer();
  try {
    const cfg = await api('/api/config');
    S.providers = cfg.providers;
    $$('#modeSeg .seg-btn').forEach((b) => {
      const p = S.providers[b.dataset.mode];
      b.classList.toggle('unavailable', p && !p.available);
      if (p && !p.available) b.title += ` — unavailable: ${p.detail}`;
    });
    $('#demoBtn').classList.toggle('disabled', !cfg.synthetic_available);
  } catch (_) { /* server will report errors on use */ }
  setMode('direct');
  const st = await api('/api/state');
  st.files.forEach(upsertFile);
  S.links = st.links; S.running = !!(st.job && st.job.running);
  st.events.forEach((ev) => { if (ev.type === 'tool') S.leaks += ev.pii_pattern_hits || 0; if (ev.type === 'agent' && ev.event === 'start') S.leaks += ev.pii_pattern_hits || 0; consoleAppend(ev); });
  if (S.running) setJob('run', `Running · ${st.job.mode}`);
  else if (st.job && st.job.finished) setJob('done', 'Done');
  scheduleRender(true);
  connect();
}
init();
