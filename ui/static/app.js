/* PrivacyGuard AI — folders + auto-redaction dashboard (vanilla ES2020, no build step) */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const INBOX = 'f_inbox';
const DONE = ['VERIFIED PASS', 'VERIFIED FAIL', 'ERROR'];
const STATUS_CLASS = {
  'QUEUED': 'b-queued', 'OCR DONE': 'b-work', 'PII DETECTED': 'b-work', 'REDACTED': 'b-work',
  'VERIFIED PASS': 'b-pass', 'VERIFIED FAIL': 'b-fail', 'ERROR': 'b-err',
};
const STATUS_TEXT = { 'VERIFIED PASS': 'pass', 'VERIFIED FAIL': 'fail', 'PII DETECTED': 'detected', 'OCR DONE': 'ocr' };
const POLICY_LABEL = { full: 'FULL BLUR', pii: 'PII ONLY' };

const S = {
  files: new Map(), folders: new Map(), folder: INBOX, selected: null, running: false, jobFolder: null,
  providers: {}, leaks: 0, filter: 'all', view: 'curtain', newPolicy: 'full',
  tf: { scale: 1, x: 0, y: 0 }, curtain: 0.5, imgSize: null,
};

// ------------------------------------------------------------------ utils
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast' + (err ? ' err' : '');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('hidden'), 3800);
}
async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts = { ...opts, body: JSON.stringify(opts.json), headers: { 'Content-Type': 'application/json' } };
    delete opts.json;
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return r.json();
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const shortId = (id) => (id ? id.slice(0, 16) : '—');
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
const folderFiles = (id = S.folder) => [...S.files.values()].filter((f) => f.folder_id === id);
const currentFolder = () => S.folders.get(S.folder) || S.folders.get(INBOX);

function highlightJson(obj) {
  const json = esc(typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2));
  return json.replace(/(&quot;(?:\\.|[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?/g,
    (m, str, colon, bool) => {
      if (str) {
        if (colon) return `<span class="j-k">${str}</span>${colon}`;
        if (/&quot;(doc|det|res)_[0-9a-f]+&quot;|&quot;cluster_\d+&quot;|&quot;FILE_\d+&quot;/.test(str)) return `<span class="j-id">${str}</span>`;
        return `<span class="j-s">${str}</span>`;
      }
      return bool ? `<span class="j-b">${m}</span>` : `<span class="j-n">${m}</span>`;
    });
}

// ------------------------------------------------------------------ folders (sidebar)
function renderFolders() {
  const list = $('#folderList');
  const folders = [...S.folders.values()].sort((a, b) => a.created - b.created);
  list.innerHTML = folders.map((fo) => {
    const files = folderFiles(fo.folder_id);
    const done = files.filter((f) => f.status === 'VERIFIED PASS').length;
    const live = S.running && (S.jobFolder === fo.folder_id || S.jobFolder === null) && files.some((f) => !DONE.includes(f.status));
    return `<li class="folder ${fo.folder_id === S.folder ? 'active' : ''}" data-folder="${fo.folder_id}">
      <span class="folder-name">${esc(fo.name)}</span>
      <span class="folder-count">${files.length ? `${done}/${files.length}` : '0'}</span>
      <span class="folder-meta"><span class="tag tag-${fo.policy}">${POLICY_LABEL[fo.policy]}</span>
        ${fo.auto ? '<span>auto</span>' : '<span>manual</span>'}${live ? '<span class="live"></span>' : ''}</span>
    </li>`;
  }).join('');
}

function renderFolderHead() {
  const fo = currentFolder();
  if (!fo) return;
  const name = $('#folderName');
  if (document.activeElement !== name) name.value = fo.name;
  name.readOnly = fo.folder_id === INBOX;
  $('#folderId').textContent = fo.folder_id;
  $$('#policySeg .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.policy === fo.policy));
  $('#autoToggle').checked = fo.auto;
  $('#deleteFolderBtn').classList.toggle('hidden', fo.folder_id === INBOX);
  $('#demoBtn').classList.toggle('hidden', !S.syntheticAvailable);

  const files = folderFiles();
  $('#runLabel').textContent = files.some((f) => f.doc_id) ? `re-run · ${POLICY_LABEL[fo.policy].toLowerCase()}` : `run · ${POLICY_LABEL[fo.policy].toLowerCase()}`;
  $('#runBtn').disabled = S.running || !files.length;
  const dl = $('#dlAll');
  const any = files.some((f) => f.has_after);
  dl.classList.toggle('disabled', !any);
  dl.href = any ? `/api/download-all?folder_id=${encodeURIComponent(fo.folder_id)}` : '#';
  $('#dropHint').textContent = fo.auto
    ? `→ ${fo.policy === 'full' ? 'every page fully blurred' : 'PII redacted'} on arrival`
    : '→ queued, press run';
}

// ------------------------------------------------------------------ documents
function linkTags(fo) {
  const map = {};
  ((fo && fo.links && fo.links.clusters) || []).forEach((c, i) => c.doc_ids.forEach((d) => { map[d] = `L${i + 1}`; }));
  return map;
}
function badge(f) {
  const working = S.running && !DONE.includes(f.status) && f.status !== 'QUEUED';
  return `<span class="badge ${STATUS_CLASS[f.status] || 'b-queued'} ${working ? 'b-run' : ''}" ${f.error ? `title="${esc(f.error)}"` : ''}>${esc(STATUS_TEXT[f.status] || f.status)}</span>`;
}
function renderFiles() {
  const list = $('#fileList');
  const files = folderFiles();
  const fo = currentFolder();
  const tags = linkTags(fo);
  const clusters = (fo && fo.links && fo.links.clusters) || [];
  $('#docCount').textContent = `${files.length} doc${files.length === 1 ? '' : 's'}`;
  $('#linkCount').textContent = clusters.length ? `${clusters.length} linked group${clusters.length === 1 ? '' : 's'}` : '';
  if (!files.length) {
    list.innerHTML = '<li class="empty-list">// empty<br>drop scans above</li>';
    return;
  }
  list.innerHTML = files.map((f) => {
    const thumb = f.has_after ? `style="background-image:url('/api/image/${f.file_id}/after?v=${f.doc_id}')"` : '';
    const meta = [f.doc_id ? shortId(f.doc_id) : (f.source === 'demo' ? 'synthetic' : 'uploaded'), f.page_count ? `${f.page_count}p` : null]
      .filter(Boolean).map(esc).join(' · ');
    const link = f.doc_id && tags[f.doc_id] ? `<span class="tag tag-link" title="Shares hashed identifiers with other documents in this folder">${tags[f.doc_id]}</span>` : '';
    return `<li class="file ${f.file_id === S.selected ? 'selected' : ''}" data-id="${f.file_id}">
      <div class="thumb" ${thumb}></div>
      <div class="min-w-0"><div class="fname" title="${esc(f.name)}">${esc(f.name)}</div><div class="fmeta">${meta}${link}</div></div>
      <div class="file-right">${badge(f)}<button class="rm" data-rm="${f.file_id}" title="Remove">✕</button></div>
    </li>`;
  }).join('');
}

// ------------------------------------------------------------------ viewer
function renderInspector() {
  const f = S.files.get(S.selected);
  $('#emptyState').classList.toggle('hidden', !!f);
  $('#curtain').classList.toggle('hidden', !f || S.view !== 'curtain');
  $('#split').classList.toggle('hidden', !f || S.view !== 'split');
  $('#pendingAfter').classList.toggle('hidden', !f || f.has_after);
  $('#inspTitle').textContent = f ? f.name : 'no document';

  const v = $('#verifyBadge');
  if (!f) { v.className = 'badge b-queued'; v.textContent = 'NOT VERIFIED'; }
  else if (f.status === 'VERIFIED PASS') { v.className = 'badge b-pass'; v.textContent = '✓ verified'; }
  else if (f.status === 'VERIFIED FAIL') { v.className = 'badge b-fail'; v.textContent = '! residual pii'; }
  else if (f.status === 'ERROR') { v.className = 'badge b-err'; v.textContent = 'error'; }
  else { v.className = `badge ${S.running && f.status !== 'QUEUED' ? 'b-work b-run' : 'b-queued'}`; v.textContent = f.status === 'QUEUED' ? 'queued' : 'processing'; }
  $('#sResidual').textContent = f && f.residual_pii_count != null ? f.residual_pii_count : '—';
  $('#sDoc').textContent = f && f.doc_id ? f.doc_id : '—';
  const dl = $('#dlOne');
  dl.classList.toggle('disabled', !(f && f.has_after));
  dl.href = f && f.has_after ? `/api/download/${f.file_id}` : '#';

  if (!f) return;
  const key = `${f.file_id}|${f.doc_id}|${f.has_after}|${f.has_before}`;
  if (renderInspector._key === key) return;
  const sameDoc = renderInspector._file === f.file_id && S.imgSize;
  renderInspector._key = key; renderInspector._file = f.file_id;
  const ver = f.doc_id || 'src';
  const before = f.has_before ? `/api/image/${f.file_id}/before?v=${ver}` : '';
  const after = f.has_after ? `/api/image/${f.file_id}/after?v=${ver}` : before;
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
  const scale = Math.min((p.w - 40) / S.imgSize.w, (p.h - 40) / S.imgSize.h, 2);
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
  window.addEventListener('resize', fit);
  $('#privacyScreen').onchange = (e) => viewer.classList.toggle('privacy', e.target.checked);
  $$('#viewSeg .seg-btn').forEach((b) => b.onclick = () => {
    $$('#viewSeg .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
    S.view = b.dataset.view; renderInspector(); fit();
  });
}

// ------------------------------------------------------------------ log
function resultSummary(tool, r) {
  if (!r) return '';
  switch (tool) {
    case 'ocr_document': return `${shortId(r.doc_id)} · ${r.word_count}w`;
    case 'detect_pii': return `${r.total_boxes} boxes`;
    case 'redact_image': return `${r.boxes_drawn} boxes drawn`;
    case 'blur_document': return `${shortId(r.doc_id)} · ${r.page_count}p blurred`;
    case 'verify_redaction': return r.status === 'PASS' ? '<span class="ok">PASS</span>' : `<span class="bad">FAIL (${r.residual_pii_count})</span>`;
    case 'link_documents': return `${(r.clusters || []).length} linked groups`;
    default: return '';
  }
}
function consoleAppend(ev) {
  let kind = 'info', cls = 'ev', label = 'info', msg = '', body = null;
  if (ev.type === 'tool') {
    kind = 'tool'; label = ev.sent_to_llm ? 'llm' : 'tool';
    const err = ev.result && ev.result.error;
    cls += err ? ' ev-err' : ' ev-tool';
    const args = Object.entries(ev.args || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ');
    const scan = ev.pii_pattern_hits ? ` <span class="bad">⚠ ${ev.pii_pattern_hits} pii pattern(s)</span>` : ' <span class="ok">✓</span>';
    msg = `#${String(ev.step).padStart(2, '0')} ${esc(ev.tool)}(${esc(args)}) → ${err ? esc(err) : resultSummary(ev.tool, ev.result)}${scan}`;
    body = highlightJson(safeParse(ev.llm_content));
  } else if (ev.type === 'agent') {
    if (ev.event === 'start') { label = 'llm'; msg = `task prompt → ${esc(ev.provider)} / ${esc(ev.model)}`; body = esc(ev.llm_user_message); }
    else if (ev.event === 'nudge') { cls += ' ev-warn'; label = 'guard'; msg = `model stopped early — nudged (${ev.pending.length} pending)`; body = esc(ev.model_said); }
    else if (ev.event === 'report') { label = 'llm'; msg = 'final report'; body = esc(ev.content); }
    else if (ev.event === 'error') { cls += ' ev-err'; label = 'error'; msg = esc(ev.message); }
    else return;
  } else if (ev.type === 'job') {
    label = 'job';
    const fo = S.folders.get(ev.folder_id);
    const scope = fo ? fo.name : 'all folders';
    if (ev.state === 'started') msg = `started · ${esc(scope)} · ${esc(ev.mode)}`;
    else { const s = ev.stats || {}; msg = `finished ${ev.seconds}s · ${s.documents_processed} docs · ${s.verified_pass} pass / ${s.verified_fail} fail`; }
  } else if (ev.type === 'log') {
    if (ev.level === 'error') { cls += ' ev-err'; label = 'error'; }
    msg = esc(ev.message);
  } else return;

  const box = $('#console');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const el = document.createElement('div');
  el.className = cls + (body ? ' has-body' : ''); el.dataset.kind = kind;
  el.innerHTML = `<div class="ev-head"><span class="ev-ts">${fmtTime(ev.ts)}</span><span class="ev-k">${label}</span><span class="msg">${msg}</span></div>${body ? `<div class="ev-body">${body}</div>` : ''}`;
  if (body) el.querySelector('.ev-head').onclick = () => el.classList.toggle('open');
  if (S.filter === 'tool' && kind !== 'tool') el.style.display = 'none';
  box.appendChild(el);
  while (box.children.length > 800) box.firstChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
  $('#logLast').innerHTML = msg;
}
function safeParse(s) { try { return JSON.parse(s); } catch (_) { return s; } }
function renderLeaks() {
  const b = $('#leakBadge');
  b.textContent = `${S.leaks} raw pii out`;
  b.className = `badge ${S.leaks ? 'b-err' : 'b-pass'}`;
}

// ------------------------------------------------------------------ events
function setFolders(list) {
  S.folders = new Map(list.map((f) => [f.folder_id, f]));
  if (!S.folders.has(S.folder)) openFolder(INBOX, false);
}
function handleEvent(ev, live = true) {
  switch (ev.type) {
    case 'file': S.files.set(ev.file.file_id, ev.file); if (!S.selected && ev.file.folder_id === S.folder) S.selected = ev.file.file_id; break;
    case 'file_removed': S.files.delete(ev.file_id); if (S.selected === ev.file_id) S.selected = (folderFiles()[0] || {}).file_id || null; break;
    case 'folders': setFolders(ev.folders); break;
    case 'reset': S.files.clear(); S.selected = null; S.leaks = 0; $('#console').innerHTML = ''; break;
    case 'tool': S.leaks += ev.pii_pattern_hits || 0; break;
    case 'agent': if (ev.event === 'start') S.leaks += ev.pii_pattern_hits || 0; break;
    case 'job':
      S.running = ev.state === 'started';
      S.jobFolder = S.running ? (ev.folder_id ?? null) : null;
      setJob(S.running ? 'run' : 'done', S.running ? `running · ${(S.folders.get(ev.folder_id) || {}).name || 'all'}` : `done · ${ev.seconds}s`);
      if (live && ev.state === 'finished') toast(`${ev.stats.verified_pass} pass · ${ev.stats.verified_fail} fail`);
      break;
  }
  consoleAppend(ev);
}
let renderQueued = false;
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderFolders(); renderFolderHead(); renderFiles(); renderInspector(); renderLeaks();
  });
}
function setJob(kind, text) {
  $('#jobPill').className = `pill pill-${kind}`;
  $('#jobText').textContent = text;
}
function connect() {
  const es = new EventSource('/api/events');
  es.onmessage = (m) => { handleEvent(JSON.parse(m.data)); scheduleRender(); };
  es.onerror = () => setJob('idle', 'reconnecting…');
  es.onopen = () => { if (!S.running) setJob('idle', 'idle'); };
}

// ------------------------------------------------------------------ actions
function openFolder(id, render = true) {
  S.folder = id;
  const files = folderFiles(id);
  if (!files.some((f) => f.file_id === S.selected)) { S.selected = (files[0] || {}).file_id || null; renderInspector._key = null; }
  if (render) scheduleRender();
}
function selectFile(id) {
  if (!S.files.has(id)) return;
  S.selected = id; scheduleRender();
}
async function patchFolder(body) {
  try { await api(`/api/folders/${S.folder}`, { method: 'PATCH', json: body }); }
  catch (e) { toast(e.message, true); scheduleRender(); }
}
async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) => /\.(png|jpe?g|pdf)$/i.test(f.name));
  if (!files.length) return toast('only png, jpg or pdf files are accepted', true);
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  fd.append('folder_id', S.folder);
  try {
    const r = await api('/api/upload', { method: 'POST', body: fd });
    toast(`${r.added.length} doc(s) added${r.auto ? ' — processing' : ''}`);
  } catch (e) { toast(e.message, true); }
}
function settings() {
  return { mode: $('#modeSelect').value, padding_px: +$('#padding').value, use_local_slm: $('#slm').checked, tts: $('#tts').checked };
}
async function saveSettings() {
  try { await api('/api/settings', { method: 'PUT', json: settings() }); } catch (e) { toast(e.message, true); }
}
function updateProviderNote() {
  const mode = $('#modeSelect').value;
  const p = S.providers[mode] || {};
  const note = $('#providerNote');
  note.classList.toggle('hidden', mode === 'direct');
  note.classList.toggle('info', p.available !== false);
  note.textContent = p.available === false ? `${p.label}: ${p.detail || 'unavailable'}` : 'agent mode: an LLM drives the MCP tools for PII folders; local CPU models are slow.';
}
async function run() {
  const fo = currentFolder();
  const p = S.providers[$('#modeSelect').value];
  if (fo.policy === 'pii' && p && p.available === false) return toast(`${p.label} is not available`, true);
  try { await api('/api/process', { method: 'POST', json: { ...settings(), folder_id: fo.folder_id } }); }
  catch (e) { toast(e.message, true); }
}

function bindUi() {
  // folders
  const form = $('#newFolderForm');
  const closeForm = () => { form.classList.add('hidden'); form.reset(); };
  $('#newFolderBtn').onclick = () => { form.classList.toggle('hidden'); if (!form.classList.contains('hidden')) $('#newFolderName').focus(); };
  $('#newFolderCancel').onclick = closeForm;
  $('#newFolderName').onkeydown = (e) => { if (e.key === 'Escape') closeForm(); };
  $$('#newFolderPolicy .seg-btn').forEach((b) => b.onclick = () => {
    S.newPolicy = b.dataset.policy;
    $$('#newFolderPolicy .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
  });
  form.onsubmit = async (e) => {
    e.preventDefault();
    try {
      const fo = await api('/api/folders', { method: 'POST', json: { name: $('#newFolderName').value, policy: S.newPolicy, auto: true } });
      S.folders.set(fo.folder_id, fo);
      closeForm(); openFolder(fo.folder_id);
    } catch (err) { toast(err.message, true); }
  };
  $('#folderList').onclick = (e) => { const li = e.target.closest('[data-folder]'); if (li) openFolder(li.dataset.folder); };
  $('#folderName').onchange = (e) => { const v = e.target.value.trim(); if (v) patchFolder({ name: v }); else scheduleRender(); };
  $('#folderName').onkeydown = (e) => { if (e.key === 'Enter') e.target.blur(); };
  $$('#policySeg .seg-btn').forEach((b) => b.onclick = () => patchFolder({ policy: b.dataset.policy }));
  $('#autoToggle').onchange = (e) => patchFolder({ auto: e.target.checked });
  $('#deleteFolderBtn').onclick = async () => {
    const fo = currentFolder();
    if (!confirm(`Delete folder "${fo.name}"? Its uploaded files are deleted from disk.`)) return;
    try { await api(`/api/folders/${fo.folder_id}`, { method: 'DELETE' }); openFolder(INBOX); } catch (e) { toast(e.message, true); }
  };

  // documents
  const dz = $('#dropZone');
  ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', (e) => uploadFiles(e.dataTransfer.files));
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => { e.preventDefault(); if (!dz.contains(e.target)) uploadFiles(e.dataTransfer.files); });
  $('#fileInput').onchange = (e) => { uploadFiles(e.target.files); e.target.value = ''; };
  $('#demoBtn').onclick = async () => {
    try {
      const r = await api(`/api/load-demo?folder_id=${encodeURIComponent(S.folder)}`, { method: 'POST' });
      toast(r.added.length ? `${r.added.length} synthetic docs added${r.auto ? ' — processing' : ''}` : 'synthetic docs already in this folder');
    } catch (e) { toast(e.message, true); }
  };
  $('#fileList').onclick = async (e) => {
    const rm = e.target.closest('[data-rm]');
    if (rm) { e.stopPropagation(); try { await api(`/api/files/${rm.dataset.rm}`, { method: 'DELETE' }); } catch (err) { toast(err.message, true); } return; }
    const li = e.target.closest('.file'); if (li) selectFile(li.dataset.id);
  };
  $('#runBtn').onclick = run;

  // settings
  $('#modeSelect').onchange = () => { updateProviderNote(); saveSettings(); };
  $('#padding').oninput = (e) => { $('#padVal').textContent = `${e.target.value}px`; };
  $('#padding').onchange = saveSettings;
  $('#slm').onchange = saveSettings;
  $('#tts').onchange = saveSettings;

  // log
  $('#logHead').onclick = (e) => { if (!e.target.closest('button')) $('#logPanel').classList.toggle('collapsed'); };
  $$('#logSeg .seg-btn').forEach((b) => b.onclick = () => {
    $$('#logSeg .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
    S.filter = b.dataset.filter;
    $$('#console .ev').forEach((el) => { el.style.display = S.filter === 'tool' && el.dataset.kind !== 'tool' ? 'none' : ''; });
  });
  $('#clearLog').onclick = () => { $('#console').innerHTML = ''; $('#logLast').textContent = ''; };

  // keyboard
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const ids = folderFiles().map((f) => f.file_id); const i = ids.indexOf(S.selected);
    if (e.key === 'ArrowDown' || e.key === 'j') { selectFile(ids[Math.min(ids.length - 1, i + 1)]); e.preventDefault(); }
    if (e.key === 'ArrowUp' || e.key === 'k') { selectFile(ids[Math.max(0, i - 1)]); e.preventDefault(); }
    if (e.key === 'f') fit();
    if (e.key === 'n') { e.preventDefault(); $('#newFolderBtn').click(); }
  });
}

async function init() {
  bindUi(); setupViewer();
  try {
    const cfg = await api('/api/config');
    S.providers = cfg.providers;
    S.syntheticAvailable = cfg.synthetic_available;
    [...$('#modeSelect').options].forEach((o) => {
      const p = S.providers[o.value];
      if (p && !p.available) { o.disabled = true; o.textContent += ' (unavailable)'; }
    });
  } catch (_) { /* server will report errors on use */ }
  const st = await api('/api/state');
  setFolders(st.folders || []);
  st.files.forEach((f) => S.files.set(f.file_id, f));
  const s = st.settings || {};
  $('#modeSelect').value = s.mode || 'direct';
  $('#padding').value = s.padding_px ?? 4; $('#padVal').textContent = `${$('#padding').value}px`;
  $('#slm').checked = !!s.use_local_slm; $('#tts').checked = !!s.tts;
  updateProviderNote();
  S.running = !!(st.job && st.job.running);
  S.jobFolder = S.running ? st.job.folder_id : null;
  st.events.forEach((ev) => {
    if (ev.type === 'tool') S.leaks += ev.pii_pattern_hits || 0;
    if (ev.type === 'agent' && ev.event === 'start') S.leaks += ev.pii_pattern_hits || 0;
    consoleAppend(ev);
  });
  if (S.running) setJob('run', 'running');
  else if (st.job && st.job.finished) setJob('done', 'done');
  openFolder(INBOX);
  connect();
}
init();
