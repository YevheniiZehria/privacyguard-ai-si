/* PrivacyGuard AI — terminal UI: login -> folder hub -> folder dashboard (vanilla ES2020, no build step) */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const ACTIVE = ['QUEUED', 'SCANNING'];
const FINISHED = ['BLURRED', 'REVIEW'];
const STATUS_LABEL = {
  QUEUED: '[ QUEUED ]', SCANNING: '[ SCANNING... ]', BLURRED: '[ BLURRED ]', REVIEW: '[ REVIEW ]', ERROR: '[ ERROR ]',
  FILLED: '[ FILLED ]', SOURCE: '[ SOURCE ]',
};
const isFilled = (d) => d.role === 'filled';
const hasResult = (d) => FINISHED.includes(d.status) || d.status === 'FILLED';     // something to preview / download
const statusLabel = (d) => (isFilled(d) && ACTIVE.includes(d.status) ? '[ FILLING... ]' : STATUS_LABEL[d.status] || esc(d.status));
const STATUS_HINT = {
  QUEUED: 'Waiting for the redaction worker',
  SCANNING: 'OCR → PII detection → redaction → verification',
  BLURRED: 'Sensitive data redacted and verified by a second OCR pass',
  REVIEW: 'Redacted, but the verification pass still found PII-like text — check the preview',
  FILLED: 'Form filled from the ID card — contains personal data (not included in the blurred .zip)',
  SOURCE: 'Stored as an auto-fill input — not scanned or redacted',
};

const DEFAULT_CALIB = { blur_level: 7, padding_px: 4, mode: 'blur' };
const S = { user: null, view: null, folderId: null, docs: [], seen: new Set(), poll: null, uploading: 0, preview: null,
  calib: { ...DEFAULT_CALIB } };

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pad = (n) => String(n).padStart(2, '0');
function fmtDate(ts) {
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function fmtSize(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = `toast mono${err ? ' err' : ''}`;
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('hidden'), 3200);
}
async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts = { ...opts, body: JSON.stringify(opts.json), headers: { 'Content-Type': 'application/json' } };
    delete opts.json;
  }
  const r = await fetch(path, { credentials: 'same-origin', ...opts });
  if (r.status === 401 && path !== '/api/login') {
    S.user = null; localStorage.removeItem('pg_user');
    if (location.hash !== '#/login') location.hash = '#/login';
    throw new Error('session expired — authenticate again');
  }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return r.json();
}
const folderUrl = (id) => `/api/folders/${encodeURIComponent(id)}`;
const docUrl = (doc) => `${folderUrl(S.folderId)}/documents/${encodeURIComponent(doc.doc_id)}`;

// ------------------------------------------------------------------ effects
function typewrite(el, text = el.dataset.text, speed = 42) {
  clearInterval(el._tw);
  el.textContent = '';
  let i = 0;
  el._tw = setInterval(() => {
    el.textContent = text.slice(0, ++i);
    if (i >= text.length) clearInterval(el._tw);
  }, speed);
}
const GLYPHS = '!<>-_\\/[]{}=+*^?#01ABCDEF';
function scramble(el, duration = 150) {
  if (el._scrambling) return;
  const orig = el.dataset.orig || el.textContent;
  el.dataset.orig = orig;
  el._scrambling = true;
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / duration);
    const fixed = Math.floor(orig.length * p);
    el.textContent = orig.split('').map((c, i) => (i < fixed || c === ' ' ? c : GLYPHS[(Math.random() * GLYPHS.length) | 0])).join('');
    if (p < 1) requestAnimationFrame(step);
    else { el.textContent = orig; el._scrambling = false; }
  };
  requestAnimationFrame(step);
}
document.addEventListener('mouseover', (e) => {
  const el = e.target.closest('[data-scramble], .folder-name');
  if (el && !el.contains(e.relatedTarget)) scramble(el);
});

// ------------------------------------------------------------------ views
function showView(name) {
  const ids = { login: '#viewLogin', hub: '#viewHub', folder: '#viewFolder' };
  const next = $(ids[name]);
  const prev = S.view && S.view !== name ? $(ids[S.view]) : null;
  const swap = () => {
    Object.values(ids).forEach((id) => $(id).classList.add('hidden'));
    $('#topbar').classList.toggle('hidden', name === 'login');
    next.classList.remove('hidden', 'leaving');
    next.style.animation = 'none'; void next.offsetWidth; next.style.animation = '';   // replay fade-in
    window.scrollTo(0, 0);
  };
  S.view = name;
  if (prev && !prev.classList.contains('hidden')) {
    prev.classList.add('leaving');
    setTimeout(swap, 200);
  } else swap();
}
function stopPolling() { clearTimeout(S.poll); S.poll = null; }

async function route() {
  stopPolling();
  closePreview();
  const hash = location.hash || '#/folders';
  if (!S.user) {
    if (hash !== '#/login') { location.hash = '#/login'; return; }
    showView('login');
    typewrite($('#loginType'));
    setTimeout(() => $('#loginUser').focus(), 250);
    return;
  }
  const m = hash.match(/^#\/folder\/(f_[0-9a-f]{8})$/);
  if (m) return openFolder(m[1]);
  if (hash !== '#/folders') { location.hash = '#/folders'; return; }
  openHub();
}

// ------------------------------------------------------------------ STEP 1 · login
async function onLogin(e) {
  e.preventDefault();
  const box = $('#loginForm'), err = $('#loginError');
  err.classList.add('hidden');
  try {
    const r = await api('/api/login', { method: 'POST', json: { username: $('#loginUser').value, passphrase: $('#loginPass').value } });
    S.user = r.username;
    localStorage.setItem('pg_user', r.username);
    $('#loginPass').value = '';
    err.classList.add('hidden');
    $('#loginType').textContent = '> ACCESS_GRANTED';
    setTimeout(() => { location.hash = '#/folders'; }, 350);
  } catch (ex) {
    err.textContent = ex.message === 'Access denied' ? 'ACCESS DENIED — check username / passphrase' : ex.message;
    err.classList.remove('hidden');
    box.classList.remove('shake'); void box.offsetWidth; box.classList.add('shake');
    $('#loginPass').select();
  }
}
async function logout() {
  try { await api('/api/logout', { method: 'POST' }); } catch (_) { /* already logged out */ }
  S.user = null; localStorage.removeItem('pg_user');
  location.hash = '#/login';
}

// ------------------------------------------------------------------ STEP 2 · folder hub
async function openHub() {
  showView('hub');
  $('#whoami').textContent = `${S.user}@localhost`;
  typewrite($('#hubType'));
  await refreshHub(true);
}
async function refreshHub(first = false) {
  let folders;
  try { folders = (await api('/api/folders')).folders; } catch (e) { return toast(e.message, true); }
  if (S.view !== 'hub') return;
  $('#folderCount').textContent = `${folders.length} folder${folders.length === 1 ? '' : 's'}`;
  const grid = $('#folderGrid');
  if (!folders.length) {
    grid.innerHTML = '<li class="empty-hub">// no document sets yet — create one to start</li>';
  } else {
    grid.innerHTML = folders.map((f, i) => `
      <li class="folder-card" data-id="${f.folder_id}" tabindex="0" style="animation-delay:${first ? i * 40 : 0}ms">
        <span class="folder-name">${esc(f.name)}</span>
        <button class="folder-del" data-del="${f.folder_id}" title="Delete folder">✕</button>
        <div class="folder-stats">
          <span>${fmtDate(f.created)}</span>
          <span><b>${f.blurred_count}</b>/${f.document_count} docs</span>
        </div>
        ${f.processing_count ? `<span class="folder-busy">scanning ${f.processing_count}…</span>` : ''}
      </li>`).join('');
  }
  if (folders.some((f) => f.processing_count)) S.poll = setTimeout(refreshHub, 2000);
}
function toggleCreate(open) {
  $('#createFolderForm').classList.toggle('hidden', !open);
  $('#createFolderBtn').classList.toggle('hidden', open);
  if (open) $('#folderNameInput').focus(); else $('#createFolderForm').reset();
}
async function onCreateFolder(e) {
  e.preventDefault();
  try {
    const f = await api('/api/folders', { method: 'POST', json: { name: $('#folderNameInput').value } });
    toggleCreate(false);
    location.hash = `#/folder/${f.folder_id}`;
  } catch (ex) { toast(ex.message, true); }
}
async function deleteFolder(id) {
  const card = $(`.folder-card[data-id="${id}"] .folder-name`);
  const name = card ? card.dataset.orig || card.textContent : id;
  if (!confirm(`Delete "${name}"? Its originals and redacted files are removed from disk.`)) return;
  try { await api(folderUrl(id), { method: 'DELETE' }); toast(`deleted ${name}`); stopPolling(); refreshHub(); }
  catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ blur calibration
const levelName = (n) => (n <= 3 ? 'LOW' : n <= 5 ? 'MED' : n <= 8 ? 'HIGH' : 'MAX');
const calibText = (c) => (c.mode === 'black' ? `SOLID BLACK · ${c.padding_px}px` : `BLUR ${levelName(c.blur_level)} ${c.blur_level}/10 · ${c.padding_px}px`);
function readCalib(root) {
  return {
    blur_level: +root.querySelector('[data-k="blur_level"]').value,
    padding_px: +root.querySelector('[data-k="padding_px"]').value,
    mode: root.querySelector('[data-k="mode"] button.active').dataset.mode,
  };
}
function paintCalib(root) {
  const c = readCalib(root);
  root.querySelector('[data-v="blur_level"]').textContent = `${levelName(c.blur_level)} · ${c.blur_level}`;
  root.querySelector('[data-v="padding_px"]').textContent = `${c.padding_px} px`;
  root.classList.toggle('black-mode', c.mode === 'black');
  const sum = root.querySelector('[data-v="summary"]');
  if (sum) sum.textContent = `default: ${calibText(S.calib)}`;
}
function fillCalib(root, cal) {
  const c = { ...DEFAULT_CALIB, ...(cal || {}) };
  root.querySelector('[data-k="blur_level"]').value = c.blur_level;
  root.querySelector('[data-k="padding_px"]').value = c.padding_px;
  root.querySelectorAll('[data-k="mode"] button').forEach((b) => b.classList.toggle('active', b.dataset.mode === c.mode));
  paintCalib(root);
}
function bindCalib(root) {
  root.querySelectorAll('.range').forEach((r) => r.addEventListener('input', () => paintCalib(root)));
  root.querySelectorAll('[data-k="mode"] button').forEach((b) => b.addEventListener('click', () => {
    root.querySelectorAll('[data-k="mode"] button').forEach((x) => x.classList.toggle('active', x === b));
    paintCalib(root);
  }));
}
async function loadFolderCalib(id) {
  try {
    const r = await api(`${folderUrl(id)}/calibration`);
    if (S.folderId !== id) return;
    S.calib = r.calibration;
  } catch (_) { S.calib = { ...DEFAULT_CALIB }; }
  fillCalib($('#calibFolder'), S.calib);
}
async function onFolderCalib(e) {
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const root = $('#calibFolder');
  if (btn.dataset.act === 'toggle') {
    const open = !root.classList.toggle('collapsed');
    btn.setAttribute('aria-expanded', String(open));
    return;
  }
  const cal = readCalib(root);
  try {
    if (btn.dataset.act === 'save') {
      S.calib = (await api(`${folderUrl(S.folderId)}/calibration`, { method: 'PUT', json: cal })).calibration;
      paintCalib(root);
      toast(`default saved — ${calibText(S.calib)}`);
    }
    if (btn.dataset.act === 'apply-all') {
      const r = await api(`${folderUrl(S.folderId)}/reapply`, { method: 'POST', json: { calibration: cal, save_as_default: true } });
      S.calib = r.calibration; paintCalib(root);
      toast(r.queued ? `re-applying ${calibText(cal)} to ${r.queued} document(s)` : 'no redacted documents yet — saved as default');
      stopPolling(); refreshFolder();
    }
  } catch (ex) { toast(ex.message, true); }
}
function paintDocCalib() {
  const p = S.preview, root = $('#calibDoc');
  if (!p) return;
  const finished = FINISHED.includes(p.doc.status), active = ACTIVE.includes(p.doc.status);
  root.classList.toggle('hidden', Boolean(p.doc.role) || !(finished || (active && p.doc.processed)));
  const st = root.querySelector('[data-v="status"]');
  st.className = `status st-${p.doc.status}`;
  st.textContent = active ? '[ RE-VERIFYING... ]' : (STATUS_LABEL[p.doc.status] || p.doc.status);
  st.title = STATUS_HINT[p.doc.status] || '';
  root.querySelector('[data-act="apply"]').classList.toggle('disabled', !finished);
}
async function onDocCalibApply() {
  const p = S.preview;
  if (!p) return;
  const root = $('#calibDoc');
  const cal = readCalib(root);
  const saveDefault = root.querySelector('[data-k="save_as_default"]').checked;
  try {
    const doc = await api(`${docUrl(p.doc)}/reapply`, { method: 'POST', json: { calibration: cal, save_as_default: saveDefault } });
    p.doc = doc; p.awaiting = true;
    if (saveDefault) { S.calib = cal; fillCalib($('#calibFolder'), cal); }
    paintDocCalib();
    toast(`re-applying ${calibText(cal)}…`);
    stopPolling(); refreshFolder();
  } catch (ex) { toast(ex.message, true); }
}
// The preview follows its document while it is re-applied: the new image loads with zoom / pan kept.
function syncPreviewDoc() {
  const p = S.preview;
  if (!p) return;
  const d = S.docs.find((x) => x.doc_id === p.doc.doc_id);
  if (!d || (d.processed === p.doc.processed && d.status === p.doc.status)) return;
  p.doc = d;
  paintDocCalib();
  if (!hasResult(d)) return;
  if (isFilled(d)) { p.pages = d.page_count || 1; renderPreview(); return; }
  $('#manualBtn').classList.remove('hidden');
  $('#manualUndo').classList.toggle('hidden', !(d.manual_boxes && d.manual_boxes.length));
  if ($('#imgBlurred').classList.contains('hidden') || !p.tf) renderPreview();
  else $('#imgBlurred').src = `${docUrl(d)}/image/blurred?page=${p.page}&v=${d.processed || Date.now()}`;
  if (p.awaiting) {
    p.awaiting = false;
    const review = d.status === 'REVIEW';
    toast(`re-verified ${STATUS_LABEL[d.status]}${review ? ` — ${d.residual_pii_count} residual hit(s): raise strength / padding` : ''}`, review);
  }
}

// ------------------------------------------------------------------ STEP 3 · folder dashboard
async function openFolder(id) {
  S.folderId = id; S.docs = []; S.seen = new Set();
  $('#docRows').innerHTML = '';
  $('#folderTitle').textContent = ''; delete $('#folderTitle').dataset.orig;
  showView('folder');
  loadFolderCalib(id);
  await refreshFolder(true);
}
async function refreshFolder(first = false) {
  const id = S.folderId;
  let data;
  try { data = await api(`${folderUrl(id)}/documents`); }
  catch (e) { toast(e.message, true); if (/Unknown folder/.test(e.message)) location.hash = '#/folders'; return; }
  if (S.view !== 'folder' || S.folderId !== id) return;
  const f = data.folder;
  if (first) {
    typewrite($('#folderTitle'), f.name, 28);
    $('#folderTitle').dataset.orig = f.name;
    document.title = `${f.name} — PrivacyGuard AI`;
  }
  $('#folderMeta').textContent = `${f.folder_id} · created ${fmtDate(f.created)} · ${f.blurred_count}/${f.document_count} blurred${f.filled_count ? ` · ${f.filled_count} filled` : ''}`;
  S.docs = data.documents;
  renderDocs();
  syncPreviewDoc();
  const busy = S.docs.some((d) => ACTIVE.includes(d.status));
  if (busy) S.poll = setTimeout(refreshFolder, 1000);
}
function renderDocs() {
  const busy = S.uploading > 0 || S.docs.some((d) => ACTIVE.includes(d.status));
  $('#dropZone').classList.toggle('scanning', busy);
  $('#dropTitle').textContent = S.uploading ? 'UPLOADING…' : busy ? 'SCANNING DOCUMENTS…' : 'DROP FILES TO REDACT';
  const done = S.docs.filter((d) => FINISHED.includes(d.status));
  const dl = $('#downloadAll');
  dl.classList.toggle('disabled', !done.length);
  dl.href = done.length ? `${folderUrl(S.folderId)}/download-all` : '#';

  $('#docEmpty').classList.toggle('hidden', S.docs.length > 0);
  const sig = JSON.stringify(S.docs.map((d) => [d.doc_id, d.status, d.role]));
  if (renderDocs._sig === sig) return;
  renderDocs._sig = sig;
  $('#docRows').innerHTML = S.docs.map((d) => {
    const isNew = !S.seen.has(d.doc_id);
    S.seen.add(d.doc_id);
    const finished = hasResult(d);
    const active = ACTIVE.includes(d.status);
    const filledHint = d.status === 'FILLED' && d.filled_fields ? ` · filled: ${d.filled_fields.join(', ')}` : '';
    const hint = d.status === 'ERROR' ? d.error : (STATUS_HINT[d.status] || '') + filledHint;
    const extra = d.status === 'FILLED'
      ? '<button class="link-act" data-act="anon" title="Copy the filled form and redact the copy">BLUR COPY</button>'
      : d.status === 'SOURCE'
        ? '<button class="link-act" data-act="redact" title="Scan and redact this document">REDACT</button>'
        : `<button class="link-act ${active ? 'disabled' : ''}" data-act="rescan" title="Run detection and redaction again">RESCAN</button>`;
    const pages = d.page_count > 1 ? ` · ${d.page_count}p` : '';
    return `<tr class="${isNew ? 'new' : ''}" data-id="${d.doc_id}">
      <td class="doc-name" title="${esc(d.name)}">${esc(d.name)}</td>
      <td class="doc-size">${fmtSize(d.size)}${pages}</td>
      <td><span class="status st-${d.status}" title="${esc(hint || '')}">${statusLabel(d)}</span></td>
      <td><div class="doc-actions">
        <button class="link-act" data-act="preview">PREVIEW</button>
        <a class="link-act ${finished ? '' : 'disabled'}" href="${finished ? `${docUrl(d)}/download` : '#'}" data-act="download">DOWNLOAD</a>
        ${extra}
        <button class="link-act danger" data-act="delete" title="Remove document">✕</button>
      </div></td>
    </tr>`;
  }).join('');
}
async function uploadFiles(list) {
  const files = [...list].filter((f) => /\.(pdf|png|jpe?g)$/i.test(f.name));
  if (!files.length) return toast('only PDF, PNG or JPG files are accepted', true);
  if (files.length < list.length) toast(`${list.length - files.length} unsupported file(s) skipped`, true);
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  S.uploading++; renderDocs();
  try {
    const r = await api(`${folderUrl(S.folderId)}/upload`, { method: 'POST', body: fd });
    toast(`${r.added.length} file(s) queued for redaction`);
  } catch (e) { toast(e.message, true); }
  S.uploading--;
  stopPolling();
  await refreshFolder();
}
async function onDocAction(e) {
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const doc = S.docs.find((d) => d.doc_id === btn.closest('tr').dataset.id);
  if (!doc) return;
  if (btn.dataset.act === 'preview') openPreview(doc);
  if (btn.dataset.act === 'rescan') {
    try { await api(`${docUrl(doc)}/rescan`, { method: 'POST' }); toast(`rescanning ${doc.name}`); stopPolling(); refreshFolder(); }
    catch (ex) { toast(ex.message, true); }
  }
  if (btn.dataset.act === 'anon' || btn.dataset.act === 'redact') {
    const anon = btn.dataset.act === 'anon';
    try {
      const r = await api(`${docUrl(doc)}/${anon ? 'anonymize-copy' : 'redact'}`, { method: 'POST' });
      toast(anon ? `anonymized copy queued: ${r.name}` : `redacting ${doc.name}`); stopPolling(); refreshFolder();
    } catch (ex) { toast(ex.message, true); }
  }
  if (btn.dataset.act === 'delete') {
    if (!confirm(`Remove "${doc.name}" and its output?`)) return;
    try { await api(docUrl(doc), { method: 'DELETE' }); stopPolling(); refreshFolder(); } catch (ex) { toast(ex.message, true); }
  }
}

// ------------------------------------------------------------------ preview
function openPreview(doc) {
  const finished = hasResult(doc);
  S.preview = { doc, page: 1, pages: doc.page_count || 1, mode: finished ? 'split' : 'original',
    tf: null, size: null, draw: false };
  $('#previewTitle').textContent = doc.name;
  $('#manualBtn').classList.toggle('hidden', !FINISHED.includes(doc.status));
  $('#manualBtn').classList.remove('active');
  $('#previewBody').classList.remove('drawing');
  $('#manualUndo').classList.toggle('hidden', !(doc.manual_boxes && doc.manual_boxes.length));
  fillCalib($('#calibDoc'), doc.calibration || S.calib);
  $('#calibDoc [data-k="save_as_default"]').checked = false;
  paintDocCalib();
  $('#preview').classList.remove('hidden');
  renderPreview();
}
function renderPreview() {
  const p = S.preview;
  if (!p) return;
  const finished = hasResult(p.doc);
  const filled = isFilled(p.doc);
  const names = filled ? { original: 'BLANK', blurred: 'FILLED' } : { original: 'ORIGINAL', blurred: 'BLURRED' };
  $('.pane-original figcaption').textContent = names.original;
  $('.pane-blurred figcaption').textContent = names.blurred;
  $$('#previewSeg button').forEach((b) => {
    if (names[b.dataset.mode]) b.textContent = names[b.dataset.mode];
    b.classList.toggle('active', b.dataset.mode === p.mode);
    b.disabled = !finished && b.dataset.mode !== 'original';
    b.style.opacity = b.disabled ? .3 : '';
  });
  $('#previewBody').className = `modal-body mode-${p.mode}`;
  $('#pager').classList.toggle('hidden', p.pages < 2);
  $('#pageLabel').textContent = `${p.page}/${p.pages}`;
  const v = p.doc.processed || p.doc.created;
  p.tf = null; p.size = null;                       // new page -> fit again once it has loaded
  for (const [id, kind] of [['#imgOriginal', 'original'], ['#imgBlurred', filled ? 'filled' : 'blurred']]) {
    const img = $(id);
    const pane = img.parentElement;
    pane.querySelector('.no-img')?.remove();
    img.classList.remove('hidden', 'sized');
    img.style.transform = '';
    img.onload = () => {
      if (!S.preview) return;
      if (!p.size) p.size = { w: img.naturalWidth, h: img.naturalHeight };
      if (!p.tf) fitPreview(); else applyPreviewTransform();
    };
    img.onerror = () => {
      img.classList.add('hidden');
      if (!pane.querySelector('.no-img')) pane.insertAdjacentHTML('beforeend', `<span class="no-img">// ${kind} not available${kind === 'original' && !finished ? ' until the scan finishes' : ''}</span>`);
    };
    img.src = `${docUrl(p.doc)}/image/${kind}?page=${p.page}&v=${v}`;
  }
}
// Both panes share one transform, so the original and redacted page stay aligned while zooming / panning.
function previewPane() {
  return [...$$('#previewBody .pane')].find((el) => el.offsetParent !== null) || $('#previewBody .pane');
}
function fitPreview() {
  const p = S.preview;
  if (!p || !p.size) return;
  const pane = previewPane();
  const pw = pane.clientWidth, ph = pane.clientHeight, m = 24;
  const scale = Math.min((pw - 2 * m) / p.size.w, (ph - 2 * m) / p.size.h, 1);   // never upscale on fit
  p.tf = { scale, x: (pw - p.size.w * scale) / 2, y: (ph - p.size.h * scale) / 2, fit: true };
  applyPreviewTransform();
}
function applyPreviewTransform() {
  const p = S.preview;
  if (!p || !p.tf) return;
  const { scale, x, y } = p.tf;
  for (const img of [$('#imgOriginal'), $('#imgBlurred')]) {
    if (!img.naturalWidth) continue;
    img.classList.add('sized');
    img.style.width = `${p.size.w}px`; img.style.height = `${p.size.h}px`;
    img.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  }
  $('#zoomVal').textContent = `${Math.round(scale * 100)}%`;
}
function zoomPreview(factor, cx, cy) {
  const p = S.preview;
  if (!p || !p.tf) return;
  const pane = previewPane();
  if (cx === undefined) { cx = pane.clientWidth / 2; cy = pane.clientHeight / 2; }
  const t = p.tf;
  const ns = Math.min(8, Math.max(0.05, t.scale * factor));
  const k = ns / t.scale;
  p.tf = { scale: ns, x: cx - (cx - t.x) * k, y: cy - (cy - t.y) * k, fit: false };
  applyPreviewTransform();
}
// ---- manual blur box: drag a rectangle on either pane; it is blurred on the redacted page right away
function setDrawMode(on) {
  if (!S.preview) return;
  S.preview.draw = on;
  $('#manualBtn').classList.toggle('active', on);
  $('#previewBody').classList.toggle('drawing', on);
  $$('.draw-hint').forEach((el) => el.remove());
  if (on) $('.pane-blurred').insertAdjacentHTML('beforeend', '<span class="draw-hint">drag over the area to blur · esc to cancel</span>');
}
function toImage(pane, clientX, clientY) {                // screen -> page pixel coordinates
  const r = pane.getBoundingClientRect(), t = S.preview.tf;
  return { x: (clientX - r.left - t.x) / t.scale, y: (clientY - r.top - t.y) / t.scale };
}
async function applyManualBox(a, b) {
  const p = S.preview;
  const x = Math.max(0, Math.round(Math.min(a.x, b.x))), y = Math.max(0, Math.round(Math.min(a.y, b.y)));
  const w = Math.round(Math.min(p.size.w, Math.max(a.x, b.x)) - x), h = Math.round(Math.min(p.size.h, Math.max(a.y, b.y)) - y);
  if (w < 3 || h < 3) return;
  try {
    const doc = await api(`${docUrl(p.doc)}/manual-redact`, { method: 'POST', json: { page: p.page, x, y, w, h } });
    p.doc = doc;
    const keep = p.tf;
    $('#imgBlurred').src = `${docUrl(doc)}/image/blurred?page=${p.page}&v=${Date.now()}`;   // reload, keep zoom
    p.tf = keep;
    $('#manualUndo').classList.remove('hidden');
    toast(`blurred ${w}×${h}px`);
    const i = S.docs.findIndex((d) => d.doc_id === doc.doc_id);
    if (i >= 0) S.docs[i] = doc;
  } catch (e) { toast(e.message, true); }
}
function setupPreviewZoom() {
  let drag = null;
  for (const pane of $$('#previewBody .pane')) {
    pane.addEventListener('wheel', (e) => {
      if (!S.preview || !S.preview.tf) return;
      e.preventDefault();
      const r = pane.getBoundingClientRect();
      zoomPreview(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    pane.addEventListener('pointerdown', (e) => {
      if (!S.preview || !S.preview.tf) return;
      pane.setPointerCapture(e.pointerId);
      if (S.preview.draw) {
        const rect = document.createElement('div');
        rect.className = 'draw-rect';
        pane.appendChild(rect);
        drag = { pane, mode: 'draw', start: toImage(pane, e.clientX, e.clientY), sx: e.clientX, sy: e.clientY, rect };
        return;
      }
      drag = { pane, mode: 'pan', x: e.clientX, y: e.clientY, tx: S.preview.tf.x, ty: S.preview.tf.y };
      pane.classList.add('panning');
    });
    pane.addEventListener('pointermove', (e) => {
      if (!drag || drag.pane !== pane || !S.preview) return;
      if (drag.mode === 'draw') {
        const r = pane.getBoundingClientRect();
        Object.assign(drag.rect.style, {
          left: `${Math.min(drag.sx, e.clientX) - r.left}px`, top: `${Math.min(drag.sy, e.clientY) - r.top}px`,
          width: `${Math.abs(e.clientX - drag.sx)}px`, height: `${Math.abs(e.clientY - drag.sy)}px`,
        });
        return;
      }
      S.preview.tf = { ...S.preview.tf, x: drag.tx + e.clientX - drag.x, y: drag.ty + e.clientY - drag.y, fit: false };
      applyPreviewTransform();
    });
    const end = (e) => {
      if (drag && drag.mode === 'draw') {
        drag.rect.remove();
        if (e.type === 'pointerup') applyManualBox(drag.start, toImage(pane, e.clientX, e.clientY));
      }
      drag = null; pane.classList.remove('panning');
    };
    pane.addEventListener('pointerup', end);
    pane.addEventListener('pointercancel', end);
    pane.addEventListener('dblclick', () => { if (!S.preview || !S.preview.draw) fitPreview(); });
  }
  $('#manualBtn').onclick = () => setDrawMode(!(S.preview && S.preview.draw));
  $('#manualUndo').onclick = async () => {
    if (!S.preview || !confirm('Remove all manual blur boxes? The document is processed again.')) return;
    try {
      const doc = await api(`${docUrl(S.preview.doc)}/manual-redact`, { method: 'DELETE' });
      S.preview.doc = doc; S.preview.awaiting = true; paintDocCalib();
      $('#manualUndo').classList.add('hidden');
      toast('manual boxes removed — re-applying'); stopPolling(); refreshFolder();
    }
    catch (e) { toast(e.message, true); }
  };
  $('#zoomIn').onclick = () => zoomPreview(1.25);
  $('#zoomOut').onclick = () => zoomPreview(0.8);
  $('#zoomFit').onclick = fitPreview;
  $('#zoom100').onclick = () => { if (S.preview && S.preview.tf) zoomPreview(1 / S.preview.tf.scale); };
  window.addEventListener('resize', () => { if (S.preview && (!S.preview.tf || S.preview.tf.fit)) fitPreview(); });
}
function closePreview() {
  S.preview = null;
  $('#preview').classList.add('hidden');
  $('#imgOriginal').removeAttribute('src'); $('#imgBlurred').removeAttribute('src');
}

// ------------------------------------------------------------------ auto-fill from ID card
const AF = { uploadTarget: null };
function afOptions(sel, keep) {
  const docs = S.docs.filter((d) => !isFilled(d) && d.status !== 'ERROR');
  sel.innerHTML = docs.length
    ? docs.map((d) => `<option value="${d.doc_id}">${esc(d.name)}${d.status === 'SOURCE' ? '' : `  ${STATUS_LABEL[d.status] || ''}`}</option>`).join('')
    : '<option value="" disabled selected>// no documents — upload one</option>';
  if (keep && docs.some((d) => d.doc_id === keep)) sel.value = keep;
}
function openAutofill() {
  const idSel = $('#afId'), formSel = $('#afFormDoc');
  afOptions(idSel, idSel.value); afOptions(formSel, formSel.value);
  if (idSel.value && idSel.value === formSel.value && formSel.options.length > 1) formSel.selectedIndex = 1;
  const date = $('[data-extra="date"]');
  if (!date.value) { const d = new Date(); date.value = `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`; }
  $('#autofill').classList.remove('hidden');
  idSel.focus();
}
const closeAutofill = () => $('#autofill').classList.add('hidden');
async function afUpload(file) {
  const fd = new FormData();
  fd.append('files', file);
  try {
    const r = await api(`${folderUrl(S.folderId)}/upload?source=1`, { method: 'POST', body: fd });
    await refreshFolder();
    const sel = AF.uploadTarget === 'id' ? $('#afId') : $('#afFormDoc');
    afOptions($('#afId'), $('#afId').value); afOptions($('#afFormDoc'), $('#afFormDoc').value);
    sel.value = r.added[0].doc_id;
    toast(`stored ${r.added[0].name} as source (not redacted)`);
  } catch (e) { toast(e.message, true); }
}
async function onAutofillSubmit(e) {
  e.preventDefault();
  const id = $('#afId').value, form = $('#afFormDoc').value;
  if (!id || !form) return toast('pick an ID card and a form', true);
  if (id === form) return toast('the ID card and the form must be different documents', true);
  const extras = {};
  $$('[data-extra]').forEach((i) => { if (i.value.trim()) extras[i.dataset.extra] = i.value.trim(); });
  try {
    const r = await api(`${folderUrl(S.folderId)}/autofill`, { method: 'POST', json: { id_doc_id: id, form_doc_id: form, extras } });
    closeAutofill();
    $$('[data-extra]').forEach((i) => { if (i.dataset.extra !== 'date') i.value = ''; });   // extras are not kept
    toast(`filling ${r.name}…`); stopPolling(); refreshFolder();
  } catch (ex) { toast(ex.message, true); }
}

// ------------------------------------------------------------------ wiring
function bind() {
  $('#loginForm').addEventListener('submit', onLogin);
  $('#logoutBtn').onclick = logout;

  $('#createFolderBtn').onclick = () => toggleCreate(true);
  $('#createFolderCancel').onclick = () => toggleCreate(false);
  $('#folderNameInput').onkeydown = (e) => { if (e.key === 'Escape') toggleCreate(false); };
  $('#createFolderForm').addEventListener('submit', onCreateFolder);
  $('#folderGrid').onclick = (e) => {
    const del = e.target.closest('[data-del]');
    if (del) { e.stopPropagation(); deleteFolder(del.dataset.del); return; }
    const card = e.target.closest('.folder-card');
    if (card) location.hash = `#/folder/${card.dataset.id}`;
  };
  $('#folderGrid').onkeydown = (e) => {
    const card = e.target.closest('.folder-card');
    if (card && e.key === 'Enter') location.hash = `#/folder/${card.dataset.id}`;
  };

  const dz = $('#dropZone');
  ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', (e) => uploadFiles(e.dataTransfer.files));
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    if (S.view === 'folder' && !dz.contains(e.target) && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
  $('#fileInput').onchange = (e) => { uploadFiles(e.target.files); e.target.value = ''; };
  $('#docRows').onclick = onDocAction;
  $('#autofillBtn').onclick = openAutofill;
  $('#afForm').addEventListener('submit', onAutofillSubmit);
  $('#autofill').addEventListener('click', (e) => {
    const act = e.target.closest('[data-af]')?.dataset.af;
    if (e.target.id === 'autofill' || act === 'close') closeAutofill();
    if (act === 'upload-id' || act === 'upload-form') { AF.uploadTarget = act === 'upload-id' ? 'id' : 'form'; $('#afFile').click(); }
  });
  $('#afFile').onchange = (e) => { if (e.target.files[0]) afUpload(e.target.files[0]); e.target.value = ''; };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#autofill').classList.contains('hidden')) closeAutofill(); });
  $('#calibDoc .calib-actions').insertAdjacentHTML('afterbegin', '<span class="status" data-v="status"></span>');
  bindCalib($('#calibFolder'));
  bindCalib($('#calibDoc'));
  $('#calibFolder').addEventListener('click', onFolderCalib);
  $('#calibDoc [data-act="apply"]').onclick = onDocCalibApply;

  $$('#previewSeg button').forEach((b) => b.onclick = () => {
    if (!S.preview) return;
    S.preview.mode = b.dataset.mode;
    $$('#previewSeg button').forEach((x) => x.classList.toggle('active', x === b));
    $('#previewBody').className = `modal-body mode-${S.preview.mode}`;
    fitPreview();                                   // pane width changes between single and side-by-side
  });
  setupPreviewZoom();
  $('#pagePrev').onclick = () => { if (S.preview && S.preview.page > 1) { S.preview.page--; renderPreview(); } };
  $('#pageNext').onclick = () => { if (S.preview && S.preview.page < S.preview.pages) { S.preview.page++; renderPreview(); } };
  $('#previewClose').onclick = closePreview;
  $('#preview').onclick = (e) => { if (e.target.id === 'preview') closePreview(); };
  document.addEventListener('keydown', (e) => {
    if (!S.preview) return;
    if (e.target.closest && e.target.closest('.calib') && e.key !== 'Escape') return;   // sliders own their keys
    if (e.key === 'Escape') { if (S.preview.draw) setDrawMode(false); else closePreview(); return; }
    if (e.key === 'b' || e.key === 'B') $('#manualBtn').click();
    if (e.key === 'ArrowRight') $('#pageNext').click();
    if (e.key === 'ArrowLeft') $('#pagePrev').click();
    if (e.key === '+' || e.key === '=') zoomPreview(1.25);
    if (e.key === '-') zoomPreview(0.8);
    if (e.key === '0') fitPreview();
    if (e.key === '1') $('#zoom100').click();
  });
  window.addEventListener('hashchange', route);
}

async function init() {
  bind();
  $('#loginUser').value = localStorage.getItem('pg_user') || '';
  try {
    const s = await api('/api/session');
    S.user = s.authenticated ? s.username : null;
  } catch (_) { S.user = null; }
  route();
}
init();
