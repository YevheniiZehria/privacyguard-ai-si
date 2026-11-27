/* PrivacyGuard AI — terminal UI: login -> folder hub -> folder dashboard (vanilla ES2020, no build step) */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const ACTIVE = ['QUEUED', 'SCANNING'];
const FINISHED = ['BLURRED', 'REVIEW'];
const STATUS_LABEL = {
  QUEUED: '[ QUEUED ]', SCANNING: '[ SCANNING... ]', BLURRED: '[ BLURRED ]', REVIEW: '[ REVIEW ]', ERROR: '[ ERROR ]',
};
const STATUS_HINT = {
  QUEUED: 'Waiting for the redaction worker',
  SCANNING: 'OCR → PII detection → redaction → verification',
  BLURRED: 'Sensitive data redacted and verified by a second OCR pass',
  REVIEW: 'Redacted, but the verification pass still found PII-like text — check the preview',
};

const S = { user: null, view: null, folderId: null, docs: [], seen: new Set(), poll: null, uploading: 0, preview: null };

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

// ------------------------------------------------------------------ STEP 3 · folder dashboard
async function openFolder(id) {
  S.folderId = id; S.docs = []; S.seen = new Set();
  $('#docRows').innerHTML = '';
  $('#folderTitle').textContent = ''; delete $('#folderTitle').dataset.orig;
  showView('folder');
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
  $('#folderMeta').textContent = `${f.folder_id} · created ${fmtDate(f.created)} · ${f.blurred_count}/${f.document_count} blurred`;
  S.docs = data.documents;
  renderDocs();
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
  const sig = JSON.stringify(S.docs.map((d) => [d.doc_id, d.status]));
  if (renderDocs._sig === sig) return;
  renderDocs._sig = sig;
  $('#docRows').innerHTML = S.docs.map((d) => {
    const isNew = !S.seen.has(d.doc_id);
    S.seen.add(d.doc_id);
    const finished = FINISHED.includes(d.status);
    const hint = d.status === 'ERROR' ? d.error : STATUS_HINT[d.status];
    const pages = d.page_count > 1 ? ` · ${d.page_count}p` : '';
    return `<tr class="${isNew ? 'new' : ''}" data-id="${d.doc_id}">
      <td class="doc-name" title="${esc(d.name)}">${esc(d.name)}</td>
      <td class="doc-size">${fmtSize(d.size)}${pages}</td>
      <td><span class="status st-${d.status}" title="${esc(hint || '')}">${STATUS_LABEL[d.status] || esc(d.status)}</span></td>
      <td><div class="doc-actions">
        <button class="link-act" data-act="preview">PREVIEW</button>
        <a class="link-act ${finished ? '' : 'disabled'}" href="${finished ? `${docUrl(d)}/download` : '#'}" data-act="download">DOWNLOAD</a>
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
  if (btn.dataset.act === 'delete') {
    if (!confirm(`Remove "${doc.name}" and its redacted output?`)) return;
    try { await api(docUrl(doc), { method: 'DELETE' }); stopPolling(); refreshFolder(); } catch (ex) { toast(ex.message, true); }
  }
}

// ------------------------------------------------------------------ preview
function openPreview(doc) {
  const finished = FINISHED.includes(doc.status);
  S.preview = { doc, page: 1, pages: doc.page_count || 1, mode: finished ? 'split' : 'original' };
  $('#previewTitle').textContent = doc.name;
  $('#preview').classList.remove('hidden');
  renderPreview();
}
function renderPreview() {
  const p = S.preview;
  if (!p) return;
  const finished = FINISHED.includes(p.doc.status);
  $$('#previewSeg button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === p.mode);
    b.disabled = !finished && b.dataset.mode !== 'original';
    b.style.opacity = b.disabled ? .3 : '';
  });
  $('#previewBody').className = `modal-body mode-${p.mode}`;
  $('#pager').classList.toggle('hidden', p.pages < 2);
  $('#pageLabel').textContent = `${p.page}/${p.pages}`;
  const v = p.doc.processed || p.doc.created;
  for (const [id, kind] of [['#imgOriginal', 'original'], ['#imgBlurred', 'blurred']]) {
    const img = $(id);
    const pane = img.parentElement;
    pane.querySelector('.no-img')?.remove();
    img.classList.remove('hidden');
    img.onerror = () => {
      img.classList.add('hidden');
      if (!pane.querySelector('.no-img')) pane.insertAdjacentHTML('beforeend', `<span class="no-img">// ${kind} not available${kind === 'original' && !finished ? ' until the scan finishes' : ''}</span>`);
    };
    img.src = `${docUrl(p.doc)}/image/${kind}?page=${p.page}&v=${v}`;
  }
}
function closePreview() {
  S.preview = null;
  $('#preview').classList.add('hidden');
  $('#imgOriginal').removeAttribute('src'); $('#imgBlurred').removeAttribute('src');
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

  $$('#previewSeg button').forEach((b) => b.onclick = () => { if (S.preview) { S.preview.mode = b.dataset.mode; renderPreview(); } });
  $('#pagePrev').onclick = () => { if (S.preview && S.preview.page > 1) { S.preview.page--; renderPreview(); } };
  $('#pageNext').onclick = () => { if (S.preview && S.preview.page < S.preview.pages) { S.preview.page++; renderPreview(); } };
  $('#previewClose').onclick = closePreview;
  $('#preview').onclick = (e) => { if (e.target.id === 'preview') closePreview(); };
  document.addEventListener('keydown', (e) => {
    if (!S.preview) return;
    if (e.key === 'Escape') closePreview();
    if (e.key === 'ArrowRight') $('#pageNext').click();
    if (e.key === 'ArrowLeft') $('#pagePrev').click();
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
