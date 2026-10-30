'use strict';
const $ = (s) => document.querySelector(s);
const cssvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------------------------------------------------------- theme
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  try { localStorage.setItem('taTheme', t); } catch (e) { /* private mode */ }
  redraw();
}
$('#theme').onclick = () => applyTheme(
  document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
let saved = null;
try { saved = localStorage.getItem('taTheme'); } catch (e) { /* ignore */ }
document.documentElement.setAttribute('data-theme', saved ||
  (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

// ---------------------------------------------------------------- state
let ready = false, busy = false;
let file = null;          // { name, bytes }
let scan = null;          // preview from the worker
let result = null;
const worker = new Worker('worker.js?v=4');

const plotMain = new Plot($('#cvMain'), { logY: true, xlabel: '2θ (deg)', ylabel: 'intensity (cps)' });
const plotResid = new Plot($('#cvResid'), { xlabel: '2θ (deg)', ylabel: 'resid / σ', yZeroLine: true, legend: false });

function redraw() {
  if (!scan) { plotMain.setSeries([]); return; }
  const s = [];
  const r = fitRange();
  const shade = 'color-mix(in srgb, ' + cssvar('--muted') + ' 22%, transparent)';
  if (isFinite(r.tmin) && r.tmin > scan.range[0]) s.push({ style: 'span', x0: scan.range[0], x1: r.tmin, color: shade });
  if (isFinite(r.tmax) && r.tmax < scan.range[1]) s.push({ style: 'span', x0: r.tmax, x1: scan.range[1], color: shade, label: 'excluded' });
  s.push({ x: scan.two_theta, y: scan.cps, color: cssvar('--data'), style: 'dots', radius: 1.3, label: 'data' });
  if (result) {
    s.push({ x: result.curves.two_theta, y: result.curves.fit, color: cssvar('--fit'), width: 1.7, label: 'fit' });
    plotResid.setSeries([{ x: result.curves.two_theta, y: result.curves.residual, color: cssvar('--accent'), width: 1 }]);
  }
  plotMain.setSeries(s);
}

// ---------------------------------------------------------------- progress / errors
function showProgress(text, frac) {
  $('#progress').hidden = false;
  $('#progText').textContent = text;
  $('#barfill').style.width = Math.round(frac * 100) + '%';
}
function showError(t) { $('#error').hidden = !t; $('#error').textContent = t || ''; }

function inputs() {
  const p = parseFloat($('#power').value), t = parseFloat($('#time').value);
  return { power: p, time: t, ok: p > 0 && t > 0 && isFinite(p) && isFinite(t) };
}
function fitRange() {
  const tmin = parseFloat($('#tmin').value), tmax = parseFloat($('#tmax').value);
  let problem = '';
  if (!isFinite(tmin) || !isFinite(tmax)) problem = 'Enter a start and an end angle.';
  else if (tmax - tmin < 0.5) problem = 'The range needs to span at least 0.5°.';
  else if (scan && (tmax <= scan.range[0] || tmin >= scan.range[1])) problem = `The range is outside the scan (${scan.range[0].toFixed(2)}–${scan.range[1].toFixed(2)}°).`;
  return { tmin, tmax, ok: !problem, problem };
}
function refresh() {
  const r = fitRange();
  const hint = $('#rangeHint');
  hint.classList.toggle('warn', !!scan && !r.ok);
  if (scan) hint.textContent = r.ok
    ? `The scan covers ${scan.range[0].toFixed(2)}–${scan.range[1].toFixed(2)}°. The fit range starts at 1.5–4.2°; change either value to fit a different part.`
    : r.problem;
  $('#run').disabled = !(ready && !busy && scan && inputs().ok && r.ok);
}
function clearResult() {
  result = null;
  $('#resultPanel').hidden = true;
  $('#residWrap').hidden = true;
  redraw();
}

// ---------------------------------------------------------------- worker messages
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'status') {
    showProgress(m.text, m.frac);
  } else if (m.type === 'ready') {
    ready = true;
    $('#progress').hidden = true;
    refresh();
    if (file) worker.postMessage({ type: 'preview', filename: file.name, bytes: file.bytes.slice() });
  } else if (m.type === 'preview') {
    scan = m.data;
    $('#tmin').value = Math.max(1.5, scan.range[0]).toFixed(2);
    $('#tmax').value = Math.min(4.2, scan.range[1]).toFixed(2);
    $('#empty').hidden = true;
    showError('');
    redraw();
    refresh();
  } else if (m.type === 'progress') {
    showProgress(m.text, m.frac);
  } else if (m.type === 'result') {
    busy = false;
    $('#progress').hidden = true;
    result = m.data;
    renderResult();
    refresh();
  } else if (m.type === 'error') {
    busy = false;
    $('#progress').hidden = true;
    showError(m.text);
    if (m.fatal) $('#run').disabled = true; else refresh();
  }
};

// ---------------------------------------------------------------- file
async function takeFile(f) {
  if (!f) return;
  if (f.size > 20 * 1024 * 1024) { showError('That file is over 20 MB; is it the right one?'); return; }
  showError('');
  scan = null;
  $('#tmin').value = ''; $('#tmax').value = '';
  clearResult();
  file = { name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) };
  $('#dropText').textContent = f.name;
  $('#drop').classList.add('has');
  refresh();
  if (ready) worker.postMessage({ type: 'preview', filename: file.name, bytes: file.bytes.slice() });
  else showProgress('Loading the fitting engine…', 0.05);
}
$('#file').onchange = (e) => takeFile(e.target.files[0]);
const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => takeFile(e.dataTransfer.files[0]));
['#power', '#time', '#tmin', '#tmax'].forEach((s) => $(s).addEventListener('input', () => { clearResult(); refresh(); }));

// ---------------------------------------------------------------- fit
$('#run').onclick = () => {
  const { power, time } = inputs();
  const { tmin, tmax } = fitRange();
  busy = true; refresh(); showError(''); clearResult();
  showProgress('starting', 0.02);
  worker.postMessage({ type: 'fit', filename: file.name, bytes: file.bytes.slice(), power, time, tmin, tmax });
};

function sig(v, n) { return Number(v).toPrecision(n); }

function renderResult() {
  const r = result, ta = r.ta;
  const err = (ta.err_A != null && ta.err_A > 0) ? ` ± ${ta.err_A.toFixed(2)}` : '';
  $('#rThick').textContent = `${ta.thickness_A.toFixed(2)}${err} Å`;
  $('#rThickNm').textContent = `${(ta.thickness_A / 10).toFixed(3)} nm`;
  $('#rRate').textContent = ta.rate_A_s.toFixed(6);
  $('#rRateW').textContent = sig(ta.rate_A_s_W, 6);
  $('#rChi').textContent = `χ²red ${r.chi2red.toFixed(2)}`;
  $('#rRange').textContent = `fitted ${r.fit_range[0].toFixed(2)}–${r.fit_range[1].toFixed(2)}°`;
  const notes = [];
  if (r.chi2red > 10) notes.push(['warn', `The fit is poor (χ²red ${r.chi2red.toFixed(1)}). Check this is the right scan and that the power and time are correct before trusting the thickness.`]);
  else if (r.chi2red > 4) notes.push(['info', `The fit is only moderate (χ²red ${r.chi2red.toFixed(1)}); the thickness is still usually reliable for a clean film.`]);
  if (r.thickness_at_bound) notes.push(['warn', 'The Ta thickness ran to the edge of its search range, so treat it with caution.']);
  $('#notes').innerHTML = '';
  notes.forEach(([k, t]) => { const d = document.createElement('div'); d.className = 'note ' + k; d.textContent = t; $('#notes').appendChild(d); });
  $('#resultPanel').hidden = false;
  $('#residWrap').hidden = false;
  redraw();
}

$('#csv').onclick = () => {
  const c = result.curves;
  const rows = ['two_theta_deg,data_cps,fit_cps,residual_sigma'];
  for (let i = 0; i < c.two_theta.length; i++) rows.push([c.two_theta[i], c.data[i], c.fit[i], c.residual[i]].join(','));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' }));
  a.download = (file.name.replace(/\.[^.]+$/, '') || 'scan') + '_ta_fit.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

showProgress('Loading the fitting engine (first visit takes ~20 s)…', 0.02);
