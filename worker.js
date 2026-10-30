/* Runs the fit in a Web Worker so the page stays responsive.
   Pyodide (Python + numpy + scipy compiled to WebAssembly) is loaded from the
   jsDelivr CDN; the fitting code itself is served alongside this file. */
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(PYODIDE + 'pyodide.js');

let py = null;
const say = (type, extra) => postMessage(Object.assign({ type }, extra));

async function boot() {
  say('status', { text: 'Loading the fitting engine (first visit takes ~20 s)…', frac: 0.05 });
  py = await loadPyodide({ indexURL: PYODIDE });
  say('status', { text: 'Loading numpy and scipy…', frac: 0.3 });
  await py.loadPackage(['numpy', 'scipy', 'micropip']);
  say('status', { text: 'Loading the X-ray optical constants…', frac: 0.7 });
  await py.runPythonAsync(`
import micropip
await micropip.install('periodictable')
`);
  say('status', { text: 'Loading the fitting code…', frac: 0.9 });
  const files = await (await fetch('py/files.json')).json();
  py.FS.mkdirTree('/app/xrrfit');
  for (const f of files) {
    const txt = await (await fetch('py/' + f)).text();
    py.FS.writeFile('/app/' + f, txt);
  }
  await py.runPythonAsync(`
import sys
sys.path.insert(0, '/app')
import ta_entry
`);
  say('ready');
}

function stash(bytes) {
  py.FS.writeFile('/tmp/scan.dat', bytes);
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'preview') {
      stash(m.bytes);
      py.globals.set('fname', m.filename);
      const out = await py.runPythonAsync("ta_entry.preview(fname, '/tmp/scan.dat')");
      say('preview', { data: JSON.parse(out) });
    } else if (m.type === 'fit') {
      stash(m.bytes);
      py.globals.set('fname', m.filename);
      py.globals.set('power', m.power);
      py.globals.set('secs', m.time);
      py.globals.set('tlo', m.tmin);
      py.globals.set('thi', m.tmax);
      py.globals.set('post', (text, frac) => say('progress', { text, frac }));
      const out = await py.runPythonAsync(
        "ta_entry.run(fname, '/tmp/scan.dat', float(power), float(secs), float(tlo), float(thi), post)");
      say('result', { data: JSON.parse(out) });
    }
  } catch (err) {
    // keep the last line of a Python traceback: that is the useful part
    let msg = String(err && err.message || err).trim().split('\n').slice(-1)[0];
    msg = msg.replace(/^[\w.]*LoadError: /, 'Could not read that file: ').replace(/^[\w.]*(Error|Exception): /, '');
    say('error', { text: msg });
  }
};

boot().catch((err) => say('error', { text: 'Could not load the fitting engine: ' + err, fatal: true }));
