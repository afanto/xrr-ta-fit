/* Minimal canvas plotting: linear or log y, multiple series, crosshair
   readout. Deliberately dependency-free so the app works with no network. */

function niceTicks(lo, hi, target) {
  if (!(isFinite(lo) && isFinite(hi)) || hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(target, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Math.abs(v) < 1e-12 ? 0 : v);
  }
  return out;
}

function decadeTicks(lo, hi) {
  const out = [];
  const a = Math.floor(lo), b = Math.ceil(hi);
  for (let e = a; e <= b; e++) if (e >= lo - 1e-9 && e <= hi + 1e-9) out.push(e);
  return out;
}

function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

class Plot {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.opts = Object.assign(
      { logY: false, xlabel: '', ylabel: '', pad: { l: 68, r: 14, t: 12, b: 44 },
        yZeroLine: false, legend: true },
      opts);
    this.series = [];
    this.hover = null;
    this._onMove = (e) => this._move(e);
    this._onLeave = () => { this.hover = null; this.draw(); };
    canvas.addEventListener('mousemove', this._onMove);
    canvas.addEventListener('mouseleave', this._onLeave);
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas.parentElement || canvas);
  }

  setSeries(series) { this.series = series; this.draw(); }

  _bounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const logY = this.opts.logY;
    for (const s of this.series) {
      if (s.decoration || !s.x || !s.x.length) continue;
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i], yv = s.y[i];
        if (!isFinite(xv) || !isFinite(yv)) continue;
        if (logY && !(yv > 0)) continue;
        if (xv < x0) x0 = xv;
        if (xv > x1) x1 = xv;
        const t = logY ? Math.log10(yv) : yv;
        if (t < y0) y0 = t;
        if (t > y1) y1 = t;
      }
    }
    if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    if (x1 === x0) x1 = x0 + 1;
    const m = (y1 - y0) * 0.06 || 0.5;
    return { x0, x1, y0: y0 - m, y1: y1 + m };
  }

  draw() {
    const cv = this.cv, ctx = cv.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = cv.getBoundingClientRect();
    const W = Math.max(rect.width, 120), H = Math.max(rect.height, 80);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue('--fg').trim() || '#222';
    const grid = css.getPropertyValue('--grid').trim() || '#ddd';
    const muted = css.getPropertyValue('--muted').trim() || '#888';

    const p = this.opts.pad;
    const b = this._bounds();
    this._b = b;
    const px = (v) => p.l + (v - b.x0) / (b.x1 - b.x0) * (W - p.l - p.r);
    const py = (v) => {
      const t = this.opts.logY ? Math.log10(v) : v;
      return H - p.b - (t - b.y0) / (b.y1 - b.y0) * (H - p.t - p.b);
    };
    this._px = px; this._py = py;

    // grid + axes
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.strokeStyle = grid; ctx.fillStyle = muted; ctx.lineWidth = 1;
    const xt = niceTicks(b.x0, b.x1, Math.max(3, Math.round(W / 90)));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of xt) {
      const X = Math.round(px(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(X, p.t); ctx.lineTo(X, H - p.b); ctx.stroke();
      ctx.fillText(fmt(t), X, H - p.b + 6);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const yt = this.opts.logY ? decadeTicks(b.y0, b.y1)
      : niceTicks(b.y0, b.y1, Math.max(3, Math.round(H / 46)));
    for (const t of yt) {
      const v = this.opts.logY ? Math.pow(10, t) : t;
      const Y = Math.round(py(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(p.l, Y); ctx.lineTo(W - p.r, Y); ctx.stroke();
      ctx.fillText(this.opts.logY ? '10' + sup(t) : fmt(t), p.l - 8, Y);
    }
    if (this.opts.yZeroLine && b.y0 < 0 && b.y1 > 0) {
      ctx.strokeStyle = muted; ctx.beginPath();
      ctx.moveTo(p.l, Math.round(py(0)) + 0.5);
      ctx.lineTo(W - p.r, Math.round(py(0)) + 0.5); ctx.stroke();
    }

    ctx.strokeStyle = muted;
    ctx.strokeRect(p.l + 0.5, p.t + 0.5, W - p.l - p.r - 1, H - p.t - p.b - 1);

    // labels
    ctx.fillStyle = fg; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    if (this.opts.xlabel) ctx.fillText(this.opts.xlabel, (p.l + W - p.r) / 2, H - 4);
    if (this.opts.ylabel) {
      ctx.save(); ctx.translate(13, (p.t + H - p.b) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'top'; ctx.fillText(this.opts.ylabel, 0, 0); ctx.restore();
    }

    // series
    ctx.save();
    ctx.beginPath();
    ctx.rect(p.l, p.t, W - p.l - p.r, H - p.t - p.b);
    ctx.clip();
    for (const s of this.series) {
      if (s.style !== 'span') continue;
      const a = px(s.x0), bx = px(s.x1);
      ctx.fillStyle = s.color;
      ctx.fillRect(Math.min(a, bx), p.t, Math.abs(bx - a), H - p.t - p.b);
      if (s.label) {
        ctx.fillStyle = muted;
        ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(s.label, (Math.min(a, bx) + Math.max(a, bx)) / 2, p.t + 5);
      }
    }
    for (const s of this.series) {
      if (s.style === 'span' || !s.x || !s.x.length) continue;
      ctx.strokeStyle = s.color; ctx.fillStyle = s.color;
      ctx.lineWidth = s.width || 1.4;
      if (s.style === 'dots') {
        const r = s.radius || 1.3;
        for (let i = 0; i < s.x.length; i++) {
          const yv = s.y[i];
          if (!isFinite(yv) || (this.opts.logY && !(yv > 0))) continue;
          ctx.beginPath();
          ctx.arc(px(s.x[i]), py(yv), r, 0, 6.2832); ctx.fill();
        }
      } else {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.x.length; i++) {
          const yv = s.y[i];
          if (!isFinite(yv) || (this.opts.logY && !(yv > 0))) { started = false; continue; }
          const X = px(s.x[i]), Y = py(yv);
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        if (s.dash) ctx.setLineDash(s.dash);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // vertical markers
    for (const s of this.series) {
      if (s.style !== 'vline') continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.2;
      ctx.setLineDash(s.dash || [5, 4]);
      const X = Math.round(px(s.at)) + 0.5;
      ctx.beginPath(); ctx.moveTo(X, p.t); ctx.lineTo(X, H - p.b); ctx.stroke();
      ctx.setLineDash([]);
      if (s.label) {
        ctx.save();
        ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = s.color;
        ctx.translate(X - 4, p.t + (s.labelY || 6));
        ctx.rotate(Math.PI / 2);
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(s.label, 0, 0);
        ctx.restore();
      }
    }
    // horizontal reference levels
    for (const s of this.series) {
      if (s.style !== 'hline') continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.1;
      ctx.setLineDash(s.dash || [4, 4]);
      const Y = Math.round(py(s.at)) + 0.5;
      ctx.beginPath(); ctx.moveTo(p.l, Y); ctx.lineTo(W - p.r, Y); ctx.stroke();
      ctx.setLineDash([]);
      if (s.label) {
        ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = s.color; ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(s.label, p.l + 6, Y - 2);
      }
    }
    // double-headed span used to show one fringe period
    for (const s of this.series) {
      if (s.style !== 'measure') continue;
      const a = px(s.x0), b2 = px(s.x1), Y = py(s.y);
      ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(a, Y); ctx.lineTo(b2, Y); ctx.stroke();
      [[a, 1], [b2, -1]].forEach(([X, dir]) => {
        ctx.beginPath();
        ctx.moveTo(X, Y); ctx.lineTo(X + 5 * dir, Y - 3.5);
        ctx.lineTo(X + 5 * dir, Y + 3.5); ctx.closePath(); ctx.fill();
        ctx.beginPath(); ctx.moveTo(X, Y - 6); ctx.lineTo(X, Y + 6); ctx.stroke();
      });
      if (s.label) {
        ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(s.label, (a + b2) / 2, Y - 8);
      }
    }
    // free-floating notes
    for (const s of this.series) {
      if (s.style !== 'note') continue;
      ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = s.color; ctx.textAlign = s.align || 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.label, px(s.x), py(s.y));
    }
    ctx.restore();

    // crosshair
    if (this.hover) {
      const { X, Y, tx, ty } = this.hover;
      ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X, p.t); ctx.lineTo(X, H - p.b); ctx.stroke();
      ctx.setLineDash([]);
      const label = `${fmt(tx)}, ${fmt(ty)}`;
      ctx.font = '11px ui-monospace, monospace';
      const w = ctx.measureText(label).width + 10;
      let lx = X + 8; if (lx + w > W - p.r) lx = X - 8 - w;
      ctx.fillStyle = css.getPropertyValue('--panel').trim() || '#fff';
      ctx.strokeStyle = muted;
      ctx.fillRect(lx, p.t + 6, w, 18); ctx.strokeRect(lx + 0.5, p.t + 6.5, w, 18);
      ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(label, lx + 5, p.t + 15);
    }

    // legend
    const named = this.series.filter((s) => s.label && !s.decoration
                                       && s.style !== 'span');
    if (this.opts.legend && named.length) {
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      let w = 0;
      for (const s of named) w = Math.max(w, ctx.measureText(s.label).width);
      w += 30;
      const h = named.length * 16 + 8;
      const lx = W - p.r - w - 8, ly = p.t + 8;
      ctx.fillStyle = (css.getPropertyValue('--panel').trim() || '#fff') + 'e6';
      ctx.strokeStyle = grid;
      ctx.fillRect(lx, ly, w, h); ctx.strokeRect(lx + 0.5, ly + 0.5, w, h);
      named.forEach((s, i) => {
        const Y = ly + 12 + i * 16;
        ctx.strokeStyle = s.color; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(lx + 7, Y); ctx.lineTo(lx + 21, Y); ctx.stroke();
        ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(s.label, lx + 26, Y);
      });
    }
  }

  _move(e) {
    if (!this._b || !this.series.length) return;
    const rect = this.cv.getBoundingClientRect();
    const X = e.clientX - rect.left;
    const b = this._b, p = this.opts.pad;
    const W = rect.width;
    if (X < p.l || X > W - p.r) { this.hover = null; this.draw(); return; }
    const tx = b.x0 + (X - p.l) / (W - p.l - p.r) * (b.x1 - b.x0);
    const ref = this.series.find((s) => s.x && s.x.length && s.style !== 'vline');
    if (!ref) return;
    let lo = 0, hi = ref.x.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ref.x[m] < tx) lo = m; else hi = m; }
    const i = Math.abs(ref.x[lo] - tx) < Math.abs(ref.x[hi] - tx) ? lo : hi;
    this.hover = { X: this._px(ref.x[i]), Y: this._py(ref.y[i]),
                   tx: ref.x[i], ty: ref.y[i] };
    this.draw();
  }
}

function sup(n) {
  const map = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴',
                5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  return String(n).split('').map((c) => map[c] || c).join('');
}
