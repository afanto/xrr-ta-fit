"""Fit driver: seed from the FFT, refine globally, then polish."""
import time

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from . import materials as M
from .model import Stack
from .seed import fft_thicknesses, scan_candidates
from . import samples as S

SYS_ERR_DEFAULT = 0.02      # systematic error floor, fraction of intensity


def prepare(data, tmin, tmax, sys_err=SYS_ERR_DEFAULT):
    tt = data['two_theta']
    m = (tt >= tmin) & (tt <= tmax) & (data['cps'] > 0)
    if m.sum() < 30:
        raise ValueError(f'only {int(m.sum())} points in 2theta = '
                         f'[{tmin}, {tmax}]; widen the fit range')
    tt = tt[m]
    cps = data['cps'][m]
    counts = np.maximum(data['counts'][m], 1.0)
    attn = data['attn'][m]
    rel = np.sqrt(attn / counts + sys_err ** 2)
    return tt, cps, rel


def _resid(pv, stack, tt, cps, rel, free, fixed):
    full = fixed.copy()
    full[free] = pv
    mod = np.maximum(stack.reflectivity(full, tt), 1e-14)
    return (np.log(mod) - np.log(cps)) / rel


def autoscale(stack, pv, tt, cps):
    """Put log_scale in the right decade before any optimisation starts."""
    i = stack.index['log_scale']
    trial = pv.copy()
    trial[i] = 0.0
    mod = stack.reflectivity(trial, tt) - 10 ** trial[stack.index['log_bg']]
    mod = np.maximum(mod, 1e-30)
    k = np.median(np.log10(cps) - np.log10(mod))
    p = stack.params[i]
    pv[i] = min(max(k, p.lo), p.hi)
    return pv


def _starting_points(stack, fixed, free, nstart, seed, candidates):
    """A structured spray of starting points for the local fits.

    Every candidate thickness assignment from the transform is crossed with a
    handful of interface/resolution variants, because those are what actually
    trap a local optimiser once the thickness is roughly right.
    """
    from scipy.stats import qmc

    nlay = len(stack.layers)
    base = fixed[free].copy()
    lo, hi = stack.lo[free], stack.hi[free]
    pos = {p: k for k, p in enumerate(free)}
    pts = []

    def variant(thick=None, **kw):
        v = base.copy()
        if thick is not None:
            for i, t in enumerate(thick, start=1):
                j = stack.index.get(f't{i}')
                if j is not None and j in pos:
                    p = stack.params[j]
                    v[pos[j]] = min(max(t, p.lo), p.hi)
        for name, val in kw.items():
            j = stack.index.get(name)
            if j is not None and j in pos:
                p = stack.params[j]
                v[pos[j]] = min(max(val, p.lo), p.hi)
        return v

    # every candidate thickness vector x a few interface/resolution variants
    sig_set = (2.0, 6.0) if nlay > 1 else (1.5, 4.0, 10.0)
    res_set = (0.01, 0.05)
    for cand in candidates:
        for sig in sig_set:
            for res in res_set:
                kw = {'sig0': sig, 'res_deg': res, 'tail_frac': 0.01}
                for i in range(1, nlay + 1):
                    kw[f'sig{i}'] = sig
                pts.append(variant(cand, **kw))

    if not nstart:
        nstart = len(pts) + 8
    nrand = max(nstart - len(pts), 0)
    if nrand:
        u = qmc.LatinHypercube(d=free.size, seed=seed).random(nrand)
        for row in u:
            v = lo + row * (hi - lo)
            # thickness and scale stay near a candidate; the rest roams
            cand = candidates[int(row[0] * len(candidates)) % len(candidates)]
            for i, t in enumerate(cand, start=1):
                j = stack.index.get(f't{i}')
                if j is not None and j in pos:
                    k = pos[j]
                    v[k] = min(max(t * (0.95 + 0.10 * row[k]), lo[k]), hi[k])
            j = stack.index.get('log_scale')
            if j is not None and j in pos:
                k = pos[j]
                v[k] = min(max(base[k] * (0.98 + 0.04 * row[k]), lo[k]), hi[k])
            pts.append(v)
    return pts


def _num(spec, key, default, label=None):
    """A missing or unparseable number falls back rather than crashing.

    The browser turns an empty or NaN field into JSON null, so this has to
    cope with None as well as absent.
    """
    v = spec.get(key, None)
    if v is None or v == '':
        return float(default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{label or key} is not a number: {v!r}')
    if not np.isfinite(f):
        return float(default)
    return f


def run(data, spec, progress=None):
    """spec: dict from the front end.  Returns a result dict."""
    wl = data['wavelength']
    tmin = _num(spec, 'tmin', max(data['two_theta'][0], 0.0), '2theta from')
    tmax = _num(spec, 'tmax', data['two_theta'][-1], '2theta to')
    sys_err = _num(spec, 'sys_err', SYS_ERR_DEFAULT, 'systematic error')
    tt, cps, rel = prepare(data, tmin, tmax, sys_err)
    step = max(1, tt.size // 700)
    ttc, cpsc, relc = tt[::step], cps[::step], rel[::step]
    # the thickness scan only needs enough points to resolve the fringes
    sstep = max(1, tt.size // 420)
    tts, cpss, rels = tt[::sstep], cps[::sstep], rel[::sstep]

    layers = spec['layers']
    nlay = len(layers)

    # ---- how thick could the whole stack be? ------------------------------
    top = M.film(layers[0]['material'], wl)
    th_c = top.theta_c(M.FILMS[layers[0]['material']]['rho'])
    peaks, amps, spectrum = fft_thicknesses(tt, cps, wl, th_c)
    # The transform is used only to bound the search, so take the largest
    # period at face value including any harmonics.  Filtering harmonics out
    # here is actively dangerous: layer thicknesses in a near-integer ratio
    # (30 A of Ru under 150 A of Co, say) look exactly like a fundamental and
    # its 5th harmonic, and discarding the real one puts the answer outside
    # the search region.  Over-estimating the bound only costs a little grid
    # resolution, so err that way.
    t_total = max(peaks) if peaks else 400.0

    # ---- candidate thicknesses from a direct scan of the cost surface ----
    # Reading periodicities off the transform is unreliable when one layer is
    # a thin cap (weak peak, easily missed).  Scanning the actual misfit is
    # slower but cannot be fooled.
    if progress:
        progress(stage='scan', peaks=peaks, t_total=t_total)

    fixed_t = [l.get('thickness') if l.get('fix_thickness') else None
               for l in layers]
    scan_hi = _num(spec, 't_max_hint',
                   min(max(t_total * 1.7, 200.0), 6000.0))
    probe = Stack(layers, spec['substrate'], wavelength=wl,
                  sample_mm=_num(spec, 'sample_mm', 25.0, 'sample length'),
                  seeds=[scan_hi / max(nlay, 1)] * nlay,
                  t_bounds=[(1.0, 1e5)] * nlay,
                  link_densities=spec.get('link_densities', True))
    pv_probe = probe.start.copy()
    for k, v in (('sig0', 3.0), ('res_deg', 0.03), ('tail_frac', 0.01),
                 ('tail_deg', 0.2), ('beam_mm', 0.3)):
        pv_probe[probe.index[k]] = v
    for i in range(1, nlay + 1):
        pv_probe[probe.index[f'sig{i}']] = 3.0
    lg = np.log(cpss)
    w = 1.0 / rels ** 2
    wlin = 1.0 / (cpss * rels) ** 2          # weights for the linear-space solve

    def scan_cost(tvec):
        """Misfit for a trial thickness, with scale and background eliminated.

        The background has to be solved for, not ignored: on a scan that
        reaches its background floor, a probe with no background is best
        fitted by an absurdly thick film whose fringes wash out, so every
        candidate piles up against the top of the grid.
        """
        p = pv_probe.copy()
        for i, t in enumerate(tvec, start=1):
            p[probe.index[f't{i}']] = t
        p[probe.index['log_scale']] = 0.0
        p[probe.index['log_bg']] = -300.0            # 10**-300 -> no background
        R = np.maximum(probe.reflectivity(p, tts), 1e-300)

        # I ~= A*R + B solved in closed form (weighted, linear space)
        S11 = np.sum(wlin * R * R)
        S1b = np.sum(wlin * R)
        Sbb = np.sum(wlin)
        S1y = np.sum(wlin * R * cpss)
        Sby = np.sum(wlin * cpss)
        det = S11 * Sbb - S1b * S1b
        if det > 0:
            A = (S1y * Sbb - Sby * S1b) / det
            B = (S11 * Sby - S1b * S1y) / det
        else:
            A, B = S1y / max(S11, 1e-300), 0.0
        if B < 0:                                    # background cannot be < 0
            B, A = 0.0, S1y / max(S11, 1e-300)
        if not (A > 0):
            return float('inf')

        lm = np.log(np.maximum(A * R + B, 1e-300))
        return float(np.sum(w * (lm - lg) ** 2))

    if all(f is not None for f in fixed_t):
        candidates = [list(fixed_t)]
    else:
        # Scan only the thicknesses that are actually free.  Scanning the
        # pinned ones as well spent the grid budget on axes that never move:
        # a trilayer with both Ru layers pinned was scanned in 3-D at 34
        # points per axis instead of 1-D at 400, coarse enough to step over
        # the right basin entirely - which is how R27 came back four times
        # too thick with chi2red 489 when the true minimum was 23.
        free_idx = [i for i in range(nlay) if fixed_t[i] is None]

        def _full(tv):
            out = list(fixed_t)
            for j, i in enumerate(free_idx):
                out[i] = tv[j]
            return out

        free_scan = scan_candidates(
            lambda tv: scan_cost(_full(tv)),
            len(free_idx), scan_hi, t_lo=float(spec.get('t_min_hint', 8.0)),
            ngrid=int(spec.get('ngrid', 0)) or None)
        candidates = [_full(c) for c in free_scan]

    # a user-supplied starting thickness is tried first
    hinted = [l.get('thickness') for l in layers]
    if any(h is not None for h in hinted):
        candidates.insert(0, [h if h is not None else candidates[0][i]
                              for i, h in enumerate(hinted)])
    seeds = candidates[0]

    tb = []
    for i in range(nlay):
        if fixed_t[i] is not None:
            tb.append((fixed_t[i] * 0.999, fixed_t[i] * 1.001))
        else:
            vals = [c[i] for c in candidates]
            tb.append((max(2.0, min(vals) * 0.6), max(vals) * 1.5 + 20.0))
    if progress:
        progress(stage='seed', peaks=peaks, seeds=seeds,
                 candidates=candidates[:6])

    stack = Stack(layers, spec['substrate'], wavelength=wl,
                  sample_mm=_num(spec, 'sample_mm', 25.0, 'sample length'),
                  fit_density=spec.get('fit_density', False),
                  fit_substrate_density=spec.get('fit_substrate_density', False),
                  seeds=seeds, rough_max=float(spec.get('rough_max', 30.0)),
                  t_bounds=tb, fixed_t=fixed_t,
                  link_densities=spec.get('link_densities', True))

    lo, hi = stack.lo, stack.hi
    vary = stack.vary_mask
    fixed = stack.start.copy()
    fixed = autoscale(stack, fixed, tt, cps)
    free = np.where(vary)[0]
    flo, fhi = lo[free], hi[free]

    # Because the FFT seed pins the thickness to within a per cent or two, a
    # spray of local fits finds the same minimum as a full global search in a
    # fraction of the time.  A global search is still available on request,
    # and is triggered automatically if the local fits disagree.
    args_c = (stack, ttc, cpsc, relc, free, fixed)
    args_f = (stack, tt, cps, rel, free, fixed)

    # Evaluation caps. A healthy coarse start converges in 10-50 evaluations
    # and the final polish in 10-20, so these are ~6x headroom. The point of
    # capping them is that an occasional start wanders somewhere pathological,
    # and an uncapped one then runs for minutes - which surfaces as a progress
    # bar frozen near the end, indistinguishable from a crash.
    COARSE_NFEV = int(_num(spec, 'coarse_nfev', 300))
    POLISH_NFEV = int(_num(spec, 'polish_nfev', 2000))

    def local(x0, args, nfev=POLISH_NFEV):
        return least_squares(_resid, np.clip(x0, flo + 1e-12, fhi - 1e-12),
                             bounds=(flo, fhi), args=args,
                             x_scale='jac', max_nfev=nfev)

    starts = _starting_points(stack, fixed, free, int(spec.get('nstart', 0)),
                              int(spec.get('seed', 1)), candidates)
    # A wall-clock budget on the refinement stage. Any one starting point can
    # in principle send the optimiser somewhere pathological; without a budget
    # that shows up as a progress bar frozen near the end, which is the worst
    # possible failure mode because it looks like a hang rather than a result.
    budget = _num(spec, 'search_budget_s', 75.0)
    t_start = time.monotonic()
    results, done = [], 0
    for i, s in enumerate(starts):
        try:
            results.append(local(s, args_c, COARSE_NFEV))
        except Exception:
            pass
        done = i + 1
        if progress:
            progress(stage='search', iteration=done, total=len(starts))
        if results and time.monotonic() - t_start > budget:
            if progress:
                progress(stage='search', iteration=len(starts),
                         total=len(starts), truncated=done)
            break
    if not results:
        raise RuntimeError('every starting point failed to converge')
    results.sort(key=lambda r: r.cost)

    # if the best few disagree about the layer thicknesses the landscape is
    # rough, so fall back to a genuine global search
    ti = [stack.index[f't{i}'] for i in range(1, nlay + 1)]
    pos = {p: k for k, p in enumerate(free)}
    comparable = [r for r in results if r.cost <= 1.3 * results[0].cost][:4]
    spread = 0.0
    for j in ti:
        if j in pos:
            v = [r.x[pos[j]] for r in comparable]
            if len(v) > 1:
                spread = max(spread, (max(v) - min(v)) / max(np.mean(v), 1e-9))

    best_chi = 2 * results[0].cost / max(ttc.size - free.size, 1)
    need_global = spread > 0.02 or best_chi > 8.0
    if spec.get('thorough') or need_global:
        if progress:
            reason = ('requested' if spec.get('thorough') else
                      (f'local fits disagreed by {100*spread:.1f}%'
                       if spread > 0.02 else
                       f'best local fit only reached chi2red = {best_chi:.1f}'))
            progress(stage='global', reason=reason)
        maxiter = int(spec.get('maxiter', 200))
        state = {'it': 0}

        def cost(pv):
            return 0.5 * np.sum(_resid(pv, *args_c) ** 2)

        def cb(xk, convergence=None):
            state['it'] += 1
            if progress:
                progress(stage='global', iteration=state['it'], total=maxiter)
            return False

        de = differential_evolution(
            cost, list(zip(flo, fhi)), maxiter=maxiter,
            popsize=int(spec.get('popsize', 18)), tol=1e-7,
            mutation=(0.3, 1.0), recombination=0.9, polish=False,
            init='sobol', seed=int(spec.get('seed', 1)), callback=cb)
        try:
            results.append(local(de.x, args_c, COARSE_NFEV))
            results.sort(key=lambda r: r.cost)
        except Exception:
            pass

    # ---- polish the best candidates on the full data ---------------------
    if progress:
        progress(stage='polish')
    best = None
    for r in results[:3]:
        try:
            ls = local(r.x, args_f)
        except Exception:
            continue
        if best is None or ls.cost < best.cost:
            best = ls
    if best is None:
        raise RuntimeError('the optimiser failed to converge')

    pv = fixed.copy()
    pv[free] = best.x
    dof = max(tt.size - free.size, 1)
    chi2red = 2 * best.cost / dof

    perr = np.full(pv.size, np.nan)
    try:
        J = best.jac
        cov = np.linalg.inv(J.T @ J) * chi2red
        perr[free] = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        pass

    # numbers the UI uses to annotate the curve and explain what set what
    g = lambda k: pv[stack.index[k]]
    n_media, thick, rough = stack.media(pv)
    delta_top = 1.0 - float(np.real(n_media[1]))
    delta_sub = 1.0 - float(np.real(n_media[-1]))
    total_A = float(sum(thick))
    th_c_top = float(np.degrees(np.arcsin(np.sqrt(max(2 * delta_top, 0)))))
    th_c_sub = float(np.degrees(np.arcsin(np.sqrt(max(2 * delta_sub, 0)))))
    explain = dict(
        wavelength=wl,
        delta_top=delta_top, delta_sub=delta_sub,
        theta_c_top=th_c_top, two_theta_c_top=2 * th_c_top,
        theta_c_sub=th_c_sub, two_theta_c_sub=2 * th_c_sub,
        total_A=total_A,
        # fringe period in 2theta, well above the critical angle
        fringe_2theta=float(np.degrees(wl / total_A)) if total_A else None,
        background=float(10 ** g('log_bg')),
        scale=float(10 ** g('log_scale')),
        res_deg=float(g('res_deg')), beam_mm=float(g('beam_mm')),
        dth=float(g('dth')),
        sigma_top=float(rough[0]),
        q_max=float(4 * np.pi / wl * np.sin(np.radians(tmax / 2))),
        materials=[m.label.split()[0] for m in stack.mats],
    )

    model = stack.reflectivity(pv, tt)
    z, prof = stack.profile(pv)
    resid = (np.log(model) - np.log(cps)) / rel

    summary = stack.summary(pv, perr)
    return dict(
        chi2red=float(chi2red), dof=int(dof),
        starts_used=int(done), starts_total=int(len(starts)),
        sample=spec.get('sample'), explain=explain,
        # deposition numbers typed into the form win over the stored preset,
        # so a sample with no preset still gets a rate table
        rates=(S.rates_from_layers(layers, [r['thickness_A'] for r in summary])
               or S.rates(spec.get('sample'),
                          [r['thickness_A'] for r in summary])),
        params=stack.describe(pv, perr),
        summary=summary,
        fft=dict(peaks=peaks, seeds=seeds, spectrum=spectrum),
        curves=dict(two_theta=tt.tolist(), data=cps.tolist(),
                    fit=model.tolist(), residual=resid.tolist()),
        profile=dict(z=z.tolist(), delta=(np.array(prof) * 1e6).tolist()),
        wavelength=wl, tmin=tmin, tmax=tmax,
        pv=pv.tolist(), names=stack.names,
    )
