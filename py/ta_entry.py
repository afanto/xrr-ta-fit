"""Entry point called from the web worker: fit one Ta film between Ru layers."""
import json
import math

import numpy as np

from xrrfit.loader import load
from xrrfit.quality import suggest_tmax
from xrrfit import fit as F

RU_THICKNESS_A = 18.72         # cap and seed, locked
MAX_POINTS = 1600


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _thin(a):
    a = np.asarray(a)
    if a.size <= MAX_POINTS:
        return a.tolist()
    return a[::int(np.ceil(a.size / MAX_POINTS))].tolist()


def preview(filename, path):
    """Parse the scan and say what range will be fitted."""
    with open(path, 'rb') as fh:
        data = load(filename, fh.read())
    try:
        cut, _ = suggest_tmax(data['two_theta'], data['cps'], data['counts'],
                              data['attn'])
    except Exception:                                  # noqa: BLE001
        cut = data['range'][1]
    return json.dumps(_clean(dict(
        npts=data['npts'], range=data['range'], tmax=cut,
        two_theta=_thin(data['two_theta']), cps=_thin(data['cps']))))


def run(filename, path, power_w, time_s, tmin, tmax, post):
    with open(path, 'rb') as fh:
        data = load(filename, fh.read())
    lo = max(float(tmin), data['range'][0])
    hi = min(float(tmax), data['range'][1])
    if hi - lo < 0.5:
        raise ValueError('The fit range is too short: it needs to span at '
                         'least 0.5 degrees inside the scan '
                         f"({data['range'][0]:.2f} to {data['range'][1]:.2f}).")
    ru = dict(material='Ru', thickness=RU_THICKNESS_A, fix_thickness=True,
              power_W=100.0, time_s=26.0)
    spec = dict(
        layers=[dict(ru), dict(material='Ta', power_W=power_w, time_s=time_s),
                dict(ru)],
        substrate='borofloat', tmin=lo, tmax=hi,
        sample_mm=25.0, sys_err=0.02, fit_density=False,
        fit_substrate_density=False, link_densities=True)

    def progress(**kw):
        stage = kw.get('stage', '')
        if stage == 'scan':
            post('scanning thickness space', 0.05)
        elif stage == 'seed':
            post('found starting points', 0.25)
        elif stage == 'search':
            it, tot = kw.get('iteration', 0), max(kw.get('total', 1), 1)
            post(f'refining ({it}/{tot})', 0.25 + 0.55 * it / tot)
        elif stage == 'global':
            if 'reason' in kw:
                post('wider search', 0.8)
            else:
                it, tot = kw.get('iteration', 0), max(kw.get('total', 1), 1)
                post(f'wider search ({it}/{tot})', 0.8 + 0.15 * it / tot)
        elif stage == 'polish':
            post('final refinement', 0.95)

    res = F.run(data, spec, progress=progress)
    ta = res['summary'][1]
    thickness = ta['thickness_A']
    at_bound = [p['name'] for p in res['params'] if p['at_bound']]
    c = res['curves']
    return json.dumps(_clean(dict(
        ta=dict(thickness_A=thickness, err_A=ta['thickness_err_A'],
                rate_A_s=thickness / time_s,
                rate_A_s_W=thickness / time_s / power_w,
                roughness_top_A=ta['roughness_top_A'],
                roughness_bot_A=ta['roughness_bot_A']),
        chi2red=res['chi2red'], dof=res['dof'],
        fit_range=[res['tmin'], res['tmax']],
        thickness_at_bound=('t2' in at_bound),
        curves=dict(two_theta=_thin(c['two_theta']), data=_thin(c['data']),
                    fit=_thin(c['fit']), residual=_thin(c['residual'])))))
