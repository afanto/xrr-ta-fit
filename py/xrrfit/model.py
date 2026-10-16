"""Turn a user-described stack into a fittable parameter model."""
import numpy as np

from . import materials as M
from .kernel import parratt, footprint, smear, sld_profile


def _finite(x):
    """None unless x is a real, finite number.

    A fixed parameter has no uncertainty, and NaN is not representable in
    JSON - json.dumps writes a bare NaN, which every browser rejects, so a
    single NaN silently breaks the whole response.
    """
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


class Param:
    __slots__ = ('name', 'label', 'unit', 'lo', 'hi', 'start', 'vary', 'group')

    def __init__(self, name, label, unit, lo, hi, start, vary=True, group=''):
        self.name, self.label, self.unit = name, label, unit
        self.lo, self.hi = float(lo), float(hi)
        self.start = float(min(max(start, lo), hi))
        self.vary, self.group = vary, group

    def as_dict(self):
        return dict(name=self.name, label=self.label, unit=self.unit,
                    lo=self.lo, hi=self.hi, start=self.start,
                    vary=self.vary, group=self.group)


class Stack:
    """Layers are given top-down; layers[0] is the free surface."""

    def __init__(self, layers, substrate_key, wavelength=M.CU_KA1,
                 sample_mm=25.0, fit_density=True, fit_substrate_density=False,
                 seeds=None, rough_max=30.0, t_bounds=None, fixed_t=None,
                 link_densities=True):
        self.wavelength = wavelength
        self.sample_mm = sample_mm
        self.layers = layers                       # list of dicts
        self.sub = M.substrate(substrate_key, wavelength)
        self.sub_key = substrate_key
        self.mats = [M.film(l['material'], wavelength) for l in layers]

        p = []
        n = len(layers)
        seeds = seeds or [None] * n
        fixed_t = fixed_t or [None] * n
        # A material that appears more than once in a stack - a Ru cap and a Ru
        # seed layer, say - came off the same target under the same conditions,
        # so giving each its own density just adds a parameter the data cannot
        # separate.  rho_key maps a layer to the density parameter it uses.
        self.rho_key = [None] * n
        seen = {}

        # roughness of the free surface
        p.append(Param('sig0', 'roughness: surface', 'Å', 0, rough_max, 4.0,
                       True, 'interfaces'))

        for i, (lay, mat) in enumerate(zip(layers, self.mats), start=1):
            t0 = lay.get('thickness') or seeds[i - 1] or 100.0
            tlo = lay.get('t_min')
            thi = lay.get('t_max')
            if tlo is None or thi is None:
                if t_bounds is not None:
                    tlo, thi = t_bounds[i - 1]
                else:
                    # a seeded layer only needs a modest window; an unseeded
                    # one gets a wide one
                    f = 0.35 if seeds[i - 1] else 0.85
                    tlo = max(3.0, t0 * (1 - f))
                    thi = t0 * (1 + f) + 20.0
            p.append(Param(f't{i}', f'thickness: {mat.label.split()[0]}', 'Å',
                           tlo, thi, t0, fixed_t[i - 1] is None, f'layer{i}'))

            mkey = lay['material']
            if link_densities and mkey in seen:
                self.rho_key[i - 1] = seen[mkey]
            else:
                rho0 = M.FILMS[mkey]['rho']
                rlo = lay.get('rho_min', 0.65 * rho0)
                rhi = lay.get('rho_max', 1.02 * rho0)
                name = f'rho{i}'
                nlabel = mat.label.split()[0]
                shared = (link_densities and
                          sum(1 for x in layers if x['material'] == mkey) > 1)
                p.append(Param(name,
                               f'density: {nlabel}' +
                               (' (shared)' if shared else ''),
                               'g/cm³', rlo, rhi, min(rho0, rhi),
                               bool(fit_density), f'layer{i}'))
                seen[mkey] = name
                self.rho_key[i - 1] = name

            below = (self.mats[i].label.split()[0] if i < n
                     else self.sub.label.split()[0])
            p.append(Param(f'sig{i}',
                           f'roughness: {mat.label.split()[0]} / {below}',
                           'Å', 0, rough_max, 4.0, True, 'interfaces'))

        srho = M.SUBSTRATES[substrate_key]['rho']
        p.append(Param('rho_sub', 'density: substrate', 'g/cm³',
                       0.85 * srho, 1.10 * srho, srho,
                       bool(fit_substrate_density), 'substrate'))

        # instrument
        p += [
            Param('log_scale', 'log10 scale factor', '', 3.0, 11.0, 7.0,
                  True, 'instrument'),
            Param('log_bg', 'log10 background', 'cps', -2.0, 5.0, 2.0,
                  True, 'instrument'),
            Param('dth', 'angular zero offset', '° (θ)', -0.03, 0.03, 0.0,
                  True, 'instrument'),
            Param('beam_mm', 'beam width (FWHM)', 'mm', 0.02, 0.8, 0.2,
                  True, 'instrument'),
            Param('res_deg', 'angular resolution (FWHM)', '° (θ)',
                  0.0, 0.15, 0.02, True, 'instrument'),
            Param('tail_frac', 'broad tail fraction', '', 0.0, 0.12, 0.01,
                  True, 'instrument'),
            Param('tail_deg', 'broad tail width (FWHM)', '° (θ)',
                  0.01, 0.8, 0.2, True, 'instrument'),
        ]
        self.params = p
        self.names = [x.name for x in p]
        self.index = {x.name: i for i, x in enumerate(p)}

    # ----------------------------------------------------------------
    @property
    def lo(self):
        return np.array([p.lo for p in self.params])

    @property
    def hi(self):
        return np.array([p.hi for p in self.params])

    @property
    def start(self):
        return np.array([p.start for p in self.params])

    @property
    def vary_mask(self):
        return np.array([p.vary for p in self.params])

    def media(self, pv):
        """-> n list (top-down), thicknesses, roughnesses"""
        g = lambda k: pv[self.index[k]]
        n = [1.0 + 0j]
        t, s = [], [g('sig0')]
        for i, mat in enumerate(self.mats, start=1):
            n.append(mat.n(g(self.rho_key[i - 1])))
            t.append(g(f't{i}'))
            s.append(g(f'sig{i}'))
        n.append(self.sub.n(g('rho_sub')))
        return n, t, s

    def reflectivity(self, pv, two_theta):
        g = lambda k: pv[self.index[k]]
        th = two_theta / 2.0 - g('dth')
        n, t, s = self.media(pv)
        R = parratt(th, self.wavelength, n, t, s)
        R = smear(th, R, g('res_deg'), g('tail_frac'), g('tail_deg'))
        R = R * footprint(th, g('beam_mm'), self.sample_mm)
        return 10 ** g('log_scale') * R + 10 ** g('log_bg')

    def profile(self, pv):
        n, t, s = self.media(pv)
        z, prof = sld_profile(n, t, s)
        return z, prof

    def describe(self, pv, perr=None):
        out = []
        for i, p in enumerate(self.params):
            out.append(dict(
                name=p.name, label=p.label, unit=p.unit,
                value=float(pv[i]),
                error=(None if perr is None or not np.isfinite(perr[i])
                       else float(perr[i])),
                lo=p.lo, hi=p.hi, vary=p.vary, group=p.group,
                at_bound=bool(p.vary and (
                    pv[i] <= p.lo + 1e-3 * (p.hi - p.lo) or
                    pv[i] >= p.hi - 1e-3 * (p.hi - p.lo)))))
        return out

    def summary(self, pv, perr=None):
        """The numbers a user actually reports."""
        rows = []
        for i, (lay, mat) in enumerate(zip(self.layers, self.mats), start=1):
            t = pv[self.index[f't{i}']]
            rho = pv[self.index[self.rho_key[i - 1]]]
            bulk = M.FILMS[lay['material']]['rho']
            rows.append(dict(
                layer=i, material=mat.label.split()[0], label=mat.label,
                thickness_A=float(t), thickness_nm=float(t) / 10.0,
                thickness_err_A=_finite(None if perr is None else
                                        perr[self.index[f't{i}']]),
                density=float(rho), bulk_density=bulk,
                density_pct_bulk=100.0 * float(rho) / bulk,
                roughness_top_A=float(pv[self.index[f'sig{i-1}']]),
                roughness_bot_A=float(pv[self.index[f'sig{i}']])))
        return rows
