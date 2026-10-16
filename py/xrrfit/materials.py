"""Material database and X-ray optical constants.

Everything downstream only needs (delta, beta) per unit mass density, so a
material is stored as a chemical formula plus a nominal bulk density; the
fitted density then scales delta and beta linearly.
"""
import periodictable as pt

CU_KA1 = 1.540598          # Angstrom


def _mix(pairs, density):
    """Weight-percent mixture -> periodictable formula."""
    args = []
    for formula, wt in pairs:
        args += [pt.formula(formula), wt]
    return pt.mix_by_weight(*args, density=density)


# --------------------------------------------------------------------------
# films the user can stack.  `rho_min`/`rho_max` bracket what a sputtered film
# can plausibly be: dense films never exceed the bulk value by more than a
# percent or so, and porous ones rarely fall below ~65 % of bulk.
# --------------------------------------------------------------------------
FILMS = {
    'Ru':   dict(formula='Ru',   rho=12.45, label='Ru  (ruthenium)'),
    'Ta':   dict(formula='Ta',   rho=16.65, label='Ta  (tantalum, bcc)'),
    'Ta_b': dict(formula='Ta',   rho=16.30, label='beta-Ta  (sputtered)'),
    'Pt':   dict(formula='Pt',   rho=21.45, label='Pt  (platinum)'),
    'Co':   dict(formula='Co',   rho=8.90,  label='Co  (cobalt)'),
    'Fe':   dict(formula='Fe',   rho=7.874, label='Fe  (iron)'),
    'CoFe': dict(formula='Co50Fe50', rho=8.30, label='CoFe  (50:50)'),
    'CoFeB': dict(formula='Co40Fe40B20', rho=8.00, label='CoFeB  (40/40/20)'),
    # Fe-rich CoFeB, the MTJ free-layer composition.  Volume additivity with
    # bulk boron gives 7.30, which is too low: boron sits interstitially in
    # the amorphous alloy rather than in its own open lattice.  Scaling the
    # 40/40/20 entry by the additivity ratio of the two compositions puts
    # this at 7.80 - an estimate, so fit the density rather than trusting it.
    'FeCoB_602020': dict(formula='Fe60Co20B20', rho=7.80,
                         label='Fe60Co20B20  (Fe-rich CoFeB)'),
    'Cu':   dict(formula='Cu',   rho=8.96,  label='Cu  (copper)'),
    'Ti':   dict(formula='Ti',   rho=4.506, label='Ti  (titanium)'),
    'Au':   dict(formula='Au',   rho=19.30, label='Au  (gold)'),
    'MgO':  dict(formula='MgO',  rho=3.58,  label='MgO'),
    # native oxides, offered as an optional surface cap
    'RuO2':  dict(formula='RuO2',  rho=6.97, label='RuO2'),
    'Ta2O5': dict(formula='Ta2O5', rho=8.20, label='Ta2O5'),
    'CoO':   dict(formula='CoO',   rho=6.44, label='CoO'),
    'Fe2O3': dict(formula='Fe2O3', rho=5.24, label='Fe2O3 (hematite)'),
    'Fe3O4': dict(formula='Fe3O4', rho=5.17, label='Fe3O4 (magnetite)'),
    'PtO2':  dict(formula='PtO2',  rho=10.2, label='PtO2'),
    'TiO2':  dict(formula='TiO2',  rho=4.23, label='TiO2'),
    'SiO2':  dict(formula='SiO2',  rho=2.20, label='SiO2'),
}

# which oxide to suggest as the native cap for each metal
NATIVE_OXIDE = {'Ru': 'RuO2', 'Ta': 'Ta2O5', 'Ta_b': 'Ta2O5', 'Pt': 'PtO2',
                'Co': 'CoO', 'Fe': 'Fe2O3', 'CoFe': 'Fe2O3',
                'CoFeB': 'Fe2O3', 'FeCoB_602020': 'Fe2O3',
                'Ti': 'TiO2', 'Cu': 'SiO2'}

# --------------------------------------------------------------------------
# substrates
# --------------------------------------------------------------------------
SUBSTRATES = {
    'borofloat': dict(
        label='Borosilicate glass (Borofloat 33 / Pyrex)', rho=2.23,
        mix=[('SiO2', 81), ('B2O3', 13), ('Na2O', 4), ('Al2O3', 2)]),
    'd263': dict(
        label='Borosilicate coverglass (Schott D263 T)', rho=2.51,
        mix=[('SiO2', 64.1), ('B2O3', 8.4), ('Al2O3', 4.2), ('Na2O', 6.4),
             ('K2O', 6.9), ('TiO2', 4.0), ('ZnO', 5.9)]),
    'sodalime': dict(
        label='Soda-lime glass (microscope slide)', rho=2.50,
        mix=[('SiO2', 73), ('Na2O', 14), ('CaO', 9), ('MgO', 4)]),
    'silica': dict(label='Fused silica / quartz (SiO2)', rho=2.20,
                   formula='SiO2'),
    'si':     dict(label='Silicon (100)', rho=2.329, formula='Si'),
    'sio2_si': dict(label='Silicon with native SiO2', rho=2.329, formula='Si'),
    'sapphire': dict(label='Sapphire (Al2O3)', rho=3.98, formula='Al2O3'),
    'mgo':    dict(label='MgO', rho=3.58, formula='MgO'),
    'gaas':   dict(label='GaAs', rho=5.317, formula='GaAs'),
}


class Material:
    """delta and beta that scale with a fittable mass density."""

    __slots__ = ('key', 'label', 'formula', 'rho0',
                 'delta_per_rho', 'beta_per_rho')

    def __init__(self, key, label, comp, rho0, wavelength):
        self.key, self.label, self.rho0 = key, label, rho0
        if isinstance(comp, str):
            mat = pt.formula(comp, density=rho0)
        else:
            mat = _mix(comp, rho0)
        self.formula = str(mat)
        re_sld, im_sld = pt.xray_sld(mat, wavelength=wavelength)[:2]
        fac = wavelength ** 2 / (2 * 3.141592653589793)
        self.delta_per_rho = re_sld * 1e-6 * fac / rho0
        self.beta_per_rho = abs(im_sld) * 1e-6 * fac / rho0

    def n(self, density):
        return 1.0 - self.delta_per_rho * density + 1j * self.beta_per_rho * density

    def delta(self, density):
        return self.delta_per_rho * density

    def theta_c(self, density):
        """Critical angle in degrees."""
        import numpy as np
        return np.degrees(np.sqrt(2 * self.delta(density)))


_cache = {}


def film(key, wavelength=CU_KA1):
    ck = ('f', key, wavelength)
    if ck not in _cache:
        d = FILMS[key]
        _cache[ck] = Material(key, d['label'], d['formula'], d['rho'], wavelength)
    return _cache[ck]


def substrate(key, wavelength=CU_KA1):
    ck = ('s', key, wavelength)
    if ck not in _cache:
        d = SUBSTRATES[key]
        comp = d.get('formula') or d['mix']
        _cache[ck] = Material(key, d['label'], comp, d['rho'], wavelength)
    return _cache[ck]


def catalogue(wavelength=CU_KA1):
    """Everything the front end needs to populate its menus."""
    metals = ['Ru', 'Ta', 'Ta_b', 'Pt', 'Co', 'Fe', 'CoFe', 'CoFeB',
              'FeCoB_602020', 'Cu', 'Ti', 'Au', 'MgO']
    oxides = ['RuO2', 'Ta2O5', 'CoO', 'Fe2O3', 'Fe3O4', 'PtO2', 'TiO2', 'SiO2']

    def entry(k):
        m = film(k, wavelength)
        return dict(key=k, label=m.label, rho=FILMS[k]['rho'],
                    delta=m.delta(FILMS[k]['rho']),
                    theta_c=m.theta_c(FILMS[k]['rho']))

    return dict(
        films=[entry(k) for k in metals],
        oxides=[entry(k) for k in oxides],
        native_oxide=NATIVE_OXIDE,
        substrates=[dict(key=k, label=SUBSTRATES[k]['label'],
                         rho=SUBSTRATES[k]['rho'],
                         delta=substrate(k, wavelength).delta(SUBSTRATES[k]['rho']))
                    for k in SUBSTRATES],
    )
