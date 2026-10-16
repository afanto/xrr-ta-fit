"""Thickness seeding from the Fourier transform of the reflectivity.

A plain FFT against q reads systematically high because refraction compresses
the fringe spacing at low q.  Transforming against
    q_c = 2k sqrt(sin^2(theta) - sin^2(theta_c))
removes that bias, which matters here because the seed sets the search bounds.
"""
import numpy as np


def _spectrum(x, y, pad=16):
    y = y - np.polyval(np.polyfit(x, y, 3), x)
    y = y * np.hanning(y.size)
    n = int(2 ** np.ceil(np.log2(max(y.size * pad, 1024))))
    F = np.abs(np.fft.rfft(y, n))
    t = 2 * np.pi * np.fft.rfftfreq(n, d=x[1] - x[0])
    return t, F


def fft_thicknesses(two_theta, cps, wavelength, theta_c_deg=0.0,
                    tmin_ang=15.0, tmax_ang=2000.0, nmax=6):
    """Candidate layer thicknesses (Angstrom), strongest peak first.

    Also returns the transform itself so the UI can show it.
    """
    th = two_theta / 2.0
    k = 2 * np.pi / wavelength

    # stay above the critical region, where the fringes actually live
    lo = max(1.6 * 2 * theta_c_deg, two_theta[0])
    m = two_theta >= lo
    if m.sum() < 64:
        m = two_theta >= two_theta[0]
    th, y = th[m], np.log10(np.maximum(cps[m], 1e-12))

    s2 = np.sin(np.radians(th)) ** 2 - np.sin(np.radians(theta_c_deg)) ** 2
    good = s2 > 0
    if good.sum() < 64:
        good = np.ones_like(s2, bool)
        s2 = np.maximum(s2, 1e-12)
    qc = 2 * k * np.sqrt(s2[good])
    yq = y[good] + 4 * np.log10(np.maximum(qc, 1e-6))

    xx = np.linspace(qc[0], qc[-1], 4096)
    yy = np.interp(xx, qc, yq)
    t, F = _spectrum(xx, yy)

    band = (t > tmin_ang) & (t < tmax_ang)
    if not band.any():
        return [], [], dict(t=[], amp=[])
    Fb, tb = F[band], t[band]

    # local maxima, strongest first, keeping peaks reasonably separated
    peaks, amps = [], []
    order = np.argsort(Fb)[::-1]
    resol = 2 * np.pi / (qc[-1] - qc[0])            # intrinsic FFT resolution
    for i in order:
        if i == 0 or i == Fb.size - 1:
            continue
        if not (Fb[i] >= Fb[i - 1] and Fb[i] >= Fb[i + 1]):
            continue
        if Fb[i] < 0.04 * Fb.max():
            break
        a, b, c = Fb[i - 1], Fb[i], Fb[i + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        tp = tb[i] + shift * (tb[1] - tb[0])
        if all(abs(tp - p) > 0.8 * resol for p in peaks):
            peaks.append(float(tp))
            amps.append(float(Fb[i]))
        if len(peaks) >= nmax:
            break

    keep = slice(0, min(tb.size, 4000))
    return peaks, amps, dict(t=tb[keep].tolist(),
                       amp=(Fb[keep] / Fb.max()).tolist(),
                       resolution=float(resol))


def drop_harmonics(peaks, amps, tol=0.06):
    """Remove peaks that are an integer multiple of a stronger peak.

    A single film produces fringe harmonics at 2t, 3t ...; left in the list
    they masquerade as a much thicker layer, and for a bilayer a harmonic can
    even satisfy the t1 + t2 = t_total test against itself.
    """
    keep = []
    for t, a in zip(peaks, amps):
        harmonic = False
        for t0, a0 in zip(peaks, amps):
            if t0 >= t or a0 <= a:
                continue
            m = t / t0
            if abs(m - round(m)) < tol * round(m) and round(m) >= 2:
                harmonic = True
                break
        if not harmonic:
            keep.append((t, a))
    return [t for t, _ in keep], [a for _, a in keep]


def grid_size(nlayers, budget=40000):
    """Points per axis so the whole scan stays inside a time budget."""
    if nlayers <= 1:
        return 400
    n = int(round(budget ** (1.0 / nlayers)))
    return int(min(max(n, 8), 72 if nlayers == 2 else 34))


def scan_candidates(cost_of_thickness, nlayers, t_hi, t_lo=8.0, ngrid=None,
                    nkeep=None):
    """Candidate thicknesses from a direct scan of the cost surface.

    Far more robust than reading periodicities off the transform: a thin
    capping or seed layer produces a weak FFT peak that is easily missed, and
    harmonics of a thick layer produce strong peaks that mean nothing.
    Scanning the actual misfit cannot be fooled by either.

    `cost_of_thickness` takes a thickness vector of length `nlayers` and
    returns a scalar.  Works for any number of layers; the grid is coarsened
    as the dimension grows so the cost stays bounded.
    """
    n = ngrid or grid_size(nlayers)
    nkeep = nkeep or (6 if nlayers == 1 else 8)
    ts = np.geomspace(t_lo, t_hi, n)

    C = np.empty((n,) * nlayers, dtype=float)
    for idx in np.ndindex(C.shape):
        v = cost_of_thickness([float(ts[i]) for i in idx])
        C[idx] = v if np.isfinite(v) else np.inf
    if not np.isfinite(C).any():
        return [[t_hi / (nlayers + 1)] * nlayers]

    # cells that are no worse than every neighbour, best first
    try:
        from scipy.ndimage import minimum_filter
        local = C <= minimum_filter(C, size=3, mode='nearest')
    except ImportError:                                   # pragma: no cover
        local = np.ones_like(C, dtype=bool)
    idxs = np.argwhere(local & np.isfinite(C))
    idxs = idxs[np.argsort(C[tuple(idxs.T)])]

    out = []
    for idx in idxs:
        t = [float(ts[i]) for i in idx]
        if all(max(abs(a - b) / max(b, 1e-9) for a, b in zip(t, o)) > 0.10
               for o in out):
            out.append(t)
        if len(out) >= nkeep:
            break
    return out or [[t_hi / (nlayers + 1)] * nlayers]


def _local_minima(c):
    m = np.ones(c.size, bool)
    m[1:-1] = (c[1:-1] <= c[:-2]) & (c[1:-1] <= c[2:])
    m[0] = c[0] <= c[1]
    m[-1] = c[-1] <= c[-2]
    idx = np.where(m)[0]
    return idx if idx.size else np.array([int(np.argmin(c))])


def seed_candidates(peaks, nlayers, tmax_hint=None):
    """Candidate thickness vectors (top-down), best guess first.

    The transform gives a set of periodicities but says nothing about which
    layer each one belongs to, so both orderings of every plausible pair are
    returned and the fit decides.  For a bilayer the spectrum typically shows
    t1, t2 and t1+t2, so pairs that add up to another peak are ranked first.
    """
    if not peaks:
        return [[100.0] * nlayers]

    if nlayers == 1:
        return [[p] for p in peaks[:3]]

    cands, seen = [], set()

    def add(a, b, rank):
        for pair in ((a, b), (b, a)):
            if min(pair) < 8 or max(pair) > 4000:
                continue
            key = (round(pair[0], 1), round(pair[1], 1))
            if key not in seen:
                seen.add(key)
                cands.append((rank, [float(pair[0]), float(pair[1])]))

    # pairs that sum to another observed peak: the classic t1, t2, t1+t2 triple
    for i in range(len(peaks)):
        for j in range(i, len(peaks)):
            for k in range(len(peaks)):
                if k in (i, j):
                    continue
                e = abs(peaks[i] + peaks[j] - peaks[k]) / peaks[k]
                if e < 0.12:
                    add(peaks[i], peaks[j], 0 + e)

    # total minus each smaller peak, in case the complement was not resolved
    tot = max(peaks)
    for p in peaks:
        if p < 0.92 * tot:
            add(p, tot - p, 1.0)

    # the two strongest peaks taken at face value
    if len(peaks) >= 2:
        add(peaks[0], peaks[1], 2.0)

    # last resort: split the dominant period
    for frac in (0.5, 0.3, 0.7):
        add(tot * frac, tot * (1 - frac), 3.0 + frac)

    cands.sort(key=lambda x: x[0])
    return [c[1] for c in cands[:10]]


def bounds_from_candidates(cands, nlayers, pad=0.30):
    """Per-layer (lo, hi) wide enough to contain every candidate."""
    out = []
    for i in range(nlayers):
        vals = [c[i] for c in cands]
        lo = max(3.0, min(vals) * (1 - pad))
        hi = max(vals) * (1 + pad) + 20.0
        out.append((lo, hi))
    return out
