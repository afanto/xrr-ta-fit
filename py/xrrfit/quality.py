"""Where does the scan stop carrying information?

Beyond some angle the fringes sink into the counting noise and the curve is
just background.  Fitting that region does not add information: it dilutes
chi-squared with points the model cannot be judged on, and it lets the
background parameter absorb structure that belongs to the film.
"""
import numpy as np


def fringe_snr(two_theta, cps, counts, attn=None, win_deg=0.7):
    """Sliding-window ratio of fringe modulation to counting noise.

    Returns (centres, snr).  The modulation is the scatter of log-intensity
    about a local straight line - that is the fringe signal.  The noise is the
    Poisson error on the same quantity.  Their ratio says whether anything is
    still oscillating above the noise.
    """
    tt = np.asarray(two_theta, float)
    y = np.log(np.maximum(np.asarray(cps, float), 1e-12))
    cnt = np.maximum(np.asarray(counts, float), 1.0)
    attn = np.ones_like(cnt) if attn is None else np.asarray(attn, float)
    sig = np.sqrt(attn / cnt)                       # relative -> log error

    step = np.median(np.diff(tt))
    half = max(int(round(win_deg / 2 / step)), 6)

    centres, snr = [], []
    for i in range(half, tt.size - half, max(half // 3, 1)):
        s = slice(i - half, i + half + 1)
        x, yy = tt[s], y[s]
        resid = yy - np.polyval(np.polyfit(x, yy, 1), x)
        modulation = resid.std()
        noise = np.sqrt(np.mean(sig[s] ** 2))
        centres.append(tt[i])
        snr.append(modulation / max(noise, 1e-9))
    return np.array(centres), np.array(snr)


def suggest_tmax(two_theta, cps, counts, attn=None, threshold=1.5,
                 min_span=1.0):
    """Largest 2theta worth fitting to.

    Walks in from the high-angle end and stops at the last window whose fringe
    modulation is still `threshold` times the counting noise.  Returns the full
    range unchanged if the fringes never die.
    """
    tt = np.asarray(two_theta, float)
    c, s = fringe_snr(tt, cps, counts, attn)
    if c.size == 0:
        return float(tt[-1]), dict(centres=[], snr=[])

    good = s >= threshold
    if good.all():
        cut = float(tt[-1])
    elif not good.any():
        cut = float(tt[-1])
    else:
        # last index where the signal was still above threshold
        last = int(np.where(good)[0][-1])
        cut = float(c[last])
        # never suggest cutting away almost everything
        if cut - tt[0] < min_span:
            cut = float(tt[-1])
    cut = min(cut, float(tt[-1]))
    return cut, dict(centres=c.tolist(), snr=s.tolist(), threshold=threshold)
