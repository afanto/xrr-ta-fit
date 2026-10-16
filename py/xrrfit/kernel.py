"""Parratt reflectivity and instrumental corrections.

Validated to ~1e-15 against an independently written Abeles transfer-matrix
implementation (see tests/test_kernel.py).
"""
import numpy as np
from scipy.special import erf


def parratt(theta_deg, wavelength, n_media, thickness, roughness):
    """Specular reflectivity with Nevot-Croce interface roughness.

    n_media    complex refractive indices, length N+2:
               [ambient, film_1 (top), ..., film_N, substrate]
    thickness  length N, Angstrom, top-down
    roughness  length N+1, Angstrom; roughness[i] is between medium i and i+1
    """
    th = np.radians(np.atleast_1d(theta_deg))
    k = 2 * np.pi / wavelength
    cos2 = np.cos(th) ** 2

    n = np.asarray(n_media, dtype=complex)
    kz = k * np.sqrt(n[:, None] ** 2 - cos2[None, :] + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)          # keep the decaying solution

    d = np.asarray(thickness, dtype=float)
    s = np.asarray(roughness, dtype=float)
    M = n.size

    j = M - 2
    R = ((kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1])
         * np.exp(-2.0 * kz[j] * kz[j + 1] * s[j] ** 2))
    for j in range(M - 3, -1, -1):
        rf = ((kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1])
              * np.exp(-2.0 * kz[j] * kz[j + 1] * s[j] ** 2))
        phase = np.exp(2j * kz[j + 1] * d[j])
        R = (rf + R * phase) / (1.0 + rf * R * phase)
    return np.abs(R) ** 2


def footprint(theta_deg, beam_fwhm_mm, sample_len_mm):
    """Fraction of a Gaussian beam intercepted by a finite sample."""
    if beam_fwhm_mm <= 0:
        return np.ones_like(theta_deg)
    sig = beam_fwhm_mm / 2.35482
    proj = sample_len_mm * np.sin(np.radians(np.abs(theta_deg)))
    return erf(proj / (2 * np.sqrt(2) * sig))


def smear(theta, y, fwhm_deg, tail_frac=0.0, tail_fwhm_deg=0.0):
    """Convolve with a narrow Gaussian core plus a weak broad component.

    The broad component stands in for the non-Gaussian wings of the beam,
    sample curvature, and the diffuse scatter an open detector accepts.
    Without it the sharp low-angle fringe minima come out much deeper than
    any real measurement.
    """
    step = theta[1] - theta[0]
    widths, weights = [], []
    if fwhm_deg > 0:
        widths.append(fwhm_deg / 2.35482 / step)
        weights.append(1.0 - tail_frac)
    if tail_frac > 0 and tail_fwhm_deg > 0:
        widths.append(tail_fwhm_deg / 2.35482 / step)
        weights.append(tail_frac)
    if not widths:
        return y

    half = int(np.ceil(4 * max(widths)))
    if half < 1:
        return y
    x = np.arange(-half, half + 1)
    kern = np.zeros_like(x, dtype=float)
    for s, w in zip(widths, weights):
        g = np.exp(-0.5 * (x / s) ** 2)
        kern += w * g / g.sum()
    kern /= kern.sum()
    ypad = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    return np.convolve(ypad, kern, mode='valid')


def sld_profile(n_media, thickness, roughness, dz=0.25, pad=40.0):
    """delta(z) with erf-broadened interfaces; z=0 at the free surface."""
    delta = np.array([1.0 - np.real(x) for x in n_media])
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(thickness, float))])
    z = np.arange(-pad, edges[-1] + pad, dz)
    prof = np.full_like(z, delta[0])
    for i, (e, sig) in enumerate(zip(edges, roughness)):
        step = 0.5 * (1 + erf((z - e) / (max(sig, 1e-6) * np.sqrt(2))))
        prof = prof + (delta[i + 1] - delta[i]) * step
    return z, prof
