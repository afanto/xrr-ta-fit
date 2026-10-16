"""Read XRR scans from PANalytical .xrdml or plain two/three-column text."""
import io
import re
import xml.etree.ElementTree as ET

import numpy as np

CU_KA1 = 1.540598


class LoadError(ValueError):
    pass


# --------------------------------------------------------------------------
def read_xrdml(data):
    """PANalytical .xrdml (any 1.x schema version)."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise LoadError(f'not valid XML: {e}')
    ns = {'x': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    def find(node, path):
        return node.find(path if not ns else
                         '/'.join('x:' + p for p in path.split('/')), ns)

    meas = find(root, 'xrdMeasurement')
    if meas is None:
        raise LoadError('no <xrdMeasurement> element')

    wl_el = find(meas, 'usedWavelength/kAlpha1')
    wavelength = float(wl_el.text) if wl_el is not None else CU_KA1

    scan = find(meas, 'scan')
    if scan is None:
        raise LoadError('no <scan> element')
    dp = find(scan, 'dataPoints')

    inten = find(dp, 'intensities')
    if inten is None:
        raise LoadError('no <intensities> element')
    counts = np.array([float(v) for v in inten.text.split()])

    axis = {}
    for pos in (dp.findall('x:positions', ns) if ns else dp.findall('positions')):
        name = pos.get('axis')
        start = find(pos, 'startPosition')
        if start is not None:
            axis[name] = (float(start.text), float(find(pos, 'endPosition').text))

    if '2Theta' in axis:
        a, b = axis['2Theta']
        two_theta = np.linspace(a, b, counts.size)
    elif 'Omega' in axis:
        a, b = axis['Omega']
        two_theta = 2 * np.linspace(a, b, counts.size)
    else:
        raise LoadError('scan has neither a 2Theta nor an Omega axis')

    ct = find(dp, 'commonCountingTime')
    if ct is not None:
        time = np.full(counts.size, float(ct.text))
    else:
        cts = find(dp, 'countingTimes')
        time = (np.array([float(v) for v in cts.text.split()])
                if cts is not None else np.ones(counts.size))

    at = find(dp, 'beamAttenuationFactors')
    attn = (np.array([float(v) for v in at.text.split()])
            if at is not None else np.ones(counts.size))

    return dict(two_theta=two_theta, cps=counts / time, attn=attn,
                counts=counts, time=time, wavelength=wavelength,
                source='xrdml')


# --------------------------------------------------------------------------
_NUM = re.compile(r'^[\s]*[-+]?[\d.]')


def read_text(data):
    """Two- or three-column ASCII: angle, intensity[, error].

    Accepts whitespace, comma or semicolon separators and any of the usual
    comment markers.  If the first column never exceeds ~15 it is assumed to
    be 2theta in degrees; a column that looks like q (max < 2) is converted.
    """
    if isinstance(data, bytes):
        data = data.decode('utf-8', 'replace')

    rows, ncol = [], None
    for line in io.StringIO(data):
        s = line.strip()
        if not s or not _NUM.match(s):
            continue
        parts = re.split(r'[\s,;]+', s)
        try:
            vals = [float(x) for x in parts if x != '']
        except ValueError:
            continue
        if len(vals) < 2:
            continue
        rows.append(vals[:3])
        ncol = max(ncol or 0, len(vals[:3]))
    if len(rows) < 20:
        raise LoadError('could not find at least 20 numeric rows of data')

    n = min(len(r) for r in rows)
    arr = np.array([r[:n] for r in rows], dtype=float)
    x, y = arr[:, 0], arr[:, 1]

    if not np.all(np.diff(x) > 0):
        order = np.argsort(x)
        arr, x, y = arr[order], x[order], y[order]

    unit = '2theta'
    if x.max() < 2.0:                       # looks like q in 1/Angstrom
        two_theta = 2 * np.degrees(np.arcsin(np.clip(
            x * CU_KA1 / (4 * np.pi), -1, 1)))
        unit = 'q'
    elif x.max() < 15.0:
        two_theta = x
    else:
        raise LoadError(f'first column runs to {x.max():.3g}; expected 2theta '
                        'in degrees (<15) or q in 1/Angstrom (<2)')

    return dict(two_theta=two_theta, cps=y, attn=np.ones_like(y),
                counts=y, time=np.ones_like(y), wavelength=CU_KA1,
                source=f'text ({unit})')


def load(filename, data):
    name = (filename or '').lower()
    if name.endswith('.xrdml') or (isinstance(data, (bytes, str))
                                   and b'<xrdMeasurement' in
                                   (data if isinstance(data, bytes)
                                    else data.encode())):
        d = read_xrdml(data)
    else:
        d = read_text(data)

    tt, y = d['two_theta'], d['cps']
    good = np.isfinite(tt) & np.isfinite(y) & (y > 0)
    if good.sum() < 20:
        raise LoadError('fewer than 20 usable (positive) data points')
    for k in ('two_theta', 'cps', 'attn', 'counts', 'time'):
        d[k] = np.asarray(d[k])[good]

    d['filename'] = filename
    d['npts'] = int(d['two_theta'].size)
    d['range'] = [float(d['two_theta'][0]), float(d['two_theta'][-1])]
    return d
