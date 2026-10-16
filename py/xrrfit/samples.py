"""Stand-in for xrrfit.samples in the public site: no presets, no rates.

The page computes the Ta rate itself from thickness / time.
"""


def rates_from_layers(layers, fitted_A):
    return None


def rates(key, fitted_A):
    return None
