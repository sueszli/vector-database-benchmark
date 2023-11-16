import numpy as np
from scipy.special import ellipkinc as ellip_F, ellipeinc as ellip_E

def ellipsoid(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates ellipsoid with semimajor axes aligned with grid dimensions\n    on grid with specified `spacing`.\n\n    Parameters\n    ----------\n    a : float\n        Length of semimajor axis aligned with x-axis.\n    b : float\n        Length of semimajor axis aligned with y-axis.\n    c : float\n        Length of semimajor axis aligned with z-axis.\n    spacing : 3-tuple of floats\n        Spacing in three spatial dimensions.\n    levelset : bool\n        If True, returns the level set for this ellipsoid (signed level\n        set about zero, with positive denoting interior) as np.float64.\n        False returns a binarized version of said level set.\n\n    Returns\n    -------\n    ellipsoid : (M, N, P) array\n        Ellipsoid centered in a correctly sized array for given `spacing`.\n        Boolean dtype unless `levelset=True`, in which case a float array is\n        returned with the level set above 0.0 representing the ellipsoid.\n\n    '
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError('Parameters a, b, and c must all be > 0')
    offset = np.r_[1, 1, 1] * np.r_[spacing]
    low = np.ceil(-np.r_[a, b, c] - offset)
    high = np.floor(np.r_[a, b, c] + offset + 1)
    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])
    (x, y, z) = np.mgrid[low[0]:high[0]:spacing[0], low[1]:high[1]:spacing[1], low[2]:high[2]:spacing[2]]
    if not levelset:
        arr = (x / float(a)) ** 2 + (y / float(b)) ** 2 + (z / float(c)) ** 2 <= 1
    else:
        arr = (x / float(a)) ** 2 + (y / float(b)) ** 2 + (z / float(c)) ** 2 - 1
    return arr

def ellipsoid_stats(a, b, c):
    if False:
        while True:
            i = 10
    '\n    Calculates analytical surface area and volume for ellipsoid with\n    semimajor axes aligned with grid dimensions of specified `spacing`.\n\n    Parameters\n    ----------\n    a : float\n        Length of semimajor axis aligned with x-axis.\n    b : float\n        Length of semimajor axis aligned with y-axis.\n    c : float\n        Length of semimajor axis aligned with z-axis.\n\n    Returns\n    -------\n    vol : float\n        Calculated volume of ellipsoid.\n    surf : float\n        Calculated surface area of ellipsoid.\n\n    '
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError('Parameters a, b, and c must all be > 0')
    abc = [a, b, c]
    abc.sort(reverse=True)
    a = abc[0]
    b = abc[1]
    c = abc[2]
    vol = 4 / 3.0 * np.pi * a * b * c
    phi = np.arcsin((1.0 - c ** 2 / a ** 2.0) ** 0.5)
    d = float((a ** 2 - c ** 2) ** 0.5)
    m = a ** 2 * (b ** 2 - c ** 2) / float(b ** 2 * (a ** 2 - c ** 2))
    F = ellip_F(phi, m)
    E = ellip_E(phi, m)
    surf = 2 * np.pi * (c ** 2 + b * c ** 2 / d * F + b * d * E)
    return (vol, surf)