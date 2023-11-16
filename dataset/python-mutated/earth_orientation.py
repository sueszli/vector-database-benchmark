"""
This module contains standard functions for earth orientation, such as
precession and nutation.

This module is (currently) not intended to be part of the public API, but
is instead primarily for internal use in `coordinates`
"""
import erfa
import numpy as np
from astropy.time import Time
from .builtin_frames.utils import get_jd12
from .matrix_utilities import matrix_transpose, rotation_matrix
jd1950 = Time('B1950').jd
jd2000 = Time('J2000').jd

def eccentricity(jd):
    if False:
        i = 10
        return i + 15
    "\n    Eccentricity of the Earth's orbit at the requested Julian Date.\n\n    Parameters\n    ----------\n    jd : scalar or array-like\n        Julian date at which to compute the eccentricity\n\n    Returns\n    -------\n    eccentricity : scalar or array\n        The eccentricity (or array of eccentricities)\n\n    References\n    ----------\n    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth\n      Seidelmann (ed), University Science Books (1992).\n    "
    T = (jd - jd1950) / 36525.0
    p = (-1.26e-07, -4.193e-05, 0.01673011)
    return np.polyval(p, T)

def mean_lon_of_perigee(jd):
    if False:
        i = 10
        return i + 15
    "\n    Computes the mean longitude of perigee of the Earth's orbit at the\n    requested Julian Date.\n\n    Parameters\n    ----------\n    jd : scalar or array-like\n        Julian date at which to compute the mean longitude of perigee\n\n    Returns\n    -------\n    mean_lon_of_perigee : scalar or array\n        Mean longitude of perigee in degrees (or array of mean longitudes)\n\n    References\n    ----------\n    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth\n      Seidelmann (ed), University Science Books (1992).\n    "
    T = (jd - jd1950) / 36525.0
    p = (0.012, 1.65, 6190.67, 1015489.951)
    return np.polyval(p, T) / 3600.0

def obliquity(jd, algorithm=2006):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the obliquity of the Earth at the requested Julian Date.\n\n    Parameters\n    ----------\n    jd : scalar or array-like\n        Julian date (TT) at which to compute the obliquity\n    algorithm : int\n        Year of algorithm based on IAU adoption. Can be 2006, 2000 or 1980.\n        The IAU 2006 algorithm is based on Hilton et al. 2006.\n        The IAU 1980 algorithm is based on the Explanatory Supplement to the\n        Astronomical Almanac (1992).\n        The IAU 2000 algorithm starts with the IAU 1980 algorithm and applies a\n        precession-rate correction from the IAU 2000 precession model.\n\n    Returns\n    -------\n    obliquity : scalar or array\n        Mean obliquity in degrees (or array of obliquities)\n\n    References\n    ----------\n    * Hilton, J. et al., 2006, Celest.Mech.Dyn.Astron. 94, 351\n    * Capitaine, N., et al., 2003, Astron.Astrophys. 400, 1145-1154\n    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth\n      Seidelmann (ed), University Science Books (1992).\n    '
    if algorithm == 2006:
        return np.rad2deg(erfa.obl06(jd, 0))
    elif algorithm == 2000:
        return np.rad2deg(erfa.obl80(jd, 0) + erfa.pr00(jd, 0)[1])
    elif algorithm == 1980:
        return np.rad2deg(erfa.obl80(jd, 0))
    else:
        raise ValueError('invalid algorithm year for computing obliquity')

def precession_matrix_Capitaine(fromepoch, toepoch):
    if False:
        print('Hello World!')
    '\n    Computes the precession matrix from one Julian epoch to another, per IAU 2006.\n\n    Parameters\n    ----------\n    fromepoch : `~astropy.time.Time`\n        The epoch to precess from.\n    toepoch : `~astropy.time.Time`\n        The epoch to precess to.\n\n    Returns\n    -------\n    pmatrix : 3x3 array\n        Precession matrix to get from ``fromepoch`` to ``toepoch``\n\n    References\n    ----------\n    Hilton, J. et al., 2006, Celest.Mech.Dyn.Astron. 94, 351\n    '
    fromepoch_to_J2000 = matrix_transpose(erfa.bp06(*get_jd12(fromepoch, 'tt'))[1])
    J2000_to_toepoch = erfa.bp06(*get_jd12(toepoch, 'tt'))[1]
    return J2000_to_toepoch @ fromepoch_to_J2000

def _precession_matrix_besselian(epoch1, epoch2):
    if False:
        for i in range(10):
            print('nop')
    "\n    Computes the precession matrix from one Besselian epoch to another using\n    Newcomb's method.\n\n    ``epoch1`` and ``epoch2`` are in Besselian year numbers.\n    "
    t1 = (epoch1 - 1850.0) / 1000.0
    t2 = (epoch2 - 1850.0) / 1000.0
    dt = t2 - t1
    zeta1 = 23035.545 + t1 * 139.72 + 0.06 * t1 * t1
    zeta2 = 30.24 - 0.27 * t1
    zeta3 = 17.995
    pzeta = (zeta3, zeta2, zeta1, 0)
    zeta = np.polyval(pzeta, dt) / 3600
    z1 = 23035.545 + t1 * 139.72 + 0.06 * t1 * t1
    z2 = 109.48 + 0.39 * t1
    z3 = 18.325
    pz = (z3, z2, z1, 0)
    z = np.polyval(pz, dt) / 3600
    theta1 = 20051.12 - 85.29 * t1 - 0.37 * t1 * t1
    theta2 = -42.65 - 0.37 * t1
    theta3 = -41.8
    ptheta = (theta3, theta2, theta1, 0)
    theta = np.polyval(ptheta, dt) / 3600
    return rotation_matrix(-z, 'z') @ rotation_matrix(theta, 'y') @ rotation_matrix(-zeta, 'z')

def nutation_components2000B(jd):
    if False:
        print('Hello World!')
    '\n    Computes nutation components following the IAU 2000B specification.\n\n    Parameters\n    ----------\n    jd : scalar\n        Julian date (TT) at which to compute the nutation components\n\n    Returns\n    -------\n    eps : float\n        epsilon in radians\n    dpsi : float\n        dpsi in radians\n    deps : float\n        depsilon in raidans\n    '
    (dpsi, deps, epsa, _, _, _, _, _) = erfa.pn00b(jd, 0)
    return (epsa, dpsi, deps)

def nutation_matrix(epoch):
    if False:
        while True:
            i = 10
    '\n    Nutation matrix generated from nutation components, IAU 2000B model.\n\n    Matrix converts from mean coordinate to true coordinate as\n    r_true = M * r_mean\n\n    Parameters\n    ----------\n    epoch : `~astropy.time.Time`\n        The epoch at which to compute the nutation matrix\n\n    Returns\n    -------\n    nmatrix : 3x3 array\n        Nutation matrix for the specified epoch\n\n    References\n    ----------\n    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth\n      Seidelmann (ed), University Science Books (1992).\n    '
    return erfa.num00b(*get_jd12(epoch, 'tt'))