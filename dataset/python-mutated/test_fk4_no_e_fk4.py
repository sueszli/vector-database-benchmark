import numpy as np
from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.coordinates.builtin_frames import FK4, FK4NoETerms
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import get_pkg_data_contents
from . import N_ACCURACY_TESTS
TOLERANCE = 1e-05

def test_fk4_no_e_fk4():
    if False:
        for i in range(10):
            print('nop')
    lines = get_pkg_data_contents('data/fk4_no_e_fk4.csv').split('\n')
    t = Table.read(lines, format='ascii', delimiter=',', guess=False)
    if N_ACCURACY_TESTS >= len(t):
        idxs = range(len(t))
    else:
        idxs = np.random.randint(len(t), size=N_ACCURACY_TESTS)
    diffarcsec1 = []
    diffarcsec2 = []
    for i in idxs:
        r = t[int(i)]
        c1 = FK4(ra=r['ra_in'] * u.deg, dec=r['dec_in'] * u.deg, obstime=Time(r['obstime']))
        c2 = c1.transform_to(FK4NoETerms())
        diff = angular_separation(c2.ra.radian, c2.dec.radian, np.radians(r['ra_fk4ne']), np.radians(r['dec_fk4ne']))
        diffarcsec1.append(np.degrees(diff) * 3600.0)
        c1 = FK4NoETerms(ra=r['ra_in'] * u.deg, dec=r['dec_in'] * u.deg, obstime=Time(r['obstime']))
        c2 = c1.transform_to(FK4())
        diff = angular_separation(c2.ra.radian, c2.dec.radian, np.radians(r['ra_fk4']), np.radians(r['dec_fk4']))
        diffarcsec2.append(np.degrees(diff) * 3600.0)
    np.testing.assert_array_less(diffarcsec1, TOLERANCE)
    np.testing.assert_array_less(diffarcsec2, TOLERANCE)