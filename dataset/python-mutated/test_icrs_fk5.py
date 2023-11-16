import numpy as np
from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.coordinates.builtin_frames import FK5, ICRS
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import get_pkg_data_contents
from . import N_ACCURACY_TESTS
TOLERANCE = 0.03

def test_icrs_fk5():
    if False:
        print('Hello World!')
    lines = get_pkg_data_contents('data/icrs_fk5.csv').split('\n')
    t = Table.read(lines, format='ascii', delimiter=',', guess=False)
    if N_ACCURACY_TESTS >= len(t):
        idxs = range(len(t))
    else:
        idxs = np.random.randint(len(t), size=N_ACCURACY_TESTS)
    diffarcsec1 = []
    diffarcsec2 = []
    for i in idxs:
        r = t[int(i)]
        c1 = ICRS(ra=r['ra_in'] * u.deg, dec=r['dec_in'] * u.deg)
        c2 = c1.transform_to(FK5(equinox=Time(r['equinox_fk5'])))
        diff = angular_separation(c2.ra.radian, c2.dec.radian, np.radians(r['ra_fk5']), np.radians(r['dec_fk5']))
        diffarcsec1.append(np.degrees(diff) * 3600.0)
        c1 = FK5(ra=r['ra_in'] * u.deg, dec=r['dec_in'] * u.deg, equinox=Time(r['equinox_fk5']))
        c2 = c1.transform_to(ICRS())
        diff = angular_separation(c2.ra.radian, c2.dec.radian, np.radians(r['ra_icrs']), np.radians(r['dec_icrs']))
        diffarcsec2.append(np.degrees(diff) * 3600.0)
    np.testing.assert_array_less(diffarcsec1, TOLERANCE)
    np.testing.assert_array_less(diffarcsec2, TOLERANCE)