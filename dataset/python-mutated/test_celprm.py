from copy import copy, deepcopy
import numpy as np
import pytest
from astropy import wcs
_WCS_UNDEFINED = 9.87654321e+107

def test_celprm_init():
    if False:
        for i in range(10):
            print('nop')
    assert wcs.WCS().wcs.cel
    assert wcs.Celprm()
    with pytest.raises(wcs.InvalidPrjParametersError):
        cel = wcs.Celprm()
        cel.set()
    cel = wcs.Celprm()
    del cel

def test_celprm_copy():
    if False:
        print('Hello World!')
    cel = wcs.Celprm()
    cel2 = copy(cel)
    cel3 = copy(cel2)
    cel.ref = [6, 8, 18, 3]
    assert np.allclose(cel.ref, cel2.ref, atol=1e-12, rtol=0) and np.allclose(cel.ref, cel3.ref, atol=1e-12, rtol=0)
    del cel, cel2, cel3
    cel = wcs.Celprm()
    cel2 = deepcopy(cel)
    cel.ref = [6, 8, 18, 3]
    assert not np.allclose(cel.ref, cel2.ref, atol=1e-12, rtol=0)
    del cel, cel2

def test_celprm_offset():
    if False:
        for i in range(10):
            print('nop')
    cel = wcs.Celprm()
    assert not cel.offset
    cel.offset = True
    assert cel.offset

def test_celprm_prj():
    if False:
        for i in range(10):
            print('nop')
    cel = wcs.Celprm()
    assert cel.prj is not None
    cel.prj.code = 'TAN'
    cel.set()
    assert cel._flag

def test_celprm_phi0():
    if False:
        print('Hello World!')
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    assert cel.phi0 == None
    assert cel._flag == 0
    cel.set()
    assert cel.phi0 == 0.0
    cel.phi0 = 0.0
    assert cel._flag
    cel.phi0 = 2.0
    assert cel._flag == 0
    cel.phi0 = None
    assert cel.phi0 == None
    assert cel._flag == 0

def test_celprm_theta0():
    if False:
        i = 10
        return i + 15
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    assert cel.theta0 == None
    assert cel._flag == 0
    cel.theta0 = 4.0
    cel.set()
    assert cel.theta0 == 4.0
    cel.theta0 = 4.0
    assert cel._flag
    cel.theta0 = 8.0
    assert cel._flag == 0
    cel.theta0 = None
    assert cel.theta0 == None
    assert cel._flag == 0

def test_celprm_ref():
    if False:
        return 10
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    cel.set()
    assert np.allclose(cel.ref, [0.0, 0.0, 180.0, 0.0], atol=1e-12, rtol=0)
    cel.phi0 = 2.0
    cel.theta0 = 4.0
    cel.ref = [123, 12]
    cel.set()
    assert np.allclose(cel.ref, [123.0, 12.0, 2, 82], atol=1e-12, rtol=0)
    cel.ref = [None, 13, None, None]
    assert np.allclose(cel.ref, [123.0, 13.0, 2, 82], atol=1e-12, rtol=0)

def test_celprm_isolat():
    if False:
        return 10
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    cel.set()
    assert cel.isolat == 0

def test_celprm_latpreq():
    if False:
        for i in range(10):
            print('nop')
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    cel.set()
    assert cel.latpreq == 0

def test_celprm_euler():
    if False:
        print('Hello World!')
    cel = wcs.Celprm()
    cel.prj.code = 'TAN'
    cel.set()
    assert np.allclose(cel.euler, [0.0, 90.0, 180.0, 0.0, 1.0], atol=1e-12, rtol=0)