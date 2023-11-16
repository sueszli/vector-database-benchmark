import copy
import numpy as np
import pytest
from astropy.constants import Constant
from astropy.units import Quantity as Q

def test_c():
    if False:
        while True:
            i = 10
    from astropy.constants.codata2010 import c
    assert c.value == 299792458.0
    assert c.si.value == 299792458.0
    assert c.cgs.value == 29979245800.0
    assert c.uncertainty == 0
    assert c.name
    assert c.reference
    assert c.unit

def test_h():
    if False:
        return 10
    from astropy.constants import h as h_current
    from astropy.constants.codata2010 import h
    assert abs(h.value - 6.62606957e-34) < 1e-43
    assert abs(h.si.value - 6.62606957e-34) < 1e-43
    assert abs(h.cgs.value - 6.62606957e-27) < 1e-36
    assert abs(h.value - h_current.value) > 4e-42
    assert h.uncertainty
    assert h.name
    assert h.reference
    assert h.unit

def test_e():
    if False:
        for i in range(10):
            print('nop')
    from astropy.constants.astropyconst13 import e as e_13
    E = Q(100.00000348276221, 'V/m')
    with pytest.raises(TypeError):
        e_13.cgs * E
    assert isinstance(e_13.si, Q)
    assert isinstance(e_13.gauss, Q)
    assert isinstance(e_13.esu, Q)
    assert e_13.gauss * E == Q(e_13.gauss.value * E.value, 'Fr V/m')
    assert e_13.esu * E == Q(e_13.esu.value * E.value, 'Fr V/m')

def test_g0():
    if False:
        i = 10
        return i + 15
    'Tests for #1263 demonstrating how g0 constant should behave.'
    from astropy.constants.astropyconst13 import g0
    assert g0.value == 9.80665
    assert g0.si.value == 9.80665
    assert g0.cgs.value == 980.665
    assert g0.uncertainty == 0
    assert g0.name
    assert g0.reference
    assert g0.unit
    assert g0.unit.physical_type == 'acceleration'

def test_b_wien():
    if False:
        i = 10
        return i + 15
    'b_wien should give the correct peak wavelength for\n    given blackbody temperature. The Sun is used in this test.\n\n    '
    from astropy import units as u
    from astropy.constants.astropyconst13 import b_wien
    t = 5778 * u.K
    w = (b_wien / t).to(u.nm)
    assert round(w.value) == 502

def test_pc():
    if False:
        i = 10
        return i + 15
    'Parsec is defined to use small-angle limit per IAU 2015 Resolution B 2.\n    iau2012 version still uses tan(parallax).\n    '
    from astropy import units as u
    from astropy.constants import iau2012
    plx = np.radians(1 / 3600)
    assert np.allclose(u.pc.to('m') / iau2012.pc.si.value, np.tan(plx) / plx, rtol=1e-14, atol=0)

def test_masses():
    if False:
        i = 10
        return i + 15
    'Ensure mass values are set up correctly.\n    https://github.com/astropy/astropy/issues/8920\n    '
    from astropy.constants import astropyconst13, astropyconst20, astropyconst40
    ref_text = "Allen's Astrophysical Quantities 4th Ed."
    assert astropyconst13.M_sun.reference == ref_text and astropyconst13.M_jup.reference == ref_text and (astropyconst13.M_earth.reference == ref_text)
    ref_text = 'IAU 2015 Resolution B 3 + CODATA 2014'
    assert astropyconst20.M_sun.reference == ref_text and astropyconst20.M_jup.reference == ref_text and (astropyconst20.M_earth.reference == ref_text)
    ref_text = 'IAU 2015 Resolution B 3 + CODATA 2018'
    assert astropyconst40.M_sun.reference == ref_text and astropyconst40.M_jup.reference == ref_text and (astropyconst40.M_earth.reference == ref_text)

def test_unit():
    if False:
        print('Hello World!')
    from astropy import units as u
    from astropy.constants import astropyconst13 as const
    for val in vars(const).values():
        if isinstance(val, Constant):
            assert not isinstance(val.unit, u.UnrecognizedUnit)

def test_copy():
    if False:
        return 10
    from astropy import constants as const
    cc = copy.deepcopy(const.c)
    assert cc == const.c
    cc = copy.copy(const.c)
    assert cc == const.c

def test_view():
    if False:
        return 10
    'Check that Constant and Quantity views can be taken (#3537, #3538).'
    from astropy.constants import c
    c2 = c.view(Constant)
    assert c2 == c
    assert c2.value == c.value
    assert c2.uncertainty == 0
    assert c2.name == c.name
    assert c2.reference == c.reference
    assert c2.unit == c.unit
    q1 = c.view(Q)
    assert q1 == c
    assert q1.value == c.value
    assert type(q1) is Q
    assert not hasattr(q1, 'reference')
    q2 = Q(c)
    assert q2 == c
    assert q2.value == c.value
    assert type(q2) is Q
    assert not hasattr(q2, 'reference')
    c3 = Q(c, subok=True)
    assert c3 == c
    assert c3.value == c.value
    assert c3.uncertainty == 0
    assert c3.name == c.name
    assert c3.reference == c.reference
    assert c3.unit == c.unit
    c4 = Q(c, subok=True, copy=False)
    assert c4 is c