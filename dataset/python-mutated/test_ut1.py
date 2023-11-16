import functools
import numpy as np
import pytest
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf
from astropy.utils.iers import iers
allclose_jd = functools.partial(np.allclose, rtol=0, atol=1e-09)
allclose_sec = functools.partial(np.allclose, rtol=1e-15, atol=0.0001)
try:
    iers.IERS_A.open()
except OSError:
    HAS_IERS_A = False
else:
    HAS_IERS_A = True

def do_ut1_prediction_tst(iers_type):
    if False:
        return 10
    tnow = Time.now()
    iers_tab = iers_type.open()
    (tnow.delta_ut1_utc, status) = iers_tab.ut1_utc(tnow, return_status=True)
    assert status == iers.FROM_IERS_A_PREDICTION
    tnow_ut1_jd = tnow.ut1.jd
    assert tnow_ut1_jd != tnow.jd
    delta_ut1_utc = tnow.delta_ut1_utc
    with iers.earth_orientation_table.set(iers_type.open()):
        (delta2, status2) = tnow.get_delta_ut1_utc(return_status=True)
        assert status2 == status
        assert delta2.to_value('s') == delta_ut1_utc
        tnow_ut1 = tnow.ut1
        assert tnow_ut1._delta_ut1_utc == delta_ut1_utc
        assert tnow_ut1.jd != tnow.jd

@pytest.mark.remote_data
class TestTimeUT1Remote:

    def setup_class(cls):
        if False:
            print('Hello World!')
        iers_conf.auto_download = True

    def teardown_class(cls):
        if False:
            while True:
                i = 10
        iers_conf.auto_download = False

    def test_utc_to_ut1(self):
        if False:
            return 10
        'Test conversion of UTC to UT1, making sure to include a leap second'
        t = Time(['2012-06-30 12:00:00', '2012-06-30 23:59:59', '2012-06-30 23:59:60', '2012-07-01 00:00:00', '2012-07-01 12:00:00'], scale='utc')
        t_ut1_jd = t.ut1.jd
        t_comp = np.array([2456108.999993208, 2456109.499981634, 2456109.4999932083, 2456109.5000047823, 2456110.0000047833])
        assert allclose_jd(t_ut1_jd, t_comp)
        t_back = t.ut1.utc
        assert allclose_jd(t.jd, t_back.jd)
        tnow = Time.now()
        tnow.ut1

    def test_ut1_iers_auto(self):
        if False:
            while True:
                i = 10
        do_ut1_prediction_tst(iers.IERS_Auto)

class TestTimeUT1:
    """Test Time.ut1 using IERS tables"""

    def test_ut1_to_utc(self):
        if False:
            print('Hello World!')
        'Also test the reverse, around the leap second\n        (round-trip test closes #2077)'
        with iers_conf.set_temp('auto_download', False):
            t = Time(['2012-06-30 12:00:00', '2012-06-30 23:59:59', '2012-07-01 00:00:00', '2012-07-01 00:00:01', '2012-07-01 12:00:00'], scale='ut1')
            t_utc_jd = t.utc.jd
            t_comp = np.array([2456109.000001005, 2456109.499983644, 2456109.4999952177, 2456109.5000067917, 2456109.9999952167])
            assert allclose_jd(t_utc_jd, t_comp)
            t_back = t.utc.ut1
            assert allclose_jd(t.jd, t_back.jd)

    def test_empty_ut1(self):
        if False:
            while True:
                i = 10
        'Testing for a zero-length Time object from UTC to UT1\n        when an empty array is passed'
        from astropy import units as u
        with iers_conf.set_temp('auto_download', False):
            t = Time(['2012-06-30 12:00:00']) + np.arange(24) * u.hour
            t_empty = t[[]].ut1
            assert isinstance(t_empty, Time)
            assert t_empty.scale == 'ut1'
            assert t_empty.size == 0

    def test_delta_ut1_utc(self):
        if False:
            i = 10
            return i + 15
        'Accessing delta_ut1_utc should try to get it from IERS\n        (closes #1924 partially)'
        with iers_conf.set_temp('auto_download', False):
            t = Time('2012-06-30 12:00:00', scale='utc')
            assert not hasattr(t, '_delta_ut1_utc')
            assert allclose_sec(t.delta_ut1_utc, -0.5868211000312497)
            assert allclose_sec(t._delta_ut1_utc, -0.5868211000312497)

class TestTimeUT1SpecificIERSTable:

    @pytest.mark.skipif(not HAS_IERS_A, reason='requires IERS_A')
    def test_ut1_iers_A(self):
        if False:
            i = 10
            return i + 15
        do_ut1_prediction_tst(iers.IERS_A)

    def test_ut1_iers_B(self):
        if False:
            print('Hello World!')
        tnow = Time.now()
        iers_b = iers.IERS_B.open()
        (delta1, status1) = tnow.get_delta_ut1_utc(iers_b, return_status=True)
        assert status1 == iers.TIME_BEYOND_IERS_RANGE
        with iers.earth_orientation_table.set(iers.IERS_B.open()):
            (delta2, status2) = tnow.get_delta_ut1_utc(return_status=True)
            assert status2 == status1
            with pytest.raises(iers.IERSRangeError):
                tnow.ut1