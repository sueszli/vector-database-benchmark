import locale
import os
import platform
import urllib.request
import erfa
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from astropy.tests.tests.test_imports import test_imports
from astropy.time import Time, TimeDelta
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.iers import iers
test_imports()
assert erfa.leap_seconds._expires is None
iers_table = iers.LeapSeconds.auto_open()
erfa.leap_seconds.update(iers_table)
assert erfa.leap_seconds._expires is not None
SYSTEM_FILE = '/usr/share/zoneinfo/leap-seconds.list'
LEAP_SECOND_LIST = get_pkg_data_filename('data/leap-seconds.list')

def test_configuration():
    if False:
        for i in range(10):
            print('nop')
    assert iers.conf.iers_leap_second_auto_url == iers.IERS_LEAP_SECOND_URL
    assert iers.conf.ietf_leap_second_auto_url == iers.IETF_LEAP_SECOND_URL

class TestReading:
    """Basic tests that leap seconds can be read."""

    def verify_day_month_year(self, ls):
        if False:
            return 10
        assert np.all(ls['day'] == 1)
        assert np.all((ls['month'] == 1) | (ls['month'] == 7) | (ls['year'] < 1970))
        assert np.all(ls['year'] >= 1960)
        t = Time({'year': ls['year'], 'month': ls['month'], 'day': ls['day']}, format='ymdhms')
        assert np.all(t == Time(ls['mjd'], format='mjd'))

    def test_read_leap_second_dat(self):
        if False:
            return 10
        ls = iers.LeapSeconds.from_iers_leap_seconds(iers.IERS_LEAP_SECOND_FILE)
        assert ls.expires >= Time('2020-06-28', scale='tai')
        assert ls['mjd'][0] == 41317
        assert ls['tai_utc'][0] == 10
        assert ls['mjd'][-1] >= 57754
        assert ls['tai_utc'][-1] >= 37
        self.verify_day_month_year(ls)

    def test_read_leap_second_dat_locale(self):
        if False:
            return 10
        current = locale.setlocale(locale.LC_ALL)
        try:
            if platform.system() == 'Darwin':
                locale.setlocale(locale.LC_ALL, 'fr_FR')
            else:
                locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
            ls = iers.LeapSeconds.from_iers_leap_seconds(iers.IERS_LEAP_SECOND_FILE)
        except locale.Error as e:
            pytest.skip(f'Locale error: {e}')
        finally:
            locale.setlocale(locale.LC_ALL, current)
        assert ls.expires >= Time('2020-06-28', scale='tai')

    def test_open_leap_second_dat(self):
        if False:
            return 10
        ls = iers.LeapSeconds.from_iers_leap_seconds(iers.IERS_LEAP_SECOND_FILE)
        ls2 = iers.LeapSeconds.open(iers.IERS_LEAP_SECOND_FILE)
        assert np.all(ls == ls2)

    @pytest.mark.parametrize('file', (LEAP_SECOND_LIST, 'file:' + urllib.request.pathname2url(LEAP_SECOND_LIST)))
    def test_read_leap_seconds_list(self, file):
        if False:
            while True:
                i = 10
        ls = iers.LeapSeconds.from_leap_seconds_list(file)
        assert ls.expires == Time('2020-06-28', scale='tai')
        assert ls['mjd'][0] == 41317
        assert ls['tai_utc'][0] == 10
        assert ls['mjd'][-1] == 57754
        assert ls['tai_utc'][-1] == 37
        self.verify_day_month_year(ls)

    @pytest.mark.parametrize('file', (LEAP_SECOND_LIST, 'file:' + urllib.request.pathname2url(LEAP_SECOND_LIST)))
    def test_open_leap_seconds_list(self, file):
        if False:
            for i in range(10):
                print('nop')
        ls = iers.LeapSeconds.from_leap_seconds_list(file)
        ls2 = iers.LeapSeconds.open(file)
        assert np.all(ls == ls2)

    @pytest.mark.skipif(not os.path.isfile(SYSTEM_FILE), reason=f'system does not have {SYSTEM_FILE}')
    def test_open_system_file(self):
        if False:
            i = 10
            return i + 15
        ls = iers.LeapSeconds.open(SYSTEM_FILE)
        expired = ls.expires < Time.now()
        if expired:
            pytest.skip('System leap second file is expired.')
        assert not expired

def make_fake_file(expiration, tmp_path):
    if False:
        print('Hello World!')
    'copy the built-in IERS file but set a different expiration date.'
    ls = iers.LeapSeconds.from_iers_leap_seconds()
    fake_file = str(tmp_path / 'fake_leap_seconds.dat')
    with open(fake_file, 'w') as fh:
        fh.write('\n'.join([f'#  File expires on {expiration}'] + str(ls).split('\n')[2:-1]))
        return fake_file

def test_fake_file(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    fake_file = make_fake_file('28 June 2345', tmp_path)
    fake = iers.LeapSeconds.from_iers_leap_seconds(fake_file)
    assert fake.expires == Time('2345-06-28', scale='tai')

class TestAutoOpenExplicitLists:

    @pytest.mark.filterwarnings(iers.IERSStaleWarning)
    def test_auto_open_simple(self):
        if False:
            for i in range(10):
                print('nop')
        ls = iers.LeapSeconds.auto_open([iers.IERS_LEAP_SECOND_FILE])
        assert ls.meta['data_url'] == iers.IERS_LEAP_SECOND_FILE

    @pytest.mark.filterwarnings(iers.IERSStaleWarning)
    def test_auto_open_erfa(self):
        if False:
            i = 10
            return i + 15
        ls = iers.LeapSeconds.auto_open(['erfa', iers.IERS_LEAP_SECOND_FILE])
        assert ls.meta['data_url'] in ['erfa', iers.IERS_LEAP_SECOND_FILE]

    @pytest.mark.filterwarnings(iers.IERSStaleWarning)
    def test_fake_future_file(self, tmp_path):
        if False:
            return 10
        fake_file = make_fake_file('28 June 2345', tmp_path)
        with iers.conf.set_temp('auto_max_age', -100000):
            ls = iers.LeapSeconds.auto_open(['erfa', iers.IERS_LEAP_SECOND_FILE, fake_file])
            assert ls.expires == Time('2345-06-28', scale='tai')
            assert ls.meta['data_url'] == str(fake_file)
            fake_url = 'file:' + urllib.request.pathname2url(fake_file)
            ls2 = iers.LeapSeconds.auto_open(['erfa', iers.IERS_LEAP_SECOND_FILE, fake_url])
            assert ls2.expires == Time('2345-06-28', scale='tai')
            assert ls2.meta['data_url'] == str(fake_url)

    def test_fake_expired_file(self, tmp_path):
        if False:
            print('Hello World!')
        fake_file1 = make_fake_file('28 June 2010', tmp_path)
        fake_file2 = make_fake_file('27 June 2012', tmp_path)
        ls = iers.LeapSeconds.auto_open([fake_file1, fake_file2, iers.IERS_LEAP_SECOND_FILE])
        assert ls.meta['data_url'] == iers.IERS_LEAP_SECOND_FILE
        with pytest.warns(iers.IERSStaleWarning):
            ls2 = iers.LeapSeconds.auto_open([fake_file1, fake_file2])
        assert ls2.meta['data_url'] == fake_file2
        assert ls2.expires == Time('2012-06-27', scale='tai')
        with iers.conf.set_temp('auto_max_age', None):
            ls3 = iers.LeapSeconds.auto_open([fake_file1, iers.IERS_LEAP_SECOND_FILE])
        assert ls3.meta['data_url'] == iers.IERS_LEAP_SECOND_FILE
        with iers.conf.set_temp('auto_max_age', None):
            ls4 = iers.LeapSeconds.auto_open([fake_file1, fake_file2])
        assert ls4.meta['data_url'] == fake_file2

@pytest.mark.remote_data
class TestRemoteURLs:

    def setup_class(cls):
        if False:
            while True:
                i = 10
        iers.conf.auto_download = True

    def teardown_class(cls):
        if False:
            while True:
                i = 10
        iers.conf.auto_download = False

    def test_iers_url(self):
        if False:
            i = 10
            return i + 15
        ls = iers.LeapSeconds.auto_open([iers.IERS_LEAP_SECOND_URL])
        assert ls.expires > Time.now()

    def test_ietf_url(self):
        if False:
            for i in range(10):
                print('nop')
        ls = iers.LeapSeconds.auto_open([iers.IETF_LEAP_SECOND_URL])
        assert ls.expires > Time.now()

class TestDefaultAutoOpen:
    """Test auto_open with different _auto_open_files."""

    def setup_method(self):
        if False:
            print('Hello World!')
        self.good_enough = iers.LeapSeconds._today() + TimeDelta(180 - iers._none_to_float(iers.conf.auto_max_age), format='jd')
        self._auto_open_files = iers.LeapSeconds._auto_open_files.copy()

    def teardown_method(self):
        if False:
            print('Hello World!')
        iers.LeapSeconds._auto_open_files = self._auto_open_files

    def remove_auto_open_files(self, *files):
        if False:
            while True:
                i = 10
        'Remove some files from the auto-opener.\n\n        The default set is restored in teardown.\n        '
        for f in files:
            iers.LeapSeconds._auto_open_files.remove(f)

    def test_erfa_found(self):
        if False:
            i = 10
            return i + 15
        with iers.conf.set_temp('auto_max_age', 100000):
            ls = iers.LeapSeconds.open()
        assert ls.meta['data_url'] == 'erfa'

    def test_builtin_found(self):
        if False:
            while True:
                i = 10
        self.remove_auto_open_files('erfa')
        with iers.conf.set_temp('auto_max_age', 100000):
            ls = iers.LeapSeconds.open()
        assert ls.meta['data_url'] == iers.IERS_LEAP_SECOND_FILE

    @pytest.mark.remote_data
    def test_builtin_not_expired(self):
        if False:
            for i in range(10):
                print('nop')
        ls = iers.LeapSeconds.open(iers.IERS_LEAP_SECOND_FILE)
        assert ls.expires > self.good_enough, 'The leap second file built in to astropy is expired. Fix with:\ncd astropy/utils/iers/data/; . update_builtin_iers.sh\nand commit as a PR (for details, see release procedure).'

    def test_fake_future_file(self, tmp_path):
        if False:
            i = 10
            return i + 15
        fake_file = make_fake_file('28 June 2345', tmp_path)
        with iers.conf.set_temp('auto_max_age', -100000), iers.conf.set_temp('system_leap_second_file', fake_file):
            ls = iers.LeapSeconds.open()
        assert ls.expires == Time('2345-06-28', scale='tai')
        assert ls.meta['data_url'] == str(fake_file)
        fake_url = 'file:' + urllib.request.pathname2url(fake_file)
        with iers.conf.set_temp('auto_max_age', -100000), iers.conf.set_temp('iers_leap_second_auto_url', fake_url):
            ls2 = iers.LeapSeconds.open()
        assert ls2.expires == Time('2345-06-28', scale='tai')
        assert ls2.meta['data_url'] == str(fake_url)

    def test_fake_expired_file(self, tmp_path):
        if False:
            i = 10
            return i + 15
        self.remove_auto_open_files('erfa', 'iers_leap_second_auto_url', 'ietf_leap_second_auto_url')
        fake_file = make_fake_file('28 June 2010', tmp_path)
        with iers.conf.set_temp('system_leap_second_file', fake_file):
            ls = iers.LeapSeconds.open()
            assert ls.meta['data_url'] == iers.IERS_LEAP_SECOND_FILE
            self.remove_auto_open_files(iers.IERS_LEAP_SECOND_FILE)
            with pytest.warns(iers.IERSStaleWarning):
                ls2 = iers.LeapSeconds.open()
            assert ls2.meta['data_url'] == fake_file
            assert ls2.expires == Time('2010-06-28', scale='tai')

    @pytest.mark.skipif(not os.path.isfile(SYSTEM_FILE), reason=f'system does not have {SYSTEM_FILE}')
    def test_system_file_used_if_not_expired(self, tmp_path):
        if False:
            print('Hello World!')
        if iers.LeapSeconds.open(SYSTEM_FILE).expires <= self.good_enough:
            pytest.skip('System leap second file is expired.')
        self.remove_auto_open_files('erfa')
        with iers.conf.set_temp('system_leap_second_file', SYSTEM_FILE):
            ls = iers.LeapSeconds.open()
            assert ls.expires > self.good_enough
            assert ls.meta['data_url'] in (iers.IERS_LEAP_SECOND_FILE, SYSTEM_FILE)
            fake_file = make_fake_file('28 June 2017', tmp_path)
            iers.LeapSeconds._auto_open_files[0] = fake_file
            ls2 = iers.LeapSeconds.open()
            assert ls2.expires > Time.now()
            assert ls2.meta['data_url'] == SYSTEM_FILE

    @pytest.mark.remote_data
    def test_auto_open_urls_always_good_enough(self):
        if False:
            i = 10
            return i + 15
        try:
            iers.conf.auto_download = True
            self.remove_auto_open_files('erfa', iers.IERS_LEAP_SECOND_FILE, 'system_leap_second_file')
            ls = iers.LeapSeconds.open()
            assert ls.expires > self.good_enough
            assert ls.meta['data_url'].startswith('http')
        finally:
            iers.conf.auto_download = False

class ERFALeapSecondsSafe:
    """Base class for tests that change the ERFA leap-second tables.

    It ensures the original state is restored.
    """

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.erfa_ls = self._erfa_ls = erfa.leap_seconds.get()
        self.erfa_expires = self._expires = erfa.leap_seconds._expires

    def teardown_method(self):
        if False:
            while True:
                i = 10
        erfa.leap_seconds.set(self.erfa_ls)
        erfa.leap_seconds._expires = self._expires

class TestFromERFA(ERFALeapSecondsSafe):

    def test_get_erfa_ls(self):
        if False:
            i = 10
            return i + 15
        ls = iers.LeapSeconds.from_erfa()
        assert ls.colnames == ['year', 'month', 'tai_utc']
        assert isinstance(ls.expires, Time)
        assert ls.expires == self.erfa_expires
        ls_array = np.array(ls['year', 'month', 'tai_utc'])
        assert np.all(ls_array == self.erfa_ls)

    def test_get_built_in_erfa_ls(self):
        if False:
            i = 10
            return i + 15
        ls = iers.LeapSeconds.from_erfa(built_in=True)
        assert ls.colnames == ['year', 'month', 'tai_utc']
        assert isinstance(ls.expires, Time)
        ls_array = np.array(ls['year', 'month', 'tai_utc'])
        assert np.all(ls_array == self.erfa_ls[:len(ls_array)])

    def test_get_modified_erfa_ls(self):
        if False:
            while True:
                i = 10
        erfa.leap_seconds.set(self.erfa_ls[:-10])
        ls = iers.LeapSeconds.from_erfa()
        assert len(ls) == len(self.erfa_ls) - 10
        ls_array = np.array(ls['year', 'month', 'tai_utc'])
        assert np.all(ls_array == self.erfa_ls[:-10])
        ls2 = iers.LeapSeconds.from_erfa(built_in=True)
        assert len(ls2) > len(ls)
        erfa.leap_seconds.set(None)
        erfa_built_in = erfa.leap_seconds.get()
        assert len(ls2) == len(erfa_built_in)
        ls2_array = np.array(ls2['year', 'month', 'tai_utc'])
        assert np.all(ls2_array == erfa_built_in)

    def test_open(self):
        if False:
            return 10
        ls = iers.LeapSeconds.open('erfa')
        assert isinstance(ls.expires, Time)
        assert ls.expires == self.erfa_expires
        ls_array = np.array(ls['year', 'month', 'tai_utc'])
        assert np.all(ls_array == self.erfa_ls)

class TestUpdateLeapSeconds(ERFALeapSecondsSafe):

    def setup_method(self):
        if False:
            print('Hello World!')
        super().setup_method()
        self.ls = iers.LeapSeconds.from_iers_leap_seconds()
        erfa.leap_seconds.set()
        self.erfa_ls = erfa.leap_seconds.get()

    def test_built_in_up_to_date(self):
        if False:
            while True:
                i = 10
        'Leap second should match between built-in and ERFA.'
        erfa_since_1970 = self.erfa_ls[self.erfa_ls['year'] > 1970]
        assert len(self.ls) >= len(erfa_since_1970), 'built-in leap seconds out of date'
        assert len(self.ls) <= len(erfa_since_1970), 'ERFA leap seconds out of date'
        overlap = np.array(self.ls['year', 'month', 'tai_utc'])
        assert np.all(overlap == erfa_since_1970.astype(overlap.dtype))

    def test_update_with_built_in(self):
        if False:
            i = 10
            return i + 15
        'An update with built-in should not do anything.'
        n_update = self.ls.update_erfa_leap_seconds()
        assert n_update == 0
        new_erfa_ls = erfa.leap_seconds.get()
        assert np.all(new_erfa_ls == self.erfa_ls)

    @pytest.mark.parametrize('n_short', (1, 3))
    def test_update(self, n_short):
        if False:
            i = 10
            return i + 15
        'Check whether we can recover removed leap seconds.'
        erfa.leap_seconds.set(self.erfa_ls[:-n_short])
        n_update = self.ls.update_erfa_leap_seconds()
        assert n_update == n_short
        new_erfa_ls = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls, self.erfa_ls)
        n_update2 = self.ls.update_erfa_leap_seconds()
        assert n_update2 == 0
        new_erfa_ls2 = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls2, self.erfa_ls)

    def test_update_initialize_erfa(self):
        if False:
            i = 10
            return i + 15
        erfa.leap_seconds.set(self.erfa_ls[:-2])
        n_update = self.ls.update_erfa_leap_seconds(initialize_erfa=True)
        assert n_update == 0
        new_erfa_ls = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls, self.erfa_ls)

    def test_update_overwrite(self):
        if False:
            return 10
        n_update = self.ls.update_erfa_leap_seconds(initialize_erfa='empty')
        assert n_update == len(self.ls)
        new_erfa_ls = erfa.leap_seconds.get()
        assert new_erfa_ls['year'].min() > 1970
        n_update2 = self.ls.update_erfa_leap_seconds()
        assert n_update2 == 0
        new_erfa_ls2 = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls2, new_erfa_ls)
        n_update3 = self.ls.update_erfa_leap_seconds(initialize_erfa=True)
        assert n_update3 == 0
        new_erfa_ls3 = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls3, self.erfa_ls)

    def test_bad_jump(self):
        if False:
            i = 10
            return i + 15
        erfa.leap_seconds.set(self.erfa_ls[:-2])
        bad = self.ls.copy()
        bad['tai_utc'][-1] = 5
        with pytest.raises(ValueError, match='jump'):
            bad.update_erfa_leap_seconds()
        assert_array_equal(erfa.leap_seconds.get(), self.erfa_ls[:-2])
        with pytest.raises(ValueError, match='jump'):
            bad.update_erfa_leap_seconds(initialize_erfa=True)
        assert_array_equal(erfa.leap_seconds.get(), self.erfa_ls)
        erfa.leap_seconds.set(self.erfa_ls[:-2])
        n_update = bad.update_erfa_leap_seconds(initialize_erfa='only')
        assert n_update == 0
        new_erfa_ls = erfa.leap_seconds.get()
        assert_array_equal(new_erfa_ls, self.erfa_ls)

    def test_bad_day(self):
        if False:
            print('Hello World!')
        erfa.leap_seconds.set(self.erfa_ls[:-2])
        bad = self.ls.copy()
        bad['day'][-1] = 5
        with pytest.raises(ValueError, match='not on 1st'):
            bad.update_erfa_leap_seconds()

    def test_bad_month(self):
        if False:
            return 10
        erfa.leap_seconds.set(self.erfa_ls[:-2])
        bad = self.ls.copy()
        bad['month'][-1] = 5
        with pytest.raises(ValueError, match='January'):
            bad.update_erfa_leap_seconds()
        assert_array_equal(erfa.leap_seconds.get(), self.erfa_ls[:-2])