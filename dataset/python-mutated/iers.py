"""
The astropy.utils.iers package provides access to the tables provided by
the International Earth Rotation and Reference Systems Service, in
particular allowing interpolation of published UT1-UTC values for given
times.  These are used in `astropy.time` to provide UT1 values.  The polar
motions are also used for determining earth orientation for
celestial-to-terrestrial coordinate transformations
(in `astropy.coordinates`).
"""
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from warnings import warn
import erfa
import numpy as np
from astropy_iers_data import IERS_A_FILE, IERS_A_README, IERS_A_URL, IERS_A_URL_MIRROR, IERS_B_FILE, IERS_B_README, IERS_B_URL, IERS_LEAP_SECOND_FILE, IERS_LEAP_SECOND_URL
from astropy_iers_data import IERS_LEAP_SECOND_URL_MIRROR as IETF_LEAP_SECOND_URL
from astropy import config as _config
from astropy import units as u
from astropy import utils
from astropy.table import MaskedColumn, QTable
from astropy.time import Time, TimeDelta
from astropy.utils.data import clear_download_cache, get_readable_fileobj, is_url_in_cache
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning
from astropy.utils.state import ScienceState
__all__ = ['Conf', 'conf', 'earth_orientation_table', 'IERS', 'IERS_B', 'IERS_A', 'IERS_Auto', 'FROM_IERS_B', 'FROM_IERS_A', 'FROM_IERS_A_PREDICTION', 'TIME_BEFORE_IERS_RANGE', 'TIME_BEYOND_IERS_RANGE', 'IERS_A_FILE', 'IERS_A_URL', 'IERS_A_URL_MIRROR', 'IERS_A_README', 'IERS_B_FILE', 'IERS_B_URL', 'IERS_B_README', 'IERSRangeError', 'IERSStaleWarning', 'IERSWarning', 'IERSDegradedAccuracyWarning', 'LeapSeconds', 'IERS_LEAP_SECOND_FILE', 'IERS_LEAP_SECOND_URL', 'IETF_LEAP_SECOND_URL']
FROM_IERS_B = 0
FROM_IERS_A = 1
FROM_IERS_A_PREDICTION = 2
TIME_BEFORE_IERS_RANGE = -1
TIME_BEYOND_IERS_RANGE = -2
MJD_ZERO = 2400000.5
INTERPOLATE_ERROR = 'interpolating from IERS_Auto using predictive values that are more\nthan {0} days old.\n\nNormally you should not see this error because this class\nautomatically downloads the latest IERS-A table.  Perhaps you are\noffline?  If you understand what you are doing then this error can be\nsuppressed by setting the auto_max_age configuration variable to\n``None``:\n\n  from astropy.utils.iers import conf\n  conf.auto_max_age = None\n'
MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

class IERSWarning(AstropyWarning):
    """
    Generic warning class for IERS.
    """

class IERSDegradedAccuracyWarning(AstropyWarning):
    """
    IERS time conversion has degraded accuracy normally due to setting
    ``conf.auto_download = False`` and ``conf.iers_degraded_accuracy = 'warn'``.
    """

class IERSStaleWarning(IERSWarning):
    """
    Downloaded IERS table may be stale.
    """

def download_file(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Overload astropy.utils.data.download_file within iers module to use a\n    custom (longer) wait time.  This just passes through ``*args`` and\n    ``**kwargs`` after temporarily setting the download_file remote timeout to\n    the local ``iers.conf.remote_timeout`` value.\n    '
    kwargs.setdefault('http_headers', {'User-Agent': 'astropy/iers', 'Accept': '*/*'})
    with utils.data.conf.set_temp('remote_timeout', conf.remote_timeout):
        return utils.data.download_file(*args, **kwargs)

def _none_to_float(value):
    if False:
        return 10
    '\n    Convert None to a valid floating point value.  Especially\n    for auto_max_age = None.\n    '
    return value if value is not None else np.finfo(float).max

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.utils.iers`.
    """
    auto_download = _config.ConfigItem(True, 'Enable auto-downloading of the latest IERS data.  If set to False then the local IERS-B file will be used by default (even if the full IERS file with predictions was already downloaded and cached). This parameter also controls whether internet resources will be queried to update the leap second table if the installed version is out of date. Default is True.')
    auto_max_age = _config.ConfigItem(30.0, 'Maximum age (days) of predictive data before auto-downloading. See "Auto refresh behavior" in astropy.utils.iers documentation for details. Default is 30.')
    iers_auto_url = _config.ConfigItem(IERS_A_URL, 'URL for auto-downloading IERS file data.')
    iers_auto_url_mirror = _config.ConfigItem(IERS_A_URL_MIRROR, 'Mirror URL for auto-downloading IERS file data.')
    remote_timeout = _config.ConfigItem(10.0, 'Remote timeout downloading IERS file data (seconds).')
    iers_degraded_accuracy = _config.ConfigItem(['error', 'warn', 'ignore'], 'IERS behavior if the range of available IERS data does not cover the times when converting time scales, potentially leading to degraded accuracy.')
    system_leap_second_file = _config.ConfigItem('', 'System file with leap seconds.')
    iers_leap_second_auto_url = _config.ConfigItem(IERS_LEAP_SECOND_URL, 'URL for auto-downloading leap seconds.')
    ietf_leap_second_auto_url = _config.ConfigItem(IETF_LEAP_SECOND_URL, 'Alternate URL for auto-downloading leap seconds.')
conf = Conf()

class IERSRangeError(IndexError):
    """
    Any error for when dates are outside of the valid range for IERS.
    """

class IERS(QTable):
    """Generic IERS table class, defining interpolation functions.

    Sub-classed from `astropy.table.QTable`.  The table should hold columns
    'MJD', 'UT1_UTC', 'dX_2000A'/'dY_2000A', and 'PM_x'/'PM_y'.
    """
    iers_table = None
    'Cached table, returned if ``open`` is called without arguments.'

    @classmethod
    def open(cls, file=None, cache=False, **kwargs):
        if False:
            print('Hello World!')
        'Open an IERS table, reading it from a file if not loaded before.\n\n        Parameters\n        ----------\n        file : str or None\n            full local or network path to the ascii file holding IERS data,\n            for passing on to the ``read`` class methods (further optional\n            arguments that are available for some IERS subclasses can be added).\n            If None, use the default location from the ``read`` class method.\n        cache : bool\n            Whether to use cache. Defaults to False, since IERS files\n            are regularly updated.\n\n        Returns\n        -------\n        IERS\n            An IERS table class instance\n\n        Notes\n        -----\n        On the first call in a session, the table will be memoized (in the\n        ``iers_table`` class attribute), and further calls to ``open`` will\n        return this stored table if ``file=None`` (the default).\n\n        If a table needs to be re-read from disk, pass on an explicit file\n        location or use the (sub-class) close method and re-open.\n\n        If the location is a network location it is first downloaded via\n        download_file.\n\n        For the IERS class itself, an IERS_B sub-class instance is opened.\n\n        '
        if file is not None or cls.iers_table is None:
            if file is not None:
                if urlparse(file).netloc:
                    kwargs.update(file=download_file(file, cache=cache))
                else:
                    kwargs.update(file=file)
            if cls is IERS:
                cls.iers_table = IERS_B.read(**kwargs)
            else:
                cls.iers_table = cls.read(**kwargs)
        return cls.iers_table

    @classmethod
    def close(cls):
        if False:
            for i in range(10):
                print('nop')
        "Remove the IERS table from the class.\n\n        This allows the table to be re-read from disk during one's session\n        (e.g., if one finds it is out of date and has updated the file).\n        "
        cls.iers_table = None

    def mjd_utc(self, jd1, jd2=0.0):
        if False:
            return 10
        'Turn a time to MJD, returning integer and fractional parts.\n\n        Parameters\n        ----------\n        jd1 : float, array, or `~astropy.time.Time`\n            first part of two-part JD, or Time object\n        jd2 : float or array, optional\n            second part of two-part JD.\n            Default is 0., ignored if jd1 is `~astropy.time.Time`.\n\n        Returns\n        -------\n        mjd : float or array\n            integer part of MJD\n        utc : float or array\n            fractional part of MJD\n        '
        try:
            (jd1, jd2) = (jd1.utc.jd1, jd1.utc.jd2)
        except Exception:
            pass
        mjd = np.floor(jd1 - MJD_ZERO + jd2)
        utc = jd1 - (MJD_ZERO + mjd) + jd2
        return (mjd, utc)

    def ut1_utc(self, jd1, jd2=0.0, return_status=False):
        if False:
            i = 10
            return i + 15
        'Interpolate UT1-UTC corrections in IERS Table for given dates.\n\n        Parameters\n        ----------\n        jd1 : float, array of float, or `~astropy.time.Time` object\n            first part of two-part JD, or Time object\n        jd2 : float or float array, optional\n            second part of two-part JD.\n            Default is 0., ignored if jd1 is `~astropy.time.Time`.\n        return_status : bool\n            Whether to return status values.  If False (default),\n            raise ``IERSRangeError`` if any time is out of the range covered\n            by the IERS table.\n\n        Returns\n        -------\n        ut1_utc : float or float array\n            UT1-UTC, interpolated in IERS Table\n        status : int or int array\n            Status values (if ``return_status``=``True``)::\n            ``iers.FROM_IERS_B``\n            ``iers.FROM_IERS_A``\n            ``iers.FROM_IERS_A_PREDICTION``\n            ``iers.TIME_BEFORE_IERS_RANGE``\n            ``iers.TIME_BEYOND_IERS_RANGE``\n        '
        return self._interpolate(jd1, jd2, ['UT1_UTC'], self.ut1_utc_source if return_status else None)

    def dcip_xy(self, jd1, jd2=0.0, return_status=False):
        if False:
            i = 10
            return i + 15
        "Interpolate CIP corrections in IERS Table for given dates.\n\n        Parameters\n        ----------\n        jd1 : float, array of float, or `~astropy.time.Time` object\n            first part of two-part JD, or Time object\n        jd2 : float or float array, optional\n            second part of two-part JD (default 0., ignored if jd1 is Time)\n        return_status : bool\n            Whether to return status values.  If False (default),\n            raise ``IERSRangeError`` if any time is out of the range covered\n            by the IERS table.\n\n        Returns\n        -------\n        D_x : `~astropy.units.Quantity` ['angle']\n            x component of CIP correction for the requested times.\n        D_y : `~astropy.units.Quantity` ['angle']\n            y component of CIP correction for the requested times\n        status : int or int array\n            Status values (if ``return_status``=``True``)::\n            ``iers.FROM_IERS_B``\n            ``iers.FROM_IERS_A``\n            ``iers.FROM_IERS_A_PREDICTION``\n            ``iers.TIME_BEFORE_IERS_RANGE``\n            ``iers.TIME_BEYOND_IERS_RANGE``\n        "
        return self._interpolate(jd1, jd2, ['dX_2000A', 'dY_2000A'], self.dcip_source if return_status else None)

    def pm_xy(self, jd1, jd2=0.0, return_status=False):
        if False:
            for i in range(10):
                print('nop')
        "Interpolate polar motions from IERS Table for given dates.\n\n        Parameters\n        ----------\n        jd1 : float, array of float, or `~astropy.time.Time` object\n            first part of two-part JD, or Time object\n        jd2 : float or float array, optional\n            second part of two-part JD.\n            Default is 0., ignored if jd1 is `~astropy.time.Time`.\n        return_status : bool\n            Whether to return status values.  If False (default),\n            raise ``IERSRangeError`` if any time is out of the range covered\n            by the IERS table.\n\n        Returns\n        -------\n        PM_x : `~astropy.units.Quantity` ['angle']\n            x component of polar motion for the requested times.\n        PM_y : `~astropy.units.Quantity` ['angle']\n            y component of polar motion for the requested times.\n        status : int or int array\n            Status values (if ``return_status``=``True``)::\n            ``iers.FROM_IERS_B``\n            ``iers.FROM_IERS_A``\n            ``iers.FROM_IERS_A_PREDICTION``\n            ``iers.TIME_BEFORE_IERS_RANGE``\n            ``iers.TIME_BEYOND_IERS_RANGE``\n        "
        return self._interpolate(jd1, jd2, ['PM_x', 'PM_y'], self.pm_source if return_status else None)

    def _check_interpolate_indices(self, indices_orig, indices_clipped, max_input_mjd):
        if False:
            return 10
        '\n        Check that the indices from interpolation match those after clipping\n        to the valid table range.  This method gets overridden in the IERS_Auto\n        class because it has different requirements.\n        '
        if np.any(indices_orig != indices_clipped):
            if conf.iers_degraded_accuracy == 'error':
                msg = '(some) times are outside of range covered by IERS table. Cannot convert with full accuracy. To allow conversion with degraded accuracy set astropy.utils.iers.conf.iers_degraded_accuracy to "warn" or "silent". For more information about setting this configuration parameter or controlling its value globally, see the Astropy configuration system documentation https://docs.astropy.org/en/stable/config/index.html.'
                raise IERSRangeError(msg)
            elif conf.iers_degraded_accuracy == 'warn':
                msg = '(some) times are outside of range covered by IERS table, accuracy is degraded.'
                warn(msg, IERSDegradedAccuracyWarning)

    def _interpolate(self, jd1, jd2, columns, source=None):
        if False:
            print('Hello World!')
        (mjd, utc) = self.mjd_utc(jd1, jd2)
        is_scalar = not hasattr(mjd, '__array__') or mjd.ndim == 0
        if is_scalar:
            mjd = np.array([mjd])
            utc = np.array([utc])
        elif mjd.size == 0:
            return np.array([])
        self._refresh_table_as_needed(mjd)
        i = np.searchsorted(self['MJD'].value, mjd, side='right')
        i1 = np.clip(i, 1, len(self) - 1)
        i0 = i1 - 1
        (mjd_0, mjd_1) = (self['MJD'][i0].value, self['MJD'][i1].value)
        results = []
        for column in columns:
            (val_0, val_1) = (self[column][i0], self[column][i1])
            d_val = val_1 - val_0
            if column == 'UT1_UTC':
                d_val -= d_val.round()
            val = val_0 + (mjd - mjd_0 + utc) / (mjd_1 - mjd_0) * d_val
            val[i == 0] = self[column][0]
            val[i == len(self)] = self[column][-1]
            if is_scalar:
                val = val[0]
            results.append(val)
        if source:
            status = source(i1)
            status[i == 0] = TIME_BEFORE_IERS_RANGE
            status[i == len(self)] = TIME_BEYOND_IERS_RANGE
            if is_scalar:
                status = status[0]
            results.append(status)
            return results
        else:
            self._check_interpolate_indices(i1, i, np.max(mjd))
            return results[0] if len(results) == 1 else results

    def _refresh_table_as_needed(self, mjd):
        if False:
            print('Hello World!')
        '\n        Potentially update the IERS table in place depending on the requested\n        time values in ``mdj`` and the time span of the table.  The base behavior\n        is not to update the table.  ``IERS_Auto`` overrides this method.\n        '
        pass

    def ut1_utc_source(self, i):
        if False:
            i = 10
            return i + 15
        'Source for UT1-UTC.  To be overridden by subclass.'
        return np.zeros_like(i)

    def dcip_source(self, i):
        if False:
            return 10
        'Source for CIP correction.  To be overridden by subclass.'
        return np.zeros_like(i)

    def pm_source(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Source for polar motion.  To be overridden by subclass.'
        return np.zeros_like(i)

    @property
    def time_now(self):
        if False:
            while True:
                i = 10
        '\n        Property to provide the current time, but also allow for explicitly setting\n        the _time_now attribute for testing purposes.\n        '
        try:
            return self._time_now
        except Exception:
            return Time.now()

    def _convert_col_for_table(self, col):
        if False:
            i = 10
            return i + 15
        if getattr(col, 'unit', None) is not None and isinstance(col, MaskedColumn):
            col = col.filled(np.nan)
        return super()._convert_col_for_table(col)

class IERS_A(IERS):
    """IERS Table class targeted to IERS A, provided by USNO.

    These include rapid turnaround and predicted times.
    See https://datacenter.iers.org/eop.php

    Notes
    -----
    The IERS A file is not part of astropy.  It can be downloaded from
    ``iers.IERS_A_URL`` or ``iers.IERS_A_URL_MIRROR``. See ``iers.__doc__``
    for instructions on use in ``Time``, etc.
    """
    iers_table = None

    @classmethod
    def _combine_a_b_columns(cls, iers_a):
        if False:
            i = 10
            return i + 15
        '\n        Return a new table with appropriate combination of IERS_A and B columns.\n        '
        table = iers_a[np.isfinite(iers_a['UT1_UTC_A']) & (iers_a['PolPMFlag_A'] != '')]
        table = cls._substitute_iers_b(table)
        b_bad = np.isnan(table['UT1_UTC_B'])
        table['UT1_UTC'] = np.where(b_bad, table['UT1_UTC_A'], table['UT1_UTC_B'])
        table['UT1Flag'] = np.where(b_bad, table['UT1Flag_A'], 'B')
        b_bad = np.isnan(table['PM_X_B']) | np.isnan(table['PM_Y_B'])
        table['PM_x'] = np.where(b_bad, table['PM_x_A'], table['PM_X_B'])
        table['PM_y'] = np.where(b_bad, table['PM_y_A'], table['PM_Y_B'])
        table['PolPMFlag'] = np.where(b_bad, table['PolPMFlag_A'], 'B')
        b_bad = np.isnan(table['dX_2000A_B']) | np.isnan(table['dY_2000A_B'])
        table['dX_2000A'] = np.where(b_bad, table['dX_2000A_A'], table['dX_2000A_B'])
        table['dY_2000A'] = np.where(b_bad, table['dY_2000A_A'], table['dY_2000A_B'])
        table['NutFlag'] = np.where(b_bad, table['NutFlag_A'], 'B')
        p_index = min(np.searchsorted(table['UT1Flag_A'], 'P'), np.searchsorted(table['PolPMFlag_A'], 'P'))
        table.meta['predictive_index'] = p_index
        table.meta['predictive_mjd'] = table['MJD'][p_index].value
        return table

    @classmethod
    def _substitute_iers_b(cls, table):
        if False:
            i = 10
            return i + 15
        return table

    @classmethod
    def read(cls, file=None, readme=None):
        if False:
            return 10
        'Read IERS-A table from a finals2000a.* file provided by USNO.\n\n        Parameters\n        ----------\n        file : str\n            full path to ascii file holding IERS-A data.\n            Defaults to ``iers.IERS_A_FILE``.\n        readme : str\n            full path to ascii file holding CDS-style readme.\n            Defaults to package version, ``iers.IERS_A_README``.\n\n        Returns\n        -------\n        ``IERS_A`` class instance\n        '
        if file is None:
            if os.path.exists('finals2000A.all'):
                file = 'finals2000A.all'
                warn("The file= argument was not specified but 'finals2000A.all' is present in the current working directory, so reading IERS data from that file. To continue reading a local file from the current working directory, specify file= explicitly otherwise a bundled file will be used in future.", AstropyDeprecationWarning)
            else:
                file = IERS_A_FILE
        if readme is None:
            readme = IERS_A_README
        iers_a = super().read(file, format='cds', readme=readme)
        table = cls._combine_a_b_columns(iers_a)
        table.meta['data_path'] = file
        table.meta['readme_path'] = readme
        return table

    def ut1_utc_source(self, i):
        if False:
            return 10
        'Set UT1-UTC source flag for entries in IERS table.'
        ut1flag = self['UT1Flag'][i]
        source = np.ones_like(i) * FROM_IERS_B
        source[ut1flag == 'I'] = FROM_IERS_A
        source[ut1flag == 'P'] = FROM_IERS_A_PREDICTION
        return source

    def dcip_source(self, i):
        if False:
            i = 10
            return i + 15
        'Set CIP correction source flag for entries in IERS table.'
        nutflag = self['NutFlag'][i]
        source = np.ones_like(i) * FROM_IERS_B
        source[nutflag == 'I'] = FROM_IERS_A
        source[nutflag == 'P'] = FROM_IERS_A_PREDICTION
        return source

    def pm_source(self, i):
        if False:
            return 10
        'Set polar motion source flag for entries in IERS table.'
        pmflag = self['PolPMFlag'][i]
        source = np.ones_like(i) * FROM_IERS_B
        source[pmflag == 'I'] = FROM_IERS_A
        source[pmflag == 'P'] = FROM_IERS_A_PREDICTION
        return source

class IERS_B(IERS):
    """IERS Table class targeted to IERS B, provided by IERS itself.

    These are final values; see https://www.iers.org/IERS/EN/Home/home_node.html

    Notes
    -----
    If the package IERS B file (```iers.IERS_B_FILE``) is out of date, a new
    version can be downloaded from ``iers.IERS_B_URL``.

    See `~astropy.utils.iers.IERS_B.read` for instructions on how to read
    a pre-2023 style IERS B file (usually named ``eopc04_IAU2000.62-now``).
    """
    iers_table = None

    @classmethod
    def read(cls, file=None, readme=None, data_start=6):
        if False:
            return 10
        'Read IERS-B table from a eopc04.* file provided by IERS.\n\n        Parameters\n        ----------\n        file : str\n            full path to ascii file holding IERS-B data.\n            Defaults to package version, ``iers.IERS_B_FILE``.\n        readme : str\n            full path to ascii file holding CDS-style readme.\n            Defaults to package version, ``iers.IERS_B_README``.\n        data_start : int\n            Starting row. Default is 6, appropriate for standard IERS files.\n\n        Returns\n        -------\n        ``IERS_B`` class instance\n\n        Notes\n        -----\n        To read a pre-2023 style IERS B file (usually named something like\n        ``eopc04_IAU2000.62-now``), do something like this example with an\n        excerpt that is used for testing::\n\n            >>> from astropy.utils.iers import IERS_B\n            >>> from astropy.utils.data import get_pkg_data_filename\n            >>> old_style_file = get_pkg_data_filename(\n            ...     "tests/data/iers_b_old_style_excerpt",\n            ...     package="astropy.utils.iers")\n            >>> iers_b = IERS_B.read(\n            ...     old_style_file,\n            ...     readme=get_pkg_data_filename("data/ReadMe.eopc04_IAU2000",\n            ...                                  package="astropy.utils.iers"),\n            ...     data_start=14)\n\n        '
        if file is None:
            file = IERS_B_FILE
        if readme is None:
            readme = IERS_B_README
        table = super().read(file, format='cds', readme=readme, data_start=data_start)
        table.meta['data_path'] = file
        table.meta['readme_path'] = readme
        return table

    def ut1_utc_source(self, i):
        if False:
            print('Hello World!')
        'Set UT1-UTC source flag for entries in IERS table.'
        return np.ones_like(i) * FROM_IERS_B

    def dcip_source(self, i):
        if False:
            i = 10
            return i + 15
        'Set CIP correction source flag for entries in IERS table.'
        return np.ones_like(i) * FROM_IERS_B

    def pm_source(self, i):
        if False:
            return 10
        'Set PM source flag for entries in IERS table.'
        return np.ones_like(i) * FROM_IERS_B

class IERS_Auto(IERS_A):
    """
    Provide most-recent IERS data and automatically handle downloading
    of updated values as necessary.
    """
    iers_table = None

    @classmethod
    def open(cls):
        if False:
            print('Hello World!')
        'If the configuration setting ``astropy.utils.iers.conf.auto_download``\n        is set to True (default), then open a recent version of the IERS-A\n        table with predictions for UT1-UTC and polar motion out to\n        approximately one year from now.  If the available version of this file\n        is older than ``astropy.utils.iers.conf.auto_max_age`` days old\n        (or non-existent) then it will be downloaded over the network and cached.\n\n        If the configuration setting ``astropy.utils.iers.conf.auto_download``\n        is set to False then ``astropy.utils.iers.IERS()`` is returned.  This\n        is normally the IERS-B table that is supplied with astropy.\n\n        On the first call in a session, the table will be memoized (in the\n        ``iers_table`` class attribute), and further calls to ``open`` will\n        return this stored table.\n\n        Returns\n        -------\n        `~astropy.table.QTable` instance\n            With IERS (Earth rotation) data columns\n\n        '
        if not conf.auto_download:
            cls.iers_table = IERS_B.open()
            return cls.iers_table
        all_urls = (conf.iers_auto_url, conf.iers_auto_url_mirror)
        if cls.iers_table is not None:
            if cls.iers_table.meta.get('data_url') in all_urls:
                return cls.iers_table
        for url in all_urls:
            try:
                filename = download_file(url, cache=True)
            except Exception as err:
                warn(f'failed to download {url}: {err}', IERSWarning)
                continue
            try:
                cls.iers_table = cls.read(file=filename)
            except Exception as err:
                warn(f'malformed IERS table from {url}: {err}', IERSWarning)
                continue
            cls.iers_table.meta['data_url'] = url
            break
        else:
            warn('unable to download valid IERS file, using local IERS-B', IERSWarning)
            cls.iers_table = IERS_B.open()
        return cls.iers_table

    def _check_interpolate_indices(self, indices_orig, indices_clipped, max_input_mjd):
        if False:
            i = 10
            return i + 15
        'Check that the indices from interpolation match those after clipping to the\n        valid table range.  The IERS_Auto class is exempted as long as it has\n        sufficiently recent available data so the clipped interpolation is\n        always within the confidence bounds of current Earth rotation\n        knowledge.\n        '
        predictive_mjd = self.meta['predictive_mjd']
        auto_max_age = _none_to_float(conf.auto_max_age)
        if max_input_mjd > predictive_mjd and self.time_now.mjd - predictive_mjd > auto_max_age:
            raise ValueError(INTERPOLATE_ERROR.format(auto_max_age))

    def _refresh_table_as_needed(self, mjd):
        if False:
            return 10
        'Potentially update the IERS table in place depending on the requested\n        time values in ``mjd`` and the time span of the table.\n\n        For IERS_Auto the behavior is that the table is refreshed from the IERS\n        server if both the following apply:\n\n        - Any of the requested IERS values are predictive.  The IERS-A table\n          contains predictive data out for a year after the available\n          definitive values.\n        - The first predictive values are at least ``conf.auto_max_age days`` old.\n          In other words the IERS-A table was created by IERS long enough\n          ago that it can be considered stale for predictions.\n        '
        max_input_mjd = np.max(mjd)
        now_mjd = self.time_now.mjd
        fpi = self.meta['predictive_index']
        predictive_mjd = self.meta['predictive_mjd']
        auto_max_age = _none_to_float(conf.auto_max_age)
        if auto_max_age < 10:
            raise ValueError('IERS auto_max_age configuration value must be larger than 10 days')
        if max_input_mjd > predictive_mjd and now_mjd - predictive_mjd > auto_max_age:
            all_urls = (conf.iers_auto_url, conf.iers_auto_url_mirror)
            try:
                filename = download_file(all_urls[0], sources=all_urls, cache='update')
            except Exception as err:
                warn(AstropyWarning(f"""failed to download {' and '.join(all_urls)}: {err}.\nA coordinate or time-related calculation might be compromised or fail because the dates are not covered by the available IERS file.  See the "IERS data access" section of the astropy documentation for additional information on working offline."""))
                return
            new_table = self.__class__.read(file=filename)
            new_table.meta['data_url'] = str(all_urls[0])
            if new_table['MJD'][-1] > self['MJD'][-1]:
                new_fpi = np.searchsorted(new_table['MJD'].value, predictive_mjd, side='right')
                n_replace = len(self) - fpi
                self[fpi:] = new_table[new_fpi:new_fpi + n_replace]
                if new_table['MJD'][new_fpi + n_replace] - self['MJD'][-1] != 1.0 * u.d:
                    raise ValueError('unexpected gap in MJD when refreshing IERS table')
                for row in new_table[new_fpi + n_replace:]:
                    self.add_row(row)
                self.meta.update(new_table.meta)
            else:
                warn(IERSStaleWarning(f'IERS_Auto predictive values are older than {conf.auto_max_age} days but downloading the latest table did not find newer values'))

    @classmethod
    def _substitute_iers_b(cls, table):
        if False:
            while True:
                i = 10
        'Substitute IERS B values with those from a real IERS B table.\n\n        IERS-A has IERS-B values included, but for reasons unknown these\n        do not match the latest IERS-B values (see comments in #4436).\n        Here, we use the bundled astropy IERS-B table to overwrite the values\n        in the downloaded IERS-A table.\n        '
        iers_b = IERS_B.open()
        mjd_b = table['MJD'][np.isfinite(table['UT1_UTC_B'])]
        i0 = np.searchsorted(iers_b['MJD'], mjd_b[0], side='left')
        i1 = np.searchsorted(iers_b['MJD'], mjd_b[-1], side='right')
        iers_b = iers_b[i0:i1]
        n_iers_b = len(iers_b)
        if n_iers_b > 0:
            if not u.allclose(table['MJD'][:n_iers_b], iers_b['MJD']):
                raise ValueError('unexpected mismatch when copying IERS-B values into IERS-A table.')
            table['UT1_UTC_B'][:n_iers_b] = iers_b['UT1_UTC']
            table['PM_X_B'][:n_iers_b] = iers_b['PM_x']
            table['PM_Y_B'][:n_iers_b] = iers_b['PM_y']
            table['dX_2000A_B'][:n_iers_b] = iers_b['dX_2000A']
            table['dY_2000A_B'][:n_iers_b] = iers_b['dY_2000A']
        return table

class earth_orientation_table(ScienceState):
    """Default IERS table for Earth rotation and reference systems service.

    These tables are used to calculate the offsets between ``UT1`` and ``UTC``
    and for conversion to Earth-based coordinate systems.

    The state itself is an IERS table, as an instance of one of the
    `~astropy.utils.iers.IERS` classes.  The default, the auto-updating
    `~astropy.utils.iers.IERS_Auto` class, should suffice for most
    purposes.

    Examples
    --------
    To temporarily use the IERS-B file packaged with astropy::

      >>> from astropy.utils import iers
      >>> from astropy.time import Time
      >>> iers_b = iers.IERS_B.open(iers.IERS_B_FILE)
      >>> with iers.earth_orientation_table.set(iers_b):
      ...     print(Time('2000-01-01').ut1.isot)
      2000-01-01T00:00:00.355

    To use the most recent IERS-A file for the whole session::

      >>> iers_a = iers.IERS_A.open(iers.IERS_A_URL)  # doctest: +SKIP
      >>> iers.earth_orientation_table.set(iers_a)  # doctest: +SKIP
      <ScienceState earth_orientation_table: <IERS_A length=17463>...>

    To go back to the default (of `~astropy.utils.iers.IERS_Auto`)::

      >>> iers.earth_orientation_table.set(None)  # doctest: +SKIP
      <ScienceState earth_orientation_table: <IERS_Auto length=17428>...>
    """
    _value = None

    @classmethod
    def validate(cls, value):
        if False:
            i = 10
            return i + 15
        if value is None:
            value = IERS_Auto.open()
        if not isinstance(value, IERS):
            raise ValueError('earth_orientation_table requires an IERS Table.')
        return value

class LeapSeconds(QTable):
    """Leap seconds class, holding TAI-UTC differences.

    The table should hold columns 'year', 'month', 'tai_utc'.

    Methods are provided to initialize the table from IERS ``Leap_Second.dat``,
    IETF/ntp ``leap-seconds.list``, or built-in ERFA/SOFA, and to update the
    list used by ERFA.

    Notes
    -----
    Astropy has a built-in ``iers.IERS_LEAP_SECONDS_FILE``. Up to date versions
    can be downloaded from ``iers.IERS_LEAP_SECONDS_URL`` or
    ``iers.LEAP_SECONDS_LIST_URL``.  Many systems also store a version
    of ``leap-seconds.list`` for use with ``ntp`` (e.g., on Debian/Ubuntu
    systems, ``/usr/share/zoneinfo/leap-seconds.list``).

    To prevent querying internet resources if the available local leap second
    file(s) are out of date, set ``iers.conf.auto_download = False``. This
    must be done prior to performing any ``Time`` scale transformations related
    to UTC (e.g. converting from UTC to TAI).
    """
    _re_expires = re.compile('^#.*File expires on[:\\s]+(\\d+\\s\\w+\\s\\d+)\\s*$')
    _expires = None
    _auto_open_files = ['erfa', IERS_LEAP_SECOND_FILE, 'system_leap_second_file', 'iers_leap_second_auto_url', 'ietf_leap_second_auto_url']
    'Files or conf attributes to try in auto_open.'

    @classmethod
    def open(cls, file=None, cache=False):
        if False:
            i = 10
            return i + 15
        "Open a leap-second list.\n\n        Parameters\n        ----------\n        file : path-like or None\n            Full local or network path to the file holding leap-second data,\n            for passing on to the various ``from_`` class methods.\n            If 'erfa', return the data used by the ERFA library.\n            If `None`, use default locations from file and configuration to\n            find a table that is not expired.\n        cache : bool\n            Whether to use cache. Defaults to False, since leap-second files\n            are regularly updated.\n\n        Returns\n        -------\n        leap_seconds : `~astropy.utils.iers.LeapSeconds`\n            Table with 'year', 'month', and 'tai_utc' columns, plus possibly\n            others.\n\n        Notes\n        -----\n        Bulletin C is released about 10 days after a possible leap second is\n        introduced, i.e., mid-January or mid-July.  Expiration days are thus\n        generally at least 150 days after the present.  For the auto-loading,\n        a list comprised of the table shipped with astropy, and files and\n        URLs in `~astropy.utils.iers.Conf` are tried, returning the first\n        that is sufficiently new, or the newest among them all.\n        "
        if file is None:
            return cls.auto_open()
        if file.lower() == 'erfa':
            return cls.from_erfa()
        if urlparse(file).netloc:
            file = download_file(file, cache=cache)
        try:
            return cls.from_iers_leap_seconds(file)
        except Exception:
            return cls.from_leap_seconds_list(file)

    @staticmethod
    def _today():
        if False:
            for i in range(10):
                print('nop')
        s = '{0.year:04d}-{0.month:02d}-{0.day:02d}'.format(datetime.now(tz=timezone.utc))
        return Time(s, scale='tai', format='iso', out_subfmt='date')

    @classmethod
    def auto_open(cls, files=None):
        if False:
            print('Hello World!')
        'Attempt to get an up-to-date leap-second list.\n\n        The routine will try the files in sequence until it finds one\n        whose expiration date is "good enough" (see below).  If none\n        are good enough, it returns the one with the most recent expiration\n        date, warning if that file is expired.\n\n        For remote files that are cached already, the cached file is tried\n        first before attempting to retrieve it again.\n\n        Parameters\n        ----------\n        files : list of path-like, optional\n            List of files/URLs to attempt to open.  By default, uses\n            ``cls._auto_open_files``.\n\n        Returns\n        -------\n        leap_seconds : `~astropy.utils.iers.LeapSeconds`\n            Up to date leap-second table\n\n        Notes\n        -----\n        Bulletin C is released about 10 days after a possible leap second is\n        introduced, i.e., mid-January or mid-July.  Expiration days are thus\n        generally at least 150 days after the present.  We look for a file\n        that expires more than 180 - `~astropy.utils.iers.Conf.auto_max_age`\n        after the present.\n        '
        offset = 180 - (30 if conf.auto_max_age is None else conf.auto_max_age)
        good_enough = cls._today() + TimeDelta(offset, format='jd')
        if files is None:
            files = [getattr(conf, f, f) for f in cls._auto_open_files]
        files = [f for f in files if f]
        trials = [(f, True) for f in files if not urlparse(f).netloc or is_url_in_cache(f)]
        if conf.auto_download:
            trials += [(f, False) for f in files if urlparse(f).netloc]
        self = None
        err_list = []
        for (f, allow_cache) in trials:
            if not allow_cache:
                clear_download_cache(f)
            try:
                trial = cls.open(f, cache=True)
            except Exception as exc:
                err_list.append(exc)
                continue
            if self is None or trial.expires > self.expires:
                self = trial
                self.meta['data_url'] = str(f)
                if self.expires > good_enough:
                    break
        if self is None:
            raise ValueError(f'none of the files could be read. The following errors were raised:\n {err_list}')
        if self.expires < self._today() and conf.auto_max_age is not None:
            warn('leap-second file is expired.', IERSStaleWarning)
        return self

    @property
    def expires(self):
        if False:
            for i in range(10):
                print('nop')
        'The limit of validity of the table.'
        return self._expires

    @classmethod
    def _read_leap_seconds(cls, file, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Read a file, identifying expiration by matching 'File expires'."
        expires = None
        with get_readable_fileobj(file) as fh:
            lines = fh.readlines()
            for line in lines:
                match = cls._re_expires.match(line)
                if match:
                    (day, month, year) = match.groups()[0].split()
                    month_nb = MONTH_ABBR.index(month[:3]) + 1
                    expires = Time(f'{year}-{month_nb:02d}-{day}', scale='tai', out_subfmt='date')
                    break
            else:
                raise ValueError(f'did not find expiration date in {file}')
        self = cls.read(lines, format='ascii.no_header', **kwargs)
        self._expires = expires
        return self

    @classmethod
    def from_iers_leap_seconds(cls, file=IERS_LEAP_SECOND_FILE):
        if False:
            print('Hello World!')
        "Create a table from a file like the IERS ``Leap_Second.dat``.\n\n        Parameters\n        ----------\n        file : path-like, optional\n            Full local or network path to the file holding leap-second data\n            in a format consistent with that used by IERS.  By default, uses\n            ``iers.IERS_LEAP_SECOND_FILE``.\n\n        Notes\n        -----\n        The file *must* contain the expiration date in a comment line, like\n        '#  File expires on 28 June 2020'\n        "
        return cls._read_leap_seconds(file, names=['mjd', 'day', 'month', 'year', 'tai_utc'])

    @classmethod
    def from_leap_seconds_list(cls, file):
        if False:
            while True:
                i = 10
        "Create a table from a file like the IETF ``leap-seconds.list``.\n\n        Parameters\n        ----------\n        file : path-like, optional\n            Full local or network path to the file holding leap-second data\n            in a format consistent with that used by IETF.  Up to date versions\n            can be retrieved from ``iers.IETF_LEAP_SECOND_URL``.\n\n        Notes\n        -----\n        The file *must* contain the expiration date in a comment line, like\n        '# File expires on:  28 June 2020'\n        "
        from astropy.io.ascii import convert_numpy
        names = ['ntp_seconds', 'tai_utc', 'comment', 'day', 'month', 'year']
        self = cls._read_leap_seconds(file, names=names, include_names=names[:2], converters={'ntp_seconds': [convert_numpy(np.int64)]})
        self['mjd'] = (self['ntp_seconds'] / 86400 + 15020).round()
        isot = Time(self['mjd'], format='mjd', scale='tai').isot
        ymd = np.array([[int(part) for part in t.partition('T')[0].split('-')] for t in isot])
        (self['year'], self['month'], self['day']) = ymd.T
        return self

    @classmethod
    def from_erfa(cls, built_in=False):
        if False:
            for i in range(10):
                print('nop')
        'Create table from the leap-second list in ERFA.\n\n        Parameters\n        ----------\n        built_in : bool\n            If `False` (default), retrieve the list currently used by ERFA,\n            which may have been updated.  If `True`, retrieve the list shipped\n            with erfa.\n        '
        current = cls(erfa.leap_seconds.get())
        current._expires = Time('{0.year:04d}-{0.month:02d}-{0.day:02d}'.format(erfa.leap_seconds.expires), scale='tai')
        if not built_in:
            return current
        try:
            erfa.leap_seconds.set(None)
            return cls.from_erfa(built_in=False)
        finally:
            erfa.leap_seconds.set(current)

    def update_erfa_leap_seconds(self, initialize_erfa=False):
        if False:
            print('Hello World!')
        "Add any leap seconds not already present to the ERFA table.\n\n        This method matches leap seconds with those present in the ERFA table,\n        and extends the latter as necessary.\n\n        Parameters\n        ----------\n        initialize_erfa : bool, or 'only', or 'empty'\n            Initialize the ERFA leap second table to its built-in value before\n            trying to expand it.  This is generally not needed but can help\n            in case it somehow got corrupted.  If equal to 'only', the ERFA\n            table is reinitialized and no attempt it made to update it.\n            If 'empty', the leap second table is emptied before updating, i.e.,\n            it is overwritten altogether (note that this may break things in\n            surprising ways, as most leap second tables do not include pre-1970\n            pseudo leap-seconds; you were warned).\n\n        Returns\n        -------\n        n_update : int\n            Number of items updated.\n\n        Raises\n        ------\n        ValueError\n            If the leap seconds in the table are not on 1st of January or July,\n            or if the matches are inconsistent.  This would normally suggest\n            a corrupted leap second table, but might also indicate that the\n            ERFA table was corrupted.  If needed, the ERFA table can be reset\n            by calling this method with an appropriate value for\n            ``initialize_erfa``.\n        "
        if initialize_erfa == 'empty':
            erfa.leap_seconds.set(self)
            return len(self)
        if initialize_erfa:
            erfa.leap_seconds.set()
            if initialize_erfa == 'only':
                return 0
        return erfa.leap_seconds.update(self)