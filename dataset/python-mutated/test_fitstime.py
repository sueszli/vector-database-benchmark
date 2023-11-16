import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.io.fits.fitstime import GLOBAL_TIME_INFO, is_time_column_keyword, time_to_fits
from astropy.table import Column, QTable, Table
from astropy.time import Time, TimeDelta
from astropy.time.core import BARYCENTRIC_SCALES
from astropy.time.formats import FITS_DEPRECATED_SCALES
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
from .conftest import FitsTestCase

class TestFitsTime(FitsTestCase):

    def setup_class(self):
        if False:
            print('Hello World!')
        self.time = np.array(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
        self.time_3d = np.array([[[1, 2], [1, 3], [3, 4]]])

    def test_is_time_column_keyword(self):
        if False:
            while True:
                i = 10
        assert is_time_column_keyword('TRPOS') is False
        assert is_time_column_keyword('TIMESYS') is False
        assert is_time_column_keyword('TRPOS12') is True

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_time_to_fits_loc(self, table_types):
        if False:
            print('Hello World!')
        '\n        Test all the unusual conditions for locations of ``Time``\n        columns in a ``Table``.\n        '
        t = table_types()
        t['a'] = Time(self.time, format='isot', scale='utc')
        t['b'] = Time(self.time, format='isot', scale='tt')
        t['a'].location = EarthLocation([1.0, 2.0], [2.0, 3.0], [3.0, 4.0], unit='Mm')
        with pytest.warns(AstropyUserWarning, match='Time Column "b" has no specified location, but global Time Position is present'):
            (table, hdr) = time_to_fits(t)
        assert (table['OBSGEO-X'] == t['a'].location.x.to_value(unit='m')).all()
        assert (table['OBSGEO-Y'] == t['a'].location.y.to_value(unit='m')).all()
        assert (table['OBSGEO-Z'] == t['a'].location.z.to_value(unit='m')).all()
        with pytest.warns(AstropyUserWarning, match='Time Column "b" has no specified location, but global Time Position is present'):
            t.write(self.temp('time.fits'), format='fits', overwrite=True)
        hdr = fits.getheader(self.temp('time.fits'), 1)
        assert hdr.get('TRPOS2', None) is None
        with pytest.warns(AstropyUserWarning, match='Time column reference position "TRPOSn" is not specified. The default value for it is "TOPOCENTER", and the observatory position has been specified.'):
            tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert (tm['a'].location == t['a'].location).all()
        assert tm['b'].location == t['b'].location
        t['a'].location = EarthLocation(1, 2, 3)
        t['b'].location = EarthLocation(2, 3, 4)
        with pytest.raises(ValueError) as err:
            (table, hdr) = time_to_fits(t)
            assert 'Multiple Time Columns with different geocentric' in str(err.value)
        t['b'].location = None
        with pytest.warns(AstropyUserWarning, match='Time Column "b" has no specified location, but global Time Position is present') as w:
            (table, hdr) = time_to_fits(t)
        assert len(w) == 1
        t['b'].location = EarthLocation(1, 2, 3)
        (table, hdr) = time_to_fits(t)
        for scale in BARYCENTRIC_SCALES:
            t.replace_column('a', getattr(t['a'], scale))
            with pytest.warns(AstropyUserWarning, match='Earth Location "TOPOCENTER" for Time Column') as w:
                (table, hdr) = time_to_fits(t)
            assert len(w) == 1
        t = table_types()
        location = EarthLocation([[[1.0, 2.0], [1.0, 3.0], [3.0, 4.0]]], [[[1.0, 2.0], [1.0, 3.0], [3.0, 4.0]]], [[[1.0, 2.0], [1.0, 3.0], [3.0, 4.0]]], unit='Mm')
        t['a'] = Time(self.time_3d, format='jd', location=location)
        (table, hdr) = time_to_fits(t)
        assert (table['OBSGEO-X'] == t['a'].location.x.to_value(unit='m')).all()
        assert (table['OBSGEO-Y'] == t['a'].location.y.to_value(unit='m')).all()
        assert (table['OBSGEO-Z'] == t['a'].location.z.to_value(unit='m')).all()
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert (tm['a'].location == t['a'].location).all()
        t['a'] = Time(self.time, location=EarthLocation([[[1.0]]], [[[2.0]]], [[[3.0]]], unit='Mm'))
        (table, hdr) = time_to_fits(t)
        assert hdr['OBSGEO-X'] == t['a'].location.x.to_value(unit='m')
        assert hdr['OBSGEO-Y'] == t['a'].location.y.to_value(unit='m')
        assert hdr['OBSGEO-Z'] == t['a'].location.z.to_value(unit='m')
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert tm['a'].location == t['a'].location

    @pytest.mark.parametrize('masked_cls', (np.ma.MaskedArray, Masked))
    @pytest.mark.parametrize('mask', (False, [True, False]))
    @pytest.mark.parametrize('serialize_method', ('jd1_jd2', 'formatted_value'))
    def test_time_to_fits_serialize_method(self, serialize_method, mask, masked_cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the data returned by ``time_to_fits`` for masked values.\n        '
        a = Time(masked_cls(self.time, mask=mask))
        b = Time(masked_cls([[1, 2], [3, 4]], mask=np.broadcast_to(mask, (2, 2))), format='cxcsec')
        assert b.masked is a.masked is (mask is not False)
        t = QTable([a, b], names=['a', 'b'])
        t.write(self.temp('time.fits'), format='fits', overwrite=True, serialize_method=serialize_method)
        tm = QTable.read(self.temp('time.fits'), format='fits', astropy_native=True)
        if mask is not False:
            assert np.all(tm['a'].mask == a.mask)
            assert np.all(tm['b'].mask == b.mask)
        if serialize_method == 'jd1_jd2':
            assert isinstance(tm['a'], Time) and np.all(tm['a'] == a)
            assert isinstance(tm['b'], Time) and np.all(tm['b'] == b)
        else:
            assert isinstance(tm['a'], Column) and np.all(tm['a'] == a.value)
            assert isinstance(tm['b'], Column) and np.all(tm['b'] == b.value)

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_time_to_fits_header(self, table_types):
        if False:
            while True:
                i = 10
        '\n        Test the header and metadata returned by ``time_to_fits``.\n        '
        t = table_types()
        t['a'] = Time(self.time, format='isot', scale='utc', location=EarthLocation(-2446354, 4237210, 4077985, unit='m'))
        t['b'] = Time([1, 2], format='cxcsec', scale='tt')
        ideal_col_hdr = {'OBSGEO-X': t['a'].location.x.value, 'OBSGEO-Y': t['a'].location.y.value, 'OBSGEO-Z': t['a'].location.z.value}
        with pytest.warns(AstropyUserWarning, match='Time Column "b" has no specified location, but global Time Position is present'):
            (table, hdr) = time_to_fits(t)
        for (key, value) in GLOBAL_TIME_INFO.items():
            assert hdr[key] == value[0]
            assert hdr.comments[key] == value[1]
            hdr.remove(key)
        for (key, value) in ideal_col_hdr.items():
            assert hdr[key] == value
            hdr.remove(key)
        coord_info = table.meta['__coordinate_columns__']
        for colname in coord_info:
            assert coord_info[colname]['coord_type'] == t[colname].scale.upper()
            assert coord_info[colname]['coord_unit'] == 'd'
        assert coord_info['a']['time_ref_pos'] == 'TOPOCENTER'
        assert coord_info['b']['time_ref_pos'] is None
        assert len(hdr) == 0

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_fits_to_time_meta(self, table_types):
        if False:
            while True:
                i = 10
        '\n        Test that the relevant global time metadata is read into\n        ``Table.meta`` as ``Time``.\n        '
        t = table_types()
        t['a'] = Time(self.time, format='isot', scale='utc')
        t.meta['DATE'] = '1999-01-01T00:00:00'
        t.meta['MJD-OBS'] = 56670
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert isinstance(tm.meta['DATE'], Time)
        assert tm.meta['DATE'].value == t.meta['DATE']
        assert tm.meta['DATE'].format == 'fits'
        assert tm.meta['DATE'].scale == 'utc'
        assert isinstance(tm.meta['MJD-OBS'], Time)
        assert tm.meta['MJD-OBS'].value == t.meta['MJD-OBS']
        assert tm.meta['MJD-OBS'].format == 'mjd'
        assert tm.meta['MJD-OBS'].scale == 'utc'
        t.meta['TIMESYS'] = 'ET'
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert isinstance(tm.meta['DATE'], Time)
        assert tm.meta['DATE'].value == t.meta['DATE']
        assert tm.meta['DATE'].scale == 'utc'
        assert isinstance(tm.meta['MJD-OBS'], Time)
        assert tm.meta['MJD-OBS'].value == t.meta['MJD-OBS']
        assert tm.meta['MJD-OBS'].scale == FITS_DEPRECATED_SCALES[t.meta['TIMESYS']]
        t['a'].info.serialize_method['fits'] = 'formatted_value'
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits')
        assert not isinstance(tm.meta['DATE'], Time)
        assert tm.meta['DATE'] == t.meta['DATE']
        assert not isinstance(tm.meta['MJD-OBS'], Time)
        assert tm.meta['MJD-OBS'] == t.meta['MJD-OBS']
        assert (tm['a'] == t['a'].value).all()

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_time_loc_unit(self, table_types):
        if False:
            return 10
        '\n        Test that ``location`` specified by using any valid unit\n        (length/angle) in ``Time`` columns gets stored in FITS\n        as ITRS Cartesian coordinates (X, Y, Z), each in m.\n        Test that it round-trips through FITS.\n        '
        t = table_types()
        t['a'] = Time(self.time, format='isot', scale='utc', location=EarthLocation(1, 2, 3, unit='km'))
        (table, hdr) = time_to_fits(t)
        assert hdr['OBSGEO-X'] == t['a'].location.x.to_value(unit='m')
        assert hdr['OBSGEO-Y'] == t['a'].location.y.to_value(unit='m')
        assert hdr['OBSGEO-Z'] == t['a'].location.z.to_value(unit='m')
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert (tm['a'].location == t['a'].location).all()
        assert tm['a'].location.x.value == t['a'].location.x.to_value(unit='m')
        assert tm['a'].location.y.value == t['a'].location.y.to_value(unit='m')
        assert tm['a'].location.z.value == t['a'].location.z.to_value(unit='m')

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_fits_to_time_index(self, table_types):
        if False:
            print('Hello World!')
        '\n        Ensure that fits_to_time works correctly if the time column is also\n        an index.\n        '
        t = table_types()
        t['a'] = Time(self.time, format='isot', scale='utc')
        t['b'] = [2, 1]
        t['c'] = [3, 4]
        t.add_index('a')
        t.add_index('b')
        t.write(self.temp('time.fits'), format='fits', overwrite=True)
        tm = table_types.read(self.temp('time.fits'), format='fits', astropy_native=True)
        assert isinstance(tm['a'], Time)
        assert len(t.indices) == 2
        assert len(tm.indices) == 0
        for name in ('a', 'b'):
            assert len(t[name].info.indices) == 1
            assert len(tm[name].info.indices) == 0

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_io_time_read_fits(self, table_types):
        if False:
            print('Hello World!')
        "\n        Test that FITS table with time columns (standard compliant)\n        can be read by io.fits as a table with Time columns.\n        This tests the following:\n\n        1. The special-case where a column has the name 'TIME' and a\n           time unit\n        2. Time from Epoch (Reference time) is appropriately converted.\n        3. Coordinate columns (corresponding to coordinate keywords in the header)\n           other than time, that is, spatial coordinates, are not mistaken\n           to be time.\n        "
        filename = self.data('chandra_time.fits')
        with pytest.warns(AstropyUserWarning, match='Time column "time" reference position will be ignored'):
            tm = table_types.read(filename, astropy_native=True)
        assert isinstance(tm['time'], Time)
        assert tm['time'].scale == 'tt'
        assert tm['time'].format == 'mjd'
        non_native = table_types.read(filename)
        ref_time = Time(non_native.meta['MJDREF'], format='mjd', scale=non_native.meta['TIMESYS'].lower())
        delta_time = TimeDelta(non_native['time'])
        assert (ref_time + delta_time == tm['time']).all()
        for colname in ['chipx', 'chipy', 'detx', 'dety', 'x', 'y']:
            assert not isinstance(tm[colname], Time)

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_io_time_read_fits_datetime(self, table_types):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that ISO-8601 Datetime String Columns are read correctly.\n        '
        c = fits.Column(name='datetime', format='A29', coord_type='TCG', time_ref_pos='GEOCENTER', array=self.time)
        bhdu = fits.BinTableHDU.from_columns([c])
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        tm = table_types.read(self.temp('time.fits'), astropy_native=True)
        assert isinstance(tm['datetime'], Time)
        assert tm['datetime'].scale == 'tcg'
        assert tm['datetime'].format == 'fits'
        assert (tm['datetime'] == self.time).all()

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_io_time_read_fits_location(self, table_types):
        if False:
            while True:
                i = 10
        '\n        Test that geocentric/geodetic observatory position is read\n        properly, as and when it is specified.\n        '
        c = fits.Column(name='datetime', format='A29', coord_type='TT', time_ref_pos='TOPOCENTER', array=self.time)
        cards = [('OBSGEO-X', -2446354), ('OBSGEO-Y', 4237210), ('OBSGEO-Z', 4077985)]
        bhdu = fits.BinTableHDU.from_columns([c], header=fits.Header(cards))
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        tm = table_types.read(self.temp('time.fits'), astropy_native=True)
        assert isinstance(tm['datetime'], Time)
        assert tm['datetime'].location.x.value == -2446354
        assert tm['datetime'].location.y.value == 4237210
        assert tm['datetime'].location.z.value == 4077985
        cards = [('OBSGEO-L', 0), ('OBSGEO-B', 0), ('OBSGEO-H', 0)]
        bhdu = fits.BinTableHDU.from_columns([c], header=fits.Header(cards))
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        tm = table_types.read(self.temp('time.fits'), astropy_native=True)
        assert isinstance(tm['datetime'], Time)
        assert tm['datetime'].location.lon.value == 0
        assert tm['datetime'].location.lat.value == 0
        assert np.isclose(tm['datetime'].location.height.value, 0, rtol=0, atol=1e-09)

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_io_time_read_fits_scale(self, table_types):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test handling of 'GPS' and 'LOCAL' time scales which are\n        recognized by the FITS standard but are not native to astropy.\n        "
        gps_time = np.array([630720013, 630720014])
        c = fits.Column(name='gps_time', format='D', unit='s', coord_type='GPS', coord_unit='s', time_ref_pos='TOPOCENTER', array=gps_time)
        cards = [('OBSGEO-L', 0), ('OBSGEO-B', 0), ('OBSGEO-H', 0)]
        bhdu = fits.BinTableHDU.from_columns([c], header=fits.Header(cards))
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        with pytest.warns(AstropyUserWarning, match='FITS recognized time scale value "GPS"') as w:
            tm = table_types.read(self.temp('time.fits'), astropy_native=True)
        assert len(w) == 1
        assert isinstance(tm['gps_time'], Time)
        assert tm['gps_time'].format == 'gps'
        assert tm['gps_time'].scale == 'tai'
        assert (tm['gps_time'].value == gps_time).all()
        local_time = np.array([1, 2])
        c = fits.Column(name='local_time', format='D', unit='d', coord_type='LOCAL', coord_unit='d', time_ref_pos='RELOCATABLE', array=local_time)
        bhdu = fits.BinTableHDU.from_columns([c])
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        tm = table_types.read(self.temp('time.fits'), astropy_native=True)
        assert isinstance(tm['local_time'], Time)
        assert tm['local_time'].format == 'mjd'
        assert tm['local_time'].scale == 'local'
        assert (tm['local_time'].value == local_time).all()

    @pytest.mark.parametrize('table_types', (Table, QTable))
    def test_io_time_read_fits_location_warnings(self, table_types):
        if False:
            return 10
        '\n        Test warnings for time column reference position.\n        '
        c = fits.Column(name='datetime', format='A29', coord_type='TT', time_ref_pos='TOPOCENTER', array=self.time)
        bhdu = fits.BinTableHDU.from_columns([c])
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        with pytest.warns(AstropyUserWarning, match='observatory position is not properly specified') as w:
            table_types.read(self.temp('time.fits'), astropy_native=True)
        assert len(w) == 1
        c = fits.Column(name='datetime', format='A29', coord_type='TT', array=self.time)
        bhdu = fits.BinTableHDU.from_columns([c])
        bhdu.writeto(self.temp('time.fits'), overwrite=True)
        table_types.read(self.temp('time.fits'), astropy_native=True)