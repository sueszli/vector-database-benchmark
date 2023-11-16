from datetime import datetime
import pytest
from numpy.testing import assert_allclose, assert_equal
from astropy import units as u
from astropy.table import Column, Table
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time, TimeDelta
from astropy.timeseries.periodograms import BoxLeastSquares, LombScargle
from astropy.timeseries.sampled import TimeSeries
from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
INPUT_TIME = Time(['2016-03-22T12:30:31', '2015-01-21T12:30:32', '2016-03-22T12:30:40'])
PLAIN_TABLE = Table([[1, 2, 11], [3, 4, 1], [1, 1, 1]], names=['a', 'b', 'c'])
CSV_FILE = get_pkg_data_filename('data/sampled.csv')

def test_empty_initialization():
    if False:
        return 10
    ts = TimeSeries()
    ts['time'] = Time([50001, 50002, 50003], format='mjd')

def test_empty_initialization_invalid():
    if False:
        i = 10
        return i + 15
    ts = TimeSeries()
    with pytest.raises(ValueError, match="TimeSeries object is invalid - expected 'time' as the first column but found 'flux'"):
        ts['flux'] = [1, 2, 3]

def test_initialize_only_time():
    if False:
        print('Hello World!')
    ts = TimeSeries(time=INPUT_TIME)
    assert ts['time'] is ts.time
    assert_equal(ts.time.isot, INPUT_TIME.isot)

def test_initialization_with_data():
    if False:
        print('Hello World!')
    ts = TimeSeries(time=INPUT_TIME, data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])
    assert_equal(ts.time.isot, INPUT_TIME.isot)
    assert_equal(ts['a'], [10, 2, 3])
    assert_equal(ts['b'], [4, 5, 6])

def test_initialize_only_data():
    if False:
        return 10
    with pytest.raises(TypeError, match="Either 'time' or 'time_start' should be specified"):
        TimeSeries(data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])

def test_initialization_with_table():
    if False:
        print('Hello World!')
    ts = TimeSeries(time=INPUT_TIME, data=PLAIN_TABLE)
    assert ts.colnames == ['time', 'a', 'b', 'c']

def test_initialization_with_time_delta():
    if False:
        return 10
    ts = TimeSeries(time_start=datetime(2018, 7, 1, 10, 10, 10), time_delta=TimeDelta(3, format='sec'), data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])
    assert_equal(ts.time.isot, ['2018-07-01T10:10:10.000', '2018-07-01T10:10:13.000', '2018-07-01T10:10:16.000'])

def test_initialization_missing_time_delta():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match="'time' is scalar, so 'time_delta' is required"):
        TimeSeries(time_start=datetime(2018, 7, 1, 10, 10, 10), data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])

def test_initialization_invalid_time_and_time_start():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match="Cannot specify both 'time' and 'time_start'"):
        TimeSeries(time=INPUT_TIME, time_start=datetime(2018, 7, 1, 10, 10, 10), data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])

def test_initialization_invalid_time_delta():
    if False:
        print('Hello World!')
    with pytest.raises(TypeError, match="'time_delta' should be a Quantity or a TimeDelta"):
        TimeSeries(time_start=datetime(2018, 7, 1, 10, 10, 10), time_delta=[1, 4, 3], data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])

def test_initialization_with_time_in_data():
    if False:
        print('Hello World!')
    data = PLAIN_TABLE.copy()
    data['time'] = INPUT_TIME
    ts1 = TimeSeries(data=data)
    assert set(ts1.colnames) == {'time', 'a', 'b', 'c'}
    assert all(ts1.time == INPUT_TIME)
    ts2 = TimeSeries(data=[[10, 2, 3], INPUT_TIME], names=['a', 'time'])
    assert set(ts2.colnames) == {'time', 'a'}
    assert all(ts2.time == INPUT_TIME)
    MESSAGE = "'time' has been given both in the table and as a keyword argument"
    with pytest.raises(TypeError, match=MESSAGE):
        TimeSeries(data=data, time=INPUT_TIME)
    with pytest.raises(TypeError, match=MESSAGE):
        TimeSeries(time=INPUT_TIME, data=[[10, 2, 3], INPUT_TIME], names=['a', 'time'])

def test_initialization_n_samples():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError, match="'n_samples' has been given both and it is not the same length as the input data."):
        TimeSeries(time=INPUT_TIME, data=PLAIN_TABLE, n_samples=1000)

def test_initialization_length_mismatch():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match="Length of 'time' \\(3\\) should match data length \\(2\\)"):
        TimeSeries(time=INPUT_TIME, data=[[10, 2], [4, 5]], names=['a', 'b'])

def test_initialization_invalid_both_time_and_time_delta():
    if False:
        print('Hello World!')
    with pytest.raises(TypeError, match="'time_delta' should not be specified since 'time' is an array"):
        TimeSeries(time=INPUT_TIME, time_delta=TimeDelta(3, format='sec'))

def test_fold():
    if False:
        while True:
            i = 10
    times = Time([1, 2, 3, 8, 9, 12], format='unix')
    ts = TimeSeries(time=times)
    ts['flux'] = [1, 4, 4, 3, 2, 3]
    tsf = ts.fold(period=3.2 * u.s)
    assert isinstance(tsf.time, TimeDelta)
    assert_allclose(tsf.time.sec, [0, 1, -1.2, 0.6, -1.6, 1.4], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_time=Time(1.6, format='unix'))
    assert isinstance(tsf.time, TimeDelta)
    assert_allclose(tsf.time.sec, [-0.6, 0.4, 1.4, 0.0, 1.0, 0.8], rtol=1e-06, atol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, wrap_phase=3.2 * u.s)
    assert isinstance(tsf.time, TimeDelta)
    assert_allclose(tsf.time.sec, [0, 1, 2, 0.6, 1.6, 1.4], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_phase=0.8 * u.s)
    assert isinstance(tsf.time, TimeDelta)
    assert_allclose(tsf.time.sec, [0.8, -1.4, -0.4, 1.4, -0.8, -1.0], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_phase=0.8 * u.s, wrap_phase=3.2 * u.s)
    assert isinstance(tsf.time, TimeDelta)
    assert_allclose(tsf.time.sec, [0.8, 1.8, 2.8, 1.4, 2.4, 2.2], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, normalize_phase=True)
    assert isinstance(tsf.time, Quantity)
    assert_allclose(tsf.time.to_value(u.one), [0, 1 / 3.2, -1.2 / 3.2, 0.6 / 3.2, -1.6 / 3.2, 1.4 / 3.2], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_time=Time(1.6, format='unix'), normalize_phase=True)
    assert isinstance(tsf.time, Quantity)
    assert_allclose(tsf.time.to_value(u.one), [-0.6 / 3.2, 0.4 / 3.2, 1.4 / 3.2, 0.0 / 3.2, 1.0 / 3.2, 0.8 / 3.2], rtol=1e-06, atol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, wrap_phase=1, normalize_phase=True)
    assert isinstance(tsf.time, Quantity)
    assert_allclose(tsf.time.to_value(u.one), [0, 1 / 3.2, 2 / 3.2, 0.6 / 3.2, 1.6 / 3.2, 1.4 / 3.2], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_phase=0.25, normalize_phase=True)
    assert isinstance(tsf.time, Quantity)
    assert_allclose(tsf.time.to_value(u.one), [0.8 / 3.2, -1.4 / 3.2, -0.4 / 3.2, 1.4 / 3.2, -0.8 / 3.2, -1.0 / 3.2], rtol=1e-06)
    tsf = ts.fold(period=3.2 * u.s, epoch_phase=0.25, wrap_phase=1, normalize_phase=True)
    assert isinstance(tsf.time, Quantity)
    assert_allclose(tsf.time.to_value(u.one), [0.8 / 3.2, 1.8 / 3.2, 2.8 / 3.2, 1.4 / 3.2, 2.4 / 3.2, 2.2 / 3.2], rtol=1e-06)

def test_fold_invalid_options():
    if False:
        return 10
    times = Time([1, 2, 3, 8, 9, 12], format='unix')
    ts = TimeSeries(time=times)
    ts['flux'] = [1, 4, 4, 3, 2, 3]
    with pytest.raises(u.UnitsError, match='period should be a Quantity in units of time'):
        ts.fold(period=3.2)
    with pytest.raises(u.UnitsError, match='period should be a Quantity in units of time'):
        ts.fold(period=3.2 * u.m)
    with pytest.raises(u.UnitsError, match='epoch_phase should be a Quantity in units of time when normalize_phase=False'):
        ts.fold(period=3.2 * u.s, epoch_phase=0.2)
    with pytest.raises(u.UnitsError, match='epoch_phase should be a dimensionless Quantity or a float when normalize_phase=True'):
        ts.fold(period=3.2 * u.s, epoch_phase=0.2 * u.s, normalize_phase=True)
    with pytest.raises(u.UnitsError, match='wrap_phase should be a Quantity in units of time when normalize_phase=False'):
        ts.fold(period=3.2 * u.s, wrap_phase=0.2)
    with pytest.raises(u.UnitsError, match='wrap_phase should be dimensionless when normalize_phase=True'):
        ts.fold(period=3.2 * u.s, wrap_phase=0.2 * u.s, normalize_phase=True)
    with pytest.raises(ValueError, match='wrap_phase should be between 0 and the period'):
        ts.fold(period=3.2 * u.s, wrap_phase=-0.1 * u.s)
    with pytest.raises(ValueError, match='wrap_phase should be between 0 and the period'):
        ts.fold(period=3.2 * u.s, wrap_phase=-4.2 * u.s)
    with pytest.raises(ValueError, match='wrap_phase should be between 0 and 1'):
        ts.fold(period=3.2 * u.s, wrap_phase=-0.1, normalize_phase=True)
    with pytest.raises(ValueError, match='wrap_phase should be between 0 and 1'):
        ts.fold(period=3.2 * u.s, wrap_phase=2.2, normalize_phase=True)

def test_pandas():
    if False:
        return 10
    pandas = pytest.importorskip('pandas')
    df1 = pandas.DataFrame()
    df1['a'] = [1, 2, 3]
    df1.set_index(pandas.DatetimeIndex(INPUT_TIME.datetime64), inplace=True)
    ts = TimeSeries.from_pandas(df1)
    assert_equal(ts.time.isot, INPUT_TIME.isot)
    assert ts.colnames == ['time', 'a']
    assert len(ts.indices) == 1
    assert (ts.indices['time'].columns[0] == INPUT_TIME).all()
    ts_tcb = TimeSeries.from_pandas(df1, time_scale='tcb')
    assert ts_tcb.time.scale == 'tcb'
    df2 = ts.to_pandas()
    assert (df2.index.values == pandas.Index(INPUT_TIME.datetime64).values).all()
    assert df2.columns == pandas.Index(['a'])
    assert (df1['a'] == df2['a']).all()
    with pytest.raises(TypeError, match='Input should be a pandas DataFrame'):
        TimeSeries.from_pandas(None)
    df4 = pandas.DataFrame()
    df4['a'] = [1, 2, 3]
    with pytest.raises(TypeError, match='DataFrame does not have a DatetimeIndex'):
        TimeSeries.from_pandas(df4)

def test_read_time_missing():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='``time_column`` should be provided since the default Table readers are being used\\.'):
        TimeSeries.read(CSV_FILE, format='csv')

def test_read_time_wrong():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match="Time column 'abc' not found in the input data\\."):
        TimeSeries.read(CSV_FILE, time_column='abc', format='csv')

def test_read():
    if False:
        for i in range(10):
            print('nop')
    timeseries = TimeSeries.read(CSV_FILE, time_column='Date', format='csv')
    assert timeseries.colnames == ['time', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert len(timeseries) == 11
    assert timeseries['time'].format == 'iso'
    assert timeseries['A'].sum() == 266.5

@pytest.mark.remote_data(source='astropy')
def test_kepler_astropy():
    if False:
        return 10
    from astropy.units import UnitsWarning
    filename = get_pkg_data_filename('timeseries/kplr010666592-2009131110544_slc.fits')
    with pytest.warns(UnitsWarning):
        timeseries = TimeSeries.read(filename, format='kepler.fits')
    assert timeseries['time'].format == 'isot'
    assert timeseries['time'].scale == 'tdb'
    assert timeseries['sap_flux'].unit.to_string() == 'electron / s'
    assert len(timeseries) == 14280
    assert len(timeseries.columns) == 20

@pytest.mark.remote_data(source='astropy')
def test_tess_astropy():
    if False:
        while True:
            i = 10
    filename = get_pkg_data_filename('timeseries/hlsp_tess-data-alerts_tess_phot_00025155310-s01_tess_v1_lc.fits')
    with pytest.warns(UserWarning, match='Ignoring 815 rows with NaN times'):
        timeseries = TimeSeries.read(filename, format='tess.fits')
    assert timeseries['time'].format == 'isot'
    assert timeseries['time'].scale == 'tdb'
    assert timeseries['sap_flux'].unit.to_string() == 'electron / s'
    assert len(timeseries) == 19261
    assert len(timeseries.columns) == 20

def test_required_columns():
    if False:
        return 10
    ts = TimeSeries(time=INPUT_TIME, data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])
    ts.copy()
    MESSAGE = "TimeSeries object is invalid - expected 'time' as the first column but found '{}'"
    with pytest.raises(ValueError, match=MESSAGE.format('c')):
        ts.copy().add_column(Column([3, 4, 5], name='c'), index=0)
    with pytest.raises(ValueError, match=MESSAGE.format('d')):
        ts.copy().add_columns([Column([3, 4, 5], name='d'), Column([3, 4, 5], name='e')], indexes=[0, 1])
    with pytest.raises(ValueError, match=MESSAGE.format('a')):
        ts.copy().keep_columns(['a', 'b'])
    with pytest.raises(ValueError, match=MESSAGE.format('a')):
        ts.copy().remove_column('time')
    with pytest.raises(ValueError, match=MESSAGE.format('b')):
        ts.copy().remove_columns(['time', 'a'])
    with pytest.raises(ValueError, match=MESSAGE.format('banana')):
        ts.copy().rename_column('time', 'banana')
    MESSAGE = "TimeSeries object is invalid - expected \\['time', 'a'\\] as the first columns but found \\['time', 'b'\\]"
    ts_2cols_required = ts.copy()
    ts_2cols_required._required_columns = ['time', 'a']
    with pytest.raises(ValueError, match=MESSAGE):
        ts_2cols_required.remove_column('a')

@pytest.mark.parametrize('cls', [BoxLeastSquares, LombScargle])
def test_periodogram(cls):
    if False:
        for i in range(10):
            print('nop')
    ts = TimeSeries(time=INPUT_TIME, data=[[10, 2, 3], [4, 5, 6]], names=['a', 'b'])
    p1 = cls.from_timeseries(ts, 'a')
    assert isinstance(p1, cls)
    assert_allclose(p1.t.jd, ts.time.jd)
    assert_equal(p1.y, ts['a'])
    assert p1.dy is None
    p2 = cls.from_timeseries(ts, 'a', uncertainty='b')
    assert_quantity_allclose(p2.dy, ts['b'])
    p3 = cls.from_timeseries(ts, 'a', uncertainty=0.1)
    assert_allclose(p3.dy, 0.1)