from datetime import timedelta
import dateutil
import numpy as np
from pandas import DataFrame, Series, date_range, period_range, timedelta_range
from pandas.tseries.frequencies import infer_freq
try:
    from pandas.plotting._matplotlib.converter import DatetimeConverter
except ImportError:
    from pandas.tseries.converter import DatetimeConverter

class DatetimeIndex:
    params = ['dst', 'repeated', 'tz_aware', 'tz_local', 'tz_naive']
    param_names = ['index_type']

    def setup(self, index_type):
        if False:
            i = 10
            return i + 15
        N = 100000
        dtidxes = {'dst': date_range(start='10/29/2000 1:00:00', end='10/29/2000 1:59:59', freq='s'), 'repeated': date_range(start='2000', periods=N / 10, freq='s').repeat(10), 'tz_aware': date_range(start='2000', periods=N, freq='s', tz='US/Eastern'), 'tz_local': date_range(start='2000', periods=N, freq='s', tz=dateutil.tz.tzlocal()), 'tz_naive': date_range(start='2000', periods=N, freq='s')}
        self.index = dtidxes[index_type]

    def time_add_timedelta(self, index_type):
        if False:
            i = 10
            return i + 15
        self.index + timedelta(minutes=2)

    def time_normalize(self, index_type):
        if False:
            print('Hello World!')
        self.index.normalize()

    def time_unique(self, index_type):
        if False:
            while True:
                i = 10
        self.index.unique()

    def time_to_time(self, index_type):
        if False:
            return 10
        self.index.time

    def time_get(self, index_type):
        if False:
            for i in range(10):
                print('nop')
        self.index[0]

    def time_timeseries_is_month_start(self, index_type):
        if False:
            i = 10
            return i + 15
        self.index.is_month_start

    def time_to_date(self, index_type):
        if False:
            print('Hello World!')
        self.index.date

    def time_to_pydatetime(self, index_type):
        if False:
            i = 10
            return i + 15
        self.index.to_pydatetime()

    def time_is_dates_only(self, index_type):
        if False:
            print('Hello World!')
        self.index._is_dates_only

class TzLocalize:
    params = [None, 'US/Eastern', 'UTC', dateutil.tz.tzutc()]
    param_names = 'tz'

    def setup(self, tz):
        if False:
            for i in range(10):
                print('nop')
        dst_rng = date_range(start='10/29/2000 1:00:00', end='10/29/2000 1:59:59', freq='s')
        self.index = date_range(start='10/29/2000', end='10/29/2000 00:59:59', freq='s')
        self.index = self.index.append(dst_rng)
        self.index = self.index.append(dst_rng)
        self.index = self.index.append(date_range(start='10/29/2000 2:00:00', end='10/29/2000 3:00:00', freq='s'))

    def time_infer_dst(self, tz):
        if False:
            while True:
                i = 10
        self.index.tz_localize(tz, ambiguous='infer')

class ResetIndex:
    params = [None, 'US/Eastern']
    param_names = 'tz'

    def setup(self, tz):
        if False:
            i = 10
            return i + 15
        idx = date_range(start='1/1/2000', periods=1000, freq='h', tz=tz)
        self.df = DataFrame(np.random.randn(1000, 2), index=idx)

    def time_reset_datetimeindex(self, tz):
        if False:
            while True:
                i = 10
        self.df.reset_index()

class InferFreq:
    params = [None, 'D', 'B']
    param_names = ['freq']

    def setup(self, freq):
        if False:
            return 10
        if freq is None:
            self.idx = date_range(start='1/1/1700', freq='D', periods=10000)
            self.idx._data._freq = None
        else:
            self.idx = date_range(start='1/1/1700', freq=freq, periods=10000)

    def time_infer_freq(self, freq):
        if False:
            i = 10
            return i + 15
        infer_freq(self.idx)

class TimeDatetimeConverter:

    def setup(self):
        if False:
            i = 10
            return i + 15
        N = 100000
        self.rng = date_range(start='1/1/2000', periods=N, freq='min')

    def time_convert(self):
        if False:
            print('Hello World!')
        DatetimeConverter.convert(self.rng, None, None)

class Iteration:
    params = [date_range, period_range, timedelta_range]
    param_names = ['time_index']

    def setup(self, time_index):
        if False:
            i = 10
            return i + 15
        N = 10 ** 6
        if time_index is timedelta_range:
            self.idx = time_index(start=0, freq='min', periods=N)
        else:
            self.idx = time_index(start='20140101', freq='min', periods=N)
        self.exit = 10000

    def time_iter(self, time_index):
        if False:
            for i in range(10):
                print('nop')
        for _ in self.idx:
            pass

    def time_iter_preexit(self, time_index):
        if False:
            print('Hello World!')
        for (i, _) in enumerate(self.idx):
            if i > self.exit:
                break

class ResampleDataFrame:
    params = ['max', 'mean', 'min']
    param_names = ['method']

    def setup(self, method):
        if False:
            print('Hello World!')
        rng = date_range(start='20130101', periods=100000, freq='50ms')
        df = DataFrame(np.random.randn(100000, 2), index=rng)
        self.resample = getattr(df.resample('1s'), method)

    def time_method(self, method):
        if False:
            i = 10
            return i + 15
        self.resample()

class ResampleSeries:
    params = (['period', 'datetime'], ['5min', '1D'], ['mean', 'ohlc'])
    param_names = ['index', 'freq', 'method']

    def setup(self, index, freq, method):
        if False:
            while True:
                i = 10
        indexes = {'period': period_range(start='1/1/2000', end='1/1/2001', freq='min'), 'datetime': date_range(start='1/1/2000', end='1/1/2001', freq='min')}
        idx = indexes[index]
        ts = Series(np.random.randn(len(idx)), index=idx)
        self.resample = getattr(ts.resample(freq), method)

    def time_resample(self, index, freq, method):
        if False:
            return 10
        self.resample()

class ResampleDatetetime64:

    def setup(self):
        if False:
            i = 10
            return i + 15
        rng3 = date_range(start='2000-01-01 00:00:00', end='2000-01-01 10:00:00', freq='555000us')
        self.dt_ts = Series(5, rng3, dtype='datetime64[ns]')

    def time_resample(self):
        if False:
            return 10
        self.dt_ts.resample('1S').last()

class AsOf:
    params = ['DataFrame', 'Series']
    param_names = ['constructor']

    def setup(self, constructor):
        if False:
            print('Hello World!')
        N = 10000
        M = 10
        rng = date_range(start='1/1/1990', periods=N, freq='53s')
        data = {'DataFrame': DataFrame(np.random.randn(N, M)), 'Series': Series(np.random.randn(N))}
        self.ts = data[constructor]
        self.ts.index = rng
        self.ts2 = self.ts.copy()
        self.ts2.iloc[250:5000] = np.nan
        self.ts3 = self.ts.copy()
        self.ts3.iloc[-5000:] = np.nan
        self.dates = date_range(start='1/1/1990', periods=N * 10, freq='5s')
        self.date = self.dates[0]
        self.date_last = self.dates[-1]
        self.date_early = self.date - timedelta(10)

    def time_asof(self, constructor):
        if False:
            i = 10
            return i + 15
        self.ts.asof(self.dates)

    def time_asof_nan(self, constructor):
        if False:
            print('Hello World!')
        self.ts2.asof(self.dates)

    def time_asof_single(self, constructor):
        if False:
            i = 10
            return i + 15
        self.ts.asof(self.date)

    def time_asof_single_early(self, constructor):
        if False:
            i = 10
            return i + 15
        self.ts.asof(self.date_early)

    def time_asof_nan_single(self, constructor):
        if False:
            i = 10
            return i + 15
        self.ts3.asof(self.date_last)

class SortIndex:
    params = [True, False]
    param_names = ['monotonic']

    def setup(self, monotonic):
        if False:
            while True:
                i = 10
        N = 10 ** 5
        idx = date_range(start='1/1/2000', periods=N, freq='s')
        self.s = Series(np.random.randn(N), index=idx)
        if not monotonic:
            self.s = self.s.sample(frac=1)

    def time_sort_index(self, monotonic):
        if False:
            print('Hello World!')
        self.s.sort_index()

    def time_get_slice(self, monotonic):
        if False:
            i = 10
            return i + 15
        self.s[:10000]

class Lookup:

    def setup(self):
        if False:
            i = 10
            return i + 15
        N = 1500000
        rng = date_range(start='1/1/2000', periods=N, freq='s')
        self.ts = Series(1, index=rng)
        self.lookup_val = rng[N // 2]

    def time_lookup_and_cleanup(self):
        if False:
            i = 10
            return i + 15
        self.ts[self.lookup_val]
        self.ts.index._cleanup()

class DatetimeAccessor:
    params = [None, 'US/Eastern', 'UTC', dateutil.tz.tzutc()]
    param_names = 'tz'

    def setup(self, tz):
        if False:
            for i in range(10):
                print('nop')
        N = 100000
        self.series = Series(date_range(start='1/1/2000', periods=N, freq='min', tz=tz))

    def time_dt_accessor(self, tz):
        if False:
            while True:
                i = 10
        self.series.dt

    def time_dt_accessor_normalize(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.series.dt.normalize()

    def time_dt_accessor_month_name(self, tz):
        if False:
            print('Hello World!')
        self.series.dt.month_name()

    def time_dt_accessor_day_name(self, tz):
        if False:
            print('Hello World!')
        self.series.dt.day_name()

    def time_dt_accessor_time(self, tz):
        if False:
            return 10
        self.series.dt.time

    def time_dt_accessor_date(self, tz):
        if False:
            print('Hello World!')
        self.series.dt.date

    def time_dt_accessor_year(self, tz):
        if False:
            print('Hello World!')
        self.series.dt.year
from .pandas_vb_common import setup