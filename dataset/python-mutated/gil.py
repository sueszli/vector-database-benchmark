from functools import wraps
import threading
import numpy as np
from pandas import DataFrame, Series, date_range, factorize, read_csv
from pandas.core.algorithms import take_nd
from .pandas_vb_common import tm
try:
    from pandas import rolling_kurt, rolling_max, rolling_mean, rolling_median, rolling_min, rolling_skew, rolling_std, rolling_var
    have_rolling_methods = True
except ImportError:
    have_rolling_methods = False
try:
    from pandas._libs import algos
except ImportError:
    from pandas import algos
from .pandas_vb_common import BaseIO

def test_parallel(num_threads=2, kwargs_list=None):
    if False:
        while True:
            i = 10
    '\n    Decorator to run the same function multiple times in parallel.\n\n    Parameters\n    ----------\n    num_threads : int, optional\n        The number of times the function is run in parallel.\n    kwargs_list : list of dicts, optional\n        The list of kwargs to update original\n        function kwargs on different threads.\n\n    Notes\n    -----\n    This decorator does not pass the return value of the decorated function.\n\n    Original from scikit-image:\n\n    https://github.com/scikit-image/scikit-image/pull/1519\n\n    '
    assert num_threads > 0
    has_kwargs_list = kwargs_list is not None
    if has_kwargs_list:
        assert len(kwargs_list) == num_threads

    def wrapper(func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def inner(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if has_kwargs_list:
                update_kwargs = lambda i: dict(kwargs, **kwargs_list[i])
            else:
                update_kwargs = lambda i: kwargs
            threads = []
            for i in range(num_threads):
                updated_kwargs = update_kwargs(i)
                thread = threading.Thread(target=func, args=args, kwargs=updated_kwargs)
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        return inner
    return wrapper

class ParallelGroupbyMethods:
    params = ([2, 4, 8], ['count', 'last', 'max', 'mean', 'min', 'prod', 'sum', 'var'])
    param_names = ['threads', 'method']

    def setup(self, threads, method):
        if False:
            print('Hello World!')
        N = 10 ** 6
        ngroups = 10 ** 3
        df = DataFrame({'key': np.random.randint(0, ngroups, size=N), 'data': np.random.randn(N)})

        @test_parallel(num_threads=threads)
        def parallel():
            if False:
                while True:
                    i = 10
            getattr(df.groupby('key')['data'], method)()
        self.parallel = parallel

        def loop():
            if False:
                while True:
                    i = 10
            getattr(df.groupby('key')['data'], method)()
        self.loop = loop

    def time_parallel(self, threads, method):
        if False:
            print('Hello World!')
        self.parallel()

    def time_loop(self, threads, method):
        if False:
            i = 10
            return i + 15
        for i in range(threads):
            self.loop()

class ParallelGroups:
    params = [2, 4, 8]
    param_names = ['threads']

    def setup(self, threads):
        if False:
            for i in range(10):
                print('nop')
        size = 2 ** 22
        ngroups = 10 ** 3
        data = Series(np.random.randint(0, ngroups, size=size))

        @test_parallel(num_threads=threads)
        def get_groups():
            if False:
                for i in range(10):
                    print('nop')
            data.groupby(data).groups
        self.get_groups = get_groups

    def time_get_groups(self, threads):
        if False:
            for i in range(10):
                print('nop')
        self.get_groups()

class ParallelTake1D:
    params = ['int64', 'float64']
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            i = 10
            return i + 15
        N = 10 ** 6
        df = DataFrame({'col': np.arange(N, dtype=dtype)})
        indexer = np.arange(100, len(df) - 100)

        @test_parallel(num_threads=2)
        def parallel_take1d():
            if False:
                for i in range(10):
                    print('nop')
            take_nd(df['col'].values, indexer)
        self.parallel_take1d = parallel_take1d

    def time_take1d(self, dtype):
        if False:
            while True:
                i = 10
        self.parallel_take1d()

class ParallelKth:
    number = 1
    repeat = 5

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        N = 10 ** 7
        k = 5 * 10 ** 5
        kwargs_list = [{'arr': np.random.randn(N)}, {'arr': np.random.randn(N)}]

        @test_parallel(num_threads=2, kwargs_list=kwargs_list)
        def parallel_kth_smallest(arr):
            if False:
                return 10
            algos.kth_smallest(arr, k)
        self.parallel_kth_smallest = parallel_kth_smallest

    def time_kth_smallest(self):
        if False:
            i = 10
            return i + 15
        self.parallel_kth_smallest()

class ParallelDatetimeFields:

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        N = 10 ** 6
        self.dti = date_range('1900-01-01', periods=N, freq='min')
        self.period = self.dti.to_period('D')

    def time_datetime_field_year(self):
        if False:
            while True:
                i = 10

        @test_parallel(num_threads=2)
        def run(dti):
            if False:
                i = 10
                return i + 15
            dti.year
        run(self.dti)

    def time_datetime_field_day(self):
        if False:
            i = 10
            return i + 15

        @test_parallel(num_threads=2)
        def run(dti):
            if False:
                i = 10
                return i + 15
            dti.day
        run(self.dti)

    def time_datetime_field_daysinmonth(self):
        if False:
            return 10

        @test_parallel(num_threads=2)
        def run(dti):
            if False:
                while True:
                    i = 10
            dti.days_in_month
        run(self.dti)

    def time_datetime_field_normalize(self):
        if False:
            while True:
                i = 10

        @test_parallel(num_threads=2)
        def run(dti):
            if False:
                i = 10
                return i + 15
            dti.normalize()
        run(self.dti)

    def time_datetime_to_period(self):
        if False:
            for i in range(10):
                print('nop')

        @test_parallel(num_threads=2)
        def run(dti):
            if False:
                for i in range(10):
                    print('nop')
            dti.to_period('s')
        run(self.dti)

    def time_period_to_datetime(self):
        if False:
            while True:
                i = 10

        @test_parallel(num_threads=2)
        def run(period):
            if False:
                while True:
                    i = 10
            period.to_timestamp()
        run(self.period)

class ParallelRolling:
    params = ['median', 'mean', 'min', 'max', 'var', 'skew', 'kurt', 'std']
    param_names = ['method']

    def setup(self, method):
        if False:
            print('Hello World!')
        win = 100
        arr = np.random.rand(100000)
        if hasattr(DataFrame, 'rolling'):
            df = DataFrame(arr).rolling(win)

            @test_parallel(num_threads=2)
            def parallel_rolling():
                if False:
                    return 10
                getattr(df, method)()
            self.parallel_rolling = parallel_rolling
        elif have_rolling_methods:
            rolling = {'median': rolling_median, 'mean': rolling_mean, 'min': rolling_min, 'max': rolling_max, 'var': rolling_var, 'skew': rolling_skew, 'kurt': rolling_kurt, 'std': rolling_std}

            @test_parallel(num_threads=2)
            def parallel_rolling():
                if False:
                    while True:
                        i = 10
                rolling[method](arr, win)
            self.parallel_rolling = parallel_rolling
        else:
            raise NotImplementedError

    def time_rolling(self, method):
        if False:
            while True:
                i = 10
        self.parallel_rolling()

class ParallelReadCSV(BaseIO):
    number = 1
    repeat = 5
    params = ['float', 'object', 'datetime']
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        rows = 10000
        cols = 50
        if dtype == 'float':
            df = DataFrame(np.random.randn(rows, cols))
        elif dtype == 'datetime':
            df = DataFrame(np.random.randn(rows, cols), index=date_range('1/1/2000', periods=rows))
        elif dtype == 'object':
            df = DataFrame('foo', index=range(rows), columns=['object%03d' for _ in range(5)])
        else:
            raise NotImplementedError
        self.fname = f'__test_{dtype}__.csv'
        df.to_csv(self.fname)

        @test_parallel(num_threads=2)
        def parallel_read_csv():
            if False:
                i = 10
                return i + 15
            read_csv(self.fname)
        self.parallel_read_csv = parallel_read_csv

    def time_read_csv(self, dtype):
        if False:
            print('Hello World!')
        self.parallel_read_csv()

class ParallelFactorize:
    number = 1
    repeat = 5
    params = [2, 4, 8]
    param_names = ['threads']

    def setup(self, threads):
        if False:
            print('Hello World!')
        strings = tm.makeStringIndex(100000)

        @test_parallel(num_threads=threads)
        def parallel():
            if False:
                print('Hello World!')
            factorize(strings)
        self.parallel = parallel

        def loop():
            if False:
                return 10
            factorize(strings)
        self.loop = loop

    def time_parallel(self, threads):
        if False:
            return 10
        self.parallel()

    def time_loop(self, threads):
        if False:
            for i in range(10):
                print('nop')
        for i in range(threads):
            self.loop()
from .pandas_vb_common import setup