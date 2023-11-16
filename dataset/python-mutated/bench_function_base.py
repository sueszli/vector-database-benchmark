from .common import Benchmark
import numpy as np
try:
    from asv_runner.benchmarks.mark import SkipNotImplemented
except ImportError:
    SkipNotImplemented = NotImplementedError

class Linspace(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.d = np.array([1, 2, 3])

    def time_linspace_scalar(self):
        if False:
            return 10
        np.linspace(0, 10, 2)

    def time_linspace_array(self):
        if False:
            for i in range(10):
                print('nop')
        np.linspace(self.d, 10, 10)

class Histogram1D(Benchmark):

    def setup(self):
        if False:
            return 10
        self.d = np.linspace(0, 100, 100000)

    def time_full_coverage(self):
        if False:
            for i in range(10):
                print('nop')
        np.histogram(self.d, 200, (0, 100))

    def time_small_coverage(self):
        if False:
            return 10
        np.histogram(self.d, 200, (50, 51))

    def time_fine_binning(self):
        if False:
            while True:
                i = 10
        np.histogram(self.d, 10000, (0, 100))

class Histogram2D(Benchmark):

    def setup(self):
        if False:
            return 10
        self.d = np.linspace(0, 100, 200000).reshape((-1, 2))

    def time_full_coverage(self):
        if False:
            print('Hello World!')
        np.histogramdd(self.d, (200, 200), ((0, 100), (0, 100)))

    def time_small_coverage(self):
        if False:
            for i in range(10):
                print('nop')
        np.histogramdd(self.d, (200, 200), ((50, 51), (50, 51)))

    def time_fine_binning(self):
        if False:
            for i in range(10):
                print('nop')
        np.histogramdd(self.d, (10000, 10000), ((0, 100), (0, 100)))

class Bincount(Benchmark):

    def setup(self):
        if False:
            return 10
        self.d = np.arange(80000, dtype=np.intp)
        self.e = self.d.astype(np.float64)

    def time_bincount(self):
        if False:
            print('Hello World!')
        np.bincount(self.d)

    def time_weights(self):
        if False:
            print('Hello World!')
        np.bincount(self.d, weights=self.e)

class Mean(Benchmark):
    param_names = ['size']
    params = [[1, 10, 100000]]

    def setup(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.array = np.arange(2 * size).reshape(2, size)

    def time_mean(self, size):
        if False:
            print('Hello World!')
        np.mean(self.array)

    def time_mean_axis(self, size):
        if False:
            return 10
        np.mean(self.array, axis=1)

class Median(Benchmark):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(10001, dtype=np.float32)
        self.tall = np.random.random((10000, 20))
        self.wide = np.random.random((20, 10000))

    def time_even(self):
        if False:
            return 10
        np.median(self.e)

    def time_odd(self):
        if False:
            print('Hello World!')
        np.median(self.o)

    def time_even_inplace(self):
        if False:
            i = 10
            return i + 15
        np.median(self.e, overwrite_input=True)

    def time_odd_inplace(self):
        if False:
            return 10
        np.median(self.o, overwrite_input=True)

    def time_even_small(self):
        if False:
            i = 10
            return i + 15
        np.median(self.e[:500], overwrite_input=True)

    def time_odd_small(self):
        if False:
            print('Hello World!')
        np.median(self.o[:500], overwrite_input=True)

    def time_tall(self):
        if False:
            print('Hello World!')
        np.median(self.tall, axis=-1)

    def time_wide(self):
        if False:
            i = 10
            return i + 15
        np.median(self.wide, axis=0)

class Percentile(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(21, dtype=np.float32)

    def time_quartile(self):
        if False:
            print('Hello World!')
        np.percentile(self.e, [25, 75])

    def time_percentile(self):
        if False:
            while True:
                i = 10
        np.percentile(self.e, [25, 35, 55, 65, 75])

    def time_percentile_small(self):
        if False:
            print('Hello World!')
        np.percentile(self.o, [25, 75])

class Select(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.d = np.arange(20000)
        self.e = self.d.copy()
        self.cond = [self.d > 4, self.d < 2]
        self.cond_large = [self.d > 4, self.d < 2] * 10

    def time_select(self):
        if False:
            for i in range(10):
                print('nop')
        np.select(self.cond, [self.d, self.e])

    def time_select_larger(self):
        if False:
            i = 10
            return i + 15
        np.select(self.cond_large, [self.d, self.e] * 10)

def memoize(f):
    if False:
        i = 10
        return i + 15
    _memoized = {}

    def wrapped(*args):
        if False:
            for i in range(10):
                print('nop')
        if args not in _memoized:
            _memoized[args] = f(*args)
        return _memoized[args].copy()
    return f

class SortGenerator:
    AREA_SIZE = 100
    BUBBLE_SIZE = 100

    @staticmethod
    @memoize
    def random(size, dtype):
        if False:
            while True:
                i = 10
        '\n        Returns a randomly-shuffled array.\n        '
        arr = np.arange(size, dtype=dtype)
        np.random.shuffle(arr)
        return arr

    @staticmethod
    @memoize
    def ordered(size, dtype):
        if False:
            return 10
        '\n        Returns an ordered array.\n        '
        return np.arange(size, dtype=dtype)

    @staticmethod
    @memoize
    def reversed(size, dtype):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns an array that's in descending order.\n        "
        dtype = np.dtype(dtype)
        try:
            with np.errstate(over='raise'):
                res = dtype.type(size - 1)
        except (OverflowError, FloatingPointError):
            raise SkipNotImplemented('Cannot construct arange for this size.')
        return np.arange(size - 1, -1, -1, dtype=dtype)

    @staticmethod
    @memoize
    def uniform(size, dtype):
        if False:
            return 10
        '\n        Returns an array that has the same value everywhere.\n        '
        return np.ones(size, dtype=dtype)

    @staticmethod
    @memoize
    def swapped_pair(size, dtype, swap_frac):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an ordered array, but one that has ``swap_frac * size``\n        pairs swapped.\n        '
        a = np.arange(size, dtype=dtype)
        for _ in range(int(size * swap_frac)):
            (x, y) = np.random.randint(0, size, 2)
            (a[x], a[y]) = (a[y], a[x])
        return a

    @staticmethod
    @memoize
    def sorted_block(size, dtype, block_size):
        if False:
            return 10
        '\n        Returns an array with blocks that are all sorted.\n        '
        a = np.arange(size, dtype=dtype)
        b = []
        if size < block_size:
            return a
        block_num = size // block_size
        for i in range(block_num):
            b.extend(a[i::block_num])
        return np.array(b)

    @classmethod
    @memoize
    def random_unsorted_area(cls, size, dtype, frac, area_size=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This type of array has random unsorted areas such that they\n        compose the fraction ``frac`` of the original array.\n        '
        if area_size is None:
            area_size = cls.AREA_SIZE
        area_num = int(size * frac / area_size)
        a = np.arange(size, dtype=dtype)
        for _ in range(area_num):
            start = np.random.randint(size - area_size)
            end = start + area_size
            np.random.shuffle(a[start:end])
        return a

    @classmethod
    @memoize
    def random_bubble(cls, size, dtype, bubble_num, bubble_size=None):
        if False:
            return 10
        '\n        This type of array has ``bubble_num`` random unsorted areas.\n        '
        if bubble_size is None:
            bubble_size = cls.BUBBLE_SIZE
        frac = bubble_size * bubble_num / size
        return cls.random_unsorted_area(size, dtype, frac, bubble_size)

class Sort(Benchmark):
    """
    This benchmark tests sorting performance with several
    different types of arrays that are likely to appear in
    real-world applications.
    """
    params = [['quick', 'merge', 'heap'], ['float64', 'int64', 'float32', 'uint32', 'int32', 'int16', 'float16'], [('random',), ('ordered',), ('reversed',), ('uniform',), ('sorted_block', 10), ('sorted_block', 100), ('sorted_block', 1000)]]
    param_names = ['kind', 'dtype', 'array_type']
    ARRAY_SIZE = 10000

    def setup(self, kind, dtype, array_type):
        if False:
            return 10
        np.random.seed(1234)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(self.ARRAY_SIZE, dtype, *array_type[1:])

    def time_sort(self, kind, dtype, array_type):
        if False:
            for i in range(10):
                print('nop')
        np.sort(self.arr, kind=kind)

    def time_argsort(self, kind, dtype, array_type):
        if False:
            return 10
        np.argsort(self.arr, kind=kind)

class Partition(Benchmark):
    params = [['float64', 'int64', 'float32', 'int32', 'int16', 'float16'], [('random',), ('ordered',), ('reversed',), ('uniform',), ('sorted_block', 10), ('sorted_block', 100), ('sorted_block', 1000)], [10, 100, 1000]]
    param_names = ['dtype', 'array_type', 'k']
    ARRAY_SIZE = 100000

    def setup(self, dtype, array_type, k):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(self.ARRAY_SIZE, dtype, *array_type[1:])

    def time_partition(self, dtype, array_type, k):
        if False:
            for i in range(10):
                print('nop')
        temp = np.partition(self.arr, k)

    def time_argpartition(self, dtype, array_type, k):
        if False:
            i = 10
            return i + 15
        temp = np.argpartition(self.arr, k)

class SortWorst(Benchmark):

    def setup(self):
        if False:
            return 10
        self.worst = np.arange(1000000)
        x = self.worst
        while x.size > 3:
            mid = x.size // 2
            (x[mid], x[-2]) = (x[-2], x[mid])
            x = x[:-2]

    def time_sort_worst(self):
        if False:
            return 10
        np.sort(self.worst)
    time_sort_worst.benchmark_name = 'bench_function_base.Sort.time_sort_worst'

class Where(Benchmark):

    def setup(self):
        if False:
            return 10
        self.d = np.arange(20000)
        self.d_o = self.d.astype(object)
        self.e = self.d.copy()
        self.e_o = self.d_o.copy()
        self.cond = self.d > 5000
        size = 1024 * 1024 // 8
        rnd_array = np.random.rand(size)
        self.rand_cond_01 = rnd_array > 0.01
        self.rand_cond_20 = rnd_array > 0.2
        self.rand_cond_30 = rnd_array > 0.3
        self.rand_cond_40 = rnd_array > 0.4
        self.rand_cond_50 = rnd_array > 0.5
        self.all_zeros = np.zeros(size, dtype=bool)
        self.all_ones = np.ones(size, dtype=bool)
        self.rep_zeros_2 = np.arange(size) % 2 == 0
        self.rep_zeros_4 = np.arange(size) % 4 == 0
        self.rep_zeros_8 = np.arange(size) % 8 == 0
        self.rep_ones_2 = np.arange(size) % 2 > 0
        self.rep_ones_4 = np.arange(size) % 4 > 0
        self.rep_ones_8 = np.arange(size) % 8 > 0

    def time_1(self):
        if False:
            for i in range(10):
                print('nop')
        np.where(self.cond)

    def time_2(self):
        if False:
            print('Hello World!')
        np.where(self.cond, self.d, self.e)

    def time_2_object(self):
        if False:
            while True:
                i = 10
        np.where(self.cond, self.d_o, self.e_o)

    def time_2_broadcast(self):
        if False:
            i = 10
            return i + 15
        np.where(self.cond, self.d, 0)

    def time_all_zeros(self):
        if False:
            while True:
                i = 10
        np.where(self.all_zeros)

    def time_random_01_percent(self):
        if False:
            return 10
        np.where(self.rand_cond_01)

    def time_random_20_percent(self):
        if False:
            while True:
                i = 10
        np.where(self.rand_cond_20)

    def time_random_30_percent(self):
        if False:
            for i in range(10):
                print('nop')
        np.where(self.rand_cond_30)

    def time_random_40_percent(self):
        if False:
            print('Hello World!')
        np.where(self.rand_cond_40)

    def time_random_50_percent(self):
        if False:
            for i in range(10):
                print('nop')
        np.where(self.rand_cond_50)

    def time_all_ones(self):
        if False:
            print('Hello World!')
        np.where(self.all_ones)

    def time_interleaved_zeros_x2(self):
        if False:
            while True:
                i = 10
        np.where(self.rep_zeros_2)

    def time_interleaved_zeros_x4(self):
        if False:
            i = 10
            return i + 15
        np.where(self.rep_zeros_4)

    def time_interleaved_zeros_x8(self):
        if False:
            while True:
                i = 10
        np.where(self.rep_zeros_8)

    def time_interleaved_ones_x2(self):
        if False:
            return 10
        np.where(self.rep_ones_2)

    def time_interleaved_ones_x4(self):
        if False:
            for i in range(10):
                print('nop')
        np.where(self.rep_ones_4)

    def time_interleaved_ones_x8(self):
        if False:
            while True:
                i = 10
        np.where(self.rep_ones_8)