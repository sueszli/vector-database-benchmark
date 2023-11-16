from .common import Benchmark, get_squares, get_squares_
import numpy as np
from io import SEEK_SET, StringIO, BytesIO

class Copy(Benchmark):
    params = ['int8', 'int16', 'float32', 'float64', 'complex64', 'complex128']
    param_names = ['type']

    def setup(self, typename):
        if False:
            return 10
        dtype = np.dtype(typename)
        self.d = np.arange(50 * 500, dtype=dtype).reshape((500, 50))
        self.e = np.arange(50 * 500, dtype=dtype).reshape((50, 500))
        self.e_d = self.e.reshape(self.d.shape)
        self.dflat = np.arange(50 * 500, dtype=dtype)

    def time_memcpy(self, typename):
        if False:
            print('Hello World!')
        self.d[...] = self.e_d

    def time_memcpy_large_out_of_place(self, typename):
        if False:
            while True:
                i = 10
        l = np.ones(1024 ** 2, dtype=np.dtype(typename))
        l.copy()

    def time_cont_assign(self, typename):
        if False:
            for i in range(10):
                print('nop')
        self.d[...] = 1

    def time_strided_copy(self, typename):
        if False:
            print('Hello World!')
        self.d[...] = self.e.T

    def time_strided_assign(self, typename):
        if False:
            return 10
        self.dflat[::2] = 2

class CopyTo(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        self.d = np.ones(50000)
        self.e = self.d.copy()
        self.m = self.d == 1
        self.im = ~self.m
        self.m8 = self.m.copy()
        self.m8[::8] = ~self.m[::8]
        self.im8 = ~self.m8

    def time_copyto(self):
        if False:
            print('Hello World!')
        np.copyto(self.d, self.e)

    def time_copyto_sparse(self):
        if False:
            return 10
        np.copyto(self.d, self.e, where=self.m)

    def time_copyto_dense(self):
        if False:
            return 10
        np.copyto(self.d, self.e, where=self.im)

    def time_copyto_8_sparse(self):
        if False:
            for i in range(10):
                print('nop')
        np.copyto(self.d, self.e, where=self.m8)

    def time_copyto_8_dense(self):
        if False:
            for i in range(10):
                print('nop')
        np.copyto(self.d, self.e, where=self.im8)

class Savez(Benchmark):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.squares = get_squares()

    def time_vb_savez_squares(self):
        if False:
            i = 10
            return i + 15
        np.savez('tmp.npz', **self.squares)

class LoadNpyOverhead(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.buffer = BytesIO()
        np.save(self.buffer, get_squares_()['float32'])

    def time_loadnpy_overhead(self):
        if False:
            print('Hello World!')
        self.buffer.seek(0, SEEK_SET)
        np.load(self.buffer)

class LoadtxtCSVComments(Benchmark):
    params = [10, int(100.0), int(10000.0), int(100000.0)]
    param_names = ['num_lines']

    def setup(self, num_lines):
        if False:
            i = 10
            return i + 15
        data = ['1,2,3 # comment'] * num_lines
        self.data_comments = StringIO('\n'.join(data))

    def time_comment_loadtxt_csv(self, num_lines):
        if False:
            print('Hello World!')
        np.loadtxt(self.data_comments, delimiter=',')
        self.data_comments.seek(0)

class LoadtxtCSVdtypes(Benchmark):
    params = (['float32', 'float64', 'int32', 'int64', 'complex128', 'str', 'object'], [10, int(100.0), int(10000.0), int(100000.0)])
    param_names = ['dtype', 'num_lines']

    def setup(self, dtype, num_lines):
        if False:
            i = 10
            return i + 15
        data = ['5, 7, 888'] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_dtypes_csv(self, dtype, num_lines):
        if False:
            for i in range(10):
                print('nop')
        np.loadtxt(self.csv_data, delimiter=',', dtype=dtype)
        self.csv_data.seek(0)

class LoadtxtCSVStructured(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        num_lines = 50000
        data = ['M, 21, 72, X, 155'] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_csv_struct_dtype(self):
        if False:
            i = 10
            return i + 15
        np.loadtxt(self.csv_data, delimiter=',', dtype=[('category_1', 'S1'), ('category_2', 'i4'), ('category_3', 'f8'), ('category_4', 'S1'), ('category_5', 'f8')])
        self.csv_data.seek(0)

class LoadtxtCSVSkipRows(Benchmark):
    params = [0, 500, 10000]
    param_names = ['skiprows']

    def setup(self, skiprows):
        if False:
            print('Hello World!')
        np.random.seed(123)
        test_array = np.random.rand(100000, 3)
        self.fname = 'test_array.csv'
        np.savetxt(fname=self.fname, X=test_array, delimiter=',')

    def time_skiprows_csv(self, skiprows):
        if False:
            i = 10
            return i + 15
        np.loadtxt(self.fname, delimiter=',', skiprows=skiprows)

class LoadtxtReadUint64Integers(Benchmark):
    params = [550, 1000, 10000]
    param_names = ['size']

    def setup(self, size):
        if False:
            return 10
        arr = np.arange(size).astype('uint64') + 2 ** 63
        self.data1 = StringIO('\n'.join(arr.astype(str).tolist()))
        arr = arr.astype(object)
        arr[500] = -1
        self.data2 = StringIO('\n'.join(arr.astype(str).tolist()))

    def time_read_uint64(self, size):
        if False:
            return 10
        np.loadtxt(self.data1)
        self.data1.seek(0)

    def time_read_uint64_neg_values(self, size):
        if False:
            print('Hello World!')
        np.loadtxt(self.data2)
        self.data2.seek(0)

class LoadtxtUseColsCSV(Benchmark):
    params = [2, [1, 3], [1, 3, 5, 7]]
    param_names = ['usecols']

    def setup(self, usecols):
        if False:
            for i in range(10):
                print('nop')
        num_lines = 5000
        data = ['0, 1, 2, 3, 4, 5, 6, 7, 8, 9'] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_usecols_csv(self, usecols):
        if False:
            i = 10
            return i + 15
        np.loadtxt(self.csv_data, delimiter=',', usecols=usecols)
        self.csv_data.seek(0)

class LoadtxtCSVDateTime(Benchmark):
    params = [20, 200, 2000, 20000]
    param_names = ['num_lines']

    def setup(self, num_lines):
        if False:
            while True:
                i = 10
        dates = np.arange('today', 20, dtype=np.datetime64)
        np.random.seed(123)
        values = np.random.rand(20)
        date_line = ''
        for (date, value) in zip(dates, values):
            date_line += str(date) + ',' + str(value) + '\n'
        data = date_line * (num_lines // 20)
        self.csv_data = StringIO(data)

    def time_loadtxt_csv_datetime(self, num_lines):
        if False:
            return 10
        X = np.loadtxt(self.csv_data, delimiter=',', dtype=[('dates', 'M8[us]'), ('values', 'float64')])
        self.csv_data.seek(0)