from .common import Benchmark
import numpy as np

class Records(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        self.l50 = np.arange(1000)
        self.fields_number = 10000
        self.arrays = [self.l50 for _ in range(self.fields_number)]
        self.formats = [self.l50.dtype.str for _ in range(self.fields_number)]
        self.formats_str = ','.join(self.formats)
        self.dtype_ = np.dtype([('field_{}'.format(i), self.l50.dtype.str) for i in range(self.fields_number)])
        self.buffer = self.l50.tostring() * self.fields_number

    def time_fromarrays_w_dtype(self):
        if False:
            return 10
        np._core.records.fromarrays(self.arrays, dtype=self.dtype_)

    def time_fromarrays_wo_dtype(self):
        if False:
            print('Hello World!')
        np._core.records.fromarrays(self.arrays)

    def time_fromarrays_formats_as_list(self):
        if False:
            while True:
                i = 10
        np._core.records.fromarrays(self.arrays, formats=self.formats)

    def time_fromarrays_formats_as_string(self):
        if False:
            while True:
                i = 10
        np._core.records.fromarrays(self.arrays, formats=self.formats_str)

    def time_fromstring_w_dtype(self):
        if False:
            print('Hello World!')
        np._core.records.fromstring(self.buffer, dtype=self.dtype_)

    def time_fromstring_formats_as_list(self):
        if False:
            return 10
        np._core.records.fromstring(self.buffer, formats=self.formats)

    def time_fromstring_formats_as_string(self):
        if False:
            for i in range(10):
                print('nop')
        np._core.records.fromstring(self.buffer, formats=self.formats_str)