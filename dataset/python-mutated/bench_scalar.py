from .common import Benchmark, TYPES1
import numpy as np

class ScalarMath(Benchmark):
    params = [TYPES1]
    param_names = ['type']

    def setup(self, typename):
        if False:
            print('Hello World!')
        self.num = np.dtype(typename).type(2)
        self.int32 = np.int32(2)
        self.int32arr = np.array(2, dtype=np.int32)

    def time_addition(self, typename):
        if False:
            print('Hello World!')
        n = self.num
        res = n + n + n + n + n + n + n + n + n + n

    def time_addition_pyint(self, typename):
        if False:
            i = 10
            return i + 15
        n = self.num
        res = n + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

    def time_multiplication(self, typename):
        if False:
            return 10
        n = self.num
        res = n * n * n * n * n * n * n * n * n * n

    def time_power_of_two(self, typename):
        if False:
            for i in range(10):
                print('nop')
        n = self.num
        res = (n ** 2, n ** 2, n ** 2, n ** 2, n ** 2, n ** 2, n ** 2, n ** 2, n ** 2, n ** 2)

    def time_abs(self, typename):
        if False:
            return 10
        n = self.num
        res = abs(abs(abs(abs(abs(abs(abs(abs(abs(abs(n))))))))))

    def time_add_int32_other(self, typename):
        if False:
            i = 10
            return i + 15
        int32 = self.int32
        other = self.num
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    def time_add_int32arr_and_other(self, typename):
        if False:
            while True:
                i = 10
        int32 = self.int32arr
        other = self.num
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    def time_add_other_and_int32arr(self, typename):
        if False:
            i = 10
            return i + 15
        int32 = self.int32arr
        other = self.num
        other + int32
        other + int32
        other + int32
        other + int32
        other + int32

class ScalarStr(Benchmark):
    params = [TYPES1]
    param_names = ['type']

    def setup(self, typename):
        if False:
            print('Hello World!')
        self.a = np.array([100] * 100, dtype=typename)

    def time_str_repr(self, typename):
        if False:
            while True:
                i = 10
        res = [str(x) for x in self.a]