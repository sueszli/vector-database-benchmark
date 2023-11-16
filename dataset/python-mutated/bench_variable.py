from Orange.data import ContinuousVariable
from .base import Benchmark, benchmark

class BenchContinuous(Benchmark):

    @benchmark()
    def bench_str_val_decimals(self):
        if False:
            i = 10
            return i + 15
        a = ContinuousVariable('a', 4)
        for _ in range(1000):
            a.str_val(1.23456)

    @benchmark()
    def bench_str_val_g(self):
        if False:
            i = 10
            return i + 15
        a = ContinuousVariable('a')
        for _ in range(1000):
            a.str_val(1.23456)