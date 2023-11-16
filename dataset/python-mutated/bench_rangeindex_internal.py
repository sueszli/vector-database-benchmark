"""Benchmarks of internal RangeIndex methods."""

def bench_column(benchmark, rangeindex):
    if False:
        return 10
    benchmark(lambda : rangeindex._column)

def bench_columns(benchmark, rangeindex):
    if False:
        while True:
            i = 10
    benchmark(lambda : rangeindex._columns)