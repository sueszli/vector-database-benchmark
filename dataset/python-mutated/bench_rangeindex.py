import pytest

@pytest.mark.pandas_incompatible
def bench_values_host(benchmark, rangeindex):
    if False:
        while True:
            i = 10
    benchmark(lambda : rangeindex.values_host)

def bench_to_numpy(benchmark, rangeindex):
    if False:
        return 10
    benchmark(rangeindex.to_numpy)

@pytest.mark.pandas_incompatible
def bench_to_arrow(benchmark, rangeindex):
    if False:
        return 10
    benchmark(rangeindex.to_arrow)

def bench_argsort(benchmark, rangeindex):
    if False:
        print('Hello World!')
    benchmark(rangeindex.argsort)

def bench_nunique(benchmark, rangeindex):
    if False:
        print('Hello World!')
    benchmark(rangeindex.nunique)

def bench_isna(benchmark, rangeindex):
    if False:
        for i in range(10):
            print('nop')
    benchmark(rangeindex.isna)

def bench_max(benchmark, rangeindex):
    if False:
        for i in range(10):
            print('nop')
    benchmark(rangeindex.max)

def bench_min(benchmark, rangeindex):
    if False:
        return 10
    benchmark(rangeindex.min)

def bench_where(benchmark, rangeindex):
    if False:
        while True:
            i = 10
    cond = rangeindex % 2 == 0
    benchmark(rangeindex.where, cond, 0)

def bench_isin(benchmark, rangeindex):
    if False:
        for i in range(10):
            print('nop')
    values = [10, 100]
    benchmark(rangeindex.isin, values)