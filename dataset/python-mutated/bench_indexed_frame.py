"""Benchmarks of IndexedFrame methods."""
import pytest
from utils import benchmark_with_object

@benchmark_with_object(cls='indexedframe', dtype='int')
@pytest.mark.parametrize('op', ['cumsum', 'cumprod', 'cummax'])
def bench_scans(benchmark, op, indexedframe):
    if False:
        while True:
            i = 10
    benchmark(getattr(indexedframe, op))

@benchmark_with_object(cls='indexedframe', dtype='int')
@pytest.mark.parametrize('op', ['sum', 'product', 'mean'])
def bench_reductions(benchmark, op, indexedframe):
    if False:
        print('Hello World!')
    benchmark(getattr(indexedframe, op))

@benchmark_with_object(cls='indexedframe', dtype='int')
def bench_drop_duplicates(benchmark, indexedframe):
    if False:
        while True:
            i = 10
    benchmark(indexedframe.drop_duplicates)

@benchmark_with_object(cls='indexedframe', dtype='int')
def bench_rangeindex_replace(benchmark, indexedframe):
    if False:
        i = 10
        return i + 15
    benchmark(indexedframe.replace, 0, 2)