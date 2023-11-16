"""Benchmarks of Index methods."""
import pytest
from config import cudf, cupy
from utils import benchmark_with_object

@pytest.mark.parametrize('N', [100, 1000000])
def bench_construction(benchmark, N):
    if False:
        while True:
            i = 10
    benchmark(cudf.Index, cupy.random.rand(N))

@benchmark_with_object(cls='index', dtype='int', nulls=False)
def bench_sort_values(benchmark, index):
    if False:
        i = 10
        return i + 15
    benchmark(index.sort_values)