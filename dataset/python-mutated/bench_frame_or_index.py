"""Benchmarks of methods that exist for both Frame and BaseIndex."""
import operator
import numpy as np
import pytest
from utils import benchmark_with_object, make_gather_map

@benchmark_with_object(cls='frame_or_index', dtype='int')
@pytest.mark.parametrize('gather_how', ['sequence', 'reverse', 'random'])
@pytest.mark.parametrize('fraction', [0.4])
def bench_take(benchmark, gather_how, fraction, frame_or_index):
    if False:
        while True:
            i = 10
    nr = len(frame_or_index)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index.take, gather_map)

@pytest.mark.pandas_incompatible
@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_argsort(benchmark, frame_or_index):
    if False:
        for i in range(10):
            print('nop')
    benchmark(frame_or_index.argsort)

@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_min(benchmark, frame_or_index):
    if False:
        i = 10
        return i + 15
    benchmark(frame_or_index.min)

@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_where(benchmark, frame_or_index):
    if False:
        i = 10
        return i + 15
    cond = frame_or_index % 2 == 0
    benchmark(frame_or_index.where, cond, 0)

@benchmark_with_object(cls='frame_or_index', dtype='int', nulls=False)
@pytest.mark.pandas_incompatible
def bench_values_host(benchmark, frame_or_index):
    if False:
        for i in range(10):
            print('nop')
    benchmark(lambda : frame_or_index.values_host)

@benchmark_with_object(cls='frame_or_index', dtype='int', nulls=False)
def bench_values(benchmark, frame_or_index):
    if False:
        i = 10
        return i + 15
    benchmark(lambda : frame_or_index.values)

@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_nunique(benchmark, frame_or_index):
    if False:
        i = 10
        return i + 15
    benchmark(frame_or_index.nunique)

@benchmark_with_object(cls='frame_or_index', dtype='int', nulls=False)
def bench_to_numpy(benchmark, frame_or_index):
    if False:
        i = 10
        return i + 15
    benchmark(frame_or_index.to_numpy)

@benchmark_with_object(cls='frame_or_index', dtype='int', nulls=False)
@pytest.mark.pandas_incompatible
def bench_to_cupy(benchmark, frame_or_index):
    if False:
        while True:
            i = 10
    benchmark(frame_or_index.to_cupy)

@benchmark_with_object(cls='frame_or_index', dtype='int')
@pytest.mark.pandas_incompatible
def bench_to_arrow(benchmark, frame_or_index):
    if False:
        while True:
            i = 10
    benchmark(frame_or_index.to_arrow)

@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_astype(benchmark, frame_or_index):
    if False:
        print('Hello World!')
    benchmark(frame_or_index.astype, float)

@pytest.mark.parametrize('ufunc', [np.add, np.logical_and])
@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_ufunc_series_binary(benchmark, frame_or_index, ufunc):
    if False:
        for i in range(10):
            print('nop')
    benchmark(ufunc, frame_or_index, frame_or_index)

@pytest.mark.parametrize('op', [operator.add, operator.mul, operator.eq])
@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_binops(benchmark, op, frame_or_index):
    if False:
        return 10
    benchmark(op, frame_or_index, frame_or_index)

@pytest.mark.parametrize('op', [operator.add, operator.mul, operator.eq])
@benchmark_with_object(cls='frame_or_index', dtype='int')
def bench_scalar_binops(benchmark, op, frame_or_index):
    if False:
        print('Hello World!')
    benchmark(op, frame_or_index, 1)