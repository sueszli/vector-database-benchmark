"""Benchmarks of MultiIndex methods."""
import numpy as np
import pandas as pd
import pytest
from config import cudf

@pytest.fixture
def pidx():
    if False:
        print('Hello World!')
    num_elements = int(1000.0)
    a = np.random.randint(0, num_elements // 10, num_elements)
    b = np.random.randint(0, num_elements // 10, num_elements)
    return pd.MultiIndex.from_arrays([a, b], names=('a', 'b'))

@pytest.fixture
def midx(pidx):
    if False:
        while True:
            i = 10
    num_elements = int(1000.0)
    a = np.random.randint(0, num_elements // 10, num_elements)
    b = np.random.randint(0, num_elements // 10, num_elements)
    df = cudf.DataFrame({'a': a, 'b': b})
    return cudf.MultiIndex.from_frame(df)

@pytest.mark.pandas_incompatible
def bench_from_pandas(benchmark, pidx):
    if False:
        return 10
    benchmark(cudf.MultiIndex.from_pandas, pidx)

def bench_constructor(benchmark, midx):
    if False:
        return 10
    benchmark(cudf.MultiIndex, codes=midx.codes, levels=midx.levels, names=midx.names)

def bench_from_frame(benchmark, midx):
    if False:
        i = 10
        return i + 15
    benchmark(cudf.MultiIndex.from_frame, midx.to_frame(index=False))

def bench_copy(benchmark, midx):
    if False:
        while True:
            i = 10
    benchmark(midx.copy, deep=False)