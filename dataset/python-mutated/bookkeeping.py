import vaex
import numpy as np
dfs = {}
dfs2 = {}

def time_create(n):
    if False:
        i = 10
        return i + 15
    arr = np.random.random(size=(n, 10))
    df = vaex.from_arrays(**{str(k): arr[k] for k in range(n)})
    return df
time_create.params = [10, 50, 100, 1000]

def setup_copy(n):
    if False:
        return 10
    dfs[n] = time_create(n)

def time_copy(n):
    if False:
        i = 10
        return i + 15
    arr = np.random.random(size=(n, 10))
    dfs[n].copy()
time_copy.setup = setup_copy
time_copy.params = time_create.params

def setup_concat(n):
    if False:
        print('Hello World!')
    dfs[n] = time_create(n)
    dfs2[n] = time_create(n)

def time_concat(n):
    if False:
        i = 10
        return i + 15
    df1 = dfs[n]
    df2 = dfs2[n]
    df1.concat(df2)
time_concat.setup = setup_concat
time_concat.params = time_create.params