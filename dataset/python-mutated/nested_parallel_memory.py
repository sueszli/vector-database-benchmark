"""
==================================================
Checkpoint using joblib.Memory and joblib.Parallel
==================================================

This example illustrates how to cache intermediate computing results using
:class:`joblib.Memory` within :class:`joblib.Parallel`.

"""
import time

def costly_compute(data, column):
    if False:
        i = 10
        return i + 15
    'Emulate a costly function by sleeping and returning a column.'
    time.sleep(2)
    return data[column]

def data_processing_mean(data, column):
    if False:
        return 10
    'Compute the mean of a column.'
    return costly_compute(data, column).mean()
import numpy as np
rng = np.random.RandomState(42)
data = rng.randn(int(10000.0), 4)
start = time.time()
results = [data_processing_mean(data, col) for col in range(data.shape[1])]
stop = time.time()
print('\nSequential processing')
print('Elapsed time for the entire processing: {:.2f} s'.format(stop - start))
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)
costly_compute_cached = memory.cache(costly_compute)

def data_processing_mean_using_cache(data, column):
    if False:
        i = 10
        return i + 15
    'Compute the mean of a column.'
    return costly_compute_cached(data, column).mean()
from joblib import Parallel, delayed
start = time.time()
results = Parallel(n_jobs=2)((delayed(data_processing_mean_using_cache)(data, col) for col in range(data.shape[1])))
stop = time.time()
print('\nFirst round - caching the data')
print('Elapsed time for the entire processing: {:.2f} s'.format(stop - start))
start = time.time()
results = Parallel(n_jobs=2)((delayed(data_processing_mean_using_cache)(data, col) for col in range(data.shape[1])))
stop = time.time()
print('\nSecond round - reloading from the cache')
print('Elapsed time for the entire processing: {:.2f} s'.format(stop - start))

def data_processing_max_using_cache(data, column):
    if False:
        i = 10
        return i + 15
    'Compute the max of a column.'
    return costly_compute_cached(data, column).max()
start = time.time()
results = Parallel(n_jobs=2)((delayed(data_processing_max_using_cache)(data, col) for col in range(data.shape[1])))
stop = time.time()
print('\nReusing intermediate checkpoints')
print('Elapsed time for the entire processing: {:.2f} s'.format(stop - start))
memory.clear(warn=False)