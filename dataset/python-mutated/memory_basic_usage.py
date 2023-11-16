"""
========================
How to use joblib.Memory
========================

This example illustrates the usage of :class:`joblib.Memory` with both
functions and methods.

"""
import time
import numpy as np

def costly_compute(data, column_index=0):
    if False:
        while True:
            i = 10
    'Simulate an expensive computation'
    time.sleep(5)
    return data[column_index]
rng = np.random.RandomState(42)
data = rng.randn(int(100000.0), 10)
start = time.time()
data_trans = costly_compute(data)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

def costly_compute_cached(data, column_index=0):
    if False:
        print('Hello World!')
    'Simulate an expensive computation'
    time.sleep(5)
    return data[column_index]
costly_compute_cached = memory.cache(costly_compute_cached)
start = time.time()
data_trans = costly_compute_cached(data)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
start = time.time()
data_trans = costly_compute_cached(data)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

def _costly_compute_cached(data, column):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(5)
    return data[column]

class Algorithm(object):
    """A class which is using the previous function."""

    def __init__(self, column=0):
        if False:
            print('Hello World!')
        self.column = column

    def transform(self, data):
        if False:
            i = 10
            return i + 15
        costly_compute = memory.cache(_costly_compute_cached)
        return costly_compute(data, self.column)
transformer = Algorithm()
start = time.time()
data_trans = transformer.transform(data)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
start = time.time()
data_trans = transformer.transform(data)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
memory.clear(warn=False)