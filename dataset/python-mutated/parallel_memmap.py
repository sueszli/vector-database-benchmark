"""
===============================
NumPy memmap in joblib.Parallel
===============================

This example illustrates some features enabled by using a memory map
(:class:`numpy.memmap`) within :class:`joblib.Parallel`. First, we show that
dumping a huge data array ahead of passing it to :class:`joblib.Parallel`
speeds up computation. Then, we show the possibility to provide write access to
original data.

"""
import numpy as np
data = np.random.random((int(10000000.0),))
window_size = int(500000.0)
slices = [slice(start, start + window_size) for start in range(0, data.size - window_size, int(100000.0))]
import time

def slow_mean(data, sl):
    if False:
        return 10
    'Simulate a time consuming processing.'
    time.sleep(0.01)
    return data[sl].mean()
tic = time.time()
results = [slow_mean(data, sl) for sl in slices]
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))
from joblib import Parallel, delayed
tic = time.time()
results = Parallel(n_jobs=2)((delayed(slow_mean)(data, sl) for sl in slices))
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))
import os
from joblib import dump, load
folder = './joblib_memmap'
try:
    os.mkdir(folder)
except FileExistsError:
    pass
data_filename_memmap = os.path.join(folder, 'data_memmap')
dump(data, data_filename_memmap)
data = load(data_filename_memmap, mmap_mode='r')
tic = time.time()
results = Parallel(n_jobs=2)((delayed(slow_mean)(data, sl) for sl in slices))
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s\n'.format(toc - tic))

def slow_mean_write_output(data, sl, output, idx):
    if False:
        while True:
            i = 10
    'Simulate a time consuming processing.'
    time.sleep(0.005)
    res_ = data[sl].mean()
    print('[Worker %d] Mean for slice %d is %f' % (os.getpid(), idx, res_))
    output[idx] = res_
output_filename_memmap = os.path.join(folder, 'output_memmap')
output = np.memmap(output_filename_memmap, dtype=data.dtype, shape=len(slices), mode='w+')
data = load(data_filename_memmap, mmap_mode='r')
Parallel(n_jobs=2)((delayed(slow_mean_write_output)(data, sl, output, idx) for (idx, sl) in enumerate(slices)))
print('\nExpected means computed in the parent process:\n {}'.format(np.array(results)))
print('\nActual means computed by the worker processes:\n {}'.format(output))
import shutil
try:
    shutil.rmtree(folder)
except:
    print('Could not clean-up automatically.')