"""
Serialization of un-picklable objects
=====================================

This example highlights the options for tempering with joblib serialization
process.

"""
import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_config
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

def func_async(i, *args):
    if False:
        i = 10
        return i + 15
    return 2 * i
print(Parallel(n_jobs=2)((delayed(func_async)(21) for _ in range(1)))[0])

def func_async(i, *args):
    if False:
        return 10
    return 2 * i
large_list = list(range(1000000))
t_start = time.time()
Parallel(n_jobs=2)((delayed(func_async)(21, large_list) for _ in range(1)))
print('With loky backend and cloudpickle serialization: {:.3f}s'.format(time.time() - t_start))
import multiprocessing as mp
if mp.get_start_method() != 'spawn':

    def func_async(i, *args):
        if False:
            print('Hello World!')
        return 2 * i
    with parallel_config('multiprocessing'):
        t_start = time.time()
        Parallel(n_jobs=2)((delayed(func_async)(21, large_list) for _ in range(1)))
        print('With multiprocessing backend and pickle serialization: {:.3f}s'.format(time.time() - t_start))
set_loky_pickler('pickle')
t_start = time.time()
Parallel(n_jobs=2)((delayed(id)(large_list) for _ in range(1)))
print('With pickle serialization: {:.3f}s'.format(time.time() - t_start))

def func_async(i, *args):
    if False:
        return 10
    return 2 * i
try:
    Parallel(n_jobs=2)((delayed(func_async)(21, large_list) for _ in range(1)))
except Exception:
    traceback.print_exc(file=sys.stdout)

@delayed
@wrap_non_picklable_objects
def func_async_wrapped(i, *args):
    if False:
        i = 10
        return i + 15
    return 2 * i
t_start = time.time()
Parallel(n_jobs=2)((func_async_wrapped(21, large_list) for _ in range(1)))
print('With pickle from stdlib and wrapper: {:.3f}s'.format(time.time() - t_start))
set_loky_pickler()