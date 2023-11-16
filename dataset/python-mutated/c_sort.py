"""
list.sort is interesting in that it calls a C function, that calls back to a
Python function. In an ideal world, we'd be able to record the time inside the
Python function _inside_ list.sort, but it's not possible currently, due to
the way that Python records frame objects.

Perhaps one day we could add some functionality to pyinstrument_cext to keep
a parallel stack containing both C and Python frames. But for now, this is
fine.
"""
import sys
import time
import numpy as np
arr = np.random.randint(0, 10, 10)

def slow_key(el):
    if False:
        return 10
    time.sleep(0.01)
    return 0
for i in range(10):
    list(arr).sort(key=slow_key)