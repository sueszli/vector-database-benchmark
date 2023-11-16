import signal
import sys
from numba import njit
import numpy as np

def sigterm_handler(signum, frame):
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('Caught SIGTERM')

@njit(parallel=True)
def busy_func_inner(a, b):
    if False:
        while True:
            i = 10
    c = a + b * np.sqrt(a) + np.sqrt(b)
    d = np.sqrt(a + b * np.sqrt(a) + np.sqrt(b))
    return c + d

def busy_func(a, b, q=None):
    if False:
        i = 10
        return i + 15
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        z = busy_func_inner(a, b)
        sys.stdout.flush()
        sys.stderr.flush()
        return z
    except Exception as e:
        if q is not None:
            q.put(e)