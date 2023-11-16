import time
import numpy as np
from astropy.utils.misc import NumpyRNGContext

def func(i):
    if False:
        for i in range(10):
            print('nop')
    "An identity function that jitters its execution time by a\n    pseudo-random amount.\n\n    FIXME: This function should be defined in test_console.py, but Astropy's\n    test runner interacts strangely with Python's `multiprocessing`\n    module. I was getting a mysterious PicklingError until I moved this\n    function into a separate module. (It worked fine in a standalone pytest\n    script.)"
    with NumpyRNGContext(i):
        time.sleep(np.random.uniform(0, 0.01))
    return i