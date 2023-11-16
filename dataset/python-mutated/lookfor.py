import numpy as np
import sys

def lookfor(what):
    if False:
        while True:
            i = 10
    "Do a keyword search on scikit-image docstrings.\n\n    Parameters\n    ----------\n    what : str\n        Words to look for.\n\n    Examples\n    --------\n    >>> import skimage\n    >>> skimage.lookfor('regular_grid')  # doctest: +SKIP\n    Search results for 'regular_grid'\n    ---------------------------------\n    skimage.lookfor\n        Do a keyword search on scikit-image docstrings.\n    skimage.util.regular_grid\n        Find `n_points` regularly spaced along `ar_shape`.\n    "
    return np.lookfor(what, sys.modules[__name__.split('.')[0]])