"""
Utilities for fetching the Goodbooks-10K dataset [1]_.

References
----------

.. [1] https://github.com/zygmuntz/goodbooks-10k
"""
import h5py
import numpy as np
from spotlight.datasets import _transport
from spotlight.interactions import Interactions

def _get_dataset():
    if False:
        return 10
    path = _transport.get_data('https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.hdf5', 'goodbooks', 'goodbooks.hdf5')
    with h5py.File(path, 'r') as data:
        return (data['ratings'][:, 0], data['ratings'][:, 1], data['ratings'][:, 2].astype(np.float32), np.arange(len(data['ratings']), dtype=np.int32))

def get_goodbooks_dataset():
    if False:
        for i in range(10):
            print('nop')
    '\n    Download and return the goodbooks-10K dataset [2]_.\n\n    Returns\n    -------\n\n    Interactions: :class:`spotlight.interactions.Interactions`\n        instance of the interactions class\n\n    References\n    ----------\n\n    .. [2] https://github.com/zygmuntz/goodbooks-10k\n    '
    return Interactions(*_get_dataset())