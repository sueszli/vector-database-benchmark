import numpy as np
from .._shared.utils import deprecate_kwarg
from .._shared.filters import gaussian

@deprecate_kwarg({'seed': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.5, rng=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate synthetic binary image with several rounded blob-like objects.\n\n    Parameters\n    ----------\n    length : int, optional\n        Linear size of output image.\n    blob_size_fraction : float, optional\n        Typical linear size of blob, as a fraction of ``length``, should be\n        smaller than 1.\n    n_dim : int, optional\n        Number of dimensions of output image.\n    volume_fraction : float, default 0.5\n        Fraction of image pixels covered by the blobs (where the output is 1).\n        Should be in [0, 1].\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n    Returns\n    -------\n    blobs : ndarray of bools\n        Output binary image\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> data.binary_blobs(length=5, blob_size_fraction=0.2)  # doctest: +SKIP\n    array([[ True, False,  True,  True,  True],\n           [ True,  True,  True, False,  True],\n           [False,  True, False,  True,  True],\n           [ True, False, False,  True,  True],\n           [ True, False, False, False,  True]])\n    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)\n    >>> # Finer structures\n    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)\n    >>> # Blobs cover a smaller volume fraction of the image\n    >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)\n\n    '
    rs = np.random.default_rng(rng)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple((indices for indices in points))] = 1
    mask = gaussian(mask, sigma=0.25 * length * blob_size_fraction, preserve_range=False)
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)