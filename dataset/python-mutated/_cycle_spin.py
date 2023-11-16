from itertools import product
import numpy as np
from .._shared import utils
from .._shared.utils import warn
try:
    import dask
    dask_available = True
except ImportError:
    dask_available = False

def _generate_shifts(ndim, multichannel, max_shifts, shift_steps=1):
    if False:
        i = 10
        return i + 15
    'Returns all combinations of shifts in n dimensions over the specified\n    max_shifts and step sizes.\n\n    Examples\n    --------\n    >>> s = list(_generate_shifts(2, False, max_shifts=(1, 2), shift_steps=1))\n    >>> print(s)\n    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]\n    '
    mc = int(multichannel)
    if np.isscalar(max_shifts):
        max_shifts = (max_shifts,) * (ndim - mc) + (0,) * mc
    elif multichannel and len(max_shifts) == ndim - 1:
        max_shifts = tuple(max_shifts) + (0,)
    elif len(max_shifts) != ndim:
        raise ValueError('max_shifts should have length ndim')
    if np.isscalar(shift_steps):
        shift_steps = (shift_steps,) * (ndim - mc) + (1,) * mc
    elif multichannel and len(shift_steps) == ndim - 1:
        shift_steps = tuple(shift_steps) + (1,)
    elif len(shift_steps) != ndim:
        raise ValueError('max_shifts should have length ndim')
    if any((s < 1 for s in shift_steps)):
        raise ValueError('shift_steps must all be >= 1')
    if multichannel and max_shifts[-1] != 0:
        raise ValueError('Multichannel cycle spinning should not have shifts along the last axis.')
    return product(*[range(0, s + 1, t) for (s, t) in zip(max_shifts, shift_steps)])

@utils.channel_as_last_axis()
def cycle_spin(x, func, max_shifts, shift_steps=1, num_workers=None, func_kw=None, *, channel_axis=None):
    if False:
        for i in range(10):
            print('nop')
    'Cycle spinning (repeatedly apply func to shifted versions of x).\n\n    Parameters\n    ----------\n    x : array-like\n        Data for input to ``func``.\n    func : function\n        A function to apply to circularly shifted versions of ``x``.  Should\n        take ``x`` as its first argument. Any additional arguments can be\n        supplied via ``func_kw``.\n    max_shifts : int or tuple\n        If an integer, shifts in ``range(0, max_shifts+1)`` will be used along\n        each axis of ``x``. If a tuple, ``range(0, max_shifts[i]+1)`` will be\n        along axis i.\n    shift_steps : int or tuple, optional\n        The step size for the shifts applied along axis, i, are::\n        ``range((0, max_shifts[i]+1, shift_steps[i]))``. If an integer is\n        provided, the same step size is used for all axes.\n    num_workers : int or None, optional\n        The number of parallel threads to use during cycle spinning. If set to\n        ``None``, the full set of available cores are used.\n    func_kw : dict, optional\n        Additional keyword arguments to supply to ``func``.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    avg_y : np.ndarray\n        The output of ``func(x, **func_kw)`` averaged over all combinations of\n        the specified axis shifts.\n\n    Notes\n    -----\n    Cycle spinning was proposed as a way to approach shift-invariance via\n    performing several circular shifts of a shift-variant transform [1]_.\n\n    For a n-level discrete wavelet transforms, one may wish to perform all\n    shifts up to ``max_shifts = 2**n - 1``. In practice, much of the benefit\n    can often be realized with only a small number of shifts per axis.\n\n    For transforms such as the blockwise discrete cosine transform, one may\n    wish to evaluate shifts up to the block size used by the transform.\n\n    References\n    ----------\n    .. [1] R.R. Coifman and D.L. Donoho.  "Translation-Invariant De-Noising".\n           Wavelets and Statistics, Lecture Notes in Statistics, vol.103.\n           Springer, New York, 1995, pp.125-150.\n           :DOI:`10.1007/978-1-4612-2544-7_9`\n\n    Examples\n    --------\n    >>> import skimage.data\n    >>> from skimage import img_as_float\n    >>> from skimage.restoration import denoise_tv_chambolle, cycle_spin\n    >>> img = img_as_float(skimage.data.camera())\n    >>> sigma = 0.1\n    >>> img = img + sigma * np.random.standard_normal(img.shape)\n    >>> denoised = cycle_spin(img, func=denoise_tv_chambolle,\n    ...                       max_shifts=3)\n\n    '
    if func_kw is None:
        func_kw = {}
    x = np.asanyarray(x)
    multichannel = channel_axis is not None
    all_shifts = _generate_shifts(x.ndim, multichannel, max_shifts, shift_steps)
    all_shifts = list(all_shifts)
    roll_axes = tuple(range(x.ndim))

    def _run_one_shift(shift):
        if False:
            while True:
                i = 10
        xs = np.roll(x, shift, axis=roll_axes)
        tmp = func(xs, **func_kw)
        return np.roll(tmp, tuple((-s for s in shift)), axis=roll_axes)
    if not dask_available and (num_workers is None or num_workers > 1):
        num_workers = 1
        warn('The optional dask dependency is not installed. The number of workers is set to 1. To silence this warning, install dask or explicitly set `num_workers=1` when calling the `cycle_spin` function')
    if num_workers == 1:
        mean = _run_one_shift(all_shifts[0])
        for shift in all_shifts[1:]:
            mean += _run_one_shift(shift)
        mean /= len(all_shifts)
    else:
        futures = [dask.delayed(_run_one_shift)(s) for s in all_shifts]
        mean = sum(futures) / len(futures)
        mean = mean.compute(num_workers=num_workers)
    return mean