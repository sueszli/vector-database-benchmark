"""Helpers for letting numpy functions interact with distributions.

The module supplies helper routines for numpy functions that propagate
distributions appropriately., for use in the ``__array_function__``
implementation of `~astropy.uncertainty.core.Distribution`.  They are not
very useful on their own, but the ones with docstrings are included in
the documentation so that there is a place to find out how the distributions
are interpreted.

"""
import numpy as np
from astropy.units.quantity_helper.function_helpers import FunctionAssigner
__all__ = ['DISTRIBUTION_SAFE_FUNCTIONS', 'DISPATCHED_FUNCTIONS', 'UNSUPPORTED_FUNCTIONS']
DISTRIBUTION_SAFE_FUNCTIONS = set()
'Set of functions that work fine on Distribution classes already.\n\nMost of these internally use `numpy.ufunc` or other functions that\nare already covered.\n'
DISPATCHED_FUNCTIONS = {}
"Dict of functions that provide the numpy function's functionality.\n\nThese are for more complicated versions where the numpy function itself\ncannot easily be used.  It should return the result of the function.\n\nIt should raise `NotImplementedError` if one of the arguments is a\ndistribution when it should not be or vice versa.\n"
FUNCTION_HELPERS = {}
'Dict of functions for which Distribution can be used after some conversions.\n\nThe `dict` is keyed by the numpy function and the values are functions\nthat take the input arguments of the numpy function and organize these\nfor passing the distribution data to the numpy function, by returning\n``args, kwargs, out``. Here, the former two are passed on, while ``out``\nis used to indicate whether there was an output argument.  If ``out`` is\nset to `True`, then no further processing should be done; otherwise, it\nit is assumed that the function operates on unwrapped distributions and\nthat the results need to be rewrapped as |Distribution|.\n\nThe function should raise `NotImplementedError` if one of the arguments is a\ndistribution when it should not be or vice versa.\n\n'
UNSUPPORTED_FUNCTIONS = set()
'Set of numpy functions that are not supported for distributions.\n\nFor most, distributions simply make no sense, but for others it may have\nbeen lack of time.  Issues or PRs for support for functions are welcome.\n'
function_helper = FunctionAssigner(FUNCTION_HELPERS)
dispatched_function = FunctionAssigner(DISPATCHED_FUNCTIONS)

def is_distribution(x):
    if False:
        while True:
            i = 10
    from astropy.uncertainty import Distribution
    return isinstance(x, Distribution)

def get_n_samples(*arrays):
    if False:
        for i in range(10):
            print('nop')
    'Get n_samples from the first Distribution amount arrays.\n\n    The logic of getting ``n_samples`` from the first |Distribution|\n    is that the code will raise an appropriate exception later if\n    distributions do not have the same ``n_samples``.\n    '
    for array in arrays:
        if is_distribution(array):
            return array.n_samples
    raise RuntimeError('no Distribution found! Please raise an issue.')

@function_helper
def empty_like(prototype, dtype=None, *args, **kwargs):
    if False:
        return 10
    dtype = prototype._get_distribution_dtype(prototype.dtype if dtype is None else dtype, prototype.n_samples)
    return ((prototype, dtype) + args, kwargs, None)

@function_helper
def broadcast_arrays(*args, subok=False):
    if False:
        return 10
    'Broadcast arrays to a common shape.\n\n    Like `numpy.broadcast_arrays`, applied to both distributions and other data.\n    Note that ``subok`` is taken to mean whether or not subclasses of\n    the distribution are allowed, i.e., for ``subok=False``,\n    `~astropy.uncertainty.NdarrayDistribution` instances will be returned.\n    '
    if not subok:
        args = tuple((arg.view(np.ndarray) if isinstance(arg, np.ndarray) else np.array(arg) for arg in args))
    return (args, {'subok': True}, True)

@function_helper
def concatenate(arrays, axis=0, out=None, dtype=None, casting='same_kind'):
    if False:
        return 10
    'Concatenate arrays.\n\n    Like `numpy.concatenate`, but any array that is not already a |Distribution|\n    is turned into one with identical samples.\n    '
    n_samples = get_n_samples(*arrays, out)
    converted = tuple((array.distribution if is_distribution(array) else np.broadcast_to(array[..., np.newaxis], array.shape + (n_samples,), subok=True) if getattr(array, 'shape', False) else array for array in arrays))
    if axis < 0:
        axis = axis - 1
    kwargs = dict(axis=axis, dtype=dtype, casting=casting)
    if out is not None:
        if is_distribution(out):
            kwargs['out'] = out.distribution
        else:
            raise NotImplementedError
    return ((converted,), kwargs, out)
__all__ += sorted((helper.__name__ for helper in set(FUNCTION_HELPERS.values()) | set(DISPATCHED_FUNCTIONS.values()) if helper.__doc__))