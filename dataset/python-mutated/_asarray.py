"""
Functions in the ``as*array`` family that promote array-likes into arrays.

`require` fits this category despite its name not matching this pattern.
"""
from .overrides import array_function_dispatch, set_array_function_like_doc, set_module
from .multiarray import array, asanyarray
__all__ = ['require']
POSSIBLE_FLAGS = {'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C', 'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F', 'A': 'A', 'ALIGNED': 'A', 'W': 'W', 'WRITEABLE': 'W', 'O': 'O', 'OWNDATA': 'O', 'E': 'E', 'ENSUREARRAY': 'E'}

@set_array_function_like_doc
@set_module('numpy')
def require(a, dtype=None, requirements=None, *, like=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return an ndarray of the provided type that satisfies requirements.\n\n    This function is useful to be sure that an array with the correct flags\n    is returned for passing to compiled code (perhaps through ctypes).\n\n    Parameters\n    ----------\n    a : array_like\n       The object to be converted to a type-and-requirement-satisfying array.\n    dtype : data-type\n       The required data-type. If None preserve the current dtype. If your\n       application requires the data to be in native byteorder, include\n       a byteorder specification as a part of the dtype specification.\n    requirements : str or sequence of str\n       The requirements list can be any of the following\n\n       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array\n       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array\n       * 'ALIGNED' ('A')      - ensure a data-type aligned array\n       * 'WRITEABLE' ('W')    - ensure a writable array\n       * 'OWNDATA' ('O')      - ensure an array that owns its own data\n       * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass\n    ${ARRAY_FUNCTION_LIKE}\n\n        .. versionadded:: 1.20.0\n\n    Returns\n    -------\n    out : ndarray\n        Array with specified requirements and type if given.\n\n    See Also\n    --------\n    asarray : Convert input to an ndarray.\n    asanyarray : Convert to an ndarray, but pass through ndarray subclasses.\n    ascontiguousarray : Convert input to a contiguous array.\n    asfortranarray : Convert input to an ndarray with column-major\n                     memory order.\n    ndarray.flags : Information about the memory layout of the array.\n\n    Notes\n    -----\n    The returned array will be guaranteed to have the listed requirements\n    by making a copy if needed.\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2,3)\n    >>> x.flags\n      C_CONTIGUOUS : True\n      F_CONTIGUOUS : False\n      OWNDATA : False\n      WRITEABLE : True\n      ALIGNED : True\n      WRITEBACKIFCOPY : False\n\n    >>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])\n    >>> y.flags\n      C_CONTIGUOUS : False\n      F_CONTIGUOUS : True\n      OWNDATA : True\n      WRITEABLE : True\n      ALIGNED : True\n      WRITEBACKIFCOPY : False\n\n    "
    if like is not None:
        return _require_with_like(like, a, dtype=dtype, requirements=requirements)
    if not requirements:
        return asanyarray(a, dtype=dtype)
    requirements = {POSSIBLE_FLAGS[x.upper()] for x in requirements}
    if 'E' in requirements:
        requirements.remove('E')
        subok = False
    else:
        subok = True
    order = 'A'
    if requirements >= {'C', 'F'}:
        raise ValueError('Cannot specify both "C" and "F" order')
    elif 'F' in requirements:
        order = 'F'
        requirements.remove('F')
    elif 'C' in requirements:
        order = 'C'
        requirements.remove('C')
    arr = array(a, dtype=dtype, order=order, copy=False, subok=subok)
    for prop in requirements:
        if not arr.flags[prop]:
            return arr.copy(order)
    return arr
_require_with_like = array_function_dispatch()(require)