"""
Low-level BLAS functions (:mod:`scipy.linalg.blas`)
===================================================

This module contains low-level functions from the BLAS library.

.. versionadded:: 0.12.0

.. note::

   The common ``overwrite_<>`` option in many routines, allows the
   input arrays to be overwritten to avoid extra memory allocation.
   However this requires the array to satisfy two conditions
   which are memory order and the data type to match exactly the
   order and the type expected by the routine.

   As an example, if you pass a double precision float array to any
   ``S....`` routine which expects single precision arguments, f2py
   will create an intermediate array to match the argument types and
   overwriting will be performed on that intermediate array.

   Similarly, if a C-contiguous array is passed, f2py will pass a
   FORTRAN-contiguous array internally. Please make sure that these
   details are satisfied. More information can be found in the f2py
   documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   find_best_blas_type

BLAS Level 1 functions
----------------------

.. autosummary::
   :toctree: generated/

   caxpy
   ccopy
   cdotc
   cdotu
   crotg
   cscal
   csrot
   csscal
   cswap
   dasum
   daxpy
   dcopy
   ddot
   dnrm2
   drot
   drotg
   drotm
   drotmg
   dscal
   dswap
   dzasum
   dznrm2
   icamax
   idamax
   isamax
   izamax
   sasum
   saxpy
   scasum
   scnrm2
   scopy
   sdot
   snrm2
   srot
   srotg
   srotm
   srotmg
   sscal
   sswap
   zaxpy
   zcopy
   zdotc
   zdotu
   zdrot
   zdscal
   zrotg
   zscal
   zswap

BLAS Level 2 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgbmv
   sgemv
   sger
   ssbmv
   sspr
   sspr2
   ssymv
   ssyr
   ssyr2
   stbmv
   stpsv
   strmv
   strsv
   dgbmv
   dgemv
   dger
   dsbmv
   dspr
   dspr2
   dsymv
   dsyr
   dsyr2
   dtbmv
   dtpsv
   dtrmv
   dtrsv
   cgbmv
   cgemv
   cgerc
   cgeru
   chbmv
   chemv
   cher
   cher2
   chpmv
   chpr
   chpr2
   ctbmv
   ctbsv
   ctpmv
   ctpsv
   ctrmv
   ctrsv
   csyr
   zgbmv
   zgemv
   zgerc
   zgeru
   zhbmv
   zhemv
   zher
   zher2
   zhpmv
   zhpr
   zhpr2
   ztbmv
   ztbsv
   ztpmv
   ztrmv
   ztrsv
   zsyr

BLAS Level 3 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgemm
   ssymm
   ssyr2k
   ssyrk
   strmm
   strsm
   dgemm
   dsymm
   dsyr2k
   dsyrk
   dtrmm
   dtrsm
   cgemm
   chemm
   cher2k
   cherk
   csymm
   csyr2k
   csyrk
   ctrmm
   ctrsm
   zgemm
   zhemm
   zher2k
   zherk
   zsymm
   zsyr2k
   zsyrk
   ztrmm
   ztrsm

"""
__all__ = ['get_blas_funcs', 'find_best_blas_type']
import numpy as _np
import functools
from scipy.linalg import _fblas
try:
    from scipy.linalg import _cblas
except ImportError:
    _cblas = None
try:
    from scipy.linalg import _fblas_64
    HAS_ILP64 = True
except ImportError:
    HAS_ILP64 = False
    _fblas_64 = None
empty_module = None
from scipy.linalg._fblas import *
del empty_module
_type_score = {x: 1 for x in '?bBhHef'}
_type_score.update({x: 2 for x in 'iIlLqQd'})
_type_score.update({'F': 3, 'D': 4, 'g': 2, 'G': 4})
_type_conv = {1: ('s', _np.dtype('float32')), 2: ('d', _np.dtype('float64')), 3: ('c', _np.dtype('complex64')), 4: ('z', _np.dtype('complex128'))}
_blas_alias = {'cnrm2': 'scnrm2', 'znrm2': 'dznrm2', 'cdot': 'cdotc', 'zdot': 'zdotc', 'cger': 'cgerc', 'zger': 'zgerc', 'sdotc': 'sdot', 'sdotu': 'sdot', 'ddotc': 'ddot', 'ddotu': 'ddot'}

def find_best_blas_type(arrays=(), dtype=None):
    if False:
        return 10
    "Find best-matching BLAS/LAPACK type.\n\n    Arrays are used to determine the optimal prefix of BLAS routines.\n\n    Parameters\n    ----------\n    arrays : sequence of ndarrays, optional\n        Arrays can be given to determine optimal prefix of BLAS\n        routines. If not given, double-precision routines will be\n        used, otherwise the most generic type in arrays will be used.\n    dtype : str or dtype, optional\n        Data-type specifier. Not used if `arrays` is non-empty.\n\n    Returns\n    -------\n    prefix : str\n        BLAS/LAPACK prefix character.\n    dtype : dtype\n        Inferred Numpy data type.\n    prefer_fortran : bool\n        Whether to prefer Fortran order routines over C order.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import scipy.linalg.blas as bla\n    >>> rng = np.random.default_rng()\n    >>> a = rng.random((10,15))\n    >>> b = np.asfortranarray(a)  # Change the memory layout order\n    >>> bla.find_best_blas_type((a,))\n    ('d', dtype('float64'), False)\n    >>> bla.find_best_blas_type((a*1j,))\n    ('z', dtype('complex128'), False)\n    >>> bla.find_best_blas_type((b,))\n    ('d', dtype('float64'), True)\n\n    "
    dtype = _np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False
    if arrays:
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            prefer_fortran = arrays[0].flags['FORTRAN']
        else:
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            if max_score == 3 and 2 in scores:
                max_score = 4
            if arrays[ind_max_score].flags['FORTRAN']:
                prefer_fortran = True
    (prefix, dtype) = _type_conv.get(max_score, ('d', _np.dtype('float64')))
    return (prefix, dtype, prefer_fortran)

def _get_funcs(names, arrays, dtype, lib_name, fmodule, cmodule, fmodule_name, cmodule_name, alias, ilp64=False):
    if False:
        i = 10
        return i + 15
    '\n    Return available BLAS/LAPACK functions.\n\n    Used also in lapack.py. See get_blas_funcs for docstring.\n    '
    funcs = []
    unpack = False
    dtype = _np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)
    if isinstance(names, str):
        names = (names,)
        unpack = True
    (prefix, dtype, prefer_fortran) = find_best_blas_type(arrays, dtype)
    if prefer_fortran:
        (module1, module2) = (module2, module1)
    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(f'{lib_name} function {func_name} could not be found')
        (func.module_name, func.typecode) = (module_name, prefix)
        func.dtype = dtype
        if not ilp64:
            func.int_dtype = _np.dtype(_np.intc)
        else:
            func.int_dtype = _np.dtype(_np.int64)
        func.prefix = prefix
        funcs.append(func)
    if unpack:
        return funcs[0]
    else:
        return funcs

def _memoize_get_funcs(func):
    if False:
        print('Hello World!')
    '\n    Memoized fast path for _get_funcs instances\n    '
    memo = {}
    func.memo = memo

    @functools.wraps(func)
    def getter(names, arrays=(), dtype=None, ilp64=False):
        if False:
            for i in range(10):
                print('nop')
        key = (names, dtype, ilp64)
        for array in arrays:
            key += (array.dtype.char, array.flags.fortran)
        try:
            value = memo.get(key)
        except TypeError:
            key = None
            value = None
        if value is not None:
            return value
        value = func(names, arrays, dtype, ilp64)
        if key is not None:
            memo[key] = value
        return value
    return getter

@_memoize_get_funcs
def get_blas_funcs(names, arrays=(), dtype=None, ilp64=False):
    if False:
        return 10
    "Return available BLAS function objects from names.\n\n    Arrays are used to determine the optimal prefix of BLAS routines.\n\n    Parameters\n    ----------\n    names : str or sequence of str\n        Name(s) of BLAS functions without type prefix.\n\n    arrays : sequence of ndarrays, optional\n        Arrays can be given to determine optimal prefix of BLAS\n        routines. If not given, double-precision routines will be\n        used, otherwise the most generic type in arrays will be used.\n\n    dtype : str or dtype, optional\n        Data-type specifier. Not used if `arrays` is non-empty.\n\n    ilp64 : {True, False, 'preferred'}, optional\n        Whether to return ILP64 routine variant.\n        Choosing 'preferred' returns ILP64 routine if available,\n        and otherwise the 32-bit routine. Default: False\n\n    Returns\n    -------\n    funcs : list\n        List containing the found function(s).\n\n\n    Notes\n    -----\n    This routine automatically chooses between Fortran/C\n    interfaces. Fortran code is used whenever possible for arrays with\n    column major order. In all other cases, C code is preferred.\n\n    In BLAS, the naming convention is that all functions start with a\n    type prefix, which depends on the type of the principal\n    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy\n    types {float32, float64, complex64, complex128} respectively.\n    The code and the dtype are stored in attributes `typecode` and `dtype`\n    of the returned functions.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import scipy.linalg as LA\n    >>> rng = np.random.default_rng()\n    >>> a = rng.random((3,2))\n    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))\n    >>> x_gemv.typecode\n    'd'\n    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))\n    >>> x_gemv.typecode\n    'z'\n\n    "
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")
    if not ilp64:
        return _get_funcs(names, arrays, dtype, 'BLAS', _fblas, _cblas, 'fblas', 'cblas', _blas_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError('BLAS ILP64 routine requested, but Scipy compiled only with 32-bit BLAS')
        return _get_funcs(names, arrays, dtype, 'BLAS', _fblas_64, None, 'fblas_64', None, _blas_alias, ilp64=True)