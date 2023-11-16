"""Common utilities for Numba operations"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
GLOBAL_USE_NUMBA: bool = False

def maybe_use_numba(engine: str | None) -> bool:
    if False:
        print('Hello World!')
    'Signal whether to use numba routines.'
    return engine == 'numba' or (engine is None and GLOBAL_USE_NUMBA)

def set_use_numba(enable: bool=False) -> None:
    if False:
        print('Hello World!')
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency('numba')
    GLOBAL_USE_NUMBA = enable

def get_jit_arguments(engine_kwargs: dict[str, bool] | None=None, kwargs: dict | None=None) -> dict[str, bool]:
    if False:
        print('Hello World!')
    '\n    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.\n\n    Parameters\n    ----------\n    engine_kwargs : dict, default None\n        user passed keyword arguments for numba.JIT\n    kwargs : dict, default None\n        user passed keyword arguments to pass into the JITed function\n\n    Returns\n    -------\n    dict[str, bool]\n        nopython, nogil, parallel\n\n    Raises\n    ------\n    NumbaUtilError\n    '
    if engine_kwargs is None:
        engine_kwargs = {}
    nopython = engine_kwargs.get('nopython', True)
    if kwargs and nopython:
        raise NumbaUtilError('numba does not support kwargs with nopython=True: https://github.com/numba/numba/issues/2916')
    nogil = engine_kwargs.get('nogil', False)
    parallel = engine_kwargs.get('parallel', False)
    return {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}

def jit_user_function(func: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    "\n    If user function is not jitted already, mark the user's function\n    as jitable.\n\n    Parameters\n    ----------\n    func : function\n        user defined function\n\n    Returns\n    -------\n    function\n        Numba JITed function, or function marked as JITable by numba\n    "
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')
    if numba.extending.is_jitted(func):
        numba_func = func
    else:
        numba_func = numba.extending.register_jitable(func)
    return numba_func