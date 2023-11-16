"""Utilities for marking deprecated functions."""
import warnings
from zipline.utils.compat import wraps

def deprecated(msg=None, stacklevel=2):
    if False:
        for i in range(10):
            print('nop')
    "\n    Used to mark a function as deprecated.\n\n    Parameters\n    ----------\n    msg : str\n        The message to display in the deprecation warning.\n    stacklevel : int\n        How far up the stack the warning needs to go, before\n        showing the relevant calling lines.\n\n    Examples\n    --------\n    @deprecated(msg='function_a is deprecated! Use function_b instead.')\n    def function_a(*args, **kwargs):\n    "

    def deprecated_dec(fn):
        if False:
            return 10

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            warnings.warn(msg or 'Function %s is deprecated.' % fn.__name__, category=DeprecationWarning, stacklevel=stacklevel)
            return fn(*args, **kwargs)
        return wrapper
    return deprecated_dec