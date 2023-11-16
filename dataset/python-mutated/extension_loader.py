import contextlib
import ctypes
import sys
_set_global_flags = hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags')

@contextlib.contextmanager
def DlopenGuard(extra_flags=ctypes.RTLD_GLOBAL):
    if False:
        for i in range(10):
            print('nop')
    if _set_global_flags:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | extra_flags)
    try:
        yield
    finally:
        if _set_global_flags:
            sys.setdlopenflags(old_flags)