"""If possible, exports all symbols with RTLD_GLOBAL.

Note that this file is only imported by pywrap_tensorflow.py if this is a static
build (meaning there is no explicit framework cc_binary shared object dependency
of _pywrap_tensorflow_internal.so). For regular (non-static) builds, RTLD_GLOBAL
is not necessary, since the dynamic dependencies of custom/contrib ops are
explicit.
"""
import ctypes
import sys
_use_rtld_global = hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags')
if _use_rtld_global:
    _default_dlopen_flags = sys.getdlopenflags()

def set_dlopen_flags():
    if False:
        i = 10
        return i + 15
    if _use_rtld_global:
        sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_GLOBAL)

def reset_dlopen_flags():
    if False:
        while True:
            i = 10
    if _use_rtld_global:
        sys.setdlopenflags(_default_dlopen_flags)