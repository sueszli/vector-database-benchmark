"""Provides some Mac / Darwin based utility functions for xonsh."""
from ctypes import byref, c_uint, create_string_buffer
from xonsh.platform import LIBC

def sysctlbyname(name, return_str=True):
    if False:
        while True:
            i = 10
    'Gets a sysctl value by name. If return_str is true, this will return\n    a string representation, else it will return the raw value.\n    '
    size = c_uint(0)
    LIBC.sysctlbyname(name, None, byref(size), None, 0)
    buf = create_string_buffer(size.value)
    LIBC.sysctlbyname(name, buf, byref(size), None, 0)
    if return_str:
        return buf.value
    else:
        return buf.raw