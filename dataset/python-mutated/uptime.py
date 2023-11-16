"""
Provides a cross-platform way to figure out the system uptime.

Should work on damned near any operating system you can realistically expect
to be asked to write Python code for.
If this module is invoked as a stand-alone script, it will print the current
uptime in a human-readable format, or display an error message if it can't,
to standard output.

This file was forked from the uptime project: https://github.com/Cairnarvon/uptime
Copyright (c) 2012, Koen Crolla, All rights reserved.
"""
import contextlib
import ctypes
import functools
import os
import struct
import sys
import time
import xonsh.lazyimps as xlimps
import xonsh.platform as xp

def _boot_time_osx() -> 'float|None':
    if False:
        for i in range(10):
            print('nop')
    'Returns the uptime on mac / darwin.'
    bt = xlimps.macutils.sysctlbyname(b'kern.boottime', return_str=False)
    if len(bt) == 4:
        bt = struct.unpack_from('@hh', bt)
    elif len(bt) == 8:
        bt = struct.unpack_from('@ii', bt)
    elif len(bt) == 16:
        bt = struct.unpack_from('@qq', bt)
    else:
        raise ValueError('length of boot time not understood: ' + repr(bt))
    bt = bt[0] + bt[1] * 1e-06
    if bt == 0.0:
        return None
    return bt

def _boot_time_linux() -> 'float|None':
    if False:
        i = 10
        return i + 15
    'A way to figure out the boot time directly on Linux.'
    bt_flag = getattr(time, 'CLOCK_BOOTTIME', None)
    if bt_flag is not None:
        return time.clock_gettime(bt_flag)
    try:
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('btime'):
                    return float(line.split()[1])
    except (OSError, IndexError):
        return None

def _boot_time_amiga() -> 'float|None':
    if False:
        while True:
            i = 10
    'Returns uptime in seconds or None, on AmigaOS.'
    try:
        return os.stat('RAM:').st_ctime
    except (NameError, OSError):
        return None

def _boot_time_beos() -> 'float|None':
    if False:
        return 10
    'Returns uptime in seconds on None, on BeOS/Haiku.'
    if not hasattr(xp.LIBC, 'system_time'):
        return None
    xp.LIBC.system_time.restype = ctypes.c_int64
    return time.time() - xp.LIBC.system_time() / 1000000.0

def _boot_time_bsd() -> 'float|None':
    if False:
        for i in range(10):
            print('nop')
    'Returns uptime in seconds or None, on BSD (including OS X).'
    with contextlib.suppress(Exception):
        ut_flag = getattr(time, 'CLOCK_UPTIME', None)
        if ut_flag is not None:
            ut = time.clock_gettime(ut_flag)
            return time.time() - ut
    if not hasattr(xp.LIBC, 'sysctlbyname'):
        return None
    sz = ctypes.c_uint(0)
    xp.LIBC.sysctlbyname(b'kern.boottime', None, ctypes.byref(sz), None, 0)
    if sz.value != struct.calcsize('@LL'):
        return None
    buf = ctypes.create_string_buffer(sz.value)
    xp.LIBC.sysctlbyname(b'kern.boottime', buf, ctypes.byref(sz), None, 0)
    (sec, usec) = struct.unpack_from('@LL', buf.raw)
    if usec > 1000000:
        usec = 0.0
    return sec + usec / 1000000.0

def _boot_time_minix():
    if False:
        i = 10
        return i + 15
    'Returns uptime in seconds or None, on MINIX.'
    try:
        with open('/proc/uptime') as f:
            up = float(f.read())
        return time.time() - up
    except (OSError, ValueError):
        return None

def _boot_time_plan9():
    if False:
        return 10
    'Returns uptime in seconds or None, on Plan 9.'
    try:
        with open('/dev/time') as f:
            (s, ns, ct, cf) = f.read().split()
        return time.time() - float(ct) / float(cf)
    except (OSError, ValueError):
        return None

def _boot_time_solaris():
    if False:
        while True:
            i = 10
    'Returns uptime in seconds or None, on Solaris.'
    try:
        kstat = ctypes.CDLL('libkstat.so')
    except (AttributeError, OSError):
        return None
    _BOOTTIME = None
    KSTAT_STRLEN = 31

    class anon_union(ctypes.Union):
        _fields_ = [('c', ctypes.c_char * 16), ('time', ctypes.c_int)]

    class kstat_named_t(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char * KSTAT_STRLEN), ('data_type', ctypes.c_char), ('value', anon_union)]
    kstat.kstat_open.restype = ctypes.c_void_p
    kstat.kstat_lookup.restype = ctypes.c_void_p
    kstat.kstat_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    kstat.kstat_read.restype = ctypes.c_int
    kstat.kstat_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kstat.kstat_data_lookup.restype = ctypes.POINTER(kstat_named_t)
    kstat.kstat_data_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    kc = kstat.kstat_open()
    if not kc:
        return None
    ksp = kstat.kstat_lookup(kc, 'unix', 0, 'system_misc')
    if ksp and kstat.kstat_read(kc, ksp, None) != -1:
        data = kstat.kstat_data_lookup(ksp, 'boot_time')
        if data:
            _BOOTTIME = data.contents.value.time
    kstat.kstat_close(kc)
    return _BOOTTIME

def _boot_time_syllable():
    if False:
        print('Hello World!')
    'Returns uptime in seconds or None, on Syllable.'
    try:
        return os.stat('/dev/pty/mst/pty0').st_mtime
    except (NameError, OSError):
        return None

def _boot_time_windows():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns uptime in seconds or None, on Windows. Warning: may return\n    incorrect answers after 49.7 days on versions older than Vista.\n    '
    uptime = None
    if hasattr(xp.LIBC, 'GetTickCount64'):
        xp.LIBC.GetTickCount64.restype = ctypes.c_uint64
        uptime = xp.LIBC.GetTickCount64() / 1000.0
    if hasattr(xp.LIBC, 'GetTickCount'):
        xp.LIBC.GetTickCount.restype = ctypes.c_uint32
        uptime = xp.LIBC.GetTickCount() / 1000.0
    if uptime:
        return time.time() - uptime
    return None

def _boot_time_monotonic():
    if False:
        while True:
            i = 10
    if hasattr(time, 'CLOCK_MONOTONIC'):
        monotime = time.clock_gettime(time.CLOCK_MONOTONIC)
    else:
        monotime = time.time()
    return time.time() - monotime

def _get_boot_time_func():
    if False:
        print('Hello World!')
    plat = sys.platform
    if plat.startswith(('amiga', 'aros12')):
        return _boot_time_amiga
    if plat.startswith(('beos5', 'haiku1')):
        return _boot_time_beos
    if plat.startswith(('cygwin', 'linux')):
        return _boot_time_linux
    if plat.startswith('minix3'):
        return _boot_time_minix
    if plat.startswith('darwin'):
        return _boot_time_osx
    if plat.startswith('sunos5'):
        return _boot_time_solaris
    if plat.startswith('syllable'):
        return _boot_time_syllable
    if plat.startswith('win'):
        return _boot_time_windows
    return _boot_time_monotonic

def uptime(args):
    if False:
        return 10
    'Returns uptime in seconds if even remotely possible, or None if not.'
    bt = boottime()
    return str(time.time() - bt)

@functools.lru_cache(None)
def boottime() -> 'float':
    if False:
        print('Hello World!')
    'Returns boot time if remotely possible, or None if not.'
    func = _get_boot_time_func()
    btime = func()
    if btime is None:
        return _boot_time_monotonic()
    return btime

def main(args=None):
    if False:
        return 10
    from xonsh.xoreutils.util import run_alias
    run_alias('uptime', args)
if __name__ == '__main__':
    main()