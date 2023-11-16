import os
import sys
import locale
from .misc import isatty
from .platform import UNIXY, WINDOWS
if UNIXY:
    DEFAULT_CONSOLE_ENCODING = 'UTF-8'
    DEFAULT_SYSTEM_ENCODING = 'UTF-8'
else:
    DEFAULT_CONSOLE_ENCODING = 'cp437'
    DEFAULT_SYSTEM_ENCODING = 'cp1252'

def get_system_encoding():
    if False:
        for i in range(10):
            print('nop')
    platform_getters = [(True, _get_python_system_encoding), (UNIXY, _get_unixy_encoding), (WINDOWS, _get_windows_system_encoding)]
    return _get_encoding(platform_getters, DEFAULT_SYSTEM_ENCODING)

def get_console_encoding():
    if False:
        while True:
            i = 10
    platform_getters = [(True, _get_stream_output_encoding), (UNIXY, _get_unixy_encoding), (WINDOWS, _get_windows_console_encoding)]
    return _get_encoding(platform_getters, DEFAULT_CONSOLE_ENCODING)

def _get_encoding(platform_getters, default):
    if False:
        print('Hello World!')
    for (platform, getter) in platform_getters:
        if platform:
            encoding = getter()
            if _is_valid(encoding):
                return encoding
    return default

def _get_python_system_encoding():
    if False:
        while True:
            i = 10
    try:
        return locale.getpreferredencoding(False)
    except ValueError:
        return None

def _get_unixy_encoding():
    if False:
        for i in range(10):
            print('nop')
    for name in ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
        if name in os.environ:
            encoding = os.environ[name].split('.')[-1]
            if _is_valid(encoding):
                return encoding
    return None

def _get_stream_output_encoding():
    if False:
        i = 10
        return i + 15
    if WINDOWS:
        return None
    for stream in (sys.__stdout__, sys.__stderr__, sys.__stdin__):
        if isatty(stream):
            encoding = getattr(stream, 'encoding', None)
            if _is_valid(encoding):
                return encoding
    return None

def _get_windows_system_encoding():
    if False:
        print('Hello World!')
    return _get_code_page('GetACP')

def _get_windows_console_encoding():
    if False:
        return 10
    return _get_code_page('GetConsoleOutputCP')

def _get_code_page(method_name):
    if False:
        while True:
            i = 10
    from ctypes import cdll
    method = getattr(cdll.kernel32, method_name)
    return 'cp%s' % method()

def _is_valid(encoding):
    if False:
        for i in range(10):
            print('nop')
    if not encoding:
        return False
    try:
        'test'.encode(encoding)
    except LookupError:
        return False
    else:
        return True