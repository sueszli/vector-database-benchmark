"""Provide helpful routines for interactive IO on the MSYS console"""
import sys
import os
__all__ = ['print_', 'is_msys']

def print_(*args, **kwds):
    if False:
        return 10
    'Print arguments in an MSYS console friendly way\n\n    Keyword arguments:\n        file, sep, end\n    '
    stream = kwds.get('file', sys.stdout)
    sep = kwds.get('sep', ' ')
    end = kwds.get('end', '\n')
    if args:
        stream.write(sep.join([str(arg) for arg in args]))
    if end:
        stream.write(end)
    try:
        stream.flush()
    except AttributeError:
        pass

def is_msys():
    if False:
        return 10
    'Return true if the execution environment is MSYS'
    try:
        return os.environ['TERM'] == 'cygwin'
    except KeyError:
        return False