"""engine.SCons.Tool.aixf77

Tool-specific initialization for IBM Visual Age f77 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/aixf77.py  2014/07/05 09:42:21 garyo'
import os.path
import f77
packages = []

def get_xlf77(env):
    if False:
        for i in range(10):
            print('nop')
    xlf77 = env.get('F77', 'xlf77')
    xlf77_r = env.get('SHF77', 'xlf77_r')
    return (None, xlf77, xlf77_r, None)

def generate(env):
    if False:
        i = 10
        return i + 15
    '\n    Add Builders and construction variables for the Visual Age FORTRAN\n    compiler to an Environment.\n    '
    (path, _f77, _shf77, version) = get_xlf77(env)
    if path:
        _f77 = os.path.join(path, _f77)
        _shf77 = os.path.join(path, _shf77)
    f77.generate(env)
    env['F77'] = _f77
    env['SHF77'] = _shf77

def exists(env):
    if False:
        return 10
    (path, _f77, _shf77, version) = get_xlf77(env)
    if path and _f77:
        xlf77 = os.path.join(path, _f77)
        if os.path.exists(xlf77):
            return xlf77
    return None