"""Tool-specific initialization for the generic Posix f90 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
from SCons.Tool.FortranCommon import add_all_to_env, add_f90_to_env
compilers = ['f90']

def generate(env):
    if False:
        while True:
            i = 10
    add_all_to_env(env)
    add_f90_to_env(env)
    fc = env.Detect(compilers) or 'f90'
    if 'F90' not in env:
        env['F90'] = fc
    if 'SHF90' not in env:
        env['SHF90'] = '$F90'
    if 'FORTRAN' not in env:
        env['FORTRAN'] = fc
    if 'SHFORTRAN' not in env:
        env['SHFORTRAN'] = '$FORTRAN'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(compilers)