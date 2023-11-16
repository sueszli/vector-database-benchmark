"""Tool-specific initialization for the generic Posix f77 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
from SCons.Tool.FortranCommon import add_all_to_env, add_f77_to_env
compilers = ['f77']

def generate(env):
    if False:
        return 10
    add_all_to_env(env)
    add_f77_to_env(env)
    fcomp = env.Detect(compilers) or 'f77'
    if 'F77' not in env:
        env['F77'] = fcomp
    if 'SHF77' not in env:
        env['SHF77'] = '$F77'
    if 'FORTRAN' not in env:
        env['FORTRAN'] = fcomp
    if 'SHFORTRAN' not in env:
        env['SHFORTRAN'] = '$FORTRAN'

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect(compilers)