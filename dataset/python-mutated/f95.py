"""Tool-specific initialization for the generic Posix f95 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
from SCons.Tool.FortranCommon import add_all_to_env, add_f95_to_env
compilers = ['f95']

def generate(env):
    if False:
        return 10
    add_all_to_env(env)
    add_f95_to_env(env)
    fcomp = env.Detect(compilers) or 'f95'
    if 'F95' not in env:
        env['F95'] = fcomp
    if 'SHF95' not in env:
        env['SHF95'] = '$F95'
    if 'FORTRAN' not in env:
        env['FORTRAN'] = fcomp
    if 'SHFORTRAN' not in env:
        env['SHFORTRAN'] = '$FORTRAN'

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect(compilers)