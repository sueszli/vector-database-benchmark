"""Tool-specific initialization for the generic Posix f08 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
from SCons.Tool.FortranCommon import add_all_to_env, add_f08_to_env
compilers = ['f08']

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    add_all_to_env(env)
    add_f08_to_env(env)
    fcomp = env.Detect(compilers) or 'f08'
    if 'F08' not in env:
        env['F08'] = fcomp
    if 'SHF08' not in env:
        env['SHF08'] = '$F08'
    if 'FORTRAN' not in env:
        env['FORTRAN'] = fcomp
    if 'SHFORTRAN' not in env:
        env['SHFORTRAN'] = '$FORTRAN'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(compilers)