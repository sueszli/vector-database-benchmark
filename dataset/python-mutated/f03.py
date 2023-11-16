"""Tool-specific initialization for the generic Posix f03 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
from SCons.Tool.FortranCommon import add_all_to_env, add_f03_to_env
compilers = ['f03']

def generate(env):
    if False:
        while True:
            i = 10
    add_all_to_env(env)
    add_f03_to_env(env)
    fcomp = env.Detect(compilers) or 'f03'
    if 'F03' not in env:
        env['F03'] = fcomp
    if 'SHF03' not in env:
        env['SHF03'] = '$F03'
    if 'FORTRAN' not in env:
        env['FORTRAN'] = fcomp
    if 'SHFORTRAN' not in env:
        env['SHFORTRAN'] = '$FORTRAN'

def exists(env):
    if False:
        print('Hello World!')
    return env.Detect(compilers)