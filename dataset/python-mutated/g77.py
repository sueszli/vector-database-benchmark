"""engine.SCons.Tool.g77

Tool-specific initialization for g77.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/g77.py  2014/07/05 09:42:21 garyo'
import SCons.Util
from SCons.Tool.FortranCommon import add_all_to_env, add_f77_to_env
compilers = ['g77', 'f77']

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for g77 to an Environment.'
    add_all_to_env(env)
    add_f77_to_env(env)
    fcomp = env.Detect(compilers) or 'g77'
    if env['PLATFORM'] in ['cygwin', 'win32']:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS')
        env['SHF77FLAGS'] = SCons.Util.CLVar('$F77FLAGS')
    else:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -fPIC')
        env['SHF77FLAGS'] = SCons.Util.CLVar('$F77FLAGS -fPIC')
    env['FORTRAN'] = fcomp
    env['SHFORTRAN'] = '$FORTRAN'
    env['F77'] = fcomp
    env['SHF77'] = '$F77'
    env['INCFORTRANPREFIX'] = '-I'
    env['INCFORTRANSUFFIX'] = ''
    env['INCF77PREFIX'] = '-I'
    env['INCF77SUFFIX'] = ''

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect(compilers)