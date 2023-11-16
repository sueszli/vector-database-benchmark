"""SCons.Tool.sunf77

Tool-specific initialization for sunf77, the Sun Studio F77 compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Util
from .FortranCommon import add_all_to_env
compilers = ['sunf77', 'f77']

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for sunf77 to an Environment.'
    add_all_to_env(env)
    fcomp = env.Detect(compilers) or 'f77'
    env['FORTRAN'] = fcomp
    env['F77'] = fcomp
    env['SHFORTRAN'] = '$FORTRAN'
    env['SHF77'] = '$F77'
    env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -KPIC')
    env['SHF77FLAGS'] = SCons.Util.CLVar('$F77FLAGS -KPIC')

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(compilers)