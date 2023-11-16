"""SCons.Tool.sunf95

Tool-specific initialization for sunf95, the Sun Studio F95 compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Util
from .FortranCommon import add_all_to_env
compilers = ['sunf95', 'f95']

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for sunf95 to an\n    Environment.'
    add_all_to_env(env)
    fcomp = env.Detect(compilers) or 'f95'
    env['FORTRAN'] = fcomp
    env['F95'] = fcomp
    env['SHFORTRAN'] = '$FORTRAN'
    env['SHF95'] = '$F95'
    env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -KPIC')
    env['SHF95FLAGS'] = SCons.Util.CLVar('$F95FLAGS -KPIC')

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(compilers)