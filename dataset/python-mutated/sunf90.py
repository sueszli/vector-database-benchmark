"""SCons.Tool.sunf90

Tool-specific initialization for sunf90, the Sun Studio F90 compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Util
from .FortranCommon import add_all_to_env
compilers = ['sunf90', 'f90']

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for sun f90 compiler to an\n    Environment.'
    add_all_to_env(env)
    fcomp = env.Detect(compilers) or 'f90'
    env['FORTRAN'] = fcomp
    env['F90'] = fcomp
    env['SHFORTRAN'] = '$FORTRAN'
    env['SHF90'] = '$F90'
    env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -KPIC')
    env['SHF90FLAGS'] = SCons.Util.CLVar('$F90FLAGS -KPIC')

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect(compilers)