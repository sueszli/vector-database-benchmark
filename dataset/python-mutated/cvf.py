"""SCons.Tool.cvf

Tool-specific initialization for the Compaq Visual Fortran compiler.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from . import fortran
compilers = ['f90']

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for compaq visual fortran to an Environment.'
    fortran.generate(env)
    env['FORTRAN'] = 'f90'
    env['FORTRANCOM'] = '$FORTRAN $FORTRANFLAGS $_FORTRANMODFLAG $_FORTRANINCFLAGS /compile_only ${SOURCES.windows} /object:${TARGET.windows}'
    env['FORTRANPPCOM'] = '$FORTRAN $FORTRANFLAGS $CPPFLAGS $_CPPDEFFLAGS $_FORTRANMODFLAG $_FORTRANINCFLAGS /compile_only ${SOURCES.windows} /object:${TARGET.windows}'
    env['SHFORTRANCOM'] = '$SHFORTRAN $SHFORTRANFLAGS $_FORTRANMODFLAG $_FORTRANINCFLAGS /compile_only ${SOURCES.windows} /object:${TARGET.windows}'
    env['SHFORTRANPPCOM'] = '$SHFORTRAN $SHFORTRANFLAGS $CPPFLAGS $_CPPDEFFLAGS $_FORTRANMODFLAG $_FORTRANINCFLAGS /compile_only ${SOURCES.windows} /object:${TARGET.windows}'
    env['OBJSUFFIX'] = '.obj'
    env['FORTRANMODDIR'] = '${TARGET.dir}'
    env['FORTRANMODDIRPREFIX'] = '/module:'
    env['FORTRANMODDIRSUFFIX'] = ''

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect(compilers)