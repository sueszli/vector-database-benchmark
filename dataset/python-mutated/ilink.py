"""SCons.Tool.ilink

Tool-specific initialization for the OS/2 ilink linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/ilink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Defaults
import SCons.Tool
import SCons.Util

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for ilink to an Environment.'
    SCons.Tool.createProgBuilder(env)
    env['LINK'] = 'ilink'
    env['LINKFLAGS'] = SCons.Util.CLVar('')
    env['LINKCOM'] = '$LINK $LINKFLAGS /O:$TARGET $SOURCES $_LIBDIRFLAGS $_LIBFLAGS'
    env['LIBDIRPREFIX'] = '/LIBPATH:'
    env['LIBDIRSUFFIX'] = ''
    env['LIBLINKPREFIX'] = ''
    env['LIBLINKSUFFIX'] = '$LIBSUFFIX'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect('ilink')