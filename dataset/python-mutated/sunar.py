"""engine.SCons.Tool.sunar

Tool-specific initialization for Solaris (Forte) ar (library archive). If CC
exists, static libraries should be built with it, so that template
instantiations can be resolved.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/sunar.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Defaults
import SCons.Tool
import SCons.Util

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for ar to an Environment.'
    SCons.Tool.createStaticLibBuilder(env)
    if env.Detect('CC'):
        env['AR'] = 'CC'
        env['ARFLAGS'] = SCons.Util.CLVar('-xar')
        env['ARCOM'] = '$AR $ARFLAGS -o $TARGET $SOURCES'
    else:
        env['AR'] = 'ar'
        env['ARFLAGS'] = SCons.Util.CLVar('r')
        env['ARCOM'] = '$AR $ARFLAGS $TARGET $SOURCES'
    env['LIBPREFIX'] = 'lib'
    env['LIBSUFFIX'] = '.a'

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect('CC') or env.Detect('ar')