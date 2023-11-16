"""SCons.Tool.tlib

XXX

"""
__revision__ = 'src/engine/SCons/Tool/tlib.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Tool
import SCons.Tool.bcc32
import SCons.Util

def generate(env):
    if False:
        return 10
    SCons.Tool.bcc32.findIt('tlib', env)
    'Add Builders and construction variables for ar to an Environment.'
    SCons.Tool.createStaticLibBuilder(env)
    env['AR'] = 'tlib'
    env['ARFLAGS'] = SCons.Util.CLVar('')
    env['ARCOM'] = '$AR $TARGET $ARFLAGS /a $SOURCES'
    env['LIBPREFIX'] = ''
    env['LIBSUFFIX'] = '.lib'

def exists(env):
    if False:
        i = 10
        return i + 15
    return SCons.Tool.bcc32.findIt('tlib', env)