"""SCons.Tool.ilink32

XXX

"""
__revision__ = 'src/engine/SCons/Tool/ilink32.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Tool
import SCons.Tool.bcc32
import SCons.Util

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for Borland ilink to an\n    Environment.'
    SCons.Tool.createSharedLibBuilder(env)
    SCons.Tool.createProgBuilder(env)
    env['LINK'] = '$CC'
    env['LINKFLAGS'] = SCons.Util.CLVar('')
    env['LINKCOM'] = '$LINK -q $LINKFLAGS -e$TARGET $SOURCES $LIBS'
    env['LIBDIRPREFIX'] = ''
    env['LIBDIRSUFFIX'] = ''
    env['LIBLINKPREFIX'] = ''
    env['LIBLINKSUFFIX'] = '$LIBSUFFIX'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return SCons.Tool.bcc32.findIt('bcc32', env)