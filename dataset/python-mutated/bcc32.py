"""SCons.Tool.bcc32

XXX

"""
__revision__ = 'src/engine/SCons/Tool/bcc32.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import os.path
import SCons.Defaults
import SCons.Tool
import SCons.Util

def findIt(program, env):
    if False:
        return 10
    borwin = env.WhereIs(program) or SCons.Util.WhereIs(program)
    if borwin:
        dir = os.path.dirname(borwin)
        env.PrependENVPath('PATH', dir)
    return borwin

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    findIt('bcc32', env)
    'Add Builders and construction variables for bcc to an\n    Environment.'
    (static_obj, shared_obj) = SCons.Tool.createObjBuilders(env)
    for suffix in ['.c', '.cpp']:
        static_obj.add_action(suffix, SCons.Defaults.CAction)
        shared_obj.add_action(suffix, SCons.Defaults.ShCAction)
        static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
        shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)
    env['CC'] = 'bcc32'
    env['CCFLAGS'] = SCons.Util.CLVar('')
    env['CFLAGS'] = SCons.Util.CLVar('')
    env['CCCOM'] = '$CC -q $CFLAGS $CCFLAGS $CPPFLAGS $_CPPDEFFLAGS $_CPPINCFLAGS -c -o$TARGET $SOURCES'
    env['SHCC'] = '$CC'
    env['SHCCFLAGS'] = SCons.Util.CLVar('$CCFLAGS')
    env['SHCFLAGS'] = SCons.Util.CLVar('$CFLAGS')
    env['SHCCCOM'] = '$SHCC -WD $SHCFLAGS $SHCCFLAGS $CPPFLAGS $_CPPDEFFLAGS $_CPPINCFLAGS -c -o$TARGET $SOURCES'
    env['CPPDEFPREFIX'] = '-D'
    env['CPPDEFSUFFIX'] = ''
    env['INCPREFIX'] = '-I'
    env['INCSUFFIX'] = ''
    env['SHOBJSUFFIX'] = '.dll'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 0
    env['CFILESUFFIX'] = '.cpp'

def exists(env):
    if False:
        while True:
            i = 10
    return findIt('bcc32', env)