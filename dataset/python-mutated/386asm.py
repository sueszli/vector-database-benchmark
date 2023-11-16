"""SCons.Tool.386asm

Tool specification for the 386ASM assembler for the Phar Lap ETS embedded
operating system.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/386asm.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from SCons.Tool.PharLapCommon import addPharLapPaths
import SCons.Util
as_module = __import__('as', globals(), locals(), [], 1)

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for ar to an Environment.'
    as_module.generate(env)
    env['AS'] = '386asm'
    env['ASFLAGS'] = SCons.Util.CLVar('')
    env['ASPPFLAGS'] = '$ASFLAGS'
    env['ASCOM'] = '$AS $ASFLAGS $SOURCES -o $TARGET'
    env['ASPPCOM'] = '$CC $ASPPFLAGS $CPPFLAGS $_CPPDEFFLAGS $_CPPINCFLAGS $SOURCES -o $TARGET'
    addPharLapPaths(env)

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect('386asm')