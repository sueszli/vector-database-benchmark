"""SCons.Tool.masm

Tool-specific initialization for the Microsoft Assembler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/masm.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Defaults
import SCons.Tool
import SCons.Util
ASSuffixes = ['.s', '.asm', '.ASM']
ASPPSuffixes = ['.spp', '.SPP', '.sx']
if SCons.Util.case_sensitive_suffixes('.s', '.S'):
    ASPPSuffixes.extend(['.S'])
else:
    ASSuffixes.extend(['.S'])

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for masm to an Environment.'
    (static_obj, shared_obj) = SCons.Tool.createObjBuilders(env)
    for suffix in ASSuffixes:
        static_obj.add_action(suffix, SCons.Defaults.ASAction)
        shared_obj.add_action(suffix, SCons.Defaults.ASAction)
        static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
        shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)
    for suffix in ASPPSuffixes:
        static_obj.add_action(suffix, SCons.Defaults.ASPPAction)
        shared_obj.add_action(suffix, SCons.Defaults.ASPPAction)
        static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
        shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)
    env['AS'] = 'ml'
    env['ASFLAGS'] = SCons.Util.CLVar('/nologo')
    env['ASPPFLAGS'] = '$ASFLAGS'
    env['ASCOM'] = '$AS $ASFLAGS /c /Fo$TARGET $SOURCES'
    env['ASPPCOM'] = '$CC $ASPPFLAGS $CPPFLAGS $_CPPDEFFLAGS $_CPPINCFLAGS /c /Fo$TARGET $SOURCES'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect('ml')