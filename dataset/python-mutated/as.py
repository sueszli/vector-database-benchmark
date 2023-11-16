"""SCons.Tool.as

Tool-specific initialization for as, the generic Posix assembler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/as.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Defaults
import SCons.Tool
import SCons.Util
assemblers = ['as']
ASSuffixes = ['.s', '.asm', '.ASM']
ASPPSuffixes = ['.spp', '.SPP', '.sx']
if SCons.Util.case_sensitive_suffixes('.s', '.S'):
    ASPPSuffixes.extend(['.S'])
else:
    ASSuffixes.extend(['.S'])

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for as to an Environment.'
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
    env['AS'] = env.Detect(assemblers) or 'as'
    env['ASFLAGS'] = SCons.Util.CLVar('')
    env['ASCOM'] = '$AS $ASFLAGS -o $TARGET $SOURCES'
    env['ASPPFLAGS'] = '$ASFLAGS'
    env['ASPPCOM'] = '$CC $ASPPFLAGS $CPPFLAGS $_CPPDEFFLAGS $_CPPINCFLAGS -c -o $TARGET $SOURCES'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(assemblers)