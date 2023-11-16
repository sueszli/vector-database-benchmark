"""SCons.Tool.m4

Tool-specific initialization for m4.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/m4.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for m4 to an Environment.'
    M4Action = SCons.Action.Action('$M4COM', '$M4COMSTR')
    bld = SCons.Builder.Builder(action=M4Action, src_suffix='.m4')
    env['BUILDERS']['M4'] = bld
    env['M4'] = 'm4'
    env['M4FLAGS'] = SCons.Util.CLVar('-E')
    env['M4COM'] = 'cd ${SOURCE.rsrcdir} && $M4 $M4FLAGS < ${SOURCE.file} > ${TARGET.abspath}'

def exists(env):
    if False:
        return 10
    return env.Detect('m4')