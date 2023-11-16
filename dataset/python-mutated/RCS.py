"""SCons.Tool.RCS.py

Tool-specific initialization for RCS.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/RCS.py  2014/07/05 09:42:21 garyo'
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add a Builder factory function and construction variables for\n    RCS to an Environment.'

    def RCSFactory(env=env):
        if False:
            print('Hello World!')
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The RCS() factory is deprecated and there is no replacement.')
        act = SCons.Action.Action('$RCS_COCOM', '$RCS_COCOMSTR')
        return SCons.Builder.Builder(action=act, env=env)
    env.RCS = RCSFactory
    env['RCS'] = 'rcs'
    env['RCS_CO'] = 'co'
    env['RCS_COFLAGS'] = SCons.Util.CLVar('')
    env['RCS_COCOM'] = '$RCS_CO $RCS_COFLAGS $TARGET'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect('rcs')