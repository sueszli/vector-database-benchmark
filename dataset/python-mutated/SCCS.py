"""SCons.Tool.SCCS.py

Tool-specific initialization for SCCS.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/SCCS.py  2014/07/05 09:42:21 garyo'
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        while True:
            i = 10
    'Add a Builder factory function and construction variables for\n    SCCS to an Environment.'

    def SCCSFactory(env=env):
        if False:
            while True:
                i = 10
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The SCCS() factory is deprecated and there is no replacement.')
        act = SCons.Action.Action('$SCCSCOM', '$SCCSCOMSTR')
        return SCons.Builder.Builder(action=act, env=env)
    env.SCCS = SCCSFactory
    env['SCCS'] = 'sccs'
    env['SCCSFLAGS'] = SCons.Util.CLVar('')
    env['SCCSGETFLAGS'] = SCons.Util.CLVar('')
    env['SCCSCOM'] = '$SCCS $SCCSFLAGS get $SCCSGETFLAGS $TARGET'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect('sccs')