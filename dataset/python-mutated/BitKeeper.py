"""SCons.Tool.BitKeeper.py

Tool-specific initialization for the BitKeeper source code control
system.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/BitKeeper.py  2014/07/05 09:42:21 garyo'
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add a Builder factory function and construction variables for\n    BitKeeper to an Environment.'

    def BitKeeperFactory(env=env):
        if False:
            return 10
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The BitKeeper() factory is deprecated and there is no replacement.')
        act = SCons.Action.Action('$BITKEEPERCOM', '$BITKEEPERCOMSTR')
        return SCons.Builder.Builder(action=act, env=env)
    env.BitKeeper = BitKeeperFactory
    env['BITKEEPER'] = 'bk'
    env['BITKEEPERGET'] = '$BITKEEPER get'
    env['BITKEEPERGETFLAGS'] = SCons.Util.CLVar('')
    env['BITKEEPERCOM'] = '$BITKEEPERGET $BITKEEPERGETFLAGS $TARGET'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect('bk')