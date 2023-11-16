"""SCons.Tool.Subversion.py

Tool-specific initialization for Subversion.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/Subversion.py  2014/07/05 09:42:21 garyo'
import os.path
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        return 10
    'Add a Builder factory function and construction variables for\n    Subversion to an Environment.'

    def SubversionFactory(repos, module='', env=env):
        if False:
            i = 10
            return i + 15
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The Subversion() factory is deprecated and there is no replacement.')
        if module != '':
            module = os.path.join(module, '')
        act = SCons.Action.Action('$SVNCOM', '$SVNCOMSTR')
        return SCons.Builder.Builder(action=act, env=env, SVNREPOSITORY=repos, SVNMODULE=module)
    env.Subversion = SubversionFactory
    env['SVN'] = 'svn'
    env['SVNFLAGS'] = SCons.Util.CLVar('')
    env['SVNCOM'] = '$SVN $SVNFLAGS cat $SVNREPOSITORY/$SVNMODULE$TARGET > $TARGET'

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect('svn')