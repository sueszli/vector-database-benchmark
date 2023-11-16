"""SCons.Tool.CVS.py

Tool-specific initialization for CVS.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/CVS.py  2014/07/05 09:42:21 garyo'
import SCons.Action
import SCons.Builder
import SCons.Util

def generate(env):
    if False:
        while True:
            i = 10
    'Add a Builder factory function and construction variables for\n    CVS to an Environment.'

    def CVSFactory(repos, module='', env=env):
        if False:
            for i in range(10):
                print('nop')
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The CVS() factory is deprecated and there is no replacement.')
        if module != '':
            module = module + '/'
            env['CVSCOM'] = '$CVS $CVSFLAGS co $CVSCOFLAGS -d ${TARGET.dir} $CVSMODULE${TARGET.posix}'
        act = SCons.Action.Action('$CVSCOM', '$CVSCOMSTR')
        return SCons.Builder.Builder(action=act, env=env, CVSREPOSITORY=repos, CVSMODULE=module)
    env.CVS = CVSFactory
    env['CVS'] = 'cvs'
    env['CVSFLAGS'] = SCons.Util.CLVar('-d $CVSREPOSITORY')
    env['CVSCOFLAGS'] = SCons.Util.CLVar('')
    env['CVSCOM'] = '$CVS $CVSFLAGS co $CVSCOFLAGS ${TARGET.posix}'

def exists(env):
    if False:
        return 10
    return env.Detect('cvs')