"""SCons.Tool.tar

Tool-specific initialization for tar.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/tar.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Node.FS
import SCons.Util
tars = ['tar', 'gtar']
TarAction = SCons.Action.Action('$TARCOM', '$TARCOMSTR')
TarBuilder = SCons.Builder.Builder(action=TarAction, source_factory=SCons.Node.FS.Entry, source_scanner=SCons.Defaults.DirScanner, suffix='$TARSUFFIX', multi=1)

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for tar to an Environment.'
    try:
        bld = env['BUILDERS']['Tar']
    except KeyError:
        bld = TarBuilder
        env['BUILDERS']['Tar'] = bld
    env['TAR'] = env.Detect(tars) or 'gtar'
    env['TARFLAGS'] = SCons.Util.CLVar('-c')
    env['TARCOM'] = '$TAR $TARFLAGS -f $TARGET $SOURCES'
    env['TARSUFFIX'] = '.tar'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect(tars)