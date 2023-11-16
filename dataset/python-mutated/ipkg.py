"""SCons.Tool.ipkg

Tool-specific initialization for ipkg.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

The ipkg tool calls the ipkg-build. Its only argument should be the 
packages fake_root.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import os
import SCons.Builder

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for ipkg to an Environment.'
    try:
        bld = env['BUILDERS']['Ipkg']
    except KeyError:
        bld = SCons.Builder.Builder(action='$IPKGCOM', suffix='$IPKGSUFFIX', source_scanner=None, target_scanner=None)
        env['BUILDERS']['Ipkg'] = bld
    env['IPKG'] = 'ipkg-build'
    env['IPKGCOM'] = '$IPKG $IPKGFLAGS ${SOURCE}'
    if env.WhereIs('id'):
        with os.popen('id -un') as p:
            env['IPKGUSER'] = p.read().strip()
        with os.popen('id -gn') as p:
            env['IPKGGROUP'] = p.read().strip()
    env['IPKGFLAGS'] = SCons.Util.CLVar('-o $IPKGUSER -g $IPKGGROUP')
    env['IPKGSUFFIX'] = '.ipk'

def exists(env):
    if False:
        i = 10
        return i + 15
    '\n    Can we find the tool\n    '
    return env.Detect('ipkg-build')