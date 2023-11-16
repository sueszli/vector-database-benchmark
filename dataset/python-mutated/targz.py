"""SCons.Tool.Packaging.targz

The targz packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import stripinstallbuilder, putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        i = 10
        return i + 15
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.gz')
    (target, source) = stripinstallbuilder(target, source, env)
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT)
    return bld(env, target, source, TARFLAGS='-zc')