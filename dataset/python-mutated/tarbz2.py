"""SCons.Tool.Packaging.tarbz2

The tarbz2 packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import stripinstallbuilder, putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        return 10
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.bz2')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT)
    (target, source) = stripinstallbuilder(target, source, env)
    return bld(env, target, source, TARFLAGS='-jc')