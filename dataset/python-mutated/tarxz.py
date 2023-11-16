"""SCons.Tool.Packaging.tarxz

The tarxz packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import stripinstallbuilder, putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        while True:
            i = 10
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.xz')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT)
    (target, source) = stripinstallbuilder(target, source, env)
    return bld(env, target, source, TARFLAGS='-Jc')