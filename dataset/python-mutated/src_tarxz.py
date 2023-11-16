"""SCons.Tool.Packaging.src_tarxz

The tarxz SRC packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        i = 10
        return i + 15
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.xz')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT, honor_install_location=0)
    return bld(env, target, source, TARFLAGS='-Jc')