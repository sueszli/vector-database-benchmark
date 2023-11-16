"""SCons.Tool.Packaging.src_tarbz2

The tarbz2 SRC packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        for i in range(10):
            print('nop')
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.bz2')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT, honor_install_location=0)
    return bld(env, target, source, TARFLAGS='-jc')