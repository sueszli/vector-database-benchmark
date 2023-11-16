"""SCons.Tool.Packaging.src_targz

The targz SRC packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        print('Hello World!')
    bld = env['BUILDERS']['Tar']
    bld.set_suffix('.tar.gz')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT, honor_install_location=0)
    return bld(env, target, source, TARFLAGS='-zc')