"""SCons.Tool.Packaging.zip

The zip SRC packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
from SCons.Tool.packaging import putintopackageroot

def package(env, target, source, PACKAGEROOT, **kw):
    if False:
        for i in range(10):
            print('nop')
    bld = env['BUILDERS']['Zip']
    bld.set_suffix('.zip')
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT, honor_install_location=0)
    return bld(env, target, source)