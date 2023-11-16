"""SCons.Tool.wix

Tool-specific initialization for wix, the Windows Installer XML Tool.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/wix.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Builder
import SCons.Action
import os

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for WiX to an Environment.'
    if not exists(env):
        return
    env['WIXCANDLEFLAGS'] = ['-nologo']
    env['WIXCANDLEINCLUDE'] = []
    env['WIXCANDLECOM'] = '$WIXCANDLE $WIXCANDLEFLAGS -I $WIXCANDLEINCLUDE -o ${TARGET} ${SOURCE}'
    env['WIXLIGHTFLAGS'].append('-nologo')
    env['WIXLIGHTCOM'] = '$WIXLIGHT $WIXLIGHTFLAGS -out ${TARGET} ${SOURCES}'
    env['WIXSRCSUF'] = '.wxs'
    env['WIXOBJSUF'] = '.wixobj'
    object_builder = SCons.Builder.Builder(action='$WIXCANDLECOM', suffix='$WIXOBJSUF', src_suffix='$WIXSRCSUF')
    linker_builder = SCons.Builder.Builder(action='$WIXLIGHTCOM', src_suffix='$WIXOBJSUF', src_builder=object_builder)
    env['BUILDERS']['WiX'] = linker_builder

def exists(env):
    if False:
        i = 10
        return i + 15
    env['WIXCANDLE'] = 'candle.exe'
    env['WIXLIGHT'] = 'light.exe'
    for path in os.environ['PATH'].split(os.pathsep):
        if not path:
            continue
        if path[0] == '"' and path[-1:] == '"':
            path = path[1:-1]
        path = os.path.normpath(path)
        try:
            files = os.listdir(path)
            if env['WIXCANDLE'] in files and env['WIXLIGHT'] in files:
                env.PrependENVPath('PATH', path)
                if 'wixui.wixlib' in files and 'WixUI_en-us.wxl' in files:
                    env['WIXLIGHTFLAGS'] = [os.path.join(path, 'wixui.wixlib'), '-loc', os.path.join(path, 'WixUI_en-us.wxl')]
                else:
                    env['WIXLIGHTFLAGS'] = []
                return 1
        except OSError:
            pass
    return None