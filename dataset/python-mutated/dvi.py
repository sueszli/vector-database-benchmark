"""SCons.Tool.dvi

Common DVI Builder definition for various other Tool modules that use it.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Builder
import SCons.Tool
DVIBuilder = None

def generate(env):
    if False:
        return 10
    try:
        env['BUILDERS']['DVI']
    except KeyError:
        global DVIBuilder
        if DVIBuilder is None:
            DVIBuilder = SCons.Builder.Builder(action={}, source_scanner=SCons.Tool.LaTeXScanner, suffix='.dvi', emitter={}, source_ext_match=None)
        env['BUILDERS']['DVI'] = DVIBuilder

def exists(env):
    if False:
        return 10
    return 1