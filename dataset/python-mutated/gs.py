"""SCons.Tool.gs

Tool-specific initialization for Ghostscript.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Action
import SCons.Builder
import SCons.Platform
import SCons.Util
platform = SCons.Platform.platform_default()
if platform == 'os2':
    gs = 'gsos2'
elif platform == 'win32':
    gs = 'gswin32c'
else:
    gs = 'gs'
GhostscriptAction = None

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for Ghostscript to an\n    Environment.'
    global GhostscriptAction
    try:
        if GhostscriptAction is None:
            GhostscriptAction = SCons.Action.Action('$GSCOM', '$GSCOMSTR')
        from SCons.Tool import pdf
        pdf.generate(env)
        bld = env['BUILDERS']['PDF']
        bld.add_action('.ps', GhostscriptAction)
    except ImportError as e:
        pass
    gsbuilder = SCons.Builder.Builder(action=SCons.Action.Action('$GSCOM', '$GSCOMSTR'))
    env['BUILDERS']['Gs'] = gsbuilder
    env['GS'] = gs
    env['GSFLAGS'] = SCons.Util.CLVar('-dNOPAUSE -dBATCH -sDEVICE=pdfwrite')
    env['GSCOM'] = '$GS $GSFLAGS -sOutputFile=$TARGET $SOURCES'

def exists(env):
    if False:
        print('Hello World!')
    if 'PS2PDF' in env:
        return env.Detect(env['PS2PDF'])
    else:
        return env.Detect(gs) or SCons.Util.WhereIs(gs)