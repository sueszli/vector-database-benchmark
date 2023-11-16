"""SCons.Tool.dvips

Tool-specific initialization for dvips.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Action
import SCons.Builder
import SCons.Tool.dvipdf
import SCons.Util

def DviPsFunction(target=None, source=None, env=None):
    if False:
        while True:
            i = 10
    result = SCons.Tool.dvipdf.DviPdfPsFunction(PSAction, target, source, env)
    return result

def DviPsStrFunction(target=None, source=None, env=None):
    if False:
        while True:
            i = 10
    'A strfunction for dvipdf that returns the appropriate\n    command string for the no_exec options.'
    if env.GetOption('no_exec'):
        result = env.subst('$PSCOM', 0, target, source)
    else:
        result = ''
    return result
PSAction = None
DVIPSAction = None
PSBuilder = None

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for dvips to an Environment.'
    global PSAction
    if PSAction is None:
        PSAction = SCons.Action.Action('$PSCOM', '$PSCOMSTR')
    global DVIPSAction
    if DVIPSAction is None:
        DVIPSAction = SCons.Action.Action(DviPsFunction, strfunction=DviPsStrFunction)
    global PSBuilder
    if PSBuilder is None:
        PSBuilder = SCons.Builder.Builder(action=PSAction, prefix='$PSPREFIX', suffix='$PSSUFFIX', src_suffix='.dvi', src_builder='DVI', single_source=True)
    env['BUILDERS']['PostScript'] = PSBuilder
    env['DVIPS'] = 'dvips'
    env['DVIPSFLAGS'] = SCons.Util.CLVar('')
    env['PSCOM'] = 'cd ${TARGET.dir} && $DVIPS $DVIPSFLAGS -o ${TARGET.file} ${SOURCE.file}'
    env['PSPREFIX'] = ''
    env['PSSUFFIX'] = '.ps'

def exists(env):
    if False:
        while True:
            i = 10
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('dvips')