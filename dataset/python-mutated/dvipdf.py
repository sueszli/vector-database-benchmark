"""SCons.Tool.dvipdf

Tool-specific initialization for dvipdf.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Action
import SCons.Defaults
import SCons.Tool.pdf
import SCons.Tool.tex
import SCons.Util
_null = SCons.Scanner.LaTeX._null

def DviPdfPsFunction(XXXDviAction, target=None, source=None, env=None):
    if False:
        return 10
    'A builder for DVI files that sets the TEXPICTS environment\n       variable before running dvi2ps or dvipdf.'
    try:
        abspath = source[0].attributes.path
    except AttributeError:
        abspath = ''
    saved_env = SCons.Scanner.LaTeX.modify_env_var(env, 'TEXPICTS', abspath)
    result = XXXDviAction(target, source, env)
    if saved_env is _null:
        try:
            del env['ENV']['TEXPICTS']
        except KeyError:
            pass
    else:
        env['ENV']['TEXPICTS'] = saved_env
    return result

def DviPdfFunction(target=None, source=None, env=None):
    if False:
        print('Hello World!')
    result = DviPdfPsFunction(PDFAction, target, source, env)
    return result

def DviPdfStrFunction(target=None, source=None, env=None):
    if False:
        i = 10
        return i + 15
    'A strfunction for dvipdf that returns the appropriate\n    command string for the no_exec options.'
    if env.GetOption('no_exec'):
        result = env.subst('$DVIPDFCOM', 0, target, source)
    else:
        result = ''
    return result
PDFAction = None
DVIPDFAction = None

def PDFEmitter(target, source, env):
    if False:
        return 10
    "Strips any .aux or .log files from the input source list.\n    These are created by the TeX Builder that in all likelihood was\n    used to generate the .dvi file we're using as input, and we only\n    care about the .dvi file.\n    "

    def strip_suffixes(n):
        if False:
            return 10
        return not SCons.Util.splitext(str(n))[1] in ['.aux', '.log']
    source = [src for src in source if strip_suffixes(src)]
    return (target, source)

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for dvipdf to an Environment.'
    global PDFAction
    if PDFAction is None:
        PDFAction = SCons.Action.Action('$DVIPDFCOM', '$DVIPDFCOMSTR')
    global DVIPDFAction
    if DVIPDFAction is None:
        DVIPDFAction = SCons.Action.Action(DviPdfFunction, strfunction=DviPdfStrFunction)
    from . import pdf
    pdf.generate(env)
    bld = env['BUILDERS']['PDF']
    bld.add_action('.dvi', DVIPDFAction)
    bld.add_emitter('.dvi', PDFEmitter)
    env['DVIPDF'] = 'dvipdf'
    env['DVIPDFFLAGS'] = SCons.Util.CLVar('')
    env['DVIPDFCOM'] = 'cd ${TARGET.dir} && $DVIPDF $DVIPDFFLAGS ${SOURCE.file} ${TARGET.file}'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('dvipdf')