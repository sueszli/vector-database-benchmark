"""SCons.Tool.pdflatex

Tool-specific initialization for pdflatex.
Generates .pdf files from .latex or .ltx files

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import SCons.Action
import SCons.Util
import SCons.Tool.pdf
import SCons.Tool.tex
PDFLaTeXAction = None

def PDFLaTeXAuxFunction(target=None, source=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    result = SCons.Tool.tex.InternalLaTeXAuxAction(PDFLaTeXAction, target, source, env)
    if result != 0:
        SCons.Tool.tex.check_file_error_message(env['PDFLATEX'])
    return result
PDFLaTeXAuxAction = None

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for pdflatex to an Environment.'
    global PDFLaTeXAction
    if PDFLaTeXAction is None:
        PDFLaTeXAction = SCons.Action.Action('$PDFLATEXCOM', '$PDFLATEXCOMSTR')
    global PDFLaTeXAuxAction
    if PDFLaTeXAuxAction is None:
        PDFLaTeXAuxAction = SCons.Action.Action(PDFLaTeXAuxFunction, strfunction=SCons.Tool.tex.TeXLaTeXStrFunction)
    env.AppendUnique(LATEXSUFFIXES=SCons.Tool.LaTeXSuffixes)
    from . import pdf
    pdf.generate(env)
    bld = env['BUILDERS']['PDF']
    bld.add_action('.ltx', PDFLaTeXAuxAction)
    bld.add_action('.latex', PDFLaTeXAuxAction)
    bld.add_emitter('.ltx', SCons.Tool.tex.tex_pdf_emitter)
    bld.add_emitter('.latex', SCons.Tool.tex.tex_pdf_emitter)
    SCons.Tool.tex.generate_common(env)

def exists(env):
    if False:
        i = 10
        return i + 15
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('pdflatex')