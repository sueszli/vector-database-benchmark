"""SCons.Tool.pdftex

Tool-specific initialization for pdftex.
Generates .pdf files from .tex files

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import os
import SCons.Action
import SCons.Util
import SCons.Tool.tex
PDFTeXAction = None
PDFLaTeXAction = None

def PDFLaTeXAuxAction(target=None, source=None, env=None):
    if False:
        print('Hello World!')
    result = SCons.Tool.tex.InternalLaTeXAuxAction(PDFLaTeXAction, target, source, env)
    return result

def PDFTeXLaTeXFunction(target=None, source=None, env=None):
    if False:
        return 10
    'A builder for TeX and LaTeX that scans the source file to\n    decide the "flavor" of the source and then executes the appropriate\n    program.'
    basedir = os.path.split(str(source[0]))[0]
    abspath = os.path.abspath(basedir)
    if SCons.Tool.tex.is_LaTeX(source, env, abspath):
        result = PDFLaTeXAuxAction(target, source, env)
        if result != 0:
            SCons.Tool.tex.check_file_error_message(env['PDFLATEX'])
    else:
        result = PDFTeXAction(target, source, env)
        if result != 0:
            SCons.Tool.tex.check_file_error_message(env['PDFTEX'])
    return result
PDFTeXLaTeXAction = None

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for pdftex to an Environment.'
    global PDFTeXAction
    if PDFTeXAction is None:
        PDFTeXAction = SCons.Action.Action('$PDFTEXCOM', '$PDFTEXCOMSTR')
    global PDFLaTeXAction
    if PDFLaTeXAction is None:
        PDFLaTeXAction = SCons.Action.Action('$PDFLATEXCOM', '$PDFLATEXCOMSTR')
    global PDFTeXLaTeXAction
    if PDFTeXLaTeXAction is None:
        PDFTeXLaTeXAction = SCons.Action.Action(PDFTeXLaTeXFunction, strfunction=SCons.Tool.tex.TeXLaTeXStrFunction)
    env.AppendUnique(LATEXSUFFIXES=SCons.Tool.LaTeXSuffixes)
    from . import pdf
    pdf.generate(env)
    bld = env['BUILDERS']['PDF']
    bld.add_action('.tex', PDFTeXLaTeXAction)
    bld.add_emitter('.tex', SCons.Tool.tex.tex_pdf_emitter)
    pdf.generate2(env)
    SCons.Tool.tex.generate_common(env)

def exists(env):
    if False:
        print('Hello World!')
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('pdftex')