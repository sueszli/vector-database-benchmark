import os
from os.path import join
import shutil
import tempfile
try:
    from subprocess import STDOUT, CalledProcessError, check_output
except ImportError:
    pass
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
__doctest_requires__ = {('preview',): ['pyglet']}

def _check_output_no_window(*args, **kwargs):
    if False:
        while True:
            i = 10
    if os.name == 'nt':
        creation_flag = 134217728
    else:
        creation_flag = 0
    return check_output(*args, creationflags=creation_flag, **kwargs)

def system_default_viewer(fname, fmt):
    if False:
        return 10
    ' Open fname with the default system viewer.\n\n    In practice, it is impossible for python to know when the system viewer is\n    done. For this reason, we ensure the passed file will not be deleted under\n    it, and this function does not attempt to block.\n    '
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-', suffix=os.path.splitext(fname)[1], delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)
    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))

def pyglet_viewer(fname, fmt):
    if False:
        for i in range(10):
            print('nop')
    try:
        from pyglet import window, image, gl
        from pyglet.window import key
        from pyglet.image.codecs import ImageDecodeException
    except ImportError:
        raise ImportError('pyglet is required for preview.\n visit https://pyglet.org/')
    try:
        img = image.load(fname)
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))
    offset = 25
    config = gl.Config(double_buffer=False)
    win = window.Window(width=img.width + 2 * offset, height=img.height + 2 * offset, caption='SymPy', resizable=False, config=config)
    win.set_vsync(False)
    try:

        def on_close():
            if False:
                while True:
                    i = 10
            win.has_exit = True
        win.on_close = on_close

        def on_key_press(symbol, modifiers):
            if False:
                return 10
            if symbol in [key.Q, key.ESCAPE]:
                on_close()
        win.on_key_press = on_key_press

        def on_expose():
            if False:
                return 10
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            img.blit((win.width - img.width) / 2, (win.height - img.height) / 2)
        win.on_expose = on_expose
        while not win.has_exit:
            win.dispatch_events()
            win.flip()
    except KeyboardInterrupt:
        pass
    win.close()

def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None, euler=True, fontsize=None, **latex_settings):
    if False:
        i = 10
        return i + 15
    '\n    Generate string of a LaTeX document rendering ``expr``.\n    '
    if preamble is None:
        actual_packages = packages + ('amsmath', 'amsfonts')
        if euler:
            actual_packages += ('euler',)
        package_includes = '\n' + '\n'.join(['\\usepackage{%s}' % p for p in actual_packages])
        if extra_preamble:
            package_includes += extra_preamble
        if not fontsize:
            fontsize = '12pt'
        elif isinstance(fontsize, int):
            fontsize = '{}pt'.format(fontsize)
        preamble = '\\documentclass[varwidth,%s]{standalone}\n%s\n\n\\begin{document}\n' % (fontsize, package_includes)
    elif packages or extra_preamble:
        raise ValueError('The "packages" or "extra_preamble" keywordsmust not be set if a custom LaTeX preamble was specified')
    if isinstance(expr, str):
        latex_string = expr
    else:
        latex_string = '$\\displaystyle ' + latex(expr, mode='plain', **latex_settings) + '$'
    return preamble + '\n' + latex_string + '\n\n' + '\\end{document}'

@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',), disable_viewers=('evince', 'gimp', 'superior-dvi-viewer'))
def preview(expr, output='png', viewer=None, euler=True, packages=(), filename=None, outputbuffer=None, preamble=None, dvioptions=None, outputTexFile=None, extra_preamble=None, fontsize=None, **latex_settings):
    if False:
        return 10
    '\n    View expression or LaTeX markup in PNG, DVI, PostScript or PDF form.\n\n    If the expr argument is an expression, it will be exported to LaTeX and\n    then compiled using the available TeX distribution.  The first argument,\n    \'expr\', may also be a LaTeX string.  The function will then run the\n    appropriate viewer for the given output format or use the user defined\n    one. By default png output is generated.\n\n    By default pretty Euler fonts are used for typesetting (they were used to\n    typeset the well known "Concrete Mathematics" book). For that to work, you\n    need the \'eulervm.sty\' LaTeX style (in Debian/Ubuntu, install the\n    texlive-fonts-extra package). If you prefer default AMS fonts or your\n    system lacks \'eulervm\' LaTeX package then unset the \'euler\' keyword\n    argument.\n\n    To use viewer auto-detection, lets say for \'png\' output, issue\n\n    >>> from sympy import symbols, preview, Symbol\n    >>> x, y = symbols("x,y")\n\n    >>> preview(x + y, output=\'png\')\n\n    This will choose \'pyglet\' by default. To select a different one, do\n\n    >>> preview(x + y, output=\'png\', viewer=\'gimp\')\n\n    The \'png\' format is considered special. For all other formats the rules\n    are slightly different. As an example we will take \'dvi\' output format. If\n    you would run\n\n    >>> preview(x + y, output=\'dvi\')\n\n    then \'view\' will look for available \'dvi\' viewers on your system\n    (predefined in the function, so it will try evince, first, then kdvi and\n    xdvi). If nothing is found, it will fall back to using a system file\n    association (via ``open`` and ``xdg-open``). To always use your system file\n    association without searching for the above readers, use\n\n    >>> from sympy.printing.preview import system_default_viewer\n    >>> preview(x + y, output=\'dvi\', viewer=system_default_viewer)\n\n    If this still does not find the viewer you want, it can be set explicitly.\n\n    >>> preview(x + y, output=\'dvi\', viewer=\'superior-dvi-viewer\')\n\n    This will skip auto-detection and will run user specified\n    \'superior-dvi-viewer\'. If ``view`` fails to find it on your system it will\n    gracefully raise an exception.\n\n    You may also enter ``\'file\'`` for the viewer argument. Doing so will cause\n    this function to return a file object in read-only mode, if ``filename``\n    is unset. However, if it was set, then \'preview\' writes the generated\n    file to this filename instead.\n\n    There is also support for writing to a ``io.BytesIO`` like object, which\n    needs to be passed to the ``outputbuffer`` argument.\n\n    >>> from io import BytesIO\n    >>> obj = BytesIO()\n    >>> preview(x + y, output=\'png\', viewer=\'BytesIO\',\n    ...         outputbuffer=obj)\n\n    The LaTeX preamble can be customized by setting the \'preamble\' keyword\n    argument. This can be used, e.g., to set a different font size, use a\n    custom documentclass or import certain set of LaTeX packages.\n\n    >>> preamble = "\\\\documentclass[10pt]{article}\\n" \\\n    ...            "\\\\usepackage{amsmath,amsfonts}\\\\begin{document}"\n    >>> preview(x + y, output=\'png\', preamble=preamble)\n\n    It is also possible to use the standard preamble and provide additional\n    information to the preamble using the ``extra_preamble`` keyword argument.\n\n    >>> from sympy import sin\n    >>> extra_preamble = "\\\\renewcommand{\\\\sin}{\\\\cos}"\n    >>> preview(sin(x), output=\'png\', extra_preamble=extra_preamble)\n\n    If the value of \'output\' is different from \'dvi\' then command line\n    options can be set (\'dvioptions\' argument) for the execution of the\n    \'dvi\'+output conversion tool. These options have to be in the form of a\n    list of strings (see ``subprocess.Popen``).\n\n    Additional keyword args will be passed to the :func:`~sympy.printing.latex.latex` call,\n    e.g., the ``symbol_names`` flag.\n\n    >>> phidd = Symbol(\'phidd\')\n    >>> preview(phidd, symbol_names={phidd: r\'\\ddot{\\varphi}\'})\n\n    For post-processing the generated TeX File can be written to a file by\n    passing the desired filename to the \'outputTexFile\' keyword\n    argument. To write the TeX code to a file named\n    ``"sample.tex"`` and run the default png viewer to display the resulting\n    bitmap, do\n\n    >>> preview(x + y, outputTexFile="sample.tex")\n\n\n    '
    if viewer is None and output == 'png':
        try:
            import pyglet
        except ImportError:
            pass
        else:
            viewer = pyglet_viewer
    if viewer is None:
        candidates = {'dvi': ['evince', 'okular', 'kdvi', 'xdvi'], 'ps': ['evince', 'okular', 'gsview', 'gv'], 'pdf': ['evince', 'okular', 'kpdf', 'acroread', 'xpdf', 'gv']}
        for candidate in candidates.get(output, []):
            path = shutil.which(candidate)
            if path is not None:
                viewer = path
                break
    if viewer is None:
        viewer = system_default_viewer
    if viewer == 'file':
        if filename is None:
            raise ValueError('filename has to be specified if viewer="file"')
    elif viewer == 'BytesIO':
        if outputbuffer is None:
            raise ValueError('outputbuffer has to be a BytesIO compatible object if viewer="BytesIO"')
    elif not callable(viewer) and (not shutil.which(viewer)):
        raise OSError('Unrecognized viewer: %s' % viewer)
    latex_main = _get_latex_main(expr, preamble=preamble, packages=packages, euler=euler, extra_preamble=extra_preamble, fontsize=fontsize, **latex_settings)
    debug('Latex code:')
    debug(latex_main)
    with tempfile.TemporaryDirectory() as workdir:
        with open(join(workdir, 'texput.tex'), 'w', encoding='utf-8') as fh:
            fh.write(latex_main)
        if outputTexFile is not None:
            shutil.copyfile(join(workdir, 'texput.tex'), outputTexFile)
        if not shutil.which('latex'):
            raise RuntimeError('latex program is not installed')
        try:
            _check_output_no_window(['latex', '-halt-on-error', '-interaction=nonstopmode', 'texput.tex'], cwd=workdir, stderr=STDOUT)
        except CalledProcessError as e:
            raise RuntimeError("'latex' exited abnormally with the following output:\n%s" % e.output)
        src = 'texput.%s' % output
        if output != 'dvi':
            commandnames = {'ps': ['dvips'], 'pdf': ['dvipdfmx', 'dvipdfm', 'dvipdf'], 'png': ['dvipng'], 'svg': ['dvisvgm']}
            try:
                cmd_variants = commandnames[output]
            except KeyError:
                raise ValueError('Invalid output format: %s' % output) from None
            for cmd_variant in cmd_variants:
                cmd_path = shutil.which(cmd_variant)
                if cmd_path:
                    cmd = [cmd_path]
                    break
            else:
                if len(cmd_variants) > 1:
                    raise RuntimeError('None of %s are installed' % ', '.join(cmd_variants))
                else:
                    raise RuntimeError('%s is not installed' % cmd_variants[0])
            defaultoptions = {'dvipng': ['-T', 'tight', '-z', '9', '--truecolor'], 'dvisvgm': ['--no-fonts']}
            commandend = {'dvips': ['-o', src, 'texput.dvi'], 'dvipdf': ['texput.dvi', src], 'dvipdfm': ['-o', src, 'texput.dvi'], 'dvipdfmx': ['-o', src, 'texput.dvi'], 'dvipng': ['-o', src, 'texput.dvi'], 'dvisvgm': ['-o', src, 'texput.dvi']}
            if dvioptions is not None:
                cmd.extend(dvioptions)
            else:
                cmd.extend(defaultoptions.get(cmd_variant, []))
            cmd.extend(commandend[cmd_variant])
            try:
                _check_output_no_window(cmd, cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError("'%s' exited abnormally with the following output:\n%s" % (' '.join(cmd), e.output))
        if viewer == 'file':
            shutil.move(join(workdir, src), filename)
        elif viewer == 'BytesIO':
            with open(join(workdir, src), 'rb') as fh:
                outputbuffer.write(fh.read())
        elif callable(viewer):
            viewer(join(workdir, src), fmt=output)
        else:
            try:
                _check_output_no_window([viewer, src], cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError("'%s %s' exited abnormally with the following output:\n%s" % (viewer, src, e.output))