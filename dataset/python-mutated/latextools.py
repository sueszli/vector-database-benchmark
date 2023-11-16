"""Tools for handling LaTeX."""
from io import BytesIO, open
import os
import tempfile
import shutil
import subprocess
from base64 import encodebytes
import textwrap
from pathlib import Path
from IPython.utils.process import find_cmd, FindCmdError
from traitlets.config import get_config
from traitlets.config.configurable import SingletonConfigurable
from traitlets import List, Bool, Unicode
from IPython.utils.py3compat import cast_unicode

class LaTeXTool(SingletonConfigurable):
    """An object to store configuration of the LaTeX tool."""

    def _config_default(self):
        if False:
            for i in range(10):
                print('nop')
        return get_config()
    backends = List(Unicode(), ['matplotlib', 'dvipng'], help='Preferred backend to draw LaTeX math equations. Backends in the list are checked one by one and the first usable one is used.  Note that `matplotlib` backend is usable only for inline style equations.  To draw  display style equations, `dvipng` backend must be specified. ').tag(config=True)
    use_breqn = Bool(True, help='Use breqn.sty to automatically break long equations. This configuration takes effect only for dvipng backend.').tag(config=True)
    packages = List(['amsmath', 'amsthm', 'amssymb', 'bm'], help="A list of packages to use for dvipng backend. 'breqn' will be automatically appended when use_breqn=True.").tag(config=True)
    preamble = Unicode(help='Additional preamble to use when generating LaTeX source for dvipng backend.').tag(config=True)

def latex_to_png(s, encode=False, backend=None, wrap=False, color='Black', scale=1.0):
    if False:
        while True:
            i = 10
    "Render a LaTeX string to PNG.\n\n    Parameters\n    ----------\n    s : str\n        The raw string containing valid inline LaTeX.\n    encode : bool, optional\n        Should the PNG data base64 encoded to make it JSON'able.\n    backend : {matplotlib, dvipng}\n        Backend for producing PNG data.\n    wrap : bool\n        If true, Automatically wrap `s` as a LaTeX equation.\n    color : string\n        Foreground color name among dvipsnames, e.g. 'Maroon' or on hex RGB\n        format, e.g. '#AA20FA'.\n    scale : float\n        Scale factor for the resulting PNG.\n    None is returned when the backend cannot be used.\n\n    "
    s = cast_unicode(s)
    allowed_backends = LaTeXTool.instance().backends
    if backend is None:
        backend = allowed_backends[0]
    if backend not in allowed_backends:
        return None
    if backend == 'matplotlib':
        f = latex_to_png_mpl
    elif backend == 'dvipng':
        f = latex_to_png_dvipng
        if color.startswith('#'):
            if len(color) == 7:
                try:
                    color = 'RGB {}'.format(' '.join([str(int(x, 16)) for x in textwrap.wrap(color[1:], 2)]))
                except ValueError as e:
                    raise ValueError('Invalid color specification {}.'.format(color)) from e
            else:
                raise ValueError('Invalid color specification {}.'.format(color))
    else:
        raise ValueError('No such backend {0}'.format(backend))
    bin_data = f(s, wrap, color, scale)
    if encode and bin_data:
        bin_data = encodebytes(bin_data)
    return bin_data

def latex_to_png_mpl(s, wrap, color='Black', scale=1.0):
    if False:
        i = 10
        return i + 15
    try:
        from matplotlib import figure, font_manager, mathtext
        from matplotlib.backends import backend_agg
        from pyparsing import ParseFatalException
    except ImportError:
        return None
    s = s.replace('$$', '$')
    if wrap:
        s = u'${0}$'.format(s)
    try:
        prop = font_manager.FontProperties(size=12)
        dpi = 120 * scale
        buffer = BytesIO()
        parser = mathtext.MathTextParser('path')
        (width, height, depth, _, _) = parser.parse(s, dpi=72, prop=prop)
        fig = figure.Figure(figsize=(width / 72, height / 72))
        fig.text(0, depth / height, s, fontproperties=prop, color=color)
        backend_agg.FigureCanvasAgg(fig)
        fig.savefig(buffer, dpi=dpi, format='png', transparent=True)
        return buffer.getvalue()
    except (ValueError, RuntimeError, ParseFatalException):
        return None

def latex_to_png_dvipng(s, wrap, color='Black', scale=1.0):
    if False:
        for i in range(10):
            print('nop')
    try:
        find_cmd('latex')
        find_cmd('dvipng')
    except FindCmdError:
        return None
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        workdir = Path(tempfile.mkdtemp())
        tmpfile = 'tmp.tex'
        dvifile = 'tmp.dvi'
        outfile = 'tmp.png'
        with workdir.joinpath(tmpfile).open('w', encoding='utf8') as f:
            f.writelines(genelatex(s, wrap))
        subprocess.check_call(['latex', '-halt-on-error', '-interaction', 'batchmode', tmpfile], cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        resolution = round(150 * scale)
        subprocess.check_call(['dvipng', '-T', 'tight', '-D', str(resolution), '-z', '9', '-bg', 'Transparent', '-o', outfile, dvifile, '-fg', color], cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        with workdir.joinpath(outfile).open('rb') as f:
            return f.read()
    except subprocess.CalledProcessError:
        return None
    finally:
        shutil.rmtree(workdir)

def kpsewhich(filename):
    if False:
        for i in range(10):
            print('nop')
    'Invoke kpsewhich command with an argument `filename`.'
    try:
        find_cmd('kpsewhich')
        proc = subprocess.Popen(['kpsewhich', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = proc.communicate()
        return stdout.strip().decode('utf8', 'replace')
    except FindCmdError:
        pass

def genelatex(body, wrap):
    if False:
        return 10
    'Generate LaTeX document for dvipng backend.'
    lt = LaTeXTool.instance()
    breqn = wrap and lt.use_breqn and kpsewhich('breqn.sty')
    yield '\\documentclass{article}'
    packages = lt.packages
    if breqn:
        packages = packages + ['breqn']
    for pack in packages:
        yield '\\usepackage{{{0}}}'.format(pack)
    yield '\\pagestyle{empty}'
    if lt.preamble:
        yield lt.preamble
    yield '\\begin{document}'
    if breqn:
        yield '\\begin{dmath*}'
        yield body
        yield '\\end{dmath*}'
    elif wrap:
        yield u'$${0}$$'.format(body)
    else:
        yield body
    yield u'\\end{document}'
_data_uri_template_png = u'<img src="data:image/png;base64,%s" alt=%s />'

def latex_to_html(s, alt='image'):
    if False:
        for i in range(10):
            print('nop')
    'Render LaTeX to HTML with embedded PNG data using data URIs.\n\n    Parameters\n    ----------\n    s : str\n        The raw string containing valid inline LateX.\n    alt : str\n        The alt text to use for the HTML.\n    '
    base64_data = latex_to_png(s, encode=True).decode('ascii')
    if base64_data:
        return _data_uri_template_png % (base64_data, alt)