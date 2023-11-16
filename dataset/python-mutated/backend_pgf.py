import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import _create_pdf_info_dict, _datetime_to_pdf
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
_log = logging.getLogger(__name__)

def _get_preamble():
    if False:
        while True:
            i = 10
    'Prepare a LaTeX preamble based on the rcParams configuration.'
    return '\n'.join(['\\def\\mathdefault#1{#1}', '\\everymath=\\expandafter{\\the\\everymath\\displaystyle}', mpl.rcParams['pgf.preamble'], '\\ifdefined\\pdftexversion\\else  % non-pdftex case.', '  \\usepackage{fontspec}', *(['  \\%s{%s}[Path=\\detokenize{%s/}]' % (command, path.name, path.parent.as_posix()) for (command, path) in zip(['setmainfont', 'setsansfont', 'setmonofont'], [pathlib.Path(fm.findfont(family)) for family in ['serif', 'sans\\-serif', 'monospace']])] if mpl.rcParams['pgf.rcfonts'] else []), '\\fi', mpl.texmanager._usepackage_if_not_loaded('underscore', option='strings')])
latex_pt_to_in = 1.0 / 72.27
latex_in_to_pt = 1.0 / latex_pt_to_in
mpl_pt_to_in = 1.0 / 72.0
mpl_in_to_pt = 1.0 / mpl_pt_to_in

def _tex_escape(text):
    if False:
        print('Hello World!')
    '\n    Do some necessary and/or useful substitutions for texts to be included in\n    LaTeX documents.\n    '
    return text.replace('−', '\\ensuremath{-}')

def _writeln(fh, line):
    if False:
        while True:
            i = 10
    fh.write(line)
    fh.write('%\n')

def _escape_and_apply_props(s, prop):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a TeX string that renders string *s* with font properties *prop*,\n    also applying any required escapes to *s*.\n    '
    commands = []
    families = {'serif': '\\rmfamily', 'sans': '\\sffamily', 'sans-serif': '\\sffamily', 'monospace': '\\ttfamily'}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif any((font.name == family for font in fm.fontManager.ttflist)):
        commands.append('\\ifdefined\\pdftexversion\\else\\setmainfont{%s}\\rmfamily\\fi' % family)
    else:
        _log.warning('Ignoring unknown font: %s', family)
    size = prop.get_size_in_points()
    commands.append('\\fontsize{%f}{%f}' % (size, size * 1.2))
    styles = {'normal': '', 'italic': '\\itshape', 'oblique': '\\slshape'}
    commands.append(styles[prop.get_style()])
    boldstyles = ['semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
    if prop.get_weight() in boldstyles:
        commands.append('\\bfseries')
    commands.append('\\selectfont')
    return '{' + ''.join(commands) + '\\catcode`\\^=\\active\\def^{\\ifmmode\\sp\\else\\^{}\\fi}' + '\\catcode`\\%=\\active\\def%{\\%}' + _tex_escape(s) + '}'

def _metadata_to_str(key, value):
    if False:
        return 10
    'Convert metadata key/value to a form that hyperref accepts.'
    if isinstance(value, datetime.datetime):
        value = _datetime_to_pdf(value)
    elif key == 'Trapped':
        value = value.name.decode('ascii')
    else:
        value = str(value)
    return f'{key}={{{value}}}'

def make_pdf_to_png_converter():
    if False:
        i = 10
        return i + 15
    'Return a function that converts a pdf file to a png file.'
    try:
        mpl._get_executable_info('pdftocairo')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(['pdftocairo', '-singlefile', '-transp', '-png', '-r', '%d' % dpi, pdffile, os.path.splitext(pngfile)[0]], stderr=subprocess.STDOUT)
    try:
        gs_info = mpl._get_executable_info('gs')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output([gs_info.executable, '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT', '-dUseCIEColor', '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE', '-sDEVICE=pngalpha', '-sOutputFile=%s' % pngfile, '-r%d' % dpi, pdffile], stderr=subprocess.STDOUT)
    raise RuntimeError('No suitable pdf to png renderer found.')

class LatexError(Exception):

    def __init__(self, message, latex_output=''):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(message)
        self.latex_output = latex_output

    def __str__(self):
        if False:
            while True:
                i = 10
        (s,) = self.args
        if self.latex_output:
            s += '\n' + self.latex_output
        return s

class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custom preamble in `.rcParams`.
    """

    @staticmethod
    def _build_latex_header():
        if False:
            for i in range(10):
                print('nop')
        latex_header = ['\\documentclass{article}', f"% !TeX program = {mpl.rcParams['pgf.texsystem']}", '\\usepackage{graphicx}', _get_preamble(), '\\begin{document}', '\\typeout{pgf_backend_query_start}']
        return '\n'.join(latex_header)

    @classmethod
    def _get_cached_or_new(cls):
        if False:
            i = 10
            return i + 15
        '\n        Return the previous LatexManager if the header and tex system did not\n        change, or a new instance otherwise.\n        '
        return cls._get_cached_or_new_impl(cls._build_latex_header())

    @classmethod
    @functools.lru_cache(1)
    def _get_cached_or_new_impl(cls, header):
        if False:
            for i in range(10):
                print('nop')
        return cls()

    def _stdin_writeln(self, s):
        if False:
            while True:
                i = 10
        if self.latex is None:
            self._setup_latex_process()
        self.latex.stdin.write(s)
        self.latex.stdin.write('\n')
        self.latex.stdin.flush()

    def _expect(self, s):
        if False:
            return 10
        s = list(s)
        chars = []
        while True:
            c = self.latex.stdout.read(1)
            chars.append(c)
            if chars[-len(s):] == s:
                break
            if not c:
                self.latex.kill()
                self.latex = None
                raise LatexError('LaTeX process halted', ''.join(chars))
        return ''.join(chars)

    def _expect_prompt(self):
        if False:
            while True:
                i = 10
        return self._expect('\n*')

    def __init__(self):
        if False:
            while True:
                i = 10
        self._tmpdir = TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self._finalize_tmpdir = weakref.finalize(self, self._tmpdir.cleanup)
        self._setup_latex_process(expect_reply=False)
        (stdout, stderr) = self.latex.communicate('\n\\makeatletter\\@@end\n')
        if self.latex.returncode != 0:
            raise LatexError(f'LaTeX errored (probably missing font or error in preamble) while processing the following input:\n{self._build_latex_header()}', stdout)
        self.latex = None
        self._get_box_metrics = functools.lru_cache(self._get_box_metrics)

    def _setup_latex_process(self, *, expect_reply=True):
        if False:
            print('Hello World!')
        try:
            self.latex = subprocess.Popen([mpl.rcParams['pgf.texsystem'], '-halt-on-error'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8', cwd=self.tmpdir)
        except FileNotFoundError as err:
            raise RuntimeError(f"{mpl.rcParams['pgf.texsystem']!r} not found; install it or change rcParams['pgf.texsystem'] to an available TeX implementation") from err
        except OSError as err:
            raise RuntimeError(f"Error starting {mpl.rcParams['pgf.texsystem']!r}") from err

        def finalize_latex(latex):
            if False:
                return 10
            latex.kill()
            latex.communicate()
        self._finalize_latex = weakref.finalize(self, finalize_latex, self.latex)
        self._stdin_writeln(self._build_latex_header())
        if expect_reply:
            self._expect('*pgf_backend_query_start')
            self._expect_prompt()

    def get_width_height_descent(self, text, prop):
        if False:
            i = 10
            return i + 15
        '\n        Get the width, total height, and descent (in TeX points) for a text\n        typeset by the current LaTeX environment.\n        '
        return self._get_box_metrics(_escape_and_apply_props(text, prop))

    def _get_box_metrics(self, tex):
        if False:
            print('Hello World!')
        "\n        Get the width, total height and descent (in TeX points) for a TeX\n        command's output in the current LaTeX environment.\n        "
        self._stdin_writeln('{\\catcode`\\^=\\active\\catcode`\\%%=\\active\\sbox0{%s}\\typeout{\\the\\wd0,\\the\\ht0,\\the\\dp0}}' % tex)
        try:
            answer = self._expect_prompt()
        except LatexError as err:
            raise ValueError('Error measuring {}\nLaTeX Output:\n{}'.format(tex, err.latex_output)) from err
        try:
            (width, height, offset) = answer.splitlines()[-3].split(',')
        except Exception as err:
            raise ValueError('Error measuring {}\nLaTeX Output:\n{}'.format(tex, answer)) from err
        (w, h, o) = (float(width[:-2]), float(height[:-2]), float(offset[:-2]))
        return (w, h + o, o)

@functools.lru_cache(1)
def _get_image_inclusion_command():
    if False:
        print('Hello World!')
    man = LatexManager._get_cached_or_new()
    man._stdin_writeln('\\includegraphics[interpolate=true]{%s}' % cbook._get_data_path('images/matplotlib.png').as_posix())
    try:
        man._expect_prompt()
        return '\\includegraphics'
    except LatexError:
        LatexManager._get_cached_or_new_impl.cache_clear()
        return '\\pgfimage'

class RendererPgf(RendererBase):

    def __init__(self, figure, fh):
        if False:
            while True:
                i = 10
        '\n        Create a new PGF renderer that translates any drawing instruction\n        into text commands to be interpreted in a latex pgfpicture environment.\n\n        Attributes\n        ----------\n        figure : `~matplotlib.figure.Figure`\n            Matplotlib figure to initialize height, width and dpi from.\n        fh : file-like\n            File handle for the output of the drawing commands.\n        '
        super().__init__()
        self.dpi = figure.dpi
        self.fh = fh
        self.figure = figure
        self.image_counter = 0

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        if False:
            for i in range(10):
                print('nop')
        _writeln(self.fh, '\\begin{pgfscope}')
        f = 1.0 / self.dpi
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        (bl, tr) = marker_path.get_extents(marker_trans).get_points()
        coords = (bl[0] * f, bl[1] * f, tr[0] * f, tr[1] * f)
        _writeln(self.fh, '\\pgfsys@defobject{currentmarker}{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}{' % coords)
        self._print_pgf_path(None, marker_path, marker_trans)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0, fill=rgbFace is not None)
        _writeln(self.fh, '}')
        maxcoord = 16383 / 72.27 * self.dpi
        clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
        for (point, code) in path.iter_segments(trans, simplify=False, clip=clip):
            (x, y) = (point[0] * f, point[1] * f)
            _writeln(self.fh, '\\begin{pgfscope}')
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (x, y))
            _writeln(self.fh, '\\pgfsys@useobject{currentmarker}{}')
            _writeln(self.fh, '\\end{pgfscope}')
        _writeln(self.fh, '\\end{pgfscope}')

    def draw_path(self, gc, path, transform, rgbFace=None):
        if False:
            return 10
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        self._print_pgf_path(gc, path, transform, rgbFace)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0, fill=rgbFace is not None)
        _writeln(self.fh, '\\end{pgfscope}')
        if gc.get_hatch():
            _writeln(self.fh, '\\begin{pgfscope}')
            self._print_pgf_path_styles(gc, rgbFace)
            self._print_pgf_clip(gc)
            self._print_pgf_path(gc, path, transform, rgbFace)
            _writeln(self.fh, '\\pgfusepath{clip}')
            _writeln(self.fh, '\\pgfsys@defobject{currentpattern}{\\pgfqpoint{0in}{0in}}{\\pgfqpoint{1in}{1in}}{')
            _writeln(self.fh, '\\begin{pgfscope}')
            _writeln(self.fh, '\\pgfpathrectangle{\\pgfqpoint{0in}{0in}}{\\pgfqpoint{1in}{1in}}')
            _writeln(self.fh, '\\pgfusepath{clip}')
            scale = mpl.transforms.Affine2D().scale(self.dpi)
            self._print_pgf_path(None, gc.get_hatch_path(), scale)
            self._pgf_path_draw(stroke=True)
            _writeln(self.fh, '\\end{pgfscope}')
            _writeln(self.fh, '}')
            f = 1.0 / self.dpi
            ((xmin, ymin), (xmax, ymax)) = path.get_extents(transform).get_points()
            (xmin, xmax) = (f * xmin, f * xmax)
            (ymin, ymax) = (f * ymin, f * ymax)
            (repx, repy) = (math.ceil(xmax - xmin), math.ceil(ymax - ymin))
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (xmin, ymin))
            for iy in range(repy):
                for ix in range(repx):
                    _writeln(self.fh, '\\pgfsys@useobject{currentpattern}{}')
                    _writeln(self.fh, '\\pgfsys@transformshift{1in}{0in}')
                _writeln(self.fh, '\\pgfsys@transformshift{-%din}{0in}' % repx)
                _writeln(self.fh, '\\pgfsys@transformshift{0in}{1in}')
            _writeln(self.fh, '\\end{pgfscope}')

    def _print_pgf_clip(self, gc):
        if False:
            for i in range(10):
                print('nop')
        f = 1.0 / self.dpi
        bbox = gc.get_clip_rectangle()
        if bbox:
            (p1, p2) = bbox.get_points()
            (w, h) = p2 - p1
            coords = (p1[0] * f, p1[1] * f, w * f, h * f)
            _writeln(self.fh, '\\pgfpathrectangle{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
            _writeln(self.fh, '\\pgfusepath{clip}')
        (clippath, clippath_trans) = gc.get_clip_path()
        if clippath is not None:
            self._print_pgf_path(gc, clippath, clippath_trans)
            _writeln(self.fh, '\\pgfusepath{clip}')

    def _print_pgf_path_styles(self, gc, rgbFace):
        if False:
            while True:
                i = 10
        capstyles = {'butt': '\\pgfsetbuttcap', 'round': '\\pgfsetroundcap', 'projecting': '\\pgfsetrectcap'}
        _writeln(self.fh, capstyles[gc.get_capstyle()])
        joinstyles = {'miter': '\\pgfsetmiterjoin', 'round': '\\pgfsetroundjoin', 'bevel': '\\pgfsetbeveljoin'}
        _writeln(self.fh, joinstyles[gc.get_joinstyle()])
        has_fill = rgbFace is not None
        if gc.get_forced_alpha():
            fillopacity = strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]
            fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0
        if has_fill:
            _writeln(self.fh, '\\definecolor{currentfill}{rgb}{%f,%f,%f}' % tuple(rgbFace[:3]))
            _writeln(self.fh, '\\pgfsetfillcolor{currentfill}')
        if has_fill and fillopacity != 1.0:
            _writeln(self.fh, '\\pgfsetfillopacity{%f}' % fillopacity)
        lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
        stroke_rgba = gc.get_rgb()
        _writeln(self.fh, '\\pgfsetlinewidth{%fpt}' % lw)
        _writeln(self.fh, '\\definecolor{currentstroke}{rgb}{%f,%f,%f}' % stroke_rgba[:3])
        _writeln(self.fh, '\\pgfsetstrokecolor{currentstroke}')
        if strokeopacity != 1.0:
            _writeln(self.fh, '\\pgfsetstrokeopacity{%f}' % strokeopacity)
        (dash_offset, dash_list) = gc.get_dashes()
        if dash_list is None:
            _writeln(self.fh, '\\pgfsetdash{}{0pt}')
        else:
            _writeln(self.fh, '\\pgfsetdash{%s}{%fpt}' % (''.join(('{%fpt}' % dash for dash in dash_list)), dash_offset))

    def _print_pgf_path(self, gc, path, transform, rgbFace=None):
        if False:
            for i in range(10):
                print('nop')
        f = 1.0 / self.dpi
        bbox = gc.get_clip_rectangle() if gc else None
        maxcoord = 16383 / 72.27 * self.dpi
        if bbox and rgbFace is None:
            (p1, p2) = bbox.get_points()
            clip = (max(p1[0], -maxcoord), max(p1[1], -maxcoord), min(p2[0], maxcoord), min(p2[1], maxcoord))
        else:
            clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
        for (points, code) in path.iter_segments(transform, clip=clip):
            if code == Path.MOVETO:
                (x, y) = tuple(points)
                _writeln(self.fh, '\\pgfpathmoveto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
            elif code == Path.CLOSEPOLY:
                _writeln(self.fh, '\\pgfpathclose')
            elif code == Path.LINETO:
                (x, y) = tuple(points)
                _writeln(self.fh, '\\pgfpathlineto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
            elif code == Path.CURVE3:
                (cx, cy, px, py) = tuple(points)
                coords = (cx * f, cy * f, px * f, py * f)
                _writeln(self.fh, '\\pgfpathquadraticcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
            elif code == Path.CURVE4:
                (c1x, c1y, c2x, c2y, px, py) = tuple(points)
                coords = (c1x * f, c1y * f, c2x * f, c2y * f, px * f, py * f)
                _writeln(self.fh, '\\pgfpathcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
        sketch_params = gc.get_sketch_params() if gc else None
        if sketch_params is not None:
            (scale, length, randomness) = sketch_params
            if scale is not None:
                length *= 0.5
                scale *= 2
                _writeln(self.fh, '\\usepgfmodule{decorations}')
                _writeln(self.fh, '\\usepgflibrary{decorations.pathmorphing}')
                _writeln(self.fh, f'\\pgfkeys{{/pgf/decoration/.cd, segment length = {length * f:f}in, amplitude = {scale * f:f}in}}')
                _writeln(self.fh, f'\\pgfmathsetseed{{{int(randomness)}}}')
                _writeln(self.fh, '\\pgfdecoratecurrentpath{random steps}')

    def _pgf_path_draw(self, stroke=True, fill=False):
        if False:
            for i in range(10):
                print('nop')
        actions = []
        if stroke:
            actions.append('stroke')
        if fill:
            actions.append('fill')
        _writeln(self.fh, '\\pgfusepath{%s}' % ','.join(actions))

    def option_scale_image(self):
        if False:
            i = 10
            return i + 15
        return True

    def option_image_nocomposite(self):
        if False:
            for i in range(10):
                print('nop')
        return not mpl.rcParams['image.composite_image']

    def draw_image(self, gc, x, y, im, transform=None):
        if False:
            for i in range(10):
                print('nop')
        (h, w) = im.shape[:2]
        if w == 0 or h == 0:
            return
        if not os.path.exists(getattr(self.fh, 'name', '')):
            raise ValueError('streamed pgf-code does not support raster graphics, consider using the pgf-to-pdf option')
        path = pathlib.Path(self.fh.name)
        fname_img = '%s-img%d.png' % (path.stem, self.image_counter)
        Image.fromarray(im[::-1]).save(path.parent / fname_img)
        self.image_counter += 1
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        f = 1.0 / self.dpi
        if transform is None:
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (x * f, y * f))
            (w, h) = (w * f, h * f)
        else:
            (tr1, tr2, tr3, tr4, tr5, tr6) = transform.frozen().to_values()
            _writeln(self.fh, '\\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}' % (tr1 * f, tr2 * f, tr3 * f, tr4 * f, (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1
        interp = str(transform is None).lower()
        _writeln(self.fh, '\\pgftext[left,bottom]{%s[interpolate=%s,width=%fin,height=%fin]{%s}}' % (_get_image_inclusion_command(), interp, w, h, fname_img))
        _writeln(self.fh, '\\end{pgfscope}')

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        if False:
            while True:
                i = 10
        self.draw_text(gc, x, y, s, prop, angle, ismath='TeX', mtext=mtext)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if False:
            print('Hello World!')
        s = _escape_and_apply_props(s, prop)
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        alpha = gc.get_alpha()
        if alpha != 1.0:
            _writeln(self.fh, '\\pgfsetfillopacity{%f}' % alpha)
            _writeln(self.fh, '\\pgfsetstrokeopacity{%f}' % alpha)
        rgb = tuple(gc.get_rgb())[:3]
        _writeln(self.fh, '\\definecolor{textcolor}{rgb}{%f,%f,%f}' % rgb)
        _writeln(self.fh, '\\pgfsetstrokecolor{textcolor}')
        _writeln(self.fh, '\\pgfsetfillcolor{textcolor}')
        s = '\\color{textcolor}' + s
        dpi = self.figure.dpi
        text_args = []
        if mtext and ((angle == 0 or mtext.get_rotation_mode() == 'anchor') and mtext.get_verticalalignment() != 'center_baseline'):
            pos = mtext.get_unitless_position()
            (x, y) = mtext.get_transform().transform(pos)
            halign = {'left': 'left', 'right': 'right', 'center': ''}
            valign = {'top': 'top', 'bottom': 'bottom', 'baseline': 'base', 'center': ''}
            text_args.extend([f'x={x / dpi:f}in', f'y={y / dpi:f}in', halign[mtext.get_horizontalalignment()], valign[mtext.get_verticalalignment()]])
        else:
            text_args.append(f'x={x / dpi:f}in, y={y / dpi:f}in, left, base')
        if angle != 0:
            text_args.append('rotate=%f' % angle)
        _writeln(self.fh, '\\pgftext[%s]{%s}' % (','.join(text_args), s))
        _writeln(self.fh, '\\end{pgfscope}')

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            print('Hello World!')
        (w, h, d) = LatexManager._get_cached_or_new().get_width_height_descent(s, prop)
        f = mpl_pt_to_in * self.dpi
        return (w * f, h * f, d * f)

    def flipy(self):
        if False:
            print('Hello World!')
        return False

    def get_canvas_width_height(self):
        if False:
            i = 10
            return i + 15
        return (self.figure.get_figwidth() * self.dpi, self.figure.get_figheight() * self.dpi)

    def points_to_pixels(self, points):
        if False:
            i = 10
            return i + 15
        return points * mpl_pt_to_in * self.dpi

class FigureCanvasPgf(FigureCanvasBase):
    filetypes = {'pgf': 'LaTeX PGF picture', 'pdf': 'LaTeX compiled PGF picture', 'png': 'Portable Network Graphics'}

    def get_default_filetype(self):
        if False:
            print('Hello World!')
        return 'pdf'

    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None):
        if False:
            i = 10
            return i + 15
        header_text = '%% Creator: Matplotlib, PGF backend\n%%\n%% To include the figure in your LaTeX document, write\n%%   \\input{<filename>.pgf}\n%%\n%% Make sure the required packages are loaded in your preamble\n%%   \\usepackage{pgf}\n%%\n%% Also ensure that all the required font packages are loaded; for instance,\n%% the lmodern package is sometimes necessary when using math font.\n%%   \\usepackage{lmodern}\n%%\n%% Figures using additional raster images can only be included by \\input if\n%% they are in the same directory as the main LaTeX file. For loading figures\n%% from other directories you can use the `import` package\n%%   \\usepackage{import}\n%%\n%% and then include the figures with\n%%   \\import{<path to file>}{<filename>.pgf}\n%%\n'
        header_info_preamble = ['%% Matplotlib used the following preamble']
        for line in _get_preamble().splitlines():
            header_info_preamble.append('%%   ' + line)
        header_info_preamble.append('%%')
        header_info_preamble = '\n'.join(header_info_preamble)
        (w, h) = (self.figure.get_figwidth(), self.figure.get_figheight())
        dpi = self.figure.dpi
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write('\n')
        _writeln(fh, '\\begingroup')
        _writeln(fh, '\\makeatletter')
        _writeln(fh, '\\begin{pgfpicture}')
        _writeln(fh, '\\pgfpathrectangle{\\pgfpointorigin}{\\pgfqpoint{%fin}{%fin}}' % (w, h))
        _writeln(fh, '\\pgfusepath{use as bounding box, clip}')
        renderer = MixedModeRenderer(self.figure, w, h, dpi, RendererPgf(self.figure, fh), bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)
        _writeln(fh, '\\end{pgfpicture}')
        _writeln(fh, '\\makeatother')
        _writeln(fh, '\\endgroup')

    def print_pgf(self, fname_or_fh, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Output pgf macros for drawing the figure so it can be included and\n        rendered in latex documents.\n        '
        with cbook.open_file_cm(fname_or_fh, 'w', encoding='utf-8') as file:
            if not cbook.file_requires_unicode(file):
                file = codecs.getwriter('utf-8')(file)
            self._print_pgf_to_fh(file, **kwargs)

    def print_pdf(self, fname_or_fh, *, metadata=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Use LaTeX to compile a pgf generated figure to pdf.'
        (w, h) = self.figure.get_size_inches()
        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        pdfinfo = ','.join((_metadata_to_str(k, v) for (k, v) in info_dict.items()))
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            self.print_pgf(tmppath / 'figure.pgf', **kwargs)
            (tmppath / 'figure.tex').write_text('\n'.join(['\\documentclass[12pt]{article}', '\\usepackage[pdfinfo={%s}]{hyperref}' % pdfinfo, '\\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}' % (w, h), '\\usepackage{pgf}', _get_preamble(), '\\begin{document}', '\\centering', '\\input{figure.pgf}', '\\end{document}']), encoding='utf-8')
            texcommand = mpl.rcParams['pgf.texsystem']
            cbook._check_and_log_subprocess([texcommand, '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'], _log, cwd=tmpdir)
            with (tmppath / 'figure.pdf').open('rb') as orig, cbook.open_file_cm(fname_or_fh, 'wb') as dest:
                shutil.copyfileobj(orig, dest)

    def print_png(self, fname_or_fh, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Use LaTeX to compile a pgf figure to pdf and convert it to png.'
        converter = make_pdf_to_png_converter()
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            pdf_path = tmppath / 'figure.pdf'
            png_path = tmppath / 'figure.png'
            self.print_pdf(pdf_path, **kwargs)
            converter(pdf_path, png_path, dpi=self.figure.dpi)
            with png_path.open('rb') as orig, cbook.open_file_cm(fname_or_fh, 'wb') as dest:
                shutil.copyfileobj(orig, dest)

    def get_renderer(self):
        if False:
            return 10
        return RendererPgf(self.figure, None)

    def draw(self):
        if False:
            return 10
        self.figure.draw_without_rendering()
        return super().draw()
FigureManagerPgf = FigureManagerBase

@_Backend.export
class _BackendPgf(_Backend):
    FigureCanvas = FigureCanvasPgf

class PdfPages:
    """
    A multi-page PDF file using the pgf backend

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Initialize:
    >>> with PdfPages('foo.pdf') as pdf:
    ...     # As many times as you like, create a figure fig and save it:
    ...     fig = plt.figure()
    ...     pdf.savefig(fig)
    ...     # When no figure is specified the current figure is saved
    ...     pdf.savefig()
    """
    _UNSET = object()

    def __init__(self, filename, *, keep_empty=_UNSET, metadata=None):
        if False:
            print('Hello World!')
        "\n        Create a new PdfPages object.\n\n        Parameters\n        ----------\n        filename : str or path-like\n            Plots using `PdfPages.savefig` will be written to a file at this\n            location. Any older file with the same name is overwritten.\n\n        keep_empty : bool, default: True\n            If set to False, then empty pdf files will be deleted automatically\n            when closed.\n\n        metadata : dict, optional\n            Information dictionary object (see PDF reference section 10.2.1\n            'Document Information Dictionary'), e.g.:\n            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.\n\n            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',\n            'Creator', 'Producer', 'CreationDate', 'ModDate', and\n            'Trapped'. Values have been predefined for 'Creator', 'Producer'\n            and 'CreationDate'. They can be removed by setting them to `None`.\n\n            Note that some versions of LaTeX engines may ignore the 'Producer'\n            key and set it to themselves.\n        "
        self._output_name = filename
        self._n_figures = 0
        if keep_empty and keep_empty is not self._UNSET:
            _api.warn_deprecated('3.8', message='Keeping empty pdf files is deprecated since %(since)s and support will be removed %(removal)s.')
        self._keep_empty = keep_empty
        self._metadata = (metadata or {}).copy()
        self._info_dict = _create_pdf_info_dict('pgf', self._metadata)
        self._file = BytesIO()
    keep_empty = _api.deprecate_privatize_attribute('3.8')

    def _write_header(self, width_inches, height_inches):
        if False:
            i = 10
            return i + 15
        pdfinfo = ','.join((_metadata_to_str(k, v) for (k, v) in self._info_dict.items()))
        latex_header = '\n'.join(['\\documentclass[12pt]{article}', '\\usepackage[pdfinfo={%s}]{hyperref}' % pdfinfo, '\\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}' % (width_inches, height_inches), '\\usepackage{pgf}', _get_preamble(), '\\setlength{\\parindent}{0pt}', '\\begin{document}%'])
        self._file.write(latex_header.encode('utf-8'))

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        self.close()

    def close(self):
        if False:
            return 10
        '\n        Finalize this object, running LaTeX in a temporary directory\n        and moving the final pdf file to *filename*.\n        '
        self._file.write(b'\\end{document}\\n')
        if self._n_figures > 0:
            self._run_latex()
        elif self._keep_empty:
            _api.warn_deprecated('3.8', message='Keeping empty pdf files is deprecated since %(since)s and support will be removed %(removal)s.')
            open(self._output_name, 'wb').close()
        self._file.close()

    def _run_latex(self):
        if False:
            return 10
        texcommand = mpl.rcParams['pgf.texsystem']
        with TemporaryDirectory() as tmpdir:
            tex_source = pathlib.Path(tmpdir, 'pdf_pages.tex')
            tex_source.write_bytes(self._file.getvalue())
            cbook._check_and_log_subprocess([texcommand, '-interaction=nonstopmode', '-halt-on-error', tex_source], _log, cwd=tmpdir)
            shutil.move(tex_source.with_suffix('.pdf'), self._output_name)

    def savefig(self, figure=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Save a `.Figure` to this file as a new page.\n\n        Any other keyword arguments are passed to `~.Figure.savefig`.\n\n        Parameters\n        ----------\n        figure : `.Figure` or int, default: the active figure\n            The figure, or index of the figure, that is saved to the file.\n        '
        if not isinstance(figure, Figure):
            if figure is None:
                manager = Gcf.get_active()
            else:
                manager = Gcf.get_fig_manager(figure)
            if manager is None:
                raise ValueError(f'No figure {figure}')
            figure = manager.canvas.figure
        with cbook._setattr_cm(figure, canvas=FigureCanvasPgf(figure)):
            (width, height) = figure.get_size_inches()
            if self._n_figures == 0:
                self._write_header(width, height)
            else:
                self._file.write(f'\\newpage\\ifdefined\\pdfpagewidth\\pdfpagewidth\\else\\pagewidth\\fi={width}in\\ifdefined\\pdfpageheight\\pdfpageheight\\else\\pageheight\\fi={height}in%%\n'.encode('ascii'))
            figure.savefig(self._file, format='pgf', **kwargs)
            self._n_figures += 1

    def get_pagecount(self):
        if False:
            while True:
                i = 10
        'Return the current number of pages in the multipage pdf file.'
        return self._n_figures