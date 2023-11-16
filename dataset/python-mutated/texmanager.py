"""
Support for embedded TeX expressions in Matplotlib.

Requirements:

* LaTeX.
* \\*Agg backends: dvipng>=1.6.
* PS backend: PSfrag, dvips, and Ghostscript>=9.0.
* PDF and SVG backends: if LuaTeX is present, it will be used to speed up some
  post-processing steps, but note that it is not used to parse the TeX string
  itself (only LaTeX is supported).

To enable TeX rendering of all text in your Matplotlib figure, set
:rc:`text.usetex` to True.

TeX and dvipng/dvips processing results are cached
in ~/.matplotlib/tex.cache for reuse between sessions.

`TexManager.get_rgba` can also be used to directly obtain raster output as RGBA
NumPy arrays.
"""
import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
_log = logging.getLogger(__name__)

def _usepackage_if_not_loaded(package, *, option=None):
    if False:
        i = 10
        return i + 15
    "\n    Output LaTeX code that loads a package (possibly with an option) if it\n    hasn't been loaded yet.\n\n    LaTeX cannot load twice a package with different options, so this helper\n    can be used to protect against users loading arbitrary packages/options in\n    their custom preamble.\n    "
    option = f'[{option}]' if option is not None else ''
    return '\\makeatletter\\@ifpackageloaded{%(package)s}{}{\\usepackage%(option)s{%(package)s}}\\makeatother' % {'package': package, 'option': option}

class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a directory.

    The cache directory is called ``tex.cache`` and is located in the directory
    returned by `.get_cachedir`.

    Repeated calls to this constructor always return the same instance.
    """
    texcache = _api.deprecate_privatize_attribute('3.8')
    _texcache = os.path.join(mpl.get_cachedir(), 'tex.cache')
    _grey_arrayd = {}
    _font_families = ('serif', 'sans-serif', 'cursive', 'monospace')
    _font_preambles = {'new century schoolbook': '\\renewcommand{\\rmdefault}{pnc}', 'bookman': '\\renewcommand{\\rmdefault}{pbk}', 'times': '\\usepackage{mathptmx}', 'palatino': '\\usepackage{mathpazo}', 'zapf chancery': '\\usepackage{chancery}', 'cursive': '\\usepackage{chancery}', 'charter': '\\usepackage{charter}', 'serif': '', 'sans-serif': '', 'helvetica': '\\usepackage{helvet}', 'avant garde': '\\usepackage{avant}', 'courier': '\\usepackage{courier}', 'monospace': '\\usepackage{type1ec}', 'computer modern roman': '\\usepackage{type1ec}', 'computer modern sans serif': '\\usepackage{type1ec}', 'computer modern typewriter': '\\usepackage{type1ec}'}
    _font_types = {'new century schoolbook': 'serif', 'bookman': 'serif', 'times': 'serif', 'palatino': 'serif', 'zapf chancery': 'cursive', 'charter': 'serif', 'helvetica': 'sans-serif', 'avant garde': 'sans-serif', 'courier': 'monospace', 'computer modern roman': 'serif', 'computer modern sans serif': 'sans-serif', 'computer modern typewriter': 'monospace'}

    @functools.lru_cache
    def __new__(cls):
        if False:
            while True:
                i = 10
        Path(cls._texcache).mkdir(parents=True, exist_ok=True)
        return object.__new__(cls)

    @classmethod
    def _get_font_family_and_reduced(cls):
        if False:
            i = 10
            return i + 15
        'Return the font family name and whether the font is reduced.'
        ff = mpl.rcParams['font.family']
        ff_val = ff[0].lower() if len(ff) == 1 else None
        if len(ff) == 1 and ff_val in cls._font_families:
            return (ff_val, False)
        elif len(ff) == 1 and ff_val in cls._font_preambles:
            return (cls._font_types[ff_val], True)
        else:
            _log.info('font.family must be one of (%s) when text.usetex is True. serif will be used by default.', ', '.join(cls._font_families))
            return ('serif', False)

    @classmethod
    def _get_font_preamble_and_command(cls):
        if False:
            return 10
        (requested_family, is_reduced_font) = cls._get_font_family_and_reduced()
        preambles = {}
        for font_family in cls._font_families:
            if is_reduced_font and font_family == requested_family:
                preambles[font_family] = cls._font_preambles[mpl.rcParams['font.family'][0].lower()]
            else:
                for font in mpl.rcParams['font.' + font_family]:
                    if font.lower() in cls._font_preambles:
                        preambles[font_family] = cls._font_preambles[font.lower()]
                        _log.debug('family: %s, font: %s, info: %s', font_family, font, cls._font_preambles[font.lower()])
                        break
                    else:
                        _log.debug('%s font is not compatible with usetex.', font)
                else:
                    _log.info('No LaTeX-compatible font found for the %s fontfamily in rcParams. Using default.', font_family)
                    preambles[font_family] = cls._font_preambles[font_family]
        cmd = {preambles[family] for family in ['serif', 'sans-serif', 'monospace']}
        if requested_family == 'cursive':
            cmd.add(preambles['cursive'])
        cmd.add('\\usepackage{type1cm}')
        preamble = '\n'.join(sorted(cmd))
        fontcmd = '\\sffamily' if requested_family == 'sans-serif' else '\\ttfamily' if requested_family == 'monospace' else '\\rmfamily'
        return (preamble, fontcmd)

    @classmethod
    def get_basefile(cls, tex, fontsize, dpi=None):
        if False:
            return 10
        '\n        Return a filename based on a hash of the string, fontsize, and dpi.\n        '
        src = cls._get_tex_source(tex, fontsize) + str(dpi)
        filehash = hashlib.md5(src.encode('utf-8')).hexdigest()
        filepath = Path(cls._texcache)
        (num_letters, num_levels) = (2, 2)
        for i in range(0, num_letters * num_levels, num_letters):
            filepath = filepath / Path(filehash[i:i + 2])
        filepath.mkdir(parents=True, exist_ok=True)
        return os.path.join(filepath, filehash)

    @classmethod
    def get_font_preamble(cls):
        if False:
            while True:
                i = 10
        '\n        Return a string containing font configuration for the tex preamble.\n        '
        (font_preamble, command) = cls._get_font_preamble_and_command()
        return font_preamble

    @classmethod
    def get_custom_preamble(cls):
        if False:
            i = 10
            return i + 15
        'Return a string containing user additions to the tex preamble.'
        return mpl.rcParams['text.latex.preamble']

    @classmethod
    def _get_tex_source(cls, tex, fontsize):
        if False:
            return 10
        'Return the complete TeX source for processing a TeX string.'
        (font_preamble, fontcmd) = cls._get_font_preamble_and_command()
        baselineskip = 1.25 * fontsize
        return '\n'.join(['\\documentclass{article}', '% Pass-through \\mathdefault, which is used in non-usetex mode', '% to use the default text font but was historically suppressed', '% in usetex mode.', '\\newcommand{\\mathdefault}[1]{#1}', font_preamble, '\\usepackage[utf8]{inputenc}', '\\DeclareUnicodeCharacter{2212}{\\ensuremath{-}}', '% geometry is loaded before the custom preamble as ', '% convert_psfrags relies on a custom preamble to change the ', '% geometry.', '\\usepackage[papersize=72in, margin=1in]{geometry}', cls.get_custom_preamble(), '% Use `underscore` package to take care of underscores in text.', '% The [strings] option allows to use underscores in file names.', _usepackage_if_not_loaded('underscore', option='strings'), '% Custom packages (e.g. newtxtext) may already have loaded ', '% textcomp with different options.', _usepackage_if_not_loaded('textcomp'), '\\pagestyle{empty}', '\\begin{document}', '% The empty hbox ensures that a page is printed even for empty', '% inputs, except when using psfrag which gets confused by it.', '% matplotlibbaselinemarker is used by dviread to detect the', "% last line's baseline.", f'\\fontsize{{{fontsize}}}{{{baselineskip}}}%', '\\ifdefined\\psfrag\\else\\hbox{}\\fi%', f'{{{fontcmd} {tex}}}%', '\\end{document}'])

    @classmethod
    def make_tex(cls, tex, fontsize):
        if False:
            while True:
                i = 10
        '\n        Generate a tex file to render the tex string at a specific font size.\n\n        Return the file name.\n        '
        texfile = cls.get_basefile(tex, fontsize) + '.tex'
        Path(texfile).write_text(cls._get_tex_source(tex, fontsize), encoding='utf-8')
        return texfile

    @classmethod
    def _run_checked_subprocess(cls, command, tex, *, cwd=None):
        if False:
            print('Hello World!')
        _log.debug(cbook._pformat_subprocess(command))
        try:
            report = subprocess.check_output(command, cwd=cwd if cwd is not None else cls._texcache, stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(f'Failed to process string with tex because {command[0]} could not be found') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('{prog} was not able to process the following string:\n{tex!r}\n\nHere is the full command invocation and its output:\n\n{format_command}\n\n{exc}\n\n'.format(prog=command[0], format_command=cbook._pformat_subprocess(command), tex=tex.encode('unicode_escape'), exc=exc.output.decode('utf-8', 'backslashreplace'))) from None
        _log.debug(report)
        return report

    @classmethod
    def make_dvi(cls, tex, fontsize):
        if False:
            for i in range(10):
                print('nop')
        "\n        Generate a dvi file containing latex's layout of tex string.\n\n        Return the file name.\n        "
        basefile = cls.get_basefile(tex, fontsize)
        dvifile = '%s.dvi' % basefile
        if not os.path.exists(dvifile):
            texfile = Path(cls.make_tex(tex, fontsize))
            cwd = Path(dvifile).parent
            with TemporaryDirectory(dir=cwd) as tmpdir:
                tmppath = Path(tmpdir)
                cls._run_checked_subprocess(['latex', '-interaction=nonstopmode', '--halt-on-error', f'--output-directory={tmppath.name}', f'{texfile.name}'], tex, cwd=cwd)
                (tmppath / Path(dvifile).name).replace(dvifile)
        return dvifile

    @classmethod
    def make_png(cls, tex, fontsize, dpi):
        if False:
            for i in range(10):
                print('nop')
        "\n        Generate a png file containing latex's rendering of tex string.\n\n        Return the file name.\n        "
        basefile = cls.get_basefile(tex, fontsize, dpi)
        pngfile = '%s.png' % basefile
        if not os.path.exists(pngfile):
            dvifile = cls.make_dvi(tex, fontsize)
            cmd = ['dvipng', '-bg', 'Transparent', '-D', str(dpi), '-T', 'tight', '-o', pngfile, dvifile]
            if getattr(mpl, '_called_from_pytest', False) and mpl._get_executable_info('dvipng').raw_version != '1.16':
                cmd.insert(1, '--freetype0')
            cls._run_checked_subprocess(cmd, tex)
        return pngfile

    @classmethod
    def get_grey(cls, tex, fontsize=None, dpi=None):
        if False:
            while True:
                i = 10
        'Return the alpha channel.'
        if not fontsize:
            fontsize = mpl.rcParams['font.size']
        if not dpi:
            dpi = mpl.rcParams['savefig.dpi']
        key = (cls._get_tex_source(tex, fontsize), dpi)
        alpha = cls._grey_arrayd.get(key)
        if alpha is None:
            pngfile = cls.make_png(tex, fontsize, dpi)
            rgba = mpl.image.imread(os.path.join(cls._texcache, pngfile))
            cls._grey_arrayd[key] = alpha = rgba[:, :, -1]
        return alpha

    @classmethod
    def get_rgba(cls, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
        if False:
            while True:
                i = 10
        '\n        Return latex\'s rendering of the tex string as an RGBA array.\n\n        Examples\n        --------\n        >>> texmanager = TexManager()\n        >>> s = r"\\TeX\\ is $\\displaystyle\\sum_n\\frac{-e^{i\\pi}}{2^n}$!"\n        >>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))\n        '
        alpha = cls.get_grey(tex, fontsize, dpi)
        rgba = np.empty((*alpha.shape, 4))
        rgba[..., :3] = mpl.colors.to_rgb(rgb)
        rgba[..., -1] = alpha
        return rgba

    @classmethod
    def get_text_width_height_descent(cls, tex, fontsize, renderer=None):
        if False:
            print('Hello World!')
        'Return width, height and descent of the text.'
        if tex.strip() == '':
            return (0, 0, 0)
        dvifile = cls.make_dvi(tex, fontsize)
        dpi_fraction = renderer.points_to_pixels(1.0) if renderer else 1
        with dviread.Dvi(dvifile, 72 * dpi_fraction) as dvi:
            (page,) = dvi
        return (page.width, page.height + page.descent, page.descent)