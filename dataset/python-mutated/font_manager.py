"""
A module for finding, managing, and using fonts across platforms.

This module provides a single `FontManager` instance, ``fontManager``, that can
be shared across backends and platforms.  The `findfont`
function returns the best TrueType (TTF) font file in the local or
system font path that matches the specified `FontProperties`
instance.  The `FontManager` also handles Adobe Font Metrics
(AFM) font files for use by the PostScript backend.
The `FontManager.addfont` function adds a custom font from a file without
installing it into your operating system.

The design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
Future versions may implement the Level 2 or 2.1 specifications.
"""
from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern, generate_fontconfig_pattern
from matplotlib.rcsetup import _validators
_log = logging.getLogger(__name__)
font_scalings = {'xx-small': 0.579, 'x-small': 0.694, 'small': 0.833, 'medium': 1.0, 'large': 1.2, 'x-large': 1.44, 'xx-large': 1.728, 'larger': 1.2, 'smaller': 0.833, None: 1.0}
stretch_dict = {'ultra-condensed': 100, 'extra-condensed': 200, 'condensed': 300, 'semi-condensed': 400, 'normal': 500, 'semi-expanded': 600, 'semi-extended': 600, 'expanded': 700, 'extended': 700, 'extra-expanded': 800, 'extra-extended': 800, 'ultra-expanded': 900, 'ultra-extended': 900}
weight_dict = {'ultralight': 100, 'light': 200, 'normal': 400, 'regular': 400, 'book': 400, 'medium': 500, 'roman': 500, 'semibold': 600, 'demibold': 600, 'demi': 600, 'bold': 700, 'heavy': 800, 'extra bold': 800, 'black': 900}
_weight_regexes = [('thin', 100), ('extralight', 200), ('ultralight', 200), ('demilight', 350), ('semilight', 350), ('light', 300), ('book', 380), ('regular', 400), ('normal', 400), ('medium', 500), ('demibold', 600), ('demi', 600), ('semibold', 600), ('extrabold', 800), ('superbold', 800), ('ultrabold', 800), ('bold', 700), ('ultrablack', 1000), ('superblack', 1000), ('extrablack', 1000), ('\\bultra', 1000), ('black', 900), ('heavy', 900)]
font_family_aliases = {'serif', 'sans-serif', 'sans serif', 'cursive', 'fantasy', 'monospace', 'sans'}
_ExceptionProxy = namedtuple('_ExceptionProxy', ['klass', 'message'])
try:
    _HOME = Path.home()
except Exception:
    _HOME = Path(os.devnull)
MSFolders = 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders'
MSFontDirectories = ['SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts', 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Fonts']
MSUserFontDirectories = [str(_HOME / 'AppData/Local/Microsoft/Windows/Fonts'), str(_HOME / 'AppData/Roaming/Microsoft/Windows/Fonts')]
X11FontDirectories = ['/usr/X11R6/lib/X11/fonts/TTF/', '/usr/X11/lib/X11/fonts', '/usr/share/fonts/', '/usr/local/share/fonts/', '/usr/lib/openoffice/share/fonts/truetype/', str(Path(os.environ.get('XDG_DATA_HOME') or _HOME / '.local/share') / 'fonts'), str(_HOME / '.fonts')]
OSXFontDirectories = ['/Library/Fonts/', '/Network/Library/Fonts/', '/System/Library/Fonts/', '/opt/local/share/fonts', str(_HOME / 'Library/Fonts')]

def get_fontext_synonyms(fontext):
    if False:
        while True:
            i = 10
    '\n    Return a list of file extensions that are synonyms for\n    the given file extension *fileext*.\n    '
    return {'afm': ['afm'], 'otf': ['otf', 'ttc', 'ttf'], 'ttc': ['otf', 'ttc', 'ttf'], 'ttf': ['otf', 'ttc', 'ttf']}[fontext]

def list_fonts(directory, extensions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of all fonts matching any of the extensions, found\n    recursively under the directory.\n    '
    extensions = ['.' + ext for ext in extensions]
    return [os.path.join(dirpath, filename) for (dirpath, _, filenames) in os.walk(directory) for filename in filenames if Path(filename).suffix.lower() in extensions]

def win32FontDirectory():
    if False:
        return 10
    '\n    Return the user-specified font directory for Win32.  This is\n    looked up from the registry key ::\n\n      \\\\HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\\Fonts\n\n    If the key is not found, ``%WINDIR%\\Fonts`` will be returned.\n    '
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MSFolders) as user:
            return winreg.QueryValueEx(user, 'Fonts')[0]
    except OSError:
        return os.path.join(os.environ['WINDIR'], 'Fonts')

def _get_win32_installed_fonts():
    if False:
        i = 10
        return i + 15
    'List the font paths known to the Windows registry.'
    import winreg
    items = set()
    for (domain, base_dirs) in [(winreg.HKEY_LOCAL_MACHINE, [win32FontDirectory()]), (winreg.HKEY_CURRENT_USER, MSUserFontDirectories)]:
        for base_dir in base_dirs:
            for reg_path in MSFontDirectories:
                try:
                    with winreg.OpenKey(domain, reg_path) as local:
                        for j in range(winreg.QueryInfoKey(local)[1]):
                            (key, value, tp) = winreg.EnumValue(local, j)
                            if not isinstance(value, str):
                                continue
                            try:
                                path = Path(base_dir, value).resolve()
                            except RuntimeError:
                                continue
                            items.add(path)
                except (OSError, MemoryError):
                    continue
    return items

@lru_cache
def _get_fontconfig_fonts():
    if False:
        return 10
    'Cache and list the font paths known to ``fc-list``.'
    try:
        if b'--format' not in subprocess.check_output(['fc-list', '--help']):
            _log.warning('Matplotlib needs fontconfig>=2.7 to query system fonts.')
            return []
        out = subprocess.check_output(['fc-list', '--format=%{file}\\n'])
    except (OSError, subprocess.CalledProcessError):
        return []
    return [Path(os.fsdecode(fname)) for fname in out.split(b'\n')]

def findSystemFonts(fontpaths=None, fontext='ttf'):
    if False:
        print('Hello World!')
    '\n    Search for fonts in the specified font paths.  If no paths are\n    given, will use a standard set of system paths, as well as the\n    list of fonts tracked by fontconfig if fontconfig is installed and\n    available.  A list of TrueType fonts are returned by default with\n    AFM fonts as an option.\n    '
    fontfiles = set()
    fontexts = get_fontext_synonyms(fontext)
    if fontpaths is None:
        if sys.platform == 'win32':
            installed_fonts = _get_win32_installed_fonts()
            fontpaths = []
        else:
            installed_fonts = _get_fontconfig_fonts()
            if sys.platform == 'darwin':
                fontpaths = [*X11FontDirectories, *OSXFontDirectories]
            else:
                fontpaths = X11FontDirectories
        fontfiles.update((str(path) for path in installed_fonts if path.suffix.lower()[1:] in fontexts))
    elif isinstance(fontpaths, str):
        fontpaths = [fontpaths]
    for path in fontpaths:
        fontfiles.update(map(os.path.abspath, list_fonts(path, fontexts)))
    return [fname for fname in fontfiles if os.path.exists(fname)]

def _fontentry_helper_repr_png(fontent):
    if False:
        i = 10
        return i + 15
    from matplotlib.figure import Figure
    fig = Figure()
    font_path = Path(fontent.fname) if fontent.fname != '' else None
    fig.text(0, 0, fontent.name, font=font_path)
    with BytesIO() as buf:
        fig.savefig(buf, bbox_inches='tight', transparent=True)
        return buf.getvalue()

def _fontentry_helper_repr_html(fontent):
    if False:
        while True:
            i = 10
    png_stream = _fontentry_helper_repr_png(fontent)
    png_b64 = b64encode(png_stream).decode()
    return f'<img src="data:image/png;base64, {png_b64}" />'
FontEntry = dataclasses.make_dataclass('FontEntry', [('fname', str, dataclasses.field(default='')), ('name', str, dataclasses.field(default='')), ('style', str, dataclasses.field(default='normal')), ('variant', str, dataclasses.field(default='normal')), ('weight', Union[str, int], dataclasses.field(default='normal')), ('stretch', str, dataclasses.field(default='normal')), ('size', str, dataclasses.field(default='medium'))], namespace={'__doc__': '\n    A class for storing Font properties.\n\n    It is used when populating the font lookup dictionary.\n    ', '_repr_html_': lambda self: _fontentry_helper_repr_html(self), '_repr_png_': lambda self: _fontentry_helper_repr_png(self)})

def ttfFontProperty(font):
    if False:
        i = 10
        return i + 15
    '\n    Extract information from a TrueType font file.\n\n    Parameters\n    ----------\n    font : `.FT2Font`\n        The TrueType font file from which information will be extracted.\n\n    Returns\n    -------\n    `FontEntry`\n        The extracted font properties.\n\n    '
    name = font.family_name
    sfnt = font.get_sfnt()
    mac_key = (1, 0, 0)
    ms_key = (3, 1, 1033)
    sfnt2 = sfnt.get((*mac_key, 2), b'').decode('latin-1').lower() or sfnt.get((*ms_key, 2), b'').decode('utf_16_be').lower()
    sfnt4 = sfnt.get((*mac_key, 4), b'').decode('latin-1').lower() or sfnt.get((*ms_key, 4), b'').decode('utf_16_be').lower()
    if sfnt4.find('oblique') >= 0:
        style = 'oblique'
    elif sfnt4.find('italic') >= 0:
        style = 'italic'
    elif sfnt2.find('regular') >= 0:
        style = 'normal'
    elif font.style_flags & ft2font.ITALIC:
        style = 'italic'
    else:
        style = 'normal'
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'
    wws_subfamily = 22
    typographic_subfamily = 16
    font_subfamily = 2
    styles = [sfnt.get((*mac_key, wws_subfamily), b'').decode('latin-1'), sfnt.get((*mac_key, typographic_subfamily), b'').decode('latin-1'), sfnt.get((*mac_key, font_subfamily), b'').decode('latin-1'), sfnt.get((*ms_key, wws_subfamily), b'').decode('utf-16-be'), sfnt.get((*ms_key, typographic_subfamily), b'').decode('utf-16-be'), sfnt.get((*ms_key, font_subfamily), b'').decode('utf-16-be')]
    styles = [*filter(None, styles)] or [font.style_name]

    def get_weight():
        if False:
            return 10
        os2 = font.get_sfnt_table('OS/2')
        if os2 and os2['version'] != 65535:
            return os2['usWeightClass']
        try:
            ps_font_info_weight = font.get_ps_font_info()['weight'].replace(' ', '') or ''
        except ValueError:
            pass
        else:
            for (regex, weight) in _weight_regexes:
                if re.fullmatch(regex, ps_font_info_weight, re.I):
                    return weight
        for style in styles:
            style = style.replace(' ', '')
            for (regex, weight) in _weight_regexes:
                if re.search(regex, style, re.I):
                    return weight
        if font.style_flags & ft2font.BOLD:
            return 700
        return 500
    weight = int(get_weight())
    if any((word in sfnt4 for word in ['narrow', 'condensed', 'cond'])):
        stretch = 'condensed'
    elif 'demi cond' in sfnt4:
        stretch = 'semi-condensed'
    elif any((word in sfnt4 for word in ['wide', 'expanded', 'extended'])):
        stretch = 'expanded'
    else:
        stretch = 'normal'
    if not font.scalable:
        raise NotImplementedError('Non-scalable fonts are not supported')
    size = 'scalable'
    return FontEntry(font.fname, name, style, variant, weight, stretch, size)

def afmFontProperty(fontpath, font):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract information from an AFM font file.\n\n    Parameters\n    ----------\n    fontpath : str\n        The filename corresponding to *font*.\n    font : AFM\n        The AFM font file from which information will be extracted.\n\n    Returns\n    -------\n    `FontEntry`\n        The extracted font properties.\n    '
    name = font.get_familyname()
    fontname = font.get_fontname().lower()
    if font.get_angle() != 0 or 'italic' in name.lower():
        style = 'italic'
    elif 'oblique' in name.lower():
        style = 'oblique'
    else:
        style = 'normal'
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'
    weight = font.get_weight().lower()
    if weight not in weight_dict:
        weight = 'normal'
    if 'demi cond' in fontname:
        stretch = 'semi-condensed'
    elif any((word in fontname for word in ['narrow', 'cond'])):
        stretch = 'condensed'
    elif any((word in fontname for word in ['wide', 'expanded', 'extended'])):
        stretch = 'expanded'
    else:
        stretch = 'normal'
    size = 'scalable'
    return FontEntry(fontpath, name, style, variant, weight, stretch, size)

class FontProperties:
    """
    A class for storing and manipulating font properties.

    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification and *math_fontfamily* for math fonts:

    - family: A list of font names in decreasing order of priority.
      The items may include a generic font family name, either 'sans-serif',
      'serif', 'cursive', 'fantasy', or 'monospace'.  In that case, the actual
      font to be used will be looked up from the associated rcParam during the
      search process in `.findfont`. Default: :rc:`font.family`

    - style: Either 'normal', 'italic' or 'oblique'.
      Default: :rc:`font.style`

    - variant: Either 'normal' or 'small-caps'.
      Default: :rc:`font.variant`

    - stretch: A numeric value in the range 0-1000 or one of
      'ultra-condensed', 'extra-condensed', 'condensed',
      'semi-condensed', 'normal', 'semi-expanded', 'expanded',
      'extra-expanded' or 'ultra-expanded'. Default: :rc:`font.stretch`

    - weight: A numeric value in the range 0-1000 or one of
      'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
      'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
      'extra bold', 'black'. Default: :rc:`font.weight`

    - size: Either a relative value of 'xx-small', 'x-small',
      'small', 'medium', 'large', 'x-large', 'xx-large' or an
      absolute font size, e.g., 10. Default: :rc:`font.size`

    - math_fontfamily: The family of fonts used to render math text.
      Supported values are: 'dejavusans', 'dejavuserif', 'cm',
      'stix', 'stixsans' and 'custom'. Default: :rc:`mathtext.fontset`

    Alternatively, a font may be specified using the absolute path to a font
    file, by using the *fname* kwarg.  However, in this case, it is typically
    simpler to just pass the path (as a `pathlib.Path`, not a `str`) to the
    *font* kwarg of the `.Text` object.

    The preferred usage of font sizes is to use the relative values,
    e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
    approach allows all text sizes to be made larger or smaller based
    on the font manager's default font size.

    This class will also accept a fontconfig_ pattern_, if it is the only
    argument provided.  This support does not depend on fontconfig; we are
    merely borrowing its pattern syntax for use here.

    .. _fontconfig: https://www.freedesktop.org/wiki/Software/fontconfig/
    .. _pattern:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

    Note that Matplotlib's internal font manager and fontconfig use a
    different algorithm to lookup fonts, so the results of the same pattern
    may be different in Matplotlib than in other applications that use
    fontconfig.
    """

    def __init__(self, family=None, style=None, variant=None, weight=None, stretch=None, size=None, fname=None, math_fontfamily=None):
        if False:
            while True:
                i = 10
        self.set_family(family)
        self.set_style(style)
        self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)
        self.set_math_fontfamily(math_fontfamily)
        if isinstance(family, str) and style is None and (variant is None) and (weight is None) and (stretch is None) and (size is None) and (fname is None):
            self.set_fontconfig_pattern(family)

    @classmethod
    def _from_any(cls, arg):
        if False:
            while True:
                i = 10
        '\n        Generic constructor which can build a `.FontProperties` from any of the\n        following:\n\n        - a `.FontProperties`: it is passed through as is;\n        - `None`: a `.FontProperties` using rc values is used;\n        - an `os.PathLike`: it is used as path to the font file;\n        - a `str`: it is parsed as a fontconfig pattern;\n        - a `dict`: it is passed as ``**kwargs`` to `.FontProperties`.\n        '
        if arg is None:
            return cls()
        elif isinstance(arg, cls):
            return arg
        elif isinstance(arg, os.PathLike):
            return cls(fname=arg)
        elif isinstance(arg, str):
            return cls(arg)
        else:
            return cls(**arg)

    def __hash__(self):
        if False:
            print('Hello World!')
        l = (tuple(self.get_family()), self.get_slant(), self.get_variant(), self.get_weight(), self.get_stretch(), self.get_size(), self.get_file(), self.get_math_fontfamily())
        return hash(l)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return hash(self) == hash(other)

    def __str__(self):
        if False:
            return 10
        return self.get_fontconfig_pattern()

    def get_family(self):
        if False:
            while True:
                i = 10
        '\n        Return a list of individual font family names or generic family names.\n\n        The font families or generic font families (which will be resolved\n        from their respective rcParams when searching for a matching font) in\n        the order of preference.\n        '
        return self._family

    def get_name(self):
        if False:
            return 10
        '\n        Return the name of the font that best matches the font properties.\n        '
        return get_font(findfont(self)).family_name

    def get_style(self):
        if False:
            return 10
        "\n        Return the font style.  Values are: 'normal', 'italic' or 'oblique'.\n        "
        return self._slant

    def get_variant(self):
        if False:
            return 10
        "\n        Return the font variant.  Values are: 'normal' or 'small-caps'.\n        "
        return self._variant

    def get_weight(self):
        if False:
            while True:
                i = 10
        "\n        Set the font weight.  Options are: A numeric value in the\n        range 0-1000 or one of 'light', 'normal', 'regular', 'book',\n        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',\n        'heavy', 'extra bold', 'black'\n        "
        return self._weight

    def get_stretch(self):
        if False:
            return 10
        "\n        Return the font stretch or width.  Options are: 'ultra-condensed',\n        'extra-condensed', 'condensed', 'semi-condensed', 'normal',\n        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.\n        "
        return self._stretch

    def get_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the font size.\n        '
        return self._size

    def get_file(self):
        if False:
            while True:
                i = 10
        '\n        Return the filename of the associated font.\n        '
        return self._file

    def get_fontconfig_pattern(self):
        if False:
            while True:
                i = 10
        "\n        Get a fontconfig_ pattern_ suitable for looking up the font as\n        specified with fontconfig's ``fc-match`` utility.\n\n        This support does not depend on fontconfig; we are merely borrowing its\n        pattern syntax for use here.\n        "
        return generate_fontconfig_pattern(self)

    def set_family(self, family):
        if False:
            return 10
        "\n        Change the font family.  Can be either an alias (generic name\n        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',\n        'fantasy', or 'monospace', a real font name or a list of real\n        font names.  Real font names are not supported when\n        :rc:`text.usetex` is `True`. Default: :rc:`font.family`\n        "
        if family is None:
            family = mpl.rcParams['font.family']
        if isinstance(family, str):
            family = [family]
        self._family = family

    def set_style(self, style):
        if False:
            return 10
        "\n        Set the font style.\n\n        Parameters\n        ----------\n        style : {'normal', 'italic', 'oblique'}, default: :rc:`font.style`\n        "
        if style is None:
            style = mpl.rcParams['font.style']
        _api.check_in_list(['normal', 'italic', 'oblique'], style=style)
        self._slant = style

    def set_variant(self, variant):
        if False:
            return 10
        "\n        Set the font variant.\n\n        Parameters\n        ----------\n        variant : {'normal', 'small-caps'}, default: :rc:`font.variant`\n        "
        if variant is None:
            variant = mpl.rcParams['font.variant']
        _api.check_in_list(['normal', 'small-caps'], variant=variant)
        self._variant = variant

    def set_weight(self, weight):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the font weight.\n\n        Parameters\n        ----------\n        weight : int or {'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}, default: :rc:`font.weight`\n            If int, must be in the range  0-1000.\n        "
        if weight is None:
            weight = mpl.rcParams['font.weight']
        if weight in weight_dict:
            self._weight = weight
            return
        try:
            weight = int(weight)
        except ValueError:
            pass
        else:
            if 0 <= weight <= 1000:
                self._weight = weight
                return
        raise ValueError(f'weight={weight!r} is invalid')

    def set_stretch(self, stretch):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the font stretch or width.\n\n        Parameters\n        ----------\n        stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}, default: :rc:`font.stretch`\n            If int, must be in the range  0-1000.\n        "
        if stretch is None:
            stretch = mpl.rcParams['font.stretch']
        if stretch in stretch_dict:
            self._stretch = stretch
            return
        try:
            stretch = int(stretch)
        except ValueError:
            pass
        else:
            if 0 <= stretch <= 1000:
                self._stretch = stretch
                return
        raise ValueError(f'stretch={stretch!r} is invalid')

    def set_size(self, size):
        if False:
            print('Hello World!')
        "\n        Set the font size.\n\n        Parameters\n        ----------\n        size : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, default: :rc:`font.size`\n            If a float, the font size in points. The string values denote\n            sizes relative to the default font size.\n        "
        if size is None:
            size = mpl.rcParams['font.size']
        try:
            size = float(size)
        except ValueError:
            try:
                scale = font_scalings[size]
            except KeyError as err:
                raise ValueError('Size is invalid. Valid font size are ' + ', '.join(map(str, font_scalings))) from err
            else:
                size = scale * FontManager.get_default_size()
        if size < 1.0:
            _log.info('Fontsize %1.2f < 1.0 pt not allowed by FreeType. Setting fontsize = 1 pt', size)
            size = 1.0
        self._size = size

    def set_file(self, file):
        if False:
            i = 10
            return i + 15
        '\n        Set the filename of the fontfile to use.  In this case, all\n        other properties will be ignored.\n        '
        self._file = os.fspath(file) if file is not None else None

    def set_fontconfig_pattern(self, pattern):
        if False:
            while True:
                i = 10
        '\n        Set the properties by parsing a fontconfig_ *pattern*.\n\n        This support does not depend on fontconfig; we are merely borrowing its\n        pattern syntax for use here.\n        '
        for (key, val) in parse_fontconfig_pattern(pattern).items():
            if type(val) is list:
                getattr(self, 'set_' + key)(val[0])
            else:
                getattr(self, 'set_' + key)(val)

    def get_math_fontfamily(self):
        if False:
            return 10
        '\n        Return the name of the font family used for math text.\n\n        The default font is :rc:`mathtext.fontset`.\n        '
        return self._math_fontfamily

    def set_math_fontfamily(self, fontfamily):
        if False:
            print('Hello World!')
        '\n        Set the font family for text in math mode.\n\n        If not set explicitly, :rc:`mathtext.fontset` will be used.\n\n        Parameters\n        ----------\n        fontfamily : str\n            The name of the font family.\n\n            Available font families are defined in the\n            :ref:`default matplotlibrc file <customizing-with-matplotlibrc-files>`.\n\n        See Also\n        --------\n        .text.Text.get_math_fontfamily\n        '
        if fontfamily is None:
            fontfamily = mpl.rcParams['mathtext.fontset']
        else:
            valid_fonts = _validators['mathtext.fontset'].valid.values()
            _api.check_in_list(valid_fonts, math_fontfamily=fontfamily)
        self._math_fontfamily = fontfamily

    def copy(self):
        if False:
            return 10
        'Return a copy of self.'
        return copy.copy(self)
    set_name = set_family
    get_slant = get_style
    set_slant = set_style
    get_size_in_points = get_size

class _JSONEncoder(json.JSONEncoder):

    def default(self, o):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(o, FontManager):
            return dict(o.__dict__, __class__='FontManager')
        elif isinstance(o, FontEntry):
            d = dict(o.__dict__, __class__='FontEntry')
            try:
                d['fname'] = str(Path(d['fname']).relative_to(mpl.get_data_path()))
            except ValueError:
                pass
            return d
        else:
            return super().default(o)

def _json_decode(o):
    if False:
        while True:
            i = 10
    cls = o.pop('__class__', None)
    if cls is None:
        return o
    elif cls == 'FontManager':
        r = FontManager.__new__(FontManager)
        r.__dict__.update(o)
        return r
    elif cls == 'FontEntry':
        r = FontEntry.__new__(FontEntry)
        r.__dict__.update(o)
        if not os.path.isabs(r.fname):
            r.fname = os.path.join(mpl.get_data_path(), r.fname)
        return r
    else:
        raise ValueError("Don't know how to deserialize __class__=%s" % cls)

def json_dump(data, filename):
    if False:
        print('Hello World!')
    "\n    Dump `FontManager` *data* as JSON to the file named *filename*.\n\n    See Also\n    --------\n    json_load\n\n    Notes\n    -----\n    File paths that are children of the Matplotlib data path (typically, fonts\n    shipped with Matplotlib) are stored relative to that data path (to remain\n    valid across virtualenvs).\n\n    This function temporarily locks the output file to prevent multiple\n    processes from overwriting one another's output.\n    "
    with cbook._lock_path(filename), open(filename, 'w') as fh:
        try:
            json.dump(data, fh, cls=_JSONEncoder, indent=2)
        except OSError as e:
            _log.warning('Could not save font_manager cache %s', e)

def json_load(filename):
    if False:
        i = 10
        return i + 15
    '\n    Load a `FontManager` from the JSON file named *filename*.\n\n    See Also\n    --------\n    json_dump\n    '
    with open(filename) as fh:
        return json.load(fh, object_hook=_json_decode)

class FontManager:
    """
    On import, the `FontManager` singleton instance creates a list of ttf and
    afm fonts and caches their `FontProperties`.  The `FontManager.findfont`
    method does a nearest neighbor search to find the font that most closely
    matches the specification.  If no good enough match is found, the default
    font is returned.

    Fonts added with the `FontManager.addfont` method will not persist in the
    cache; therefore, `addfont` will need to be called every time Matplotlib is
    imported. This method should only be used if and when a font cannot be
    installed on your operating system by other means.

    Notes
    -----
    The `FontManager.addfont` method must be called on the global `FontManager`
    instance.

    Example usage::

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_dirs = ["/resources/fonts"]  # The path to the custom font file.
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    """
    __version__ = 330

    def __init__(self, size=None, weight='normal'):
        if False:
            while True:
                i = 10
        self._version = self.__version__
        self.__default_weight = weight
        self.default_size = size
        paths = [cbook._get_data_path('fonts', subdir) for subdir in ['ttf', 'afm', 'pdfcorefonts']]
        _log.debug('font search path %s', paths)
        self.defaultFamily = {'ttf': 'DejaVu Sans', 'afm': 'Helvetica'}
        self.afmlist = []
        self.ttflist = []
        timer = threading.Timer(5, lambda : _log.warning('Matplotlib is building the font cache; this may take a moment.'))
        timer.start()
        try:
            for fontext in ['afm', 'ttf']:
                for path in [*findSystemFonts(paths, fontext=fontext), *findSystemFonts(fontext=fontext)]:
                    try:
                        self.addfont(path)
                    except OSError as exc:
                        _log.info('Failed to open font file %s: %s', path, exc)
                    except Exception as exc:
                        _log.info('Failed to extract font properties from %s: %s', path, exc)
        finally:
            timer.cancel()

    def addfont(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cache the properties of the font at *path* to make it available to the\n        `FontManager`.  The type of font is inferred from the path suffix.\n\n        Parameters\n        ----------\n        path : str or path-like\n\n        Notes\n        -----\n        This method is useful for adding a custom font without installing it in\n        your operating system. See the `FontManager` singleton instance for\n        usage and caveats about this function.\n        '
        path = os.fsdecode(path)
        if Path(path).suffix.lower() == '.afm':
            with open(path, 'rb') as fh:
                font = _afm.AFM(fh)
            prop = afmFontProperty(path, font)
            self.afmlist.append(prop)
        else:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)
            self.ttflist.append(prop)
        self._findfont_cached.cache_clear()

    @property
    def defaultFont(self):
        if False:
            print('Hello World!')
        return {ext: self.findfont(family, fontext=ext) for (ext, family) in self.defaultFamily.items()}

    def get_default_weight(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the default font weight.\n        '
        return self.__default_weight

    @staticmethod
    def get_default_size():
        if False:
            return 10
        '\n        Return the default font size.\n        '
        return mpl.rcParams['font.size']

    def set_default_weight(self, weight):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the default font weight.  The initial value is 'normal'.\n        "
        self.__default_weight = weight

    @staticmethod
    def _expand_aliases(family):
        if False:
            print('Hello World!')
        if family in ('sans', 'sans serif'):
            family = 'sans-serif'
        return mpl.rcParams['font.' + family]

    def score_family(self, families, family2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a match score between the list of font families in\n        *families* and the font family name *family2*.\n\n        An exact match at the head of the list returns 0.0.\n\n        A match further down the list will return between 0 and 1.\n\n        No match will return 1.0.\n        '
        if not isinstance(families, (list, tuple)):
            families = [families]
        elif len(families) == 0:
            return 1.0
        family2 = family2.lower()
        step = 1 / len(families)
        for (i, family1) in enumerate(families):
            family1 = family1.lower()
            if family1 in font_family_aliases:
                options = [*map(str.lower, self._expand_aliases(family1))]
                if family2 in options:
                    idx = options.index(family2)
                    return (i + idx / len(options)) * step
            elif family1 == family2:
                return i * step
        return 1.0

    def score_style(self, style1, style2):
        if False:
            while True:
                i = 10
        "\n        Return a match score between *style1* and *style2*.\n\n        An exact match returns 0.0.\n\n        A match between 'italic' and 'oblique' returns 0.1.\n\n        No match returns 1.0.\n        "
        if style1 == style2:
            return 0.0
        elif style1 in ('italic', 'oblique') and style2 in ('italic', 'oblique'):
            return 0.1
        return 1.0

    def score_variant(self, variant1, variant2):
        if False:
            while True:
                i = 10
        '\n        Return a match score between *variant1* and *variant2*.\n\n        An exact match returns 0.0, otherwise 1.0.\n        '
        if variant1 == variant2:
            return 0.0
        else:
            return 1.0

    def score_stretch(self, stretch1, stretch2):
        if False:
            while True:
                i = 10
        '\n        Return a match score between *stretch1* and *stretch2*.\n\n        The result is the absolute value of the difference between the\n        CSS numeric values of *stretch1* and *stretch2*, normalized\n        between 0.0 and 1.0.\n        '
        try:
            stretchval1 = int(stretch1)
        except ValueError:
            stretchval1 = stretch_dict.get(stretch1, 500)
        try:
            stretchval2 = int(stretch2)
        except ValueError:
            stretchval2 = stretch_dict.get(stretch2, 500)
        return abs(stretchval1 - stretchval2) / 1000.0

    def score_weight(self, weight1, weight2):
        if False:
            return 10
        '\n        Return a match score between *weight1* and *weight2*.\n\n        The result is 0.0 if both weight1 and weight 2 are given as strings\n        and have the same value.\n\n        Otherwise, the result is the absolute value of the difference between\n        the CSS numeric values of *weight1* and *weight2*, normalized between\n        0.05 and 1.0.\n        '
        if cbook._str_equal(weight1, weight2):
            return 0.0
        w1 = weight1 if isinstance(weight1, Number) else weight_dict[weight1]
        w2 = weight2 if isinstance(weight2, Number) else weight_dict[weight2]
        return 0.95 * (abs(w1 - w2) / 1000) + 0.05

    def score_size(self, size1, size2):
        if False:
            print('Hello World!')
        "\n        Return a match score between *size1* and *size2*.\n\n        If *size2* (the size specified in the font file) is 'scalable', this\n        function always returns 0.0, since any font size can be generated.\n\n        Otherwise, the result is the absolute distance between *size1* and\n        *size2*, normalized so that the usual range of font sizes (6pt -\n        72pt) will lie between 0.0 and 1.0.\n        "
        if size2 == 'scalable':
            return 0.0
        try:
            sizeval1 = float(size1)
        except ValueError:
            sizeval1 = self.default_size * font_scalings[size1]
        try:
            sizeval2 = float(size2)
        except ValueError:
            return 1.0
        return abs(sizeval1 - sizeval2) / 72

    def findfont(self, prop, fontext='ttf', directory=None, fallback_to_default=True, rebuild_if_missing=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find a font that most closely matches the given font properties.\n\n        Parameters\n        ----------\n        prop : str or `~matplotlib.font_manager.FontProperties`\n            The font properties to search for. This can be either a\n            `.FontProperties` object or a string defining a\n            `fontconfig patterns`_.\n\n        fontext : {\'ttf\', \'afm\'}, default: \'ttf\'\n            The extension of the font file:\n\n            - \'ttf\': TrueType and OpenType fonts (.ttf, .ttc, .otf)\n            - \'afm\': Adobe Font Metrics (.afm)\n\n        directory : str, optional\n            If given, only search this directory and its subdirectories.\n\n        fallback_to_default : bool\n            If True, will fall back to the default font family (usually\n            "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.\n\n        rebuild_if_missing : bool\n            Whether to rebuild the font cache and search again if the first\n            match appears to point to a nonexisting font (i.e., the font cache\n            contains outdated entries).\n\n        Returns\n        -------\n        str\n            The filename of the best matching font.\n\n        Notes\n        -----\n        This performs a nearest neighbor search.  Each font is given a\n        similarity score to the target font properties.  The first font with\n        the highest score is returned.  If no matches below a certain\n        threshold are found, the default font (usually DejaVu Sans) is\n        returned.\n\n        The result is cached, so subsequent lookups don\'t have to\n        perform the O(n) nearest neighbor search.\n\n        See the `W3C Cascading Style Sheet, Level 1\n        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation\n        for a description of the font finding algorithm.\n\n        .. _fontconfig patterns:\n           https://www.freedesktop.org/software/fontconfig/fontconfig-user.html\n        '
        rc_params = tuple((tuple(mpl.rcParams[key]) for key in ['font.serif', 'font.sans-serif', 'font.cursive', 'font.fantasy', 'font.monospace']))
        ret = self._findfont_cached(prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params)
        if isinstance(ret, _ExceptionProxy):
            raise ret.klass(ret.message)
        return ret

    def get_font_names(self):
        if False:
            print('Hello World!')
        'Return the list of available fonts.'
        return list({font.name for font in self.ttflist})

    def _find_fonts_by_props(self, prop, fontext='ttf', directory=None, fallback_to_default=True, rebuild_if_missing=True):
        if False:
            i = 10
            return i + 15
        '\n        Find font families that most closely match the given properties.\n\n        Parameters\n        ----------\n        prop : str or `~matplotlib.font_manager.FontProperties`\n            The font properties to search for. This can be either a\n            `.FontProperties` object or a string defining a\n            `fontconfig patterns`_.\n\n        fontext : {\'ttf\', \'afm\'}, default: \'ttf\'\n            The extension of the font file:\n\n            - \'ttf\': TrueType and OpenType fonts (.ttf, .ttc, .otf)\n            - \'afm\': Adobe Font Metrics (.afm)\n\n        directory : str, optional\n            If given, only search this directory and its subdirectories.\n\n        fallback_to_default : bool\n            If True, will fall back to the default font family (usually\n            "DejaVu Sans" or "Helvetica") if none of the families were found.\n\n        rebuild_if_missing : bool\n            Whether to rebuild the font cache and search again if the first\n            match appears to point to a nonexisting font (i.e., the font cache\n            contains outdated entries).\n\n        Returns\n        -------\n        list[str]\n            The paths of the fonts found\n\n        Notes\n        -----\n        This is an extension/wrapper of the original findfont API, which only\n        returns a single font for given font properties. Instead, this API\n        returns a dict containing multiple fonts and their filepaths\n        which closely match the given font properties.  Since this internally\n        uses the original API, there\'s no change to the logic of performing the\n        nearest neighbor search.  See `findfont` for more details.\n        '
        prop = FontProperties._from_any(prop)
        fpaths = []
        for family in prop.get_family():
            cprop = prop.copy()
            cprop.set_family(family)
            try:
                fpaths.append(self.findfont(cprop, fontext, directory, fallback_to_default=False, rebuild_if_missing=rebuild_if_missing))
            except ValueError:
                if family in font_family_aliases:
                    _log.warning('findfont: Generic family %r not found because none of the following families were found: %s', family, ', '.join(self._expand_aliases(family)))
                else:
                    _log.warning('findfont: Font family %r not found.', family)
        if not fpaths:
            if fallback_to_default:
                dfamily = self.defaultFamily[fontext]
                cprop = prop.copy()
                cprop.set_family(dfamily)
                fpaths.append(self.findfont(cprop, fontext, directory, fallback_to_default=True, rebuild_if_missing=rebuild_if_missing))
            else:
                raise ValueError('Failed to find any font, and fallback to the default font was disabled')
        return fpaths

    @lru_cache(1024)
    def _findfont_cached(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params):
        if False:
            for i in range(10):
                print('nop')
        prop = FontProperties._from_any(prop)
        fname = prop.get_file()
        if fname is not None:
            return fname
        if fontext == 'afm':
            fontlist = self.afmlist
        else:
            fontlist = self.ttflist
        best_score = 1e+64
        best_font = None
        _log.debug('findfont: Matching %s.', prop)
        for font in fontlist:
            if directory is not None and Path(directory) not in Path(font.fname).parents:
                continue
            score = self.score_family(prop.get_family(), font.name) * 10 + self.score_style(prop.get_style(), font.style) + self.score_variant(prop.get_variant(), font.variant) + self.score_weight(prop.get_weight(), font.weight) + self.score_stretch(prop.get_stretch(), font.stretch) + self.score_size(prop.get_size(), font.size)
            _log.debug('findfont: score(%s) = %s', font, score)
            if score < best_score:
                best_score = score
                best_font = font
            if score == 0:
                break
        if best_font is None or best_score >= 10.0:
            if fallback_to_default:
                _log.warning('findfont: Font family %s not found. Falling back to %s.', prop.get_family(), self.defaultFamily[fontext])
                for family in map(str.lower, prop.get_family()):
                    if family in font_family_aliases:
                        _log.warning('findfont: Generic family %r not found because none of the following families were found: %s', family, ', '.join(self._expand_aliases(family)))
                default_prop = prop.copy()
                default_prop.set_family(self.defaultFamily[fontext])
                return self.findfont(default_prop, fontext, directory, fallback_to_default=False)
            else:
                return _ExceptionProxy(ValueError, f'Failed to find font {prop}, and fallback to the default font was disabled')
        else:
            _log.debug('findfont: Matching %s to %s (%r) with score of %f.', prop, best_font.name, best_font.fname, best_score)
            result = best_font.fname
        if not os.path.isfile(result):
            if rebuild_if_missing:
                _log.info('findfont: Found a missing font file.  Rebuilding cache.')
                new_fm = _load_fontmanager(try_read_cache=False)
                vars(self).update(vars(new_fm))
                return self.findfont(prop, fontext, directory, rebuild_if_missing=False)
            else:
                return _ExceptionProxy(ValueError, 'No valid font could be found')
        return _cached_realpath(result)

@lru_cache
def is_opentype_cff_font(filename):
    if False:
        while True:
            i = 10
    '\n    Return whether the given font is a Postscript Compact Font Format Font\n    embedded in an OpenType wrapper.  Used by the PostScript and PDF backends\n    that cannot subset these fonts.\n    '
    if os.path.splitext(filename)[1].lower() == '.otf':
        with open(filename, 'rb') as fd:
            return fd.read(4) == b'OTTO'
    else:
        return False

@lru_cache(64)
def _get_font(font_filepaths, hinting_factor, *, _kerning_factor, thread_id):
    if False:
        return 10
    (first_fontpath, *rest) = font_filepaths
    return ft2font.FT2Font(first_fontpath, hinting_factor, _fallback_list=[ft2font.FT2Font(fpath, hinting_factor, _kerning_factor=_kerning_factor) for fpath in rest], _kerning_factor=_kerning_factor)
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_get_font.cache_clear)

@lru_cache(64)
def _cached_realpath(path):
    if False:
        print('Hello World!')
    return os.path.realpath(path)

def get_font(font_filepaths, hinting_factor=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get an `.ft2font.FT2Font` object given a list of file paths.\n\n    Parameters\n    ----------\n    font_filepaths : Iterable[str, Path, bytes], str, Path, bytes\n        Relative or absolute paths to the font files to be used.\n\n        If a single string, bytes, or `pathlib.Path`, then it will be treated\n        as a list with that entry only.\n\n        If more than one filepath is passed, then the returned FT2Font object\n        will fall back through the fonts, in the order given, to find a needed\n        glyph.\n\n    Returns\n    -------\n    `.ft2font.FT2Font`\n\n    '
    if isinstance(font_filepaths, (str, Path, bytes)):
        paths = (_cached_realpath(font_filepaths),)
    else:
        paths = tuple((_cached_realpath(fname) for fname in font_filepaths))
    if hinting_factor is None:
        hinting_factor = mpl.rcParams['text.hinting_factor']
    return _get_font(paths, hinting_factor, _kerning_factor=mpl.rcParams['text.kerning_factor'], thread_id=threading.get_ident())

def _load_fontmanager(*, try_read_cache=True):
    if False:
        for i in range(10):
            print('nop')
    fm_path = Path(mpl.get_cachedir(), f'fontlist-v{FontManager.__version__}.json')
    if try_read_cache:
        try:
            fm = json_load(fm_path)
        except Exception:
            pass
        else:
            if getattr(fm, '_version', object()) == FontManager.__version__:
                _log.debug('Using fontManager instance from %s', fm_path)
                return fm
    fm = FontManager()
    json_dump(fm, fm_path)
    _log.info('generated new fontManager')
    return fm
fontManager = _load_fontmanager()
findfont = fontManager.findfont
get_font_names = fontManager.get_font_names