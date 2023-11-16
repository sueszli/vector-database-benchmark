"""
A python interface to Adobe Font Metrics Files.

Although a number of other Python implementations exist, and may be more
complete than this, it was decided not to go with them because they were
either:

1) copyrighted or used a non-BSD compatible license
2) had too many dependencies and a free standing lib was needed
3) did more than needed and it was easier to write afresh rather than
   figure out how to get just what was needed.

It is pretty easy to use, and has no external dependencies:

>>> import matplotlib as mpl
>>> from pathlib import Path
>>> afm_path = Path(mpl.get_data_path(), 'fonts', 'afm', 'ptmr8a.afm')
>>>
>>> from matplotlib.afm import AFM
>>> with afm_path.open('rb') as fh:
...     afm = AFM(fh)
>>> afm.string_width_height('What the heck?')
(6220.0, 694)
>>> afm.get_fontname()
'Times-Roman'
>>> afm.get_kern_dist('A', 'f')
0
>>> afm.get_kern_dist('A', 'y')
-92.0
>>> afm.get_bbox_char('!')
[130, -9, 238, 676]

As in the Adobe Font Metrics File Format Specification, all dimensions
are given in units of 1/1000 of the scale factor (point size) of the font
being used.
"""
from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
_log = logging.getLogger(__name__)

def _to_int(x):
    if False:
        for i in range(10):
            print('nop')
    return int(float(x))

def _to_float(x):
    if False:
        while True:
            i = 10
    if isinstance(x, bytes):
        x = x.decode('latin-1')
    return float(x.replace(',', '.'))

def _to_str(x):
    if False:
        return 10
    return x.decode('utf8')

def _to_list_of_ints(s):
    if False:
        i = 10
        return i + 15
    s = s.replace(b',', b' ')
    return [_to_int(val) for val in s.split()]

def _to_list_of_floats(s):
    if False:
        return 10
    return [_to_float(val) for val in s.split()]

def _to_bool(s):
    if False:
        while True:
            i = 10
    if s.lower().strip() in (b'false', b'0', b'no'):
        return False
    else:
        return True

def _parse_header(fh):
    if False:
        print('Hello World!')
    "\n    Read the font metrics header (up to the char metrics) and returns\n    a dictionary mapping *key* to *val*.  *val* will be converted to the\n    appropriate python type as necessary; e.g.:\n\n        * 'False'->False\n        * '0'->0\n        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]\n\n    Dictionary keys are\n\n      StartFontMetrics, FontName, FullName, FamilyName, Weight,\n      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,\n      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,\n      XHeight, Ascender, Descender, StartCharMetrics\n    "
    header_converters = {b'StartFontMetrics': _to_float, b'FontName': _to_str, b'FullName': _to_str, b'FamilyName': _to_str, b'Weight': _to_str, b'ItalicAngle': _to_float, b'IsFixedPitch': _to_bool, b'FontBBox': _to_list_of_ints, b'UnderlinePosition': _to_float, b'UnderlineThickness': _to_float, b'Version': _to_str, b'Notice': lambda x: x, b'EncodingScheme': _to_str, b'CapHeight': _to_float, b'Capheight': _to_float, b'XHeight': _to_float, b'Ascender': _to_float, b'Descender': _to_float, b'StdHW': _to_float, b'StdVW': _to_float, b'StartCharMetrics': _to_int, b'CharacterSet': _to_str, b'Characters': _to_int}
    d = {}
    first_line = True
    for line in fh:
        line = line.rstrip()
        if line.startswith(b'Comment'):
            continue
        lst = line.split(b' ', 1)
        key = lst[0]
        if first_line:
            if key != b'StartFontMetrics':
                raise RuntimeError('Not an AFM file')
            first_line = False
        if len(lst) == 2:
            val = lst[1]
        else:
            val = b''
        try:
            converter = header_converters[key]
        except KeyError:
            _log.error('Found an unknown keyword in AFM header (was %r)', key)
            continue
        try:
            d[key] = converter(val)
        except ValueError:
            _log.error('Value error parsing header in AFM: %s, %s', key, val)
            continue
        if key == b'StartCharMetrics':
            break
    else:
        raise RuntimeError('Bad parse')
    return d
CharMetrics = namedtuple('CharMetrics', 'width, name, bbox')
CharMetrics.__doc__ = '\n    Represents the character metrics of a single character.\n\n    Notes\n    -----\n    The fields do currently only describe a subset of character metrics\n    information defined in the AFM standard.\n    '
CharMetrics.width.__doc__ = 'The character width (WX).'
CharMetrics.name.__doc__ = 'The character name (N).'
CharMetrics.bbox.__doc__ = '\n    The bbox of the character (B) as a tuple (*llx*, *lly*, *urx*, *ury*).'

def _parse_char_metrics(fh):
    if False:
        return 10
    '\n    Parse the given filehandle for character metrics information and return\n    the information as dicts.\n\n    It is assumed that the file cursor is on the line behind\n    \'StartCharMetrics\'.\n\n    Returns\n    -------\n    ascii_d : dict\n         A mapping "ASCII num of the character" to `.CharMetrics`.\n    name_d : dict\n         A mapping "character name" to `.CharMetrics`.\n\n    Notes\n    -----\n    This function is incomplete per the standard, but thus far parses\n    all the sample afm files tried.\n    '
    required_keys = {'C', 'WX', 'N', 'B'}
    ascii_d = {}
    name_d = {}
    for line in fh:
        line = _to_str(line.rstrip())
        if line.startswith('EndCharMetrics'):
            return (ascii_d, name_d)
        vals = dict((s.strip().split(' ', 1) for s in line.split(';') if s))
        if not required_keys.issubset(vals):
            raise RuntimeError('Bad char metrics line: %s' % line)
        num = _to_int(vals['C'])
        wx = _to_float(vals['WX'])
        name = vals['N']
        bbox = _to_list_of_floats(vals['B'])
        bbox = list(map(int, bbox))
        metrics = CharMetrics(wx, name, bbox)
        if name == 'Euro':
            num = 128
        elif name == 'minus':
            num = ord('−')
        if num != -1:
            ascii_d[num] = metrics
        name_d[name] = metrics
    raise RuntimeError('Bad parse')

def _parse_kern_pairs(fh):
    if False:
        i = 10
        return i + 15
    "\n    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and\n    values are the kern pair value.  For example, a kern pairs line like\n    ``KPX A y -50``\n\n    will be represented as::\n\n      d[ ('A', 'y') ] = -50\n\n    "
    line = next(fh)
    if not line.startswith(b'StartKernPairs'):
        raise RuntimeError('Bad start of kern pairs data: %s' % line)
    d = {}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(b'EndKernPairs'):
            next(fh)
            return d
        vals = line.split()
        if len(vals) != 4 or vals[0] != b'KPX':
            raise RuntimeError('Bad kern pairs line: %s' % line)
        (c1, c2, val) = (_to_str(vals[1]), _to_str(vals[2]), _to_float(vals[3]))
        d[c1, c2] = val
    raise RuntimeError('Bad kern pairs parse')
CompositePart = namedtuple('CompositePart', 'name, dx, dy')
CompositePart.__doc__ = '\n    Represents the information on a composite element of a composite char.'
CompositePart.name.__doc__ = "Name of the part, e.g. 'acute'."
CompositePart.dx.__doc__ = 'x-displacement of the part from the origin.'
CompositePart.dy.__doc__ = 'y-displacement of the part from the origin.'

def _parse_composites(fh):
    if False:
        while True:
            i = 10
    "\n    Parse the given filehandle for composites information return them as a\n    dict.\n\n    It is assumed that the file cursor is on the line behind 'StartComposites'.\n\n    Returns\n    -------\n    dict\n        A dict mapping composite character names to a parts list. The parts\n        list is a list of `.CompositePart` entries describing the parts of\n        the composite.\n\n    Examples\n    --------\n    A composite definition line::\n\n      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;\n\n    will be represented as::\n\n      composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),\n                              CompositePart(name='acute', dx=160, dy=170)]\n\n    "
    composites = {}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(b'EndComposites'):
            return composites
        vals = line.split(b';')
        cc = vals[0].split()
        (name, _num_parts) = (cc[1], _to_int(cc[2]))
        pccParts = []
        for s in vals[1:-1]:
            pcc = s.split()
            part = CompositePart(pcc[1], _to_float(pcc[2]), _to_float(pcc[3]))
            pccParts.append(part)
        composites[name] = pccParts
    raise RuntimeError('Bad composites parse')

def _parse_optional(fh):
    if False:
        print('Hello World!')
    '\n    Parse the optional fields for kern pair data and composites.\n\n    Returns\n    -------\n    kern_data : dict\n        A dict containing kerning information. May be empty.\n        See `._parse_kern_pairs`.\n    composites : dict\n        A dict containing composite information. May be empty.\n        See `._parse_composites`.\n    '
    optional = {b'StartKernData': _parse_kern_pairs, b'StartComposites': _parse_composites}
    d = {b'StartKernData': {}, b'StartComposites': {}}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        key = line.split()[0]
        if key in optional:
            d[key] = optional[key](fh)
    return (d[b'StartKernData'], d[b'StartComposites'])

class AFM:

    def __init__(self, fh):
        if False:
            print('Hello World!')
        'Parse the AFM file in file object *fh*.'
        self._header = _parse_header(fh)
        (self._metrics, self._metrics_by_name) = _parse_char_metrics(fh)
        (self._kern, self._composite) = _parse_optional(fh)

    def get_bbox_char(self, c, isord=False):
        if False:
            i = 10
            return i + 15
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox

    def string_width_height(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the string width (including kerning) and string height\n        as a (*w*, *h*) tuple.\n        '
        if not len(s):
            return (0, 0)
        total_width = 0
        namelast = None
        miny = 1000000000.0
        maxy = 0
        for c in s:
            if c == '\n':
                continue
            (wx, name, bbox) = self._metrics[ord(c)]
            total_width += wx + self._kern.get((namelast, name), 0)
            (l, b, w, h) = bbox
            miny = min(miny, b)
            maxy = max(maxy, b + h)
            namelast = name
        return (total_width, maxy - miny)

    def get_str_bbox_and_descent(self, s):
        if False:
            i = 10
            return i + 15
        'Return the string bounding box and the maximal descent.'
        if not len(s):
            return (0, 0, 0, 0, 0)
        total_width = 0
        namelast = None
        miny = 1000000000.0
        maxy = 0
        left = 0
        if not isinstance(s, str):
            s = _to_str(s)
        for c in s:
            if c == '\n':
                continue
            name = uni2type1.get(ord(c), f'uni{ord(c):04X}')
            try:
                (wx, _, bbox) = self._metrics_by_name[name]
            except KeyError:
                name = 'question'
                (wx, _, bbox) = self._metrics_by_name[name]
            total_width += wx + self._kern.get((namelast, name), 0)
            (l, b, w, h) = bbox
            left = min(left, l)
            miny = min(miny, b)
            maxy = max(maxy, b + h)
            namelast = name
        return (left, miny, total_width, maxy - miny, -miny)

    def get_str_bbox(self, s):
        if False:
            while True:
                i = 10
        'Return the string bounding box.'
        return self.get_str_bbox_and_descent(s)[:4]

    def get_name_char(self, c, isord=False):
        if False:
            i = 10
            return i + 15
        "Get the name of the character, i.e., ';' is 'semicolon'."
        if not isord:
            c = ord(c)
        return self._metrics[c].name

    def get_width_char(self, c, isord=False):
        if False:
            return 10
        '\n        Get the width of the character from the character metric WX field.\n        '
        if not isord:
            c = ord(c)
        return self._metrics[c].width

    def get_width_from_char_name(self, name):
        if False:
            while True:
                i = 10
        'Get the width of the character from a type1 character name.'
        return self._metrics_by_name[name].width

    def get_height_char(self, c, isord=False):
        if False:
            i = 10
            return i + 15
        'Get the bounding box (ink) height of character *c* (space is 0).'
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox[-1]

    def get_kern_dist(self, c1, c2):
        if False:
            return 10
        '\n        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.\n        '
        (name1, name2) = (self.get_name_char(c1), self.get_name_char(c2))
        return self.get_kern_dist_from_name(name1, name2)

    def get_kern_dist_from_name(self, name1, name2):
        if False:
            return 10
        '\n        Return the kerning pair distance (possibly 0) for chars\n        *name1* and *name2*.\n        '
        return self._kern.get((name1, name2), 0)

    def get_fontname(self):
        if False:
            i = 10
            return i + 15
        "Return the font name, e.g., 'Times-Roman'."
        return self._header[b'FontName']

    @property
    def postscript_name(self):
        if False:
            return 10
        return self.get_fontname()

    def get_fullname(self):
        if False:
            while True:
                i = 10
        "Return the font full name, e.g., 'Times-Roman'."
        name = self._header.get(b'FullName')
        if name is None:
            name = self._header[b'FontName']
        return name

    def get_familyname(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the font family name, e.g., 'Times'."
        name = self._header.get(b'FamilyName')
        if name is not None:
            return name
        name = self.get_fullname()
        extras = '(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$'
        return re.sub(extras, '', name)

    @property
    def family_name(self):
        if False:
            return 10
        "The font family name, e.g., 'Times'."
        return self.get_familyname()

    def get_weight(self):
        if False:
            i = 10
            return i + 15
        "Return the font weight, e.g., 'Bold' or 'Roman'."
        return self._header[b'Weight']

    def get_angle(self):
        if False:
            return 10
        'Return the fontangle as float.'
        return self._header[b'ItalicAngle']

    def get_capheight(self):
        if False:
            while True:
                i = 10
        'Return the cap height as float.'
        return self._header[b'CapHeight']

    def get_xheight(self):
        if False:
            while True:
                i = 10
        'Return the xheight as float.'
        return self._header[b'XHeight']

    def get_underline_thickness(self):
        if False:
            print('Hello World!')
        'Return the underline thickness as float.'
        return self._header[b'UnderlineThickness']

    def get_horizontal_stem_width(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the standard horizontal stem width as float, or *None* if\n        not specified in AFM file.\n        '
        return self._header.get(b'StdHW', None)

    def get_vertical_stem_width(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the standard vertical stem width as float, or *None* if\n        not specified in AFM file.\n        '
        return self._header.get(b'StdVW', None)