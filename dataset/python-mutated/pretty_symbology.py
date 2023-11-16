"""Symbolic primitives + unicode/ASCII abstraction for pretty.py"""
import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
unicode_warnings = ''

def U(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a unicode character by name or, None if not found.\n\n    This exists because older versions of Python use older unicode databases.\n    '
    try:
        return unicodedata.lookup(name)
    except KeyError:
        global unicode_warnings
        unicode_warnings += "No '%s' in unicodedata\n" % name
        return None
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
__all__ = ['greek_unicode', 'sub', 'sup', 'xsym', 'vobj', 'hobj', 'pretty_symbol', 'annotated', 'center_pad', 'center']
_use_unicode = False

def pretty_use_unicode(flag=None):
    if False:
        i = 10
        return i + 15
    'Set whether pretty-printer should use unicode by default'
    global _use_unicode
    global unicode_warnings
    if flag is None:
        return _use_unicode
    if flag and unicode_warnings:
        warnings.warn(unicode_warnings)
        unicode_warnings = ''
    use_unicode_prev = _use_unicode
    _use_unicode = flag
    return use_unicode_prev

def pretty_try_use_unicode():
    if False:
        return 10
    'See if unicode output is available and leverage it if possible'
    encoding = getattr(sys.stdout, 'encoding', None)
    if encoding is None:
        return
    symbols = []
    symbols += greek_unicode.values()
    symbols += atoms_table.values()
    for s in symbols:
        if s is None:
            return
        try:
            s.encode(encoding)
        except UnicodeEncodeError:
            return
    pretty_use_unicode(True)

def xstr(*args):
    if False:
        i = 10
        return i + 15
    sympy_deprecation_warning('\n        The sympy.printing.pretty.pretty_symbology.xstr() function is\n        deprecated. Use str() instead.\n        ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
    return str(*args)
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())
greek_letters = list(greeks)
greek_letters[greek_letters.index('lambda')] = 'lamda'
greek_unicode = {L: g(L) for L in greek_letters}
greek_unicode.update(((L[0].upper() + L[1:], G(L)) for L in greek_letters))
greek_unicode['lambda'] = greek_unicode['lamda']
greek_unicode['Lambda'] = greek_unicode['Lamda']
greek_unicode['varsigma'] = 'ς'
b = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
B = lambda l: U('MATHEMATICAL BOLD CAPITAL %s' % l.upper())
bold_unicode = {l: b(l) for l in ascii_lowercase}
bold_unicode.update(((L, B(L)) for L in ascii_uppercase))
gb = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
GB = lambda l: U('MATHEMATICAL BOLD CAPITAL  %s' % l.upper())
greek_bold_letters = list(greeks)
greek_bold_letters[greek_bold_letters.index('lambda')] = 'lamda'
greek_bold_unicode = {L: g(L) for L in greek_bold_letters}
greek_bold_unicode.update(((L[0].upper() + L[1:], G(L)) for L in greek_bold_letters))
greek_bold_unicode['lambda'] = greek_unicode['lamda']
greek_bold_unicode['Lambda'] = greek_unicode['Lamda']
greek_bold_unicode['varsigma'] = '𝛓'
digit_2txt = {'0': 'ZERO', '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR', '5': 'FIVE', '6': 'SIX', '7': 'SEVEN', '8': 'EIGHT', '9': 'NINE'}
symb_2txt = {'+': 'PLUS SIGN', '-': 'MINUS', '=': 'EQUALS SIGN', '(': 'LEFT PARENTHESIS', ')': 'RIGHT PARENTHESIS', '[': 'LEFT SQUARE BRACKET', ']': 'RIGHT SQUARE BRACKET', '{': 'LEFT CURLY BRACKET', '}': 'RIGHT CURLY BRACKET', '{}': 'CURLY BRACKET', 'sum': 'SUMMATION', 'int': 'INTEGRAL'}
LSUB = lambda letter: U('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
GSUB = lambda letter: U('GREEK SUBSCRIPT SMALL LETTER %s' % letter.upper())
DSUB = lambda digit: U('SUBSCRIPT %s' % digit_2txt[digit])
SSUB = lambda symb: U('SUBSCRIPT %s' % symb_2txt[symb])
LSUP = lambda letter: U('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
DSUP = lambda digit: U('SUPERSCRIPT %s' % digit_2txt[digit])
SSUP = lambda symb: U('SUPERSCRIPT %s' % symb_2txt[symb])
sub = {}
sup = {}
for l in 'aeioruvxhklmnpst':
    sub[l] = LSUB(l)
for l in 'in':
    sup[l] = LSUP(l)
for gl in ['beta', 'gamma', 'rho', 'phi', 'chi']:
    sub[gl] = GSUB(gl)
for d in [str(i) for i in range(10)]:
    sub[d] = DSUB(d)
    sup[d] = DSUP(d)
for s in '+-=()':
    sub[s] = SSUB(s)
    sup[s] = SSUP(s)
modifier_dict = {'mathring': lambda s: center_accent(s, '̊'), 'ddddot': lambda s: center_accent(s, '⃜'), 'dddot': lambda s: center_accent(s, '⃛'), 'ddot': lambda s: center_accent(s, '̈'), 'dot': lambda s: center_accent(s, '̇'), 'check': lambda s: center_accent(s, '̌'), 'breve': lambda s: center_accent(s, '̆'), 'acute': lambda s: center_accent(s, '́'), 'grave': lambda s: center_accent(s, '̀'), 'tilde': lambda s: center_accent(s, '̃'), 'hat': lambda s: center_accent(s, '̂'), 'bar': lambda s: center_accent(s, '̅'), 'vec': lambda s: center_accent(s, '⃗'), 'prime': lambda s: s + '′', 'prm': lambda s: s + '′', 'norm': lambda s: '‖' + s + '‖', 'avg': lambda s: '⟨' + s + '⟩', 'abs': lambda s: '|' + s + '|', 'mag': lambda s: '|' + s + '|'}
HUP = lambda symb: U('%s UPPER HOOK' % symb_2txt[symb])
CUP = lambda symb: U('%s UPPER CORNER' % symb_2txt[symb])
MID = lambda symb: U('%s MIDDLE PIECE' % symb_2txt[symb])
EXT = lambda symb: U('%s EXTENSION' % symb_2txt[symb])
HLO = lambda symb: U('%s LOWER HOOK' % symb_2txt[symb])
CLO = lambda symb: U('%s LOWER CORNER' % symb_2txt[symb])
TOP = lambda symb: U('%s TOP' % symb_2txt[symb])
BOT = lambda symb: U('%s BOTTOM' % symb_2txt[symb])
_xobj_unicode = {'(': ((EXT('('), HUP('('), HLO('(')), '('), ')': ((EXT(')'), HUP(')'), HLO(')')), ')'), '[': ((EXT('['), CUP('['), CLO('[')), '['), ']': ((EXT(']'), CUP(']'), CLO(']')), ']'), '{': ((EXT('{}'), HUP('{'), HLO('{'), MID('{')), '{'), '}': ((EXT('{}'), HUP('}'), HLO('}'), MID('}')), '}'), '|': U('BOX DRAWINGS LIGHT VERTICAL'), '<': ((U('BOX DRAWINGS LIGHT VERTICAL'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')), '<'), '>': ((U('BOX DRAWINGS LIGHT VERTICAL'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), '>'), 'lfloor': ((EXT('['), EXT('['), CLO('[')), U('LEFT FLOOR')), 'rfloor': ((EXT(']'), EXT(']'), CLO(']')), U('RIGHT FLOOR')), 'lceil': ((EXT('['), CUP('['), EXT('[')), U('LEFT CEILING')), 'rceil': ((EXT(']'), CUP(']'), EXT(']')), U('RIGHT CEILING')), 'int': ((EXT('int'), U('TOP HALF INTEGRAL'), U('BOTTOM HALF INTEGRAL')), U('INTEGRAL')), 'sum': ((U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'), '_', U('OVERLINE'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), U('N-ARY SUMMATION')), '-': U('BOX DRAWINGS LIGHT HORIZONTAL'), '_': U('LOW LINE'), '/': U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'), '\\': U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')}
_xobj_ascii = {'(': (('|', '/', '\\'), '('), ')': (('|', '\\', '/'), ')'), '[': (('[', '[', '['), '['), ']': ((']', ']', ']'), ']'), '{': (('|', '/', '\\', '<'), '{'), '}': (('|', '\\', '/', '>'), '}'), '|': '|', '<': (('|', '/', '\\'), '<'), '>': (('|', '\\', '/'), '>'), 'int': (' | ', '  /', '/  '), '-': '-', '_': '_', '/': '/', '\\': '\\'}

def xobj(symb, length):
    if False:
        i = 10
        return i + 15
    'Construct spatial object of given length.\n\n    return: [] of equal-length strings\n    '
    if length <= 0:
        raise ValueError('Length should be greater than 0')
    if _use_unicode:
        _xobj = _xobj_unicode
    else:
        _xobj = _xobj_ascii
    vinfo = _xobj[symb]
    c1 = top = bot = mid = None
    if not isinstance(vinfo, tuple):
        ext = vinfo
    else:
        if isinstance(vinfo[0], tuple):
            vlong = vinfo[0]
            c1 = vinfo[1]
        else:
            vlong = vinfo
        ext = vlong[0]
        try:
            top = vlong[1]
            bot = vlong[2]
            mid = vlong[3]
        except IndexError:
            pass
    if c1 is None:
        c1 = ext
    if top is None:
        top = ext
    if bot is None:
        bot = ext
    if mid is not None:
        if length % 2 == 0:
            length += 1
    else:
        mid = ext
    if length == 1:
        return c1
    res = []
    next = (length - 2) // 2
    nmid = length - 2 - next * 2
    res += [top]
    res += [ext] * next
    res += [mid] * nmid
    res += [ext] * next
    res += [bot]
    return res

def vobj(symb, height):
    if False:
        for i in range(10):
            print('nop')
    'Construct vertical object of a given height\n\n       see: xobj\n    '
    return '\n'.join(xobj(symb, height))

def hobj(symb, width):
    if False:
        return 10
    'Construct horizontal object of a given width\n\n       see: xobj\n    '
    return ''.join(xobj(symb, width))
root = {2: U('SQUARE ROOT'), 3: U('CUBE ROOT'), 4: U('FOURTH ROOT')}
VF = lambda txt: U('VULGAR FRACTION %s' % txt)
frac = {(1, 2): VF('ONE HALF'), (1, 3): VF('ONE THIRD'), (2, 3): VF('TWO THIRDS'), (1, 4): VF('ONE QUARTER'), (3, 4): VF('THREE QUARTERS'), (1, 5): VF('ONE FIFTH'), (2, 5): VF('TWO FIFTHS'), (3, 5): VF('THREE FIFTHS'), (4, 5): VF('FOUR FIFTHS'), (1, 6): VF('ONE SIXTH'), (5, 6): VF('FIVE SIXTHS'), (1, 8): VF('ONE EIGHTH'), (3, 8): VF('THREE EIGHTHS'), (5, 8): VF('FIVE EIGHTHS'), (7, 8): VF('SEVEN EIGHTHS')}
_xsym = {'==': ('=', '='), '<': ('<', '<'), '>': ('>', '>'), '<=': ('<=', U('LESS-THAN OR EQUAL TO')), '>=': ('>=', U('GREATER-THAN OR EQUAL TO')), '!=': ('!=', U('NOT EQUAL TO')), ':=': (':=', ':='), '+=': ('+=', '+='), '-=': ('-=', '-='), '*=': ('*=', '*='), '/=': ('/=', '/='), '%=': ('%=', '%='), '*': ('*', U('DOT OPERATOR')), '-->': ('-->', U('EM DASH') + U('EM DASH') + U('BLACK RIGHT-POINTING TRIANGLE') if U('EM DASH') and U('BLACK RIGHT-POINTING TRIANGLE') else None), '==>': ('==>', U('BOX DRAWINGS DOUBLE HORIZONTAL') + U('BOX DRAWINGS DOUBLE HORIZONTAL') + U('BLACK RIGHT-POINTING TRIANGLE') if U('BOX DRAWINGS DOUBLE HORIZONTAL') and U('BOX DRAWINGS DOUBLE HORIZONTAL') and U('BLACK RIGHT-POINTING TRIANGLE') else None), '.': ('*', U('RING OPERATOR'))}

def xsym(sym):
    if False:
        return 10
    "get symbology for a 'character'"
    op = _xsym[sym]
    if _use_unicode:
        return op[1]
    else:
        return op[0]
atoms_table = {'Exp1': U('SCRIPT SMALL E'), 'Pi': U('GREEK SMALL LETTER PI'), 'Infinity': U('INFINITY'), 'NegativeInfinity': U('INFINITY') and '-' + U('INFINITY'), 'ImaginaryUnit': U('DOUBLE-STRUCK ITALIC SMALL I'), 'EmptySet': U('EMPTY SET'), 'Naturals': U('DOUBLE-STRUCK CAPITAL N'), 'Naturals0': U('DOUBLE-STRUCK CAPITAL N') and U('DOUBLE-STRUCK CAPITAL N') + U('SUBSCRIPT ZERO'), 'Integers': U('DOUBLE-STRUCK CAPITAL Z'), 'Rationals': U('DOUBLE-STRUCK CAPITAL Q'), 'Reals': U('DOUBLE-STRUCK CAPITAL R'), 'Complexes': U('DOUBLE-STRUCK CAPITAL C'), 'Union': U('UNION'), 'SymmetricDifference': U('INCREMENT'), 'Intersection': U('INTERSECTION'), 'Ring': U('RING OPERATOR'), 'Modifier Letter Low Ring': U('Modifier Letter Low Ring'), 'EmptySequence': 'EmptySequence'}

def pretty_atom(atom_name, default=None, printer=None):
    if False:
        return 10
    'return pretty representation of an atom'
    if _use_unicode:
        if printer is not None and atom_name == 'ImaginaryUnit' and (printer._settings['imaginary_unit'] == 'j'):
            return U('DOUBLE-STRUCK ITALIC SMALL J')
        else:
            return atoms_table[atom_name]
    else:
        if default is not None:
            return default
        raise KeyError('only unicode')

def pretty_symbol(symb_name, bold_name=False):
    if False:
        return 10
    'return pretty representation of a symbol'
    if not _use_unicode:
        return symb_name
    (name, sups, subs) = split_super_sub(symb_name)

    def translate(s, bold_name):
        if False:
            for i in range(10):
                print('nop')
        if bold_name:
            gG = greek_bold_unicode.get(s)
        else:
            gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        for key in sorted(modifier_dict.keys(), key=lambda k: len(k), reverse=True):
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)], bold_name))
        if bold_name:
            return ''.join([bold_unicode[c] for c in s])
        return s
    name = translate(name, bold_name)

    def pretty_list(l, mapping):
        if False:
            while True:
                i = 10
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result
    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None
    if pretty_subs is None:
        if subs:
            name += '_' + '_'.join([translate(s, bold_name) for s in subs])
        if sups:
            name += '__' + '__'.join([translate(s, bold_name) for s in sups])
        return name
    else:
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)
    return ''.join([name, sups_result, subs_result])

def annotated(letter):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a stylised drawing of the letter ``letter``, together with\n    information on how to put annotations (super- and subscripts to the\n    left and to the right) on it.\n\n    See pretty.py functions _print_meijerg, _print_hyper on how to use this\n    information.\n    '
    ucode_pics = {'F': (2, 0, 2, 0, '┌─\n├─\n╵'), 'G': (3, 0, 3, 1, '╭─╮\n│╶┐\n╰─╯')}
    ascii_pics = {'F': (3, 0, 3, 0, ' _\n|_\n|\n'), 'G': (3, 0, 3, 1, ' __\n/__\n\\_|')}
    if _use_unicode:
        return ucode_pics[letter]
    else:
        return ascii_pics[letter]
_remove_combining = dict.fromkeys(list(range(ord('̀'), ord('ͯ'))) + list(range(ord('⃐'), ord('⃰'))))

def is_combining(sym):
    if False:
        i = 10
        return i + 15
    'Check whether symbol is a unicode modifier. '
    return ord(sym) in _remove_combining

def center_accent(string, accent):
    if False:
        print('Hello World!')
    '\n    Returns a string with accent inserted on the middle character. Useful to\n    put combining accents on symbol names, including multi-character names.\n\n    Parameters\n    ==========\n\n    string : string\n        The string to place the accent in.\n    accent : string\n        The combining accent to insert\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Combining_character\n    .. [2] https://en.wikipedia.org/wiki/Combining_Diacritical_Marks\n\n    '
    midpoint = len(string) // 2 + 1
    firstpart = string[:midpoint]
    secondpart = string[midpoint:]
    return firstpart + accent + secondpart

def line_width(line):
    if False:
        print('Hello World!')
    'Unicode combining symbols (modifiers) are not ever displayed as\n    separate symbols and thus should not be counted\n    '
    return len(line.translate(_remove_combining))

def is_subscriptable_in_unicode(subscript):
    if False:
        return 10
    "\n    Checks whether a string is subscriptable in unicode or not.\n\n    Parameters\n    ==========\n\n    subscript: the string which needs to be checked\n\n    Examples\n    ========\n\n    >>> from sympy.printing.pretty.pretty_symbology import is_subscriptable_in_unicode\n    >>> is_subscriptable_in_unicode('abc')\n    False\n    >>> is_subscriptable_in_unicode('123')\n    True\n\n    "
    return all((character in sub for character in subscript))

def center_pad(wstring, wtarget, fillchar=' '):
    if False:
        while True:
            i = 10
    '\n    Return the padding strings necessary to center a string of\n    wstring characters wide in a wtarget wide space.\n\n    The line_width wstring should always be less or equal to wtarget\n    or else a ValueError will be raised.\n    '
    if wstring > wtarget:
        raise ValueError('not enough space for string')
    wdelta = wtarget - wstring
    wleft = wdelta // 2
    wright = wdelta - wleft
    left = fillchar * wleft
    right = fillchar * wright
    return (left, right)

def center(string, width, fillchar=' '):
    if False:
        while True:
            i = 10
    'Return a centered string of length determined by `line_width`\n    that uses `fillchar` for padding.\n    '
    (left, right) = center_pad(line_width(string), width, fillchar)
    return ''.join([left, string, right])