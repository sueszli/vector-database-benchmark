"""
A module for parsing and generating `fontconfig patterns`_.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
"""
from functools import lru_cache, partial
import re
from pyparsing import Group, Optional, ParseException, Regex, StringEnd, Suppress, ZeroOrMore
from matplotlib import _api
_family_punc = '\\\\\\-:,'
_family_unescape = partial(re.compile('\\\\(?=[%s])' % _family_punc).sub, '')
_family_escape = partial(re.compile('(?=[%s])' % _family_punc).sub, '\\\\')
_value_punc = '\\\\=_:,'
_value_unescape = partial(re.compile('\\\\(?=[%s])' % _value_punc).sub, '')
_value_escape = partial(re.compile('(?=[%s])' % _value_punc).sub, '\\\\')
_CONSTANTS = {'thin': ('weight', 'light'), 'extralight': ('weight', 'light'), 'ultralight': ('weight', 'light'), 'light': ('weight', 'light'), 'book': ('weight', 'book'), 'regular': ('weight', 'regular'), 'normal': ('weight', 'normal'), 'medium': ('weight', 'medium'), 'demibold': ('weight', 'demibold'), 'semibold': ('weight', 'semibold'), 'bold': ('weight', 'bold'), 'extrabold': ('weight', 'extra bold'), 'black': ('weight', 'black'), 'heavy': ('weight', 'heavy'), 'roman': ('slant', 'normal'), 'italic': ('slant', 'italic'), 'oblique': ('slant', 'oblique'), 'ultracondensed': ('width', 'ultra-condensed'), 'extracondensed': ('width', 'extra-condensed'), 'condensed': ('width', 'condensed'), 'semicondensed': ('width', 'semi-condensed'), 'expanded': ('width', 'expanded'), 'extraexpanded': ('width', 'extra-expanded'), 'ultraexpanded': ('width', 'ultra-expanded')}

@lru_cache
def _make_fontconfig_parser():
    if False:
        i = 10
        return i + 15

    def comma_separated(elem):
        if False:
            return 10
        return elem + ZeroOrMore(Suppress(',') + elem)
    family = Regex(f'([^{_family_punc}]|(\\\\[{_family_punc}]))*')
    size = Regex('([0-9]+\\.?[0-9]*|\\.[0-9]+)')
    name = Regex('[a-z]+')
    value = Regex(f'([^{_value_punc}]|(\\\\[{_value_punc}]))*')
    prop = Group(name + Suppress('=') + comma_separated(value) | name)
    return Optional(comma_separated(family)('families')) + Optional('-' + comma_separated(size)('sizes')) + ZeroOrMore(':' + prop('properties*')) + StringEnd()

@lru_cache
def parse_fontconfig_pattern(pattern):
    if False:
        i = 10
        return i + 15
    '\n    Parse a fontconfig *pattern* into a dict that can initialize a\n    `.font_manager.FontProperties` object.\n    '
    parser = _make_fontconfig_parser()
    try:
        parse = parser.parseString(pattern)
    except ParseException as err:
        raise ValueError('\n' + ParseException.explain(err, 0)) from None
    parser.resetCache()
    props = {}
    if 'families' in parse:
        props['family'] = [*map(_family_unescape, parse['families'])]
    if 'sizes' in parse:
        props['size'] = [*parse['sizes']]
    for prop in parse.get('properties', []):
        if len(prop) == 1:
            if prop[0] not in _CONSTANTS:
                _api.warn_deprecated('3.7', message=f'Support for unknown constants ({prop[0]!r}) is deprecated since %(since)s and will be removed %(removal)s.')
                continue
            prop = _CONSTANTS[prop[0]]
        (k, *v) = prop
        props.setdefault(k, []).extend(map(_value_unescape, v))
    return props

def generate_fontconfig_pattern(d):
    if False:
        for i in range(10):
            print('nop')
    'Convert a `.FontProperties` to a fontconfig pattern string.'
    kvs = [(k, getattr(d, f'get_{k}')()) for k in ['style', 'variant', 'weight', 'stretch', 'file', 'size']]
    return ','.join((_family_escape(f) for f in d.get_family())) + ''.join((f':{k}={_value_escape(str(v))}' for (k, v) in kvs if v is not None))