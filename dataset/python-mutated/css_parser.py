"""CSS selector parser."""
from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
UNICODE_REPLACEMENT_CHAR = 65533
PSEUDO_SIMPLE = {':any-link', ':empty', ':first-child', ':first-of-type', ':in-range', ':out-of-range', ':last-child', ':last-of-type', ':link', ':only-child', ':only-of-type', ':root', ':checked', ':default', ':disabled', ':enabled', ':indeterminate', ':optional', ':placeholder-shown', ':read-only', ':read-write', ':required', ':scope', ':defined'}
PSEUDO_SIMPLE_NO_MATCH = {':active', ':current', ':focus', ':focus-visible', ':focus-within', ':future', ':host', ':hover', ':local-link', ':past', ':paused', ':playing', ':target', ':target-within', ':user-invalid', ':visited'}
PSEUDO_COMPLEX = {':contains', ':-soup-contains', ':-soup-contains-own', ':has', ':is', ':matches', ':not', ':where'}
PSEUDO_COMPLEX_NO_MATCH = {':current', ':host', ':host-context'}
PSEUDO_SPECIAL = {':dir', ':lang', ':nth-child', ':nth-last-child', ':nth-last-of-type', ':nth-of-type'}
PSEUDO_SUPPORTED = PSEUDO_SIMPLE | PSEUDO_SIMPLE_NO_MATCH | PSEUDO_COMPLEX | PSEUDO_COMPLEX_NO_MATCH | PSEUDO_SPECIAL
NEWLINE = '(?:\\r\\n|(?!\\r\\n)[\\n\\f\\r])'
WS = '(?:[ \\t]|{})'.format(NEWLINE)
COMMENTS = '(?:/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/)'
WSC = '(?:{ws}|{comments})'.format(ws=WS, comments=COMMENTS)
CSS_ESCAPES = '(?:\\\\(?:[a-f0-9]{{1,6}}{ws}?|[^\\r\\n\\f]|$))'.format(ws=WS)
CSS_STRING_ESCAPES = '(?:\\\\(?:[a-f0-9]{{1,6}}{ws}?|[^\\r\\n\\f]|$|{nl}))'.format(ws=WS, nl=NEWLINE)
IDENTIFIER = '\n(?:(?:-?(?:[^\\x00-\\x2f\\x30-\\x40\\x5B-\\x5E\\x60\\x7B-\\x9f]|{esc})+|--)\n(?:[^\\x00-\\x2c\\x2e\\x2f\\x3A-\\x40\\x5B-\\x5E\\x60\\x7B-\\x9f]|{esc})*)\n'.format(esc=CSS_ESCAPES)
NTH = '(?:[-+])?(?:[0-9]+n?|n)(?:(?<=n){ws}*(?:[-+]){ws}*(?:[0-9]+))?'.format(ws=WSC)
VALUE = '\n(?:"(?:\\\\(?:.|{nl})|[^\\\\"\\r\\n\\f]+)*?"|\'(?:\\\\(?:.|{nl})|[^\\\\\'\\r\\n\\f]+)*?\'|{ident}+)\n'.format(nl=NEWLINE, ident=IDENTIFIER)
ATTR = '\n(?:{ws}*(?P<cmp>[!~^|*$]?=){ws}*(?P<value>{value})(?:{ws}*(?P<case>[is]))?)?{ws}*\\]\n'.format(ws=WSC, value=VALUE)
PAT_ID = '\\#{ident}'.format(ident=IDENTIFIER)
PAT_CLASS = '\\.{ident}'.format(ident=IDENTIFIER)
PAT_TAG = '(?P<tag_ns>(?:{ident}|\\*)?\\|)?(?P<tag_name>{ident}|\\*)'.format(ident=IDENTIFIER)
PAT_ATTR = '\n\\[{ws}*(?P<attr_ns>(?:{ident}|\\*)?\\|)?(?P<attr_name>{ident}){attr}\n'.format(ws=WSC, ident=IDENTIFIER, attr=ATTR)
PAT_PSEUDO_CLASS = '(?P<name>:{ident})(?P<open>\\({ws}*)?'.format(ws=WSC, ident=IDENTIFIER)
PAT_PSEUDO_CLASS_SPECIAL = '(?P<name>:{ident})(?P<open>\\({ws}*)'.format(ws=WSC, ident=IDENTIFIER)
PAT_PSEUDO_CLASS_CUSTOM = '(?P<name>:(?=--){ident})'.format(ident=IDENTIFIER)
PAT_PSEUDO_CLOSE = '{ws}*\\)'.format(ws=WSC)
PAT_PSEUDO_ELEMENT = ':{}'.format(PAT_PSEUDO_CLASS)
PAT_AT_RULE = '@P{ident}'.format(ident=IDENTIFIER)
PAT_PSEUDO_NTH_CHILD = '\n(?P<pseudo_nth_child>{name}\n(?P<nth_child>{nth}|even|odd))(?:{wsc}*\\)|(?P<of>{comments}*{ws}{wsc}*of{comments}*{ws}{wsc}*))\n'.format(name=PAT_PSEUDO_CLASS_SPECIAL, wsc=WSC, comments=COMMENTS, ws=WS, nth=NTH)
PAT_PSEUDO_NTH_TYPE = '\n(?P<pseudo_nth_type>{name}\n(?P<nth_type>{nth}|even|odd)){ws}*\\)\n'.format(name=PAT_PSEUDO_CLASS_SPECIAL, ws=WSC, nth=NTH)
PAT_PSEUDO_LANG = '{name}(?P<values>{value}(?:{ws}*,{ws}*{value})*){ws}*\\)'.format(name=PAT_PSEUDO_CLASS_SPECIAL, ws=WSC, value=VALUE)
PAT_PSEUDO_DIR = '{name}(?P<dir>ltr|rtl){ws}*\\)'.format(name=PAT_PSEUDO_CLASS_SPECIAL, ws=WSC)
PAT_COMBINE = '{wsc}*?(?P<relation>[,+>~]|{ws}(?![,+>~])){wsc}*'.format(ws=WS, wsc=WSC)
PAT_PSEUDO_CONTAINS = '{name}(?P<values>{value}(?:{ws}*,{ws}*{value})*){ws}*\\)'.format(name=PAT_PSEUDO_CLASS_SPECIAL, ws=WSC, value=VALUE)
RE_CSS_ESC = re.compile('(?:(\\\\[a-f0-9]{{1,6}}{ws}?)|(\\\\[^\\r\\n\\f])|(\\\\$))'.format(ws=WSC), re.I)
RE_CSS_STR_ESC = re.compile('(?:(\\\\[a-f0-9]{{1,6}}{ws}?)|(\\\\[^\\r\\n\\f])|(\\\\$)|(\\\\{nl}))'.format(ws=WS, nl=NEWLINE), re.I)
RE_NTH = re.compile('(?P<s1>[-+])?(?P<a>[0-9]+n?|n)(?:(?<=n){ws}*(?P<s2>[-+]){ws}*(?P<b>[0-9]+))?'.format(ws=WSC), re.I)
RE_VALUES = re.compile('(?:(?P<value>{value})|(?P<split>{ws}*,{ws}*))'.format(ws=WSC, value=VALUE), re.X)
RE_WS = re.compile(WS)
RE_WS_BEGIN = re.compile('^{}*'.format(WSC))
RE_WS_END = re.compile('{}*$'.format(WSC))
RE_CUSTOM = re.compile('^{}$'.format(PAT_PSEUDO_CLASS_CUSTOM), re.X)
COMMA_COMBINATOR = ','
WS_COMBINATOR = ' '
FLG_PSEUDO = 1
FLG_NOT = 2
FLG_RELATIVE = 4
FLG_DEFAULT = 8
FLG_HTML = 16
FLG_INDETERMINATE = 32
FLG_OPEN = 64
FLG_IN_RANGE = 128
FLG_OUT_OF_RANGE = 256
FLG_PLACEHOLDER_SHOWN = 512
FLG_FORGIVE = 1024
_MAXCACHE = 500

@lru_cache(maxsize=_MAXCACHE)
def _cached_css_compile(pattern: str, namespaces: ct.Namespaces | None, custom: ct.CustomSelectors | None, flags: int) -> cm.SoupSieve:
    if False:
        while True:
            i = 10
    'Cached CSS compile.'
    custom_selectors = process_custom(custom)
    return cm.SoupSieve(pattern, CSSParser(pattern, custom=custom_selectors, flags=flags).process_selectors(), namespaces, custom, flags)

def _purge_cache() -> None:
    if False:
        return 10
    'Purge the cache.'
    _cached_css_compile.cache_clear()

def process_custom(custom: ct.CustomSelectors | None) -> dict[str, str | ct.SelectorList]:
    if False:
        while True:
            i = 10
    'Process custom.'
    custom_selectors = {}
    if custom is not None:
        for (key, value) in custom.items():
            name = util.lower(key)
            if RE_CUSTOM.match(name) is None:
                raise SelectorSyntaxError("The name '{}' is not a valid custom pseudo-class name".format(name))
            if name in custom_selectors:
                raise KeyError("The custom selector '{}' has already been registered".format(name))
            custom_selectors[css_unescape(name)] = value
    return custom_selectors

def css_unescape(content: str, string: bool=False) -> str:
    if False:
        while True:
            i = 10
    '\n    Unescape CSS value.\n\n    Strings allow for spanning the value on multiple strings by escaping a new line.\n    '

    def replace(m: Match[str]) -> str:
        if False:
            i = 10
            return i + 15
        'Replace with the appropriate substitute.'
        if m.group(1):
            codepoint = int(m.group(1)[1:], 16)
            if codepoint == 0:
                codepoint = UNICODE_REPLACEMENT_CHAR
            value = chr(codepoint)
        elif m.group(2):
            value = m.group(2)[1:]
        elif m.group(3):
            value = '�'
        else:
            value = ''
        return value
    return (RE_CSS_ESC if not string else RE_CSS_STR_ESC).sub(replace, content)

def escape(ident: str) -> str:
    if False:
        while True:
            i = 10
    'Escape identifier.'
    string = []
    length = len(ident)
    start_dash = length > 0 and ident[0] == '-'
    if length == 1 and start_dash:
        string.append('\\{}'.format(ident))
    else:
        for (index, c) in enumerate(ident):
            codepoint = ord(c)
            if codepoint == 0:
                string.append('�')
            elif 1 <= codepoint <= 31 or codepoint == 127:
                string.append('\\{:x} '.format(codepoint))
            elif (index == 0 or (start_dash and index == 1)) and 48 <= codepoint <= 57:
                string.append('\\{:x} '.format(codepoint))
            elif codepoint in (45, 95) or codepoint >= 128 or 48 <= codepoint <= 57 or (48 <= codepoint <= 57) or (65 <= codepoint <= 90) or (97 <= codepoint <= 122):
                string.append(c)
            else:
                string.append('\\{}'.format(c))
    return ''.join(string)

class SelectorPattern:
    """Selector pattern."""

    def __init__(self, name: str, pattern: str) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize.'
        self.name = name
        self.re_pattern = re.compile(pattern, re.I | re.X | re.U)

    def get_name(self) -> str:
        if False:
            return 10
        'Get name.'
        return self.name

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        if False:
            print('Hello World!')
        'Match the selector.'
        return self.re_pattern.match(selector, index)

class SpecialPseudoPattern(SelectorPattern):
    """Selector pattern."""

    def __init__(self, patterns: tuple[tuple[str, tuple[str, ...], str, type[SelectorPattern]], ...]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        self.patterns = {}
        for p in patterns:
            name = p[0]
            pattern = p[3](name, p[2])
            for pseudo in p[1]:
                self.patterns[pseudo] = pattern
        self.matched_name = None
        self.re_pseudo_name = re.compile(PAT_PSEUDO_CLASS_SPECIAL, re.I | re.X | re.U)

    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get name.'
        return '' if self.matched_name is None else self.matched_name.get_name()

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        if False:
            print('Hello World!')
        'Match the selector.'
        pseudo = None
        m = self.re_pseudo_name.match(selector, index)
        if m:
            name = util.lower(css_unescape(m.group('name')))
            pattern = self.patterns.get(name)
            if pattern:
                pseudo = pattern.match(selector, index, flags)
                if pseudo:
                    self.matched_name = pattern
        return pseudo

class _Selector:
    """
    Intermediate selector class.

    This stores selector data for a compound selector as we are acquiring them.
    Once we are done collecting the data for a compound selector, we freeze
    the data in an object that can be pickled and hashed.
    """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Initialize.'
        self.tag = kwargs.get('tag', None)
        self.ids = kwargs.get('ids', [])
        self.classes = kwargs.get('classes', [])
        self.attributes = kwargs.get('attributes', [])
        self.nth = kwargs.get('nth', [])
        self.selectors = kwargs.get('selectors', [])
        self.relations = kwargs.get('relations', [])
        self.rel_type = kwargs.get('rel_type', None)
        self.contains = kwargs.get('contains', [])
        self.lang = kwargs.get('lang', [])
        self.flags = kwargs.get('flags', 0)
        self.no_match = kwargs.get('no_match', False)

    def _freeze_relations(self, relations: list[_Selector]) -> ct.SelectorList:
        if False:
            for i in range(10):
                print('nop')
        'Freeze relation.'
        if relations:
            sel = relations[0]
            sel.relations.extend(relations[1:])
            return ct.SelectorList([sel.freeze()])
        else:
            return ct.SelectorList()

    def freeze(self) -> ct.Selector | ct.SelectorNull:
        if False:
            print('Hello World!')
        'Freeze self.'
        if self.no_match:
            return ct.SelectorNull()
        else:
            return ct.Selector(self.tag, tuple(self.ids), tuple(self.classes), tuple(self.attributes), tuple(self.nth), tuple(self.selectors), self._freeze_relations(self.relations), self.rel_type, tuple(self.contains), tuple(self.lang), self.flags)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        'String representation.'
        return '_Selector(tag={!r}, ids={!r}, classes={!r}, attributes={!r}, nth={!r}, selectors={!r}, relations={!r}, rel_type={!r}, contains={!r}, lang={!r}, flags={!r}, no_match={!r})'.format(self.tag, self.ids, self.classes, self.attributes, self.nth, self.selectors, self.relations, self.rel_type, self.contains, self.lang, self.flags, self.no_match)
    __repr__ = __str__

class CSSParser:
    """Parse CSS selectors."""
    css_tokens = (SelectorPattern('pseudo_close', PAT_PSEUDO_CLOSE), SpecialPseudoPattern((('pseudo_contains', (':contains', ':-soup-contains', ':-soup-contains-own'), PAT_PSEUDO_CONTAINS, SelectorPattern), ('pseudo_nth_child', (':nth-child', ':nth-last-child'), PAT_PSEUDO_NTH_CHILD, SelectorPattern), ('pseudo_nth_type', (':nth-of-type', ':nth-last-of-type'), PAT_PSEUDO_NTH_TYPE, SelectorPattern), ('pseudo_lang', (':lang',), PAT_PSEUDO_LANG, SelectorPattern), ('pseudo_dir', (':dir',), PAT_PSEUDO_DIR, SelectorPattern))), SelectorPattern('pseudo_class_custom', PAT_PSEUDO_CLASS_CUSTOM), SelectorPattern('pseudo_class', PAT_PSEUDO_CLASS), SelectorPattern('pseudo_element', PAT_PSEUDO_ELEMENT), SelectorPattern('at_rule', PAT_AT_RULE), SelectorPattern('id', PAT_ID), SelectorPattern('class', PAT_CLASS), SelectorPattern('tag', PAT_TAG), SelectorPattern('attribute', PAT_ATTR), SelectorPattern('combine', PAT_COMBINE))

    def __init__(self, selector: str, custom: dict[str, str | ct.SelectorList] | None=None, flags: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        self.pattern = selector.replace('\x00', '�')
        self.flags = flags
        self.debug = self.flags & util.DEBUG
        self.custom = {} if custom is None else custom

    def parse_attribute_selector(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            i = 10
            return i + 15
        'Create attribute selector from the returned regex match.'
        inverse = False
        op = m.group('cmp')
        case = util.lower(m.group('case')) if m.group('case') else None
        ns = css_unescape(m.group('attr_ns')[:-1]) if m.group('attr_ns') else ''
        attr = css_unescape(m.group('attr_name'))
        is_type = False
        pattern2 = None
        value = ''
        if case:
            flags = (re.I if case == 'i' else 0) | re.DOTALL
        elif util.lower(attr) == 'type':
            flags = re.I | re.DOTALL
            is_type = True
        else:
            flags = re.DOTALL
        if op:
            if m.group('value').startswith(('"', "'")):
                value = css_unescape(m.group('value')[1:-1], True)
            else:
                value = css_unescape(m.group('value'))
        if not op:
            pattern = None
        elif op.startswith('^'):
            pattern = re.compile('^%s.*' % re.escape(value), flags)
        elif op.startswith('$'):
            pattern = re.compile('.*?%s$' % re.escape(value), flags)
        elif op.startswith('*'):
            pattern = re.compile('.*?%s.*' % re.escape(value), flags)
        elif op.startswith('~'):
            value = '[^\\s\\S]' if not value or RE_WS.search(value) else re.escape(value)
            pattern = re.compile('.*?(?:(?<=^)|(?<=[ \\t\\r\\n\\f]))%s(?=(?:[ \\t\\r\\n\\f]|$)).*' % value, flags)
        elif op.startswith('|'):
            pattern = re.compile('^%s(?:-.*)?$' % re.escape(value), flags)
        else:
            pattern = re.compile('^%s$' % re.escape(value), flags)
            if op.startswith('!'):
                inverse = True
        if is_type and pattern:
            pattern2 = re.compile(pattern.pattern)
        sel_attr = ct.SelectorAttribute(attr, ns, pattern, pattern2)
        if inverse:
            sub_sel = _Selector()
            sub_sel.attributes.append(sel_attr)
            not_list = ct.SelectorList([sub_sel.freeze()], True, False)
            sel.selectors.append(not_list)
        else:
            sel.attributes.append(sel_attr)
        has_selector = True
        return has_selector

    def parse_tag_pattern(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            print('Hello World!')
        'Parse tag pattern from regex match.'
        prefix = css_unescape(m.group('tag_ns')[:-1]) if m.group('tag_ns') else None
        tag = css_unescape(m.group('tag_name'))
        sel.tag = ct.SelectorTag(tag, prefix)
        has_selector = True
        return has_selector

    def parse_pseudo_class_custom(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            print('Hello World!')
        '\n        Parse custom pseudo class alias.\n\n        Compile custom selectors as we need them. When compiling a custom selector,\n        set it to `None` in the dictionary so we can avoid an infinite loop.\n        '
        pseudo = util.lower(css_unescape(m.group('name')))
        selector = self.custom.get(pseudo)
        if selector is None:
            raise SelectorSyntaxError("Undefined custom selector '{}' found at position {}".format(pseudo, m.end(0)), self.pattern, m.end(0))
        if not isinstance(selector, ct.SelectorList):
            del self.custom[pseudo]
            selector = CSSParser(selector, custom=self.custom, flags=self.flags).process_selectors(flags=FLG_PSEUDO)
            self.custom[pseudo] = selector
        sel.selectors.append(selector)
        has_selector = True
        return has_selector

    def parse_pseudo_class(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], is_html: bool) -> tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Parse pseudo class.'
        complex_pseudo = False
        pseudo = util.lower(css_unescape(m.group('name')))
        if m.group('open'):
            complex_pseudo = True
        if complex_pseudo and pseudo in PSEUDO_COMPLEX:
            has_selector = self.parse_pseudo_open(sel, pseudo, has_selector, iselector, m.end(0))
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE:
            if pseudo == ':root':
                sel.flags |= ct.SEL_ROOT
            elif pseudo == ':defined':
                sel.flags |= ct.SEL_DEFINED
                is_html = True
            elif pseudo == ':scope':
                sel.flags |= ct.SEL_SCOPE
            elif pseudo == ':empty':
                sel.flags |= ct.SEL_EMPTY
            elif pseudo in (':link', ':any-link'):
                sel.selectors.append(CSS_LINK)
            elif pseudo == ':checked':
                sel.selectors.append(CSS_CHECKED)
            elif pseudo == ':default':
                sel.selectors.append(CSS_DEFAULT)
            elif pseudo == ':indeterminate':
                sel.selectors.append(CSS_INDETERMINATE)
            elif pseudo == ':disabled':
                sel.selectors.append(CSS_DISABLED)
            elif pseudo == ':enabled':
                sel.selectors.append(CSS_ENABLED)
            elif pseudo == ':required':
                sel.selectors.append(CSS_REQUIRED)
            elif pseudo == ':optional':
                sel.selectors.append(CSS_OPTIONAL)
            elif pseudo == ':read-only':
                sel.selectors.append(CSS_READ_ONLY)
            elif pseudo == ':read-write':
                sel.selectors.append(CSS_READ_WRITE)
            elif pseudo == ':in-range':
                sel.selectors.append(CSS_IN_RANGE)
            elif pseudo == ':out-of-range':
                sel.selectors.append(CSS_OUT_OF_RANGE)
            elif pseudo == ':placeholder-shown':
                sel.selectors.append(CSS_PLACEHOLDER_SHOWN)
            elif pseudo == ':first-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()))
            elif pseudo == ':last-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, True, ct.SelectorList()))
            elif pseudo == ':first-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()))
            elif pseudo == ':last-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, True, ct.SelectorList()))
            elif pseudo == ':only-child':
                sel.nth.extend([ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, False, True, ct.SelectorList())])
            elif pseudo == ':only-of-type':
                sel.nth.extend([ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, True, True, ct.SelectorList())])
            has_selector = True
        elif complex_pseudo and pseudo in PSEUDO_COMPLEX_NO_MATCH:
            self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            sel.no_match = True
            has_selector = True
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE_NO_MATCH:
            sel.no_match = True
            has_selector = True
        elif pseudo in PSEUDO_SUPPORTED:
            raise SelectorSyntaxError("Invalid syntax for pseudo class '{}'".format(pseudo), self.pattern, m.start(0))
        else:
            raise NotImplementedError("'{}' pseudo-class is not implemented at this time".format(pseudo))
        return (has_selector, is_html)

    def parse_pseudo_nth(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]]) -> bool:
        if False:
            return 10
        'Parse `nth` pseudo.'
        mdict = m.groupdict()
        if mdict.get('pseudo_nth_child'):
            postfix = '_child'
        else:
            postfix = '_type'
        mdict['name'] = util.lower(css_unescape(mdict['name']))
        content = util.lower(mdict.get('nth' + postfix))
        if content == 'even':
            s1 = 2
            s2 = 0
            var = True
        elif content == 'odd':
            s1 = 2
            s2 = 1
            var = True
        else:
            nth_parts = cast(Match[str], RE_NTH.match(content))
            _s1 = '-' if nth_parts.group('s1') and nth_parts.group('s1') == '-' else ''
            a = nth_parts.group('a')
            var = a.endswith('n')
            if a.startswith('n'):
                _s1 += '1'
            elif var:
                _s1 += a[:-1]
            else:
                _s1 += a
            _s2 = '-' if nth_parts.group('s2') and nth_parts.group('s2') == '-' else ''
            if nth_parts.group('b'):
                _s2 += nth_parts.group('b')
            else:
                _s2 = '0'
            s1 = int(_s1, 10)
            s2 = int(_s2, 10)
        pseudo_sel = mdict['name']
        if postfix == '_child':
            if m.group('of'):
                nth_sel = self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            else:
                nth_sel = CSS_NTH_OF_S_DEFAULT
            if pseudo_sel == ':nth-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, False, nth_sel))
            elif pseudo_sel == ':nth-last-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, True, nth_sel))
        elif pseudo_sel == ':nth-of-type':
            sel.nth.append(ct.SelectorNth(s1, var, s2, True, False, ct.SelectorList()))
        elif pseudo_sel == ':nth-last-of-type':
            sel.nth.append(ct.SelectorNth(s1, var, s2, True, True, ct.SelectorList()))
        has_selector = True
        return has_selector

    def parse_pseudo_open(self, sel: _Selector, name: str, has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], index: int) -> bool:
        if False:
            print('Hello World!')
        'Parse pseudo with opening bracket.'
        flags = FLG_PSEUDO | FLG_OPEN
        if name == ':not':
            flags |= FLG_NOT
        elif name == ':has':
            flags |= FLG_RELATIVE
        elif name in (':where', ':is'):
            flags |= FLG_FORGIVE
        sel.selectors.append(self.parse_selectors(iselector, index, flags))
        has_selector = True
        return has_selector

    def parse_has_combinator(self, sel: _Selector, m: Match[str], has_selector: bool, selectors: list[_Selector], rel_type: str, index: int) -> tuple[bool, _Selector, str]:
        if False:
            for i in range(10):
                print('nop')
        'Parse combinator tokens.'
        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if combinator == COMMA_COMBINATOR:
            sel.rel_type = rel_type
            selectors[-1].relations.append(sel)
            rel_type = ':' + WS_COMBINATOR
            selectors.append(_Selector())
        else:
            if has_selector:
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            elif rel_type[1:] != WS_COMBINATOR:
                raise SelectorSyntaxError('The multiple combinators at position {}'.format(index), self.pattern, index)
            rel_type = ':' + combinator
        sel = _Selector()
        has_selector = False
        return (has_selector, sel, rel_type)

    def parse_combinator(self, sel: _Selector, m: Match[str], has_selector: bool, selectors: list[_Selector], relations: list[_Selector], is_pseudo: bool, is_forgive: bool, index: int) -> tuple[bool, _Selector]:
        if False:
            i = 10
            return i + 15
        'Parse combinator tokens.'
        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if not has_selector:
            if not is_forgive or combinator != COMMA_COMBINATOR:
                raise SelectorSyntaxError("The combinator '{}' at position {}, must have a selector before it".format(combinator, index), self.pattern, index)
            if combinator == COMMA_COMBINATOR:
                sel.no_match = True
                del relations[:]
                selectors.append(sel)
        elif combinator == COMMA_COMBINATOR:
            if not sel.tag and (not is_pseudo):
                sel.tag = ct.SelectorTag('*', None)
            sel.relations.extend(relations)
            selectors.append(sel)
            del relations[:]
        else:
            sel.relations.extend(relations)
            sel.rel_type = combinator
            del relations[:]
            relations.append(sel)
        sel = _Selector()
        has_selector = False
        return (has_selector, sel)

    def parse_class_id(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            print('Hello World!')
        'Parse HTML classes and ids.'
        selector = m.group(0)
        if selector.startswith('.'):
            sel.classes.append(css_unescape(selector[1:]))
        else:
            sel.ids.append(css_unescape(selector[1:]))
        has_selector = True
        return has_selector

    def parse_pseudo_contains(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Parse contains.'
        pseudo = util.lower(css_unescape(m.group('name')))
        if pseudo == ':contains':
            warnings.warn("The pseudo class ':contains' is deprecated, ':-soup-contains' should be used moving forward.", FutureWarning)
        contains_own = pseudo == ':-soup-contains-own'
        values = css_unescape(m.group('values'))
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(("'", '"')):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)
            patterns.append(value)
        sel.contains.append(ct.SelectorContains(patterns, contains_own))
        has_selector = True
        return has_selector

    def parse_pseudo_lang(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            print('Hello World!')
        'Parse pseudo language.'
        values = m.group('values')
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(('"', "'")):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)
            patterns.append(value)
        sel.lang.append(ct.SelectorLang(patterns))
        has_selector = True
        return has_selector

    def parse_pseudo_dir(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        if False:
            while True:
                i = 10
        'Parse pseudo direction.'
        value = ct.SEL_DIR_LTR if util.lower(m.group('dir')) == 'ltr' else ct.SEL_DIR_RTL
        sel.flags |= value
        has_selector = True
        return has_selector

    def parse_selectors(self, iselector: Iterator[tuple[str, Match[str]]], index: int=0, flags: int=0) -> ct.SelectorList:
        if False:
            return 10
        'Parse selectors.'
        sel = _Selector()
        selectors = []
        has_selector = False
        closed = False
        relations = []
        rel_type = ':' + WS_COMBINATOR
        is_open = bool(flags & FLG_OPEN)
        is_pseudo = bool(flags & FLG_PSEUDO)
        is_relative = bool(flags & FLG_RELATIVE)
        is_not = bool(flags & FLG_NOT)
        is_html = bool(flags & FLG_HTML)
        is_default = bool(flags & FLG_DEFAULT)
        is_indeterminate = bool(flags & FLG_INDETERMINATE)
        is_in_range = bool(flags & FLG_IN_RANGE)
        is_out_of_range = bool(flags & FLG_OUT_OF_RANGE)
        is_placeholder_shown = bool(flags & FLG_PLACEHOLDER_SHOWN)
        is_forgive = bool(flags & FLG_FORGIVE)
        if self.debug:
            if is_pseudo:
                print('    is_pseudo: True')
            if is_open:
                print('    is_open: True')
            if is_relative:
                print('    is_relative: True')
            if is_not:
                print('    is_not: True')
            if is_html:
                print('    is_html: True')
            if is_default:
                print('    is_default: True')
            if is_indeterminate:
                print('    is_indeterminate: True')
            if is_in_range:
                print('    is_in_range: True')
            if is_out_of_range:
                print('    is_out_of_range: True')
            if is_placeholder_shown:
                print('    is_placeholder_shown: True')
            if is_forgive:
                print('    is_forgive: True')
        if is_relative:
            selectors.append(_Selector())
        try:
            while True:
                (key, m) = next(iselector)
                if key == 'at_rule':
                    raise NotImplementedError('At-rules found at position {}'.format(m.start(0)))
                elif key == 'pseudo_class_custom':
                    has_selector = self.parse_pseudo_class_custom(sel, m, has_selector)
                elif key == 'pseudo_class':
                    (has_selector, is_html) = self.parse_pseudo_class(sel, m, has_selector, iselector, is_html)
                elif key == 'pseudo_element':
                    raise NotImplementedError('Pseudo-element found at position {}'.format(m.start(0)))
                elif key == 'pseudo_contains':
                    has_selector = self.parse_pseudo_contains(sel, m, has_selector)
                elif key in ('pseudo_nth_type', 'pseudo_nth_child'):
                    has_selector = self.parse_pseudo_nth(sel, m, has_selector, iselector)
                elif key == 'pseudo_lang':
                    has_selector = self.parse_pseudo_lang(sel, m, has_selector)
                elif key == 'pseudo_dir':
                    has_selector = self.parse_pseudo_dir(sel, m, has_selector)
                    is_html = True
                elif key == 'pseudo_close':
                    if not has_selector:
                        if not is_forgive:
                            raise SelectorSyntaxError('Expected a selector at position {}'.format(m.start(0)), self.pattern, m.start(0))
                        sel.no_match = True
                    if is_open:
                        closed = True
                        break
                    else:
                        raise SelectorSyntaxError('Unmatched pseudo-class close at position {}'.format(m.start(0)), self.pattern, m.start(0))
                elif key == 'combine':
                    if is_relative:
                        (has_selector, sel, rel_type) = self.parse_has_combinator(sel, m, has_selector, selectors, rel_type, index)
                    else:
                        (has_selector, sel) = self.parse_combinator(sel, m, has_selector, selectors, relations, is_pseudo, is_forgive, index)
                elif key == 'attribute':
                    has_selector = self.parse_attribute_selector(sel, m, has_selector)
                elif key == 'tag':
                    if has_selector:
                        raise SelectorSyntaxError('Tag name found at position {} instead of at the start'.format(m.start(0)), self.pattern, m.start(0))
                    has_selector = self.parse_tag_pattern(sel, m, has_selector)
                elif key in ('class', 'id'):
                    has_selector = self.parse_class_id(sel, m, has_selector)
                index = m.end(0)
        except StopIteration:
            pass
        if is_open and (not closed):
            raise SelectorSyntaxError('Unclosed pseudo-class at position {}'.format(index), self.pattern, index)
        if has_selector:
            if not sel.tag and (not is_pseudo):
                sel.tag = ct.SelectorTag('*', None)
            if is_relative:
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            else:
                sel.relations.extend(relations)
                del relations[:]
                selectors.append(sel)
        elif is_forgive and (not selectors or not relations):
            sel.no_match = True
            del relations[:]
            selectors.append(sel)
            has_selector = True
        if not has_selector:
            raise SelectorSyntaxError('Expected a selector at position {}'.format(index), self.pattern, index)
        if is_default:
            selectors[-1].flags = ct.SEL_DEFAULT
        if is_indeterminate:
            selectors[-1].flags = ct.SEL_INDETERMINATE
        if is_in_range:
            selectors[-1].flags = ct.SEL_IN_RANGE
        if is_out_of_range:
            selectors[-1].flags = ct.SEL_OUT_OF_RANGE
        if is_placeholder_shown:
            selectors[-1].flags = ct.SEL_PLACEHOLDER_SHOWN
        return ct.SelectorList([s.freeze() for s in selectors], is_not, is_html)

    def selector_iter(self, pattern: str) -> Iterator[tuple[str, Match[str]]]:
        if False:
            while True:
                i = 10
        'Iterate selector tokens.'
        m = RE_WS_BEGIN.search(pattern)
        index = m.end(0) if m else 0
        m = RE_WS_END.search(pattern)
        end = m.start(0) - 1 if m else len(pattern) - 1
        if self.debug:
            print('## PARSING: {!r}'.format(pattern))
        while index <= end:
            m = None
            for v in self.css_tokens:
                m = v.match(pattern, index, self.flags)
                if m:
                    name = v.get_name()
                    if self.debug:
                        print("TOKEN: '{}' --> {!r} at position {}".format(name, m.group(0), m.start(0)))
                    index = m.end(0)
                    yield (name, m)
                    break
            if m is None:
                c = pattern[index]
                if c == '[':
                    msg = 'Malformed attribute selector at position {}'.format(index)
                elif c == '.':
                    msg = 'Malformed class selector at position {}'.format(index)
                elif c == '#':
                    msg = 'Malformed id selector at position {}'.format(index)
                elif c == ':':
                    msg = 'Malformed pseudo-class selector at position {}'.format(index)
                else:
                    msg = 'Invalid character {!r} position {}'.format(c, index)
                raise SelectorSyntaxError(msg, self.pattern, index)
        if self.debug:
            print('## END PARSING')

    def process_selectors(self, index: int=0, flags: int=0) -> ct.SelectorList:
        if False:
            for i in range(10):
                print('nop')
        'Process selectors.'
        return self.parse_selectors(self.selector_iter(self.pattern), index, flags)
CSS_LINK = CSSParser('html|*:is(a, area)[href]').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_CHECKED = CSSParser('\n    html|*:is(input[type=checkbox], input[type=radio])[checked], html|option[selected]\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_DEFAULT = CSSParser('\n    :checked,\n\n    /*\n    This pattern must be at the end.\n    Special logic is applied to the last selector.\n    */\n    html|form html|*:is(button, input)[type="submit"]\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_DEFAULT)
CSS_INDETERMINATE = CSSParser('\n    html|input[type="checkbox"][indeterminate],\n    html|input[type="radio"]:is(:not([name]), [name=""]):not([checked]),\n    html|progress:not([value]),\n\n    /*\n    This pattern must be at the end.\n    Special logic is applied to the last selector.\n    */\n    html|input[type="radio"][name]:not([name=\'\']):not([checked])\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_INDETERMINATE)
CSS_DISABLED = CSSParser('\n    html|*:is(input:not([type=hidden]), button, select, textarea, fieldset, optgroup, option, fieldset)[disabled],\n    html|optgroup[disabled] > html|option,\n    html|fieldset[disabled] > html|*:is(input:not([type=hidden]), button, select, textarea, fieldset),\n    html|fieldset[disabled] >\n        html|*:not(legend:nth-of-type(1)) html|*:is(input:not([type=hidden]), button, select, textarea, fieldset)\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_ENABLED = CSSParser('\n    html|*:is(input:not([type=hidden]), button, select, textarea, fieldset, optgroup, option, fieldset):not(:disabled)\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_REQUIRED = CSSParser('html|*:is(input, textarea, select)[required]').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_OPTIONAL = CSSParser('html|*:is(input, textarea, select):not([required])').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_PLACEHOLDER_SHOWN = CSSParser('\n    html|input:is(\n        :not([type]),\n        [type=""],\n        [type=text],\n        [type=search],\n        [type=url],\n        [type=tel],\n        [type=email],\n        [type=password],\n        [type=number]\n    )[placeholder]:not([placeholder=\'\']):is(:not([value]), [value=""]),\n    html|textarea[placeholder]:not([placeholder=\'\'])\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_PLACEHOLDER_SHOWN)
CSS_NTH_OF_S_DEFAULT = CSSParser('*|*').process_selectors(flags=FLG_PSEUDO)
CSS_READ_WRITE = CSSParser('\n    html|*:is(\n        textarea,\n        input:is(\n            :not([type]),\n            [type=""],\n            [type=text],\n            [type=search],\n            [type=url],\n            [type=tel],\n            [type=email],\n            [type=number],\n            [type=password],\n            [type=date],\n            [type=datetime-local],\n            [type=month],\n            [type=time],\n            [type=week]\n        )\n    ):not([readonly], :disabled),\n    html|*:is([contenteditable=""], [contenteditable="true" i])\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_READ_ONLY = CSSParser('\n    html|*:not(:read-write)\n    ').process_selectors(flags=FLG_PSEUDO | FLG_HTML)
CSS_IN_RANGE = CSSParser('\n    html|input:is(\n        [type="date"],\n        [type="month"],\n        [type="week"],\n        [type="time"],\n        [type="datetime-local"],\n        [type="number"],\n        [type="range"]\n    ):is(\n        [min],\n        [max]\n    )\n    ').process_selectors(flags=FLG_PSEUDO | FLG_IN_RANGE | FLG_HTML)
CSS_OUT_OF_RANGE = CSSParser('\n    html|input:is(\n        [type="date"],\n        [type="month"],\n        [type="week"],\n        [type="time"],\n        [type="datetime-local"],\n        [type="number"],\n        [type="range"]\n    ):is(\n        [min],\n        [max]\n    )\n    ').process_selectors(flags=FLG_PSEUDO | FLG_OUT_OF_RANGE | FLG_HTML)