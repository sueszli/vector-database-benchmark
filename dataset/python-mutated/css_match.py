"""CSS matcher."""
from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4
from typing import Iterator, Iterable, Any, Callable, Sequence, cast
RE_NOT_EMPTY = re.compile('[^ \t\r\n\x0c]')
RE_NOT_WS = re.compile('[^ \t\r\n\x0c]+')
REL_PARENT = ' '
REL_CLOSE_PARENT = '>'
REL_SIBLING = '~'
REL_CLOSE_SIBLING = '+'
REL_HAS_PARENT = ': '
REL_HAS_CLOSE_PARENT = ':>'
REL_HAS_SIBLING = ':~'
REL_HAS_CLOSE_SIBLING = ':+'
NS_XHTML = 'http://www.w3.org/1999/xhtml'
NS_XML = 'http://www.w3.org/XML/1998/namespace'
DIR_FLAGS = ct.SEL_DIR_LTR | ct.SEL_DIR_RTL
RANGES = ct.SEL_IN_RANGE | ct.SEL_OUT_OF_RANGE
DIR_MAP = {'ltr': ct.SEL_DIR_LTR, 'rtl': ct.SEL_DIR_RTL, 'auto': 0}
RE_NUM = re.compile('^(?P<value>-?(?:[0-9]{1,}(\\.[0-9]+)?|\\.[0-9]+))$')
RE_TIME = re.compile('^(?P<hour>[0-9]{2}):(?P<minutes>[0-9]{2})$')
RE_MONTH = re.compile('^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})$')
RE_WEEK = re.compile('^(?P<year>[0-9]{4,})-W(?P<week>[0-9]{2})$')
RE_DATE = re.compile('^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})$')
RE_DATETIME = re.compile('^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T(?P<hour>[0-9]{2}):(?P<minutes>[0-9]{2})$')
RE_WILD_STRIP = re.compile('(?:(?:-\\*-)(?:\\*(?:-|$))*|-\\*$)')
MONTHS_30 = (4, 6, 9, 11)
FEB = 2
SHORT_MONTH = 30
LONG_MONTH = 31
FEB_MONTH = 28
FEB_LEAP_MONTH = 29
DAYS_IN_WEEK = 7

class _FakeParent:
    """
    Fake parent class.

    When we have a fragment with no `BeautifulSoup` document object,
    we can't evaluate `nth` selectors properly.  Create a temporary
    fake parent so we can traverse the root element as a child.
    """

    def __init__(self, element: bs4.Tag) -> None:
        if False:
            return 10
        'Initialize.'
        self.contents = [element]

    def __len__(self) -> bs4.PageElement:
        if False:
            for i in range(10):
                print('nop')
        'Length.'
        return len(self.contents)

class _DocumentNav:
    """Navigate a Beautiful Soup document."""

    @classmethod
    def assert_valid_input(cls, tag: Any) -> None:
        if False:
            while True:
                i = 10
        'Check if valid input tag or document.'
        if not cls.is_tag(tag):
            raise TypeError("Expected a BeautifulSoup 'Tag', but instead received type {}".format(type(tag)))

    @staticmethod
    def is_doc(obj: bs4.Tag) -> bool:
        if False:
            return 10
        'Is `BeautifulSoup` object.'
        return isinstance(obj, bs4.BeautifulSoup)

    @staticmethod
    def is_tag(obj: bs4.PageElement) -> bool:
        if False:
            return 10
        'Is tag.'
        return isinstance(obj, bs4.Tag)

    @staticmethod
    def is_declaration(obj: bs4.PageElement) -> bool:
        if False:
            i = 10
            return i + 15
        'Is declaration.'
        return isinstance(obj, bs4.Declaration)

    @staticmethod
    def is_cdata(obj: bs4.PageElement) -> bool:
        if False:
            while True:
                i = 10
        'Is CDATA.'
        return isinstance(obj, bs4.CData)

    @staticmethod
    def is_processing_instruction(obj: bs4.PageElement) -> bool:
        if False:
            while True:
                i = 10
        'Is processing instruction.'
        return isinstance(obj, bs4.ProcessingInstruction)

    @staticmethod
    def is_navigable_string(obj: bs4.PageElement) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is navigable string.'
        return isinstance(obj, bs4.NavigableString)

    @staticmethod
    def is_special_string(obj: bs4.PageElement) -> bool:
        if False:
            return 10
        'Is special string.'
        return isinstance(obj, (bs4.Comment, bs4.Declaration, bs4.CData, bs4.ProcessingInstruction, bs4.Doctype))

    @classmethod
    def is_content_string(cls, obj: bs4.PageElement) -> bool:
        if False:
            print('Hello World!')
        'Check if node is content string.'
        return cls.is_navigable_string(obj) and (not cls.is_special_string(obj))

    @staticmethod
    def create_fake_parent(el: bs4.Tag) -> _FakeParent:
        if False:
            return 10
        'Create fake parent for a given element.'
        return _FakeParent(el)

    @staticmethod
    def is_xml_tree(el: bs4.Tag) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if element (or document) is from a XML tree.'
        return bool(el._is_xml)

    def is_iframe(self, el: bs4.Tag) -> bool:
        if False:
            print('Hello World!')
        'Check if element is an `iframe`.'
        return bool((el.name if self.is_xml_tree(el) else util.lower(el.name)) == 'iframe' and self.is_html_tag(el))

    def is_root(self, el: bs4.Tag) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether element is a root element.\n\n        We check that the element is the root of the tree (which we have already pre-calculated),\n        and we check if it is the root element under an `iframe`.\n        '
        root = self.root and self.root is el
        if not root:
            parent = self.get_parent(el)
            root = parent is not None and self.is_html and self.is_iframe(parent)
        return root

    def get_contents(self, el: bs4.Tag, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        if False:
            print('Hello World!')
        'Get contents or contents in reverse.'
        if not no_iframe or not self.is_iframe(el):
            for content in el.contents:
                yield content

    def get_children(self, el: bs4.Tag, start: int | None=None, reverse: bool=False, tags: bool=True, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        if False:
            while True:
                i = 10
        'Get children.'
        if not no_iframe or not self.is_iframe(el):
            last = len(el.contents) - 1
            if start is None:
                index = last if reverse else 0
            else:
                index = start
            end = -1 if reverse else last + 1
            incr = -1 if reverse else 1
            if 0 <= index <= last:
                while index != end:
                    node = el.contents[index]
                    index += incr
                    if not tags or self.is_tag(node):
                        yield node

    def get_descendants(self, el: bs4.Tag, tags: bool=True, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        if False:
            return 10
        'Get descendants.'
        if not no_iframe or not self.is_iframe(el):
            next_good = None
            for child in el.descendants:
                if next_good is not None:
                    if child is not next_good:
                        continue
                    next_good = None
                is_tag = self.is_tag(child)
                if no_iframe and is_tag and self.is_iframe(child):
                    if child.next_sibling is not None:
                        next_good = child.next_sibling
                    else:
                        last_child = child
                        while self.is_tag(last_child) and last_child.contents:
                            last_child = last_child.contents[-1]
                        next_good = last_child.next_element
                    yield child
                    if next_good is None:
                        break
                    continue
                if not tags or is_tag:
                    yield child

    def get_parent(self, el: bs4.Tag, no_iframe: bool=False) -> bs4.Tag:
        if False:
            while True:
                i = 10
        'Get parent.'
        parent = el.parent
        if no_iframe and parent is not None and self.is_iframe(parent):
            parent = None
        return parent

    @staticmethod
    def get_tag_name(el: bs4.Tag) -> str | None:
        if False:
            return 10
        'Get tag.'
        return cast('str | None', el.name)

    @staticmethod
    def get_prefix_name(el: bs4.Tag) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Get prefix.'
        return cast('str | None', el.prefix)

    @staticmethod
    def get_uri(el: bs4.Tag) -> str | None:
        if False:
            return 10
        'Get namespace `URI`.'
        return cast('str | None', el.namespace)

    @classmethod
    def get_next(cls, el: bs4.Tag, tags: bool=True) -> bs4.PageElement:
        if False:
            return 10
        'Get next sibling tag.'
        sibling = el.next_sibling
        while tags and (not cls.is_tag(sibling)) and (sibling is not None):
            sibling = sibling.next_sibling
        return sibling

    @classmethod
    def get_previous(cls, el: bs4.Tag, tags: bool=True) -> bs4.PageElement:
        if False:
            for i in range(10):
                print('nop')
        'Get previous sibling tag.'
        sibling = el.previous_sibling
        while tags and (not cls.is_tag(sibling)) and (sibling is not None):
            sibling = sibling.previous_sibling
        return sibling

    @staticmethod
    def has_html_ns(el: bs4.Tag) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Check if element has an HTML namespace.\n\n        This is a bit different than whether a element is treated as having an HTML namespace,\n        like we do in the case of `is_html_tag`.\n        '
        ns = getattr(el, 'namespace') if el else None
        return bool(ns and ns == NS_XHTML)

    @staticmethod
    def split_namespace(el: bs4.Tag, attr_name: str) -> tuple[str | None, str | None]:
        if False:
            while True:
                i = 10
        'Return namespace and attribute name without the prefix.'
        return (getattr(attr_name, 'namespace', None), getattr(attr_name, 'name', None))

    @classmethod
    def normalize_value(cls, value: Any) -> str | Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        'Normalize the value to be a string or list of strings.'
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode('utf8')
        if isinstance(value, Sequence):
            new_value = []
            for v in value:
                if not isinstance(v, (str, bytes)) and isinstance(v, Sequence):
                    new_value.append(str(v))
                else:
                    new_value.append(cast(str, cls.normalize_value(v)))
            return new_value
        return str(value)

    @classmethod
    def get_attribute_by_name(cls, el: bs4.Tag, name: str, default: str | Sequence[str] | None=None) -> str | Sequence[str] | None:
        if False:
            i = 10
            return i + 15
        'Get attribute by name.'
        value = default
        if el._is_xml:
            try:
                value = cls.normalize_value(el.attrs[name])
            except KeyError:
                pass
        else:
            for (k, v) in el.attrs.items():
                if util.lower(k) == name:
                    value = cls.normalize_value(v)
                    break
        return value

    @classmethod
    def iter_attributes(cls, el: bs4.Tag) -> Iterator[tuple[str, str | Sequence[str] | None]]:
        if False:
            for i in range(10):
                print('nop')
        'Iterate attributes.'
        for (k, v) in el.attrs.items():
            yield (k, cls.normalize_value(v))

    @classmethod
    def get_classes(cls, el: bs4.Tag) -> Sequence[str]:
        if False:
            while True:
                i = 10
        'Get classes.'
        classes = cls.get_attribute_by_name(el, 'class', [])
        if isinstance(classes, str):
            classes = RE_NOT_WS.findall(classes)
        return cast(Sequence[str], classes)

    def get_text(self, el: bs4.Tag, no_iframe: bool=False) -> str:
        if False:
            print('Hello World!')
        'Get text.'
        return ''.join([node for node in self.get_descendants(el, tags=False, no_iframe=no_iframe) if self.is_content_string(node)])

    def get_own_text(self, el: bs4.Tag, no_iframe: bool=False) -> list[str]:
        if False:
            while True:
                i = 10
        'Get Own Text.'
        return [node for node in self.get_contents(el, no_iframe=no_iframe) if self.is_content_string(node)]

class Inputs:
    """Class for parsing and validating input items."""

    @staticmethod
    def validate_day(year: int, month: int, day: int) -> bool:
        if False:
            i = 10
            return i + 15
        'Validate day.'
        max_days = LONG_MONTH
        if month == FEB:
            max_days = FEB_LEAP_MONTH if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else FEB_MONTH
        elif month in MONTHS_30:
            max_days = SHORT_MONTH
        return 1 <= day <= max_days

    @staticmethod
    def validate_week(year: int, week: int) -> bool:
        if False:
            while True:
                i = 10
        'Validate week.'
        max_week = datetime.strptime('{}-{}-{}'.format(12, 31, year), '%m-%d-%Y').isocalendar()[1]
        if max_week == 1:
            max_week = 53
        return 1 <= week <= max_week

    @staticmethod
    def validate_month(month: int) -> bool:
        if False:
            print('Hello World!')
        'Validate month.'
        return 1 <= month <= 12

    @staticmethod
    def validate_year(year: int) -> bool:
        if False:
            return 10
        'Validate year.'
        return 1 <= year

    @staticmethod
    def validate_hour(hour: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Validate hour.'
        return 0 <= hour <= 23

    @staticmethod
    def validate_minutes(minutes: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Validate minutes.'
        return 0 <= minutes <= 59

    @classmethod
    def parse_value(cls, itype: str, value: str | None) -> tuple[float, ...] | None:
        if False:
            i = 10
            return i + 15
        'Parse the input value.'
        parsed = None
        if value is None:
            return value
        if itype == 'date':
            m = RE_DATE.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                day = int(m.group('day'), 10)
                if cls.validate_year(year) and cls.validate_month(month) and cls.validate_day(year, month, day):
                    parsed = (year, month, day)
        elif itype == 'month':
            m = RE_MONTH.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                if cls.validate_year(year) and cls.validate_month(month):
                    parsed = (year, month)
        elif itype == 'week':
            m = RE_WEEK.match(value)
            if m:
                year = int(m.group('year'), 10)
                week = int(m.group('week'), 10)
                if cls.validate_year(year) and cls.validate_week(year, week):
                    parsed = (year, week)
        elif itype == 'time':
            m = RE_TIME.match(value)
            if m:
                hour = int(m.group('hour'), 10)
                minutes = int(m.group('minutes'), 10)
                if cls.validate_hour(hour) and cls.validate_minutes(minutes):
                    parsed = (hour, minutes)
        elif itype == 'datetime-local':
            m = RE_DATETIME.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                day = int(m.group('day'), 10)
                hour = int(m.group('hour'), 10)
                minutes = int(m.group('minutes'), 10)
                if cls.validate_year(year) and cls.validate_month(month) and cls.validate_day(year, month, day) and cls.validate_hour(hour) and cls.validate_minutes(minutes):
                    parsed = (year, month, day, hour, minutes)
        elif itype in ('number', 'range'):
            m = RE_NUM.match(value)
            if m:
                parsed = (float(m.group('value')),)
        return parsed

class CSSMatch(_DocumentNav):
    """Perform CSS matching."""

    def __init__(self, selectors: ct.SelectorList, scope: bs4.Tag, namespaces: ct.Namespaces | None, flags: int) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize.'
        self.assert_valid_input(scope)
        self.tag = scope
        self.cached_meta_lang = []
        self.cached_default_forms = []
        self.cached_indeterminate_forms = []
        self.selectors = selectors
        self.namespaces = {} if namespaces is None else namespaces
        self.flags = flags
        self.iframe_restrict = False
        doc = scope
        parent = self.get_parent(doc)
        while parent:
            doc = parent
            parent = self.get_parent(doc)
        root = None
        if not self.is_doc(doc):
            root = doc
        else:
            for child in self.get_children(doc):
                root = child
                break
        self.root = root
        self.scope = scope if scope is not doc else root
        self.has_html_namespace = self.has_html_ns(root)
        self.is_xml = self.is_xml_tree(doc)
        self.is_html = not self.is_xml or self.has_html_namespace

    def supports_namespaces(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if namespaces are supported in the HTML type.'
        return self.is_xml or self.has_html_namespace

    def get_tag_ns(self, el: bs4.Tag) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get tag namespace.'
        if self.supports_namespaces():
            namespace = ''
            ns = self.get_uri(el)
            if ns:
                namespace = ns
        else:
            namespace = NS_XHTML
        return namespace

    def is_html_tag(self, el: bs4.Tag) -> bool:
        if False:
            return 10
        'Check if tag is in HTML namespace.'
        return self.get_tag_ns(el) == NS_XHTML

    def get_tag(self, el: bs4.Tag) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Get tag.'
        name = self.get_tag_name(el)
        return util.lower(name) if name is not None and (not self.is_xml) else name

    def get_prefix(self, el: bs4.Tag) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Get prefix.'
        prefix = self.get_prefix_name(el)
        return util.lower(prefix) if prefix is not None and (not self.is_xml) else prefix

    def find_bidi(self, el: bs4.Tag) -> int | None:
        if False:
            for i in range(10):
                print('nop')
        'Get directionality from element text.'
        for node in self.get_children(el, tags=False):
            if self.is_tag(node):
                direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(node, 'dir', '')), None)
                if self.get_tag(node) in ('bdi', 'script', 'style', 'textarea', 'iframe') or not self.is_html_tag(node) or direction is not None:
                    continue
                value = self.find_bidi(node)
                if value is not None:
                    return value
                continue
            if self.is_special_string(node):
                continue
            for c in node:
                bidi = unicodedata.bidirectional(c)
                if bidi in ('AL', 'R', 'L'):
                    return ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
        return None

    def extended_language_filter(self, lang_range: str, lang_tag: str) -> bool:
        if False:
            while True:
                i = 10
        'Filter the language tags.'
        match = True
        lang_range = RE_WILD_STRIP.sub('-', lang_range).lower()
        ranges = lang_range.split('-')
        subtags = lang_tag.lower().split('-')
        length = len(ranges)
        slength = len(subtags)
        rindex = 0
        sindex = 0
        r = ranges[rindex]
        s = subtags[sindex]
        if length == 1 and slength == 1 and (not r) and (r == s):
            return True
        if r != '*' and r != s or (r == '*' and slength == 1 and (not s)):
            match = False
        rindex += 1
        sindex += 1
        while match and rindex < length:
            r = ranges[rindex]
            try:
                s = subtags[sindex]
            except IndexError:
                match = False
                continue
            if not r:
                match = False
                continue
            elif s == r:
                rindex += 1
            elif len(s) == 1:
                match = False
                continue
            sindex += 1
        return match

    def match_attribute_name(self, el: bs4.Tag, attr: str, prefix: str | None) -> str | Sequence[str] | None:
        if False:
            i = 10
            return i + 15
        'Match attribute name and return value if it exists.'
        value = None
        if self.supports_namespaces():
            value = None
            if prefix:
                ns = self.namespaces.get(prefix)
                if ns is None and prefix != '*':
                    return None
            else:
                ns = None
            for (k, v) in self.iter_attributes(el):
                (namespace, name) = self.split_namespace(el, k)
                if ns is None:
                    if self.is_xml and attr == k or (not self.is_xml and util.lower(attr) == util.lower(k)):
                        value = v
                        break
                    continue
                if namespace is None or (ns != namespace and prefix != '*'):
                    continue
                if util.lower(attr) != util.lower(name) if not self.is_xml else attr != name:
                    continue
                value = v
                break
        else:
            for (k, v) in self.iter_attributes(el):
                if util.lower(attr) != util.lower(k):
                    continue
                value = v
                break
        return value

    def match_namespace(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        if False:
            return 10
        'Match the namespace of the element.'
        match = True
        namespace = self.get_tag_ns(el)
        default_namespace = self.namespaces.get('')
        tag_ns = '' if tag.prefix is None else self.namespaces.get(tag.prefix)
        if tag.prefix is None and (default_namespace is not None and namespace != default_namespace):
            match = False
        elif tag.prefix is not None and tag.prefix == '' and namespace:
            match = False
        elif tag.prefix and tag.prefix != '*' and (tag_ns is None or namespace != tag_ns):
            match = False
        return match

    def match_attributes(self, el: bs4.Tag, attributes: tuple[ct.SelectorAttribute, ...]) -> bool:
        if False:
            return 10
        'Match attributes.'
        match = True
        if attributes:
            for a in attributes:
                temp = self.match_attribute_name(el, a.attribute, a.prefix)
                pattern = a.xml_type_pattern if self.is_xml and a.xml_type_pattern else a.pattern
                if temp is None:
                    match = False
                    break
                value = temp if isinstance(temp, str) else ' '.join(temp)
                if pattern is None:
                    continue
                elif pattern.match(value) is None:
                    match = False
                    break
        return match

    def match_tagname(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Match tag name.'
        name = util.lower(tag.name) if not self.is_xml and tag.name is not None else tag.name
        return not (name is not None and name not in (self.get_tag(el), '*'))

    def match_tag(self, el: bs4.Tag, tag: ct.SelectorTag | None) -> bool:
        if False:
            while True:
                i = 10
        'Match the tag.'
        match = True
        if tag is not None:
            if not self.match_namespace(el, tag):
                match = False
            if not self.match_tagname(el, tag):
                match = False
        return match

    def match_past_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Match past relationship.'
        found = False
        if isinstance(relation[0], ct.SelectorNull):
            return found
        if relation[0].rel_type == REL_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            while not found and parent:
                found = self.match_selectors(parent, relation)
                parent = self.get_parent(parent, no_iframe=self.iframe_restrict)
        elif relation[0].rel_type == REL_CLOSE_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            if parent:
                found = self.match_selectors(parent, relation)
        elif relation[0].rel_type == REL_SIBLING:
            sibling = self.get_previous(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_previous(sibling)
        elif relation[0].rel_type == REL_CLOSE_SIBLING:
            sibling = self.get_previous(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_future_child(self, parent: bs4.Tag, relation: ct.SelectorList, recursive: bool=False) -> bool:
        if False:
            print('Hello World!')
        'Match future child.'
        match = False
        if recursive:
            children = self.get_descendants
        else:
            children = self.get_children
        for child in children(parent, no_iframe=self.iframe_restrict):
            match = self.match_selectors(child, relation)
            if match:
                break
        return match

    def match_future_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        if False:
            return 10
        'Match future relationship.'
        found = False
        if isinstance(relation[0], ct.SelectorNull):
            return found
        if relation[0].rel_type == REL_HAS_PARENT:
            found = self.match_future_child(el, relation, True)
        elif relation[0].rel_type == REL_HAS_CLOSE_PARENT:
            found = self.match_future_child(el, relation)
        elif relation[0].rel_type == REL_HAS_SIBLING:
            sibling = self.get_next(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_next(sibling)
        elif relation[0].rel_type == REL_HAS_CLOSE_SIBLING:
            sibling = self.get_next(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        if False:
            return 10
        'Match relationship to other elements.'
        found = False
        if isinstance(relation[0], ct.SelectorNull) or relation[0].rel_type is None:
            return found
        if relation[0].rel_type.startswith(':'):
            found = self.match_future_relations(el, relation)
        else:
            found = self.match_past_relations(el, relation)
        return found

    def match_id(self, el: bs4.Tag, ids: tuple[str, ...]) -> bool:
        if False:
            i = 10
            return i + 15
        "Match element's ID."
        found = True
        for i in ids:
            if i != self.get_attribute_by_name(el, 'id', ''):
                found = False
                break
        return found

    def match_classes(self, el: bs4.Tag, classes: tuple[str, ...]) -> bool:
        if False:
            while True:
                i = 10
        "Match element's classes."
        current_classes = self.get_classes(el)
        found = True
        for c in classes:
            if c not in current_classes:
                found = False
                break
        return found

    def match_root(self, el: bs4.Tag) -> bool:
        if False:
            return 10
        'Match element as root.'
        is_root = self.is_root(el)
        if is_root:
            sibling = self.get_previous(el, tags=False)
            while is_root and sibling is not None:
                if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                    is_root = False
                else:
                    sibling = self.get_previous(sibling, tags=False)
        if is_root:
            sibling = self.get_next(el, tags=False)
            while is_root and sibling is not None:
                if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                    is_root = False
                else:
                    sibling = self.get_next(sibling, tags=False)
        return is_root

    def match_scope(self, el: bs4.Tag) -> bool:
        if False:
            while True:
                i = 10
        'Match element as scope.'
        return self.scope is el

    def match_nth_tag_type(self, el: bs4.Tag, child: bs4.Tag) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Match tag type for `nth` matches.'
        return self.get_tag(child) == self.get_tag(el) and self.get_tag_ns(child) == self.get_tag_ns(el)

    def match_nth(self, el: bs4.Tag, nth: bs4.Tag) -> bool:
        if False:
            return 10
        'Match `nth` elements.'
        matched = True
        for n in nth:
            matched = False
            if n.selectors and (not self.match_selectors(el, n.selectors)):
                break
            parent = self.get_parent(el)
            if parent is None:
                parent = self.create_fake_parent(el)
            last = n.last
            last_index = len(parent) - 1
            index = last_index if last else 0
            relative_index = 0
            a = n.a
            b = n.b
            var = n.n
            count = 0
            count_incr = 1
            factor = -1 if last else 1
            idx = last_idx = a * count + b if var else a
            if var:
                adjust = None
                while idx < 1 or idx > last_index:
                    if idx < 0:
                        diff_low = 0 - idx
                        if adjust is not None and adjust == 1:
                            break
                        adjust = -1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = 0 - idx
                        if diff >= diff_low:
                            break
                    else:
                        diff_high = idx - last_index
                        if adjust is not None and adjust == -1:
                            break
                        adjust = 1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = idx - last_index
                        if diff >= diff_high:
                            break
                        diff_high = diff
                lowest = count
                if a < 0:
                    while idx >= 1:
                        lowest = count
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                    count_incr = -1
                count = lowest
                idx = last_idx = a * count + b if var else a
            while 1 <= idx <= last_index + 1:
                child = None
                for child in self.get_children(parent, start=index, reverse=factor < 0, tags=False):
                    index += factor
                    if not self.is_tag(child):
                        continue
                    if n.selectors and (not self.match_selectors(child, n.selectors)):
                        continue
                    if n.of_type and (not self.match_nth_tag_type(el, child)):
                        continue
                    relative_index += 1
                    if relative_index == idx:
                        if child is el:
                            matched = True
                        else:
                            break
                    if child is el:
                        break
                if child is el:
                    break
                last_idx = idx
                count += count_incr
                if count < 0:
                    break
                idx = a * count + b if var else a
                if last_idx == idx:
                    break
            if not matched:
                break
        return matched

    def match_empty(self, el: bs4.Tag) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if element is empty (if requested).'
        is_empty = True
        for child in self.get_children(el, tags=False):
            if self.is_tag(child):
                is_empty = False
                break
            elif self.is_content_string(child) and RE_NOT_EMPTY.search(child):
                is_empty = False
                break
        return is_empty

    def match_subselectors(self, el: bs4.Tag, selectors: tuple[ct.SelectorList, ...]) -> bool:
        if False:
            print('Hello World!')
        'Match selectors.'
        match = True
        for sel in selectors:
            if not self.match_selectors(el, sel):
                match = False
        return match

    def match_contains(self, el: bs4.Tag, contains: tuple[ct.SelectorContains, ...]) -> bool:
        if False:
            i = 10
            return i + 15
        'Match element if it contains text.'
        match = True
        content = None
        for contain_list in contains:
            if content is None:
                if contain_list.own:
                    content = self.get_own_text(el, no_iframe=self.is_html)
                else:
                    content = self.get_text(el, no_iframe=self.is_html)
            found = False
            for text in contain_list.text:
                if contain_list.own:
                    for c in content:
                        if text in c:
                            found = True
                            break
                    if found:
                        break
                elif text in content:
                    found = True
                    break
            if not found:
                match = False
        return match

    def match_default(self, el: bs4.Tag) -> bool:
        if False:
            i = 10
            return i + 15
        'Match default.'
        match = False
        form = None
        parent = self.get_parent(el, no_iframe=True)
        while parent and form is None:
            if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                form = parent
            else:
                parent = self.get_parent(parent, no_iframe=True)
        found_form = False
        for (f, t) in self.cached_default_forms:
            if f is form:
                found_form = True
                if t is el:
                    match = True
                break
        if not found_form:
            for child in self.get_descendants(form, no_iframe=True):
                name = self.get_tag(child)
                if name == 'form':
                    break
                if name in ('input', 'button'):
                    v = self.get_attribute_by_name(child, 'type', '')
                    if v and util.lower(v) == 'submit':
                        self.cached_default_forms.append((form, child))
                        if el is child:
                            match = True
                        break
        return match

    def match_indeterminate(self, el: bs4.Tag) -> bool:
        if False:
            while True:
                i = 10
        'Match default.'
        match = False
        name = cast(str, self.get_attribute_by_name(el, 'name'))

        def get_parent_form(el: bs4.Tag) -> bs4.Tag | None:
            if False:
                for i in range(10):
                    print('nop')
            "Find this input's form."
            form = None
            parent = self.get_parent(el, no_iframe=True)
            while form is None:
                if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                    form = parent
                    break
                last_parent = parent
                parent = self.get_parent(parent, no_iframe=True)
                if parent is None:
                    form = last_parent
                    break
            return form
        form = get_parent_form(el)
        found_form = False
        for (f, n, i) in self.cached_indeterminate_forms:
            if f is form and n == name:
                found_form = True
                if i is True:
                    match = True
                break
        if not found_form:
            checked = False
            for child in self.get_descendants(form, no_iframe=True):
                if child is el:
                    continue
                tag_name = self.get_tag(child)
                if tag_name == 'input':
                    is_radio = False
                    check = False
                    has_name = False
                    for (k, v) in self.iter_attributes(child):
                        if util.lower(k) == 'type' and util.lower(v) == 'radio':
                            is_radio = True
                        elif util.lower(k) == 'name' and v == name:
                            has_name = True
                        elif util.lower(k) == 'checked':
                            check = True
                        if is_radio and check and has_name and (get_parent_form(child) is form):
                            checked = True
                            break
                if checked:
                    break
            if not checked:
                match = True
            self.cached_indeterminate_forms.append((form, name, match))
        return match

    def match_lang(self, el: bs4.Tag, langs: tuple[ct.SelectorLang, ...]) -> bool:
        if False:
            print('Hello World!')
        'Match languages.'
        match = False
        has_ns = self.supports_namespaces()
        root = self.root
        has_html_namespace = self.has_html_namespace
        parent = el
        found_lang = None
        last = None
        while not found_lang:
            has_html_ns = self.has_html_ns(parent)
            for (k, v) in self.iter_attributes(parent):
                (attr_ns, attr) = self.split_namespace(parent, k)
                if (not has_ns or has_html_ns) and (util.lower(k) if not self.is_xml else k) == 'lang' or (has_ns and (not has_html_ns) and (attr_ns == NS_XML) and ((util.lower(attr) if not self.is_xml and attr is not None else attr) == 'lang')):
                    found_lang = v
                    break
            last = parent
            parent = self.get_parent(parent, no_iframe=self.is_html)
            if parent is None:
                root = last
                has_html_namespace = self.has_html_ns(root)
                parent = last
                break
        if found_lang is None and self.cached_meta_lang:
            for cache in self.cached_meta_lang:
                if root is cache[0]:
                    found_lang = cache[1]
        if found_lang is None and (not self.is_xml or (has_html_namespace and root.name == 'html')):
            found = False
            for tag in ('html', 'head'):
                found = False
                for child in self.get_children(parent, no_iframe=self.is_html):
                    if self.get_tag(child) == tag and self.is_html_tag(child):
                        found = True
                        parent = child
                        break
                if not found:
                    break
            if found:
                for child in parent:
                    if self.is_tag(child) and self.get_tag(child) == 'meta' and self.is_html_tag(parent):
                        c_lang = False
                        content = None
                        for (k, v) in self.iter_attributes(child):
                            if util.lower(k) == 'http-equiv' and util.lower(v) == 'content-language':
                                c_lang = True
                            if util.lower(k) == 'content':
                                content = v
                            if c_lang and content:
                                found_lang = content
                                self.cached_meta_lang.append((cast(str, root), cast(str, found_lang)))
                                break
                    if found_lang is not None:
                        break
                if found_lang is None:
                    self.cached_meta_lang.append((cast(str, root), ''))
        if found_lang is not None:
            for patterns in langs:
                match = False
                for pattern in patterns:
                    if self.extended_language_filter(pattern, cast(str, found_lang)):
                        match = True
                if not match:
                    break
        return match

    def match_dir(self, el: bs4.Tag, directionality: int) -> bool:
        if False:
            return 10
        'Check directionality.'
        if directionality & ct.SEL_DIR_LTR and directionality & ct.SEL_DIR_RTL:
            return False
        if el is None or not self.is_html_tag(el):
            return False
        direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(el, 'dir', '')), None)
        if direction not in (None, 0):
            return direction == directionality
        is_root = self.is_root(el)
        if is_root and direction is None:
            return ct.SEL_DIR_LTR == directionality
        name = self.get_tag(el)
        is_input = name == 'input'
        is_textarea = name == 'textarea'
        is_bdi = name == 'bdi'
        itype = util.lower(self.get_attribute_by_name(el, 'type', '')) if is_input else ''
        if is_input and itype == 'tel' and (direction is None):
            return ct.SEL_DIR_LTR == directionality
        if (is_input and itype in ('text', 'search', 'tel', 'url', 'email') or is_textarea) and direction == 0:
            if is_textarea:
                temp = []
                for node in self.get_contents(el, no_iframe=True):
                    if self.is_content_string(node):
                        temp.append(node)
                value = ''.join(temp)
            else:
                value = cast(str, self.get_attribute_by_name(el, 'value', ''))
            if value:
                for c in value:
                    bidi = unicodedata.bidirectional(c)
                    if bidi in ('AL', 'R', 'L'):
                        direction = ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
                        return direction == directionality
                return ct.SEL_DIR_LTR == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)
        if is_bdi and direction is None or direction == 0:
            direction = self.find_bidi(el)
            if direction is not None:
                return direction == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)
        return self.match_dir(self.get_parent(el, no_iframe=True), directionality)

    def match_range(self, el: bs4.Tag, condition: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Match range.\n\n        Behavior is modeled after what we see in browsers. Browsers seem to evaluate\n        if the value is out of range, and if not, it is in range. So a missing value\n        will not evaluate out of range; therefore, value is in range. Personally, I\n        feel like this should evaluate as neither in or out of range.\n        '
        out_of_range = False
        itype = util.lower(self.get_attribute_by_name(el, 'type'))
        mn = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'min', None)))
        mx = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'max', None)))
        if mn is None and mx is None:
            return False
        value = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'value', None)))
        if value is not None:
            if itype in ('date', 'datetime-local', 'month', 'week', 'number', 'range'):
                if mn is not None and value < mn:
                    out_of_range = True
                if not out_of_range and mx is not None and (value > mx):
                    out_of_range = True
            elif itype == 'time':
                if mn is not None and mx is not None and (mn > mx):
                    if value < mn and value > mx:
                        out_of_range = True
                else:
                    if mn is not None and value < mn:
                        out_of_range = True
                    if not out_of_range and mx is not None and (value > mx):
                        out_of_range = True
        return not out_of_range if condition & ct.SEL_IN_RANGE else out_of_range

    def match_defined(self, el: bs4.Tag) -> bool:
        if False:
            return 10
        "\n        Match defined.\n\n        `:defined` is related to custom elements in a browser.\n\n        - If the document is XML (not XHTML), all tags will match.\n        - Tags that are not custom (don't have a hyphen) are marked defined.\n        - If the tag has a prefix (without or without a namespace), it will not match.\n\n        This is of course requires the parser to provide us with the proper prefix and namespace info,\n        if it doesn't, there is nothing we can do.\n        "
        name = self.get_tag(el)
        return name is not None and (name.find('-') == -1 or name.find(':') != -1 or self.get_prefix(el) is not None)

    def match_placeholder_shown(self, el: bs4.Tag) -> bool:
        if False:
            while True:
                i = 10
        '\n        Match placeholder shown according to HTML spec.\n\n        - text area should be checked if they have content. A single newline does not count as content.\n\n        '
        match = False
        content = self.get_text(el)
        if content in ('', '\n'):
            match = True
        return match

    def match_selectors(self, el: bs4.Tag, selectors: ct.SelectorList) -> bool:
        if False:
            while True:
                i = 10
        'Check if element matches one of the selectors.'
        match = False
        is_not = selectors.is_not
        is_html = selectors.is_html
        if is_html:
            namespaces = self.namespaces
            iframe_restrict = self.iframe_restrict
            self.namespaces = {'html': NS_XHTML}
            self.iframe_restrict = True
        if not is_html or self.is_html:
            for selector in selectors:
                match = is_not
                if isinstance(selector, ct.SelectorNull):
                    continue
                if not self.match_tag(el, selector.tag):
                    continue
                if selector.flags & ct.SEL_DEFINED and (not self.match_defined(el)):
                    continue
                if selector.flags & ct.SEL_ROOT and (not self.match_root(el)):
                    continue
                if selector.flags & ct.SEL_SCOPE and (not self.match_scope(el)):
                    continue
                if selector.flags & ct.SEL_PLACEHOLDER_SHOWN and (not self.match_placeholder_shown(el)):
                    continue
                if not self.match_nth(el, selector.nth):
                    continue
                if selector.flags & ct.SEL_EMPTY and (not self.match_empty(el)):
                    continue
                if selector.ids and (not self.match_id(el, selector.ids)):
                    continue
                if selector.classes and (not self.match_classes(el, selector.classes)):
                    continue
                if not self.match_attributes(el, selector.attributes):
                    continue
                if selector.flags & RANGES and (not self.match_range(el, selector.flags & RANGES)):
                    continue
                if selector.lang and (not self.match_lang(el, selector.lang)):
                    continue
                if selector.selectors and (not self.match_subselectors(el, selector.selectors)):
                    continue
                if selector.relation and (not self.match_relations(el, selector.relation)):
                    continue
                if selector.flags & ct.SEL_DEFAULT and (not self.match_default(el)):
                    continue
                if selector.flags & ct.SEL_INDETERMINATE and (not self.match_indeterminate(el)):
                    continue
                if selector.flags & DIR_FLAGS and (not self.match_dir(el, selector.flags & DIR_FLAGS)):
                    continue
                if selector.contains and (not self.match_contains(el, selector.contains)):
                    continue
                match = not is_not
                break
        if is_html:
            self.namespaces = namespaces
            self.iframe_restrict = iframe_restrict
        return match

    def select(self, limit: int=0) -> Iterator[bs4.Tag]:
        if False:
            while True:
                i = 10
        'Match all tags under the targeted tag.'
        lim = None if limit < 1 else limit
        for child in self.get_descendants(self.tag):
            if self.match(child):
                yield child
                if lim is not None:
                    lim -= 1
                    if lim < 1:
                        break

    def closest(self) -> bs4.Tag | None:
        if False:
            return 10
        'Match closest ancestor.'
        current = self.tag
        closest = None
        while closest is None and current is not None:
            if self.match(current):
                closest = current
            else:
                current = self.get_parent(current)
        return closest

    def filter(self) -> list[bs4.Tag]:
        if False:
            i = 10
            return i + 15
        "Filter tag's children."
        return [tag for tag in self.get_contents(self.tag) if not self.is_navigable_string(tag) and self.match(tag)]

    def match(self, el: bs4.Tag) -> bool:
        if False:
            while True:
                i = 10
        'Match.'
        return not self.is_doc(el) and self.is_tag(el) and self.match_selectors(el, self.selectors)

class SoupSieve(ct.Immutable):
    """Compiled Soup Sieve selector matching object."""
    pattern: str
    selectors: ct.SelectorList
    namespaces: ct.Namespaces | None
    custom: dict[str, str]
    flags: int
    __slots__ = ('pattern', 'selectors', 'namespaces', 'custom', 'flags', '_hash')

    def __init__(self, pattern: str, selectors: ct.SelectorList, namespaces: ct.Namespaces | None, custom: ct.CustomSelectors | None, flags: int):
        if False:
            while True:
                i = 10
        'Initialize.'
        super().__init__(pattern=pattern, selectors=selectors, namespaces=namespaces, custom=custom, flags=flags)

    def match(self, tag: bs4.Tag) -> bool:
        if False:
            while True:
                i = 10
        'Match.'
        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).match(tag)

    def closest(self, tag: bs4.Tag) -> bs4.Tag:
        if False:
            print('Hello World!')
        'Match closest ancestor.'
        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).closest()

    def filter(self, iterable: Iterable[bs4.Tag]) -> list[bs4.Tag]:
        if False:
            print('Hello World!')
        '\n        Filter.\n\n        `CSSMatch` can cache certain searches for tags of the same document,\n        so if we are given a tag, all tags are from the same document,\n        and we can take advantage of the optimization.\n\n        Any other kind of iterable could have tags from different documents or detached tags,\n        so for those, we use a new `CSSMatch` for each item in the iterable.\n        '
        if CSSMatch.is_tag(iterable):
            return CSSMatch(self.selectors, iterable, self.namespaces, self.flags).filter()
        else:
            return [node for node in iterable if not CSSMatch.is_navigable_string(node) and self.match(node)]

    def select_one(self, tag: bs4.Tag) -> bs4.Tag:
        if False:
            return 10
        'Select a single tag.'
        tags = self.select(tag, limit=1)
        return tags[0] if tags else None

    def select(self, tag: bs4.Tag, limit: int=0) -> list[bs4.Tag]:
        if False:
            print('Hello World!')
        'Select the specified tags.'
        return list(self.iselect(tag, limit))

    def iselect(self, tag: bs4.Tag, limit: int=0) -> Iterator[bs4.Tag]:
        if False:
            print('Hello World!')
        'Iterate the specified tags.'
        for el in CSSMatch(self.selectors, tag, self.namespaces, self.flags).select(limit):
            yield el

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'Representation.'
        return 'SoupSieve(pattern={!r}, namespaces={!r}, custom={!r}, flags={!r})'.format(self.pattern, self.namespaces, self.custom, self.flags)
    __str__ = __repr__
ct.pickle_register(SoupSieve)