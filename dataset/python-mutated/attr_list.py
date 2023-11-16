"""
 Adds attribute list syntax. Inspired by
[Maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
feature of the same name.

See the [documentation](https://Python-Markdown.github.io/extensions/attr_list)
for details.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

def _handle_double_quote(s, t):
    if False:
        return 10
    (k, v) = t.split('=', 1)
    return (k, v.strip('"'))

def _handle_single_quote(s, t):
    if False:
        i = 10
        return i + 15
    (k, v) = t.split('=', 1)
    return (k, v.strip("'"))

def _handle_key_value(s, t):
    if False:
        return 10
    return t.split('=', 1)

def _handle_word(s, t):
    if False:
        return 10
    if t.startswith('.'):
        return ('.', t[1:])
    if t.startswith('#'):
        return ('id', t[1:])
    return (t, t)
_scanner = re.Scanner([('[^ =]+=".*?"', _handle_double_quote), ("[^ =]+='.*?'", _handle_single_quote), ('[^ =]+=[^ =]+', _handle_key_value), ('[^ =]+', _handle_word), (' ', None)])

def get_attrs(str: str) -> list[tuple[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    ' Parse attribute list and return a list of attribute tuples. '
    return _scanner.scan(str)[0]

def isheader(elem: Element) -> bool:
    if False:
        i = 10
        return i + 15
    return elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

class AttrListTreeprocessor(Treeprocessor):
    BASE_RE = '\\{\\:?[ ]*([^\\}\\n ][^\\}\\n]*)[ ]*\\}'
    HEADER_RE = re.compile('[ ]+{}[ ]*$'.format(BASE_RE))
    BLOCK_RE = re.compile('\\n[ ]*{}[ ]*$'.format(BASE_RE))
    INLINE_RE = re.compile('^{}'.format(BASE_RE))
    NAME_RE = re.compile('[^A-Z_a-z\\u00c0-\\u00d6\\u00d8-\\u00f6\\u00f8-\\u02ff\\u0370-\\u037d\\u037f-\\u1fff\\u200c-\\u200d\\u2070-\\u218f\\u2c00-\\u2fef\\u3001-\\ud7ff\\uf900-\\ufdcf\\ufdf0-\\ufffd\\:\\-\\.0-9\\u00b7\\u0300-\\u036f\\u203f-\\u2040]+')

    def run(self, doc: Element) -> None:
        if False:
            return 10
        for elem in doc.iter():
            if self.md.is_block_level(elem.tag):
                RE = self.BLOCK_RE
                if isheader(elem) or elem.tag in ['dt', 'td', 'th']:
                    RE = self.HEADER_RE
                if len(elem) and elem.tag == 'li':
                    pos = None
                    for (i, child) in enumerate(elem):
                        if child.tag in ['ul', 'ol']:
                            pos = i
                            break
                    if pos is None and elem[-1].tail:
                        m = RE.search(elem[-1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem[-1].tail = elem[-1].tail[:m.start()]
                    elif pos is not None and pos > 0 and elem[pos - 1].tail:
                        m = RE.search(elem[pos - 1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem[pos - 1].tail = elem[pos - 1].tail[:m.start()]
                    elif elem.text:
                        m = RE.search(elem.text)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem.text = elem.text[:m.start()]
                elif len(elem) and elem[-1].tail:
                    m = RE.search(elem[-1].tail)
                    if m:
                        self.assign_attrs(elem, m.group(1))
                        elem[-1].tail = elem[-1].tail[:m.start()]
                        if isheader(elem):
                            elem[-1].tail = elem[-1].tail.rstrip('#').rstrip()
                elif elem.text:
                    m = RE.search(elem.text)
                    if m:
                        self.assign_attrs(elem, m.group(1))
                        elem.text = elem.text[:m.start()]
                        if isheader(elem):
                            elem.text = elem.text.rstrip('#').rstrip()
            elif elem.tail:
                m = self.INLINE_RE.match(elem.tail)
                if m:
                    self.assign_attrs(elem, m.group(1))
                    elem.tail = elem.tail[m.end():]

    def assign_attrs(self, elem: Element, attrs: str) -> None:
        if False:
            while True:
                i = 10
        ' Assign `attrs` to element. '
        for (k, v) in get_attrs(attrs):
            if k == '.':
                cls = elem.get('class')
                if cls:
                    elem.set('class', '{} {}'.format(cls, v))
                else:
                    elem.set('class', v)
            else:
                elem.set(self.sanitize_name(k), v)

    def sanitize_name(self, name: str) -> str:
        if False:
            print('Hello World!')
        '\n        Sanitize name as \'an XML Name, minus the ":"\'.\n        See https://www.w3.org/TR/REC-xml-names/#NT-NCName\n        '
        return self.NAME_RE.sub('_', name)

class AttrListExtension(Extension):
    """ Attribute List extension for Python-Markdown """

    def extendMarkdown(self, md):
        if False:
            while True:
                i = 10
        md.treeprocessors.register(AttrListTreeprocessor(md), 'attr_list', 8)
        md.registerExtension(self)

def makeExtension(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    return AttrListExtension(**kwargs)