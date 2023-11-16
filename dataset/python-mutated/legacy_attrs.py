"""
An extension to Python Markdown which implements legacy attributes.

Prior to Python-Markdown version 3.0, the Markdown class had an `enable_attributes`
keyword which was on by default and provided for attributes to be defined for elements
using the format `{@key=value}`. This extension is provided as a replacement for
backward compatibility. New documents should be authored using `attr_lists`. However,
numerous documents exist which have been using the old attribute format for many
years. This extension can be used to continue to render those documents correctly.
"""
from __future__ import annotations
import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import xml.etree.ElementTree as etree
ATTR_RE = re.compile('\\{@([^\\}]*)=([^\\}]*)}')

class LegacyAttrs(Treeprocessor):

    def run(self, doc: etree.Element) -> None:
        if False:
            i = 10
            return i + 15
        'Find and set values of attributes ({@key=value}). '
        for el in doc.iter():
            alt = el.get('alt', None)
            if alt is not None:
                el.set('alt', self.handleAttributes(el, alt))
            if el.text and isString(el.text):
                el.text = self.handleAttributes(el, el.text)
            if el.tail and isString(el.tail):
                el.tail = self.handleAttributes(el, el.tail)

    def handleAttributes(self, el: etree.Element, txt: str) -> str:
        if False:
            while True:
                i = 10
        ' Set attributes and return text without definitions. '

        def attributeCallback(match: re.Match[str]):
            if False:
                return 10
            el.set(match.group(1), match.group(2).replace('\n', ' '))
        return ATTR_RE.sub(attributeCallback, txt)

class LegacyAttrExtension(Extension):

    def extendMarkdown(self, md):
        if False:
            for i in range(10):
                print('nop')
        ' Add `LegacyAttrs` to Markdown instance. '
        md.treeprocessors.register(LegacyAttrs(md), 'legacyattrs', 15)

def makeExtension(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    return LegacyAttrExtension(**kwargs)