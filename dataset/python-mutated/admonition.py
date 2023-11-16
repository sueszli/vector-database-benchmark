"""
Adds rST-style admonitions. Inspired by [rST][] feature with the same name.

[rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

See the [documentation](https://Python-Markdown.github.io/extensions/admonition)
for details.
"""
from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from markdown import blockparser

class AdmonitionExtension(Extension):
    """ Admonition extension for Python-Markdown. """

    def extendMarkdown(self, md):
        if False:
            return 10
        ' Add Admonition to Markdown instance. '
        md.registerExtension(self)
        md.parser.blockprocessors.register(AdmonitionProcessor(md.parser), 'admonition', 105)

class AdmonitionProcessor(BlockProcessor):
    CLASSNAME = 'admonition'
    CLASSNAME_TITLE = 'admonition-title'
    RE = re.compile('(?:^|\\n)!!! ?([\\w\\-]+(?: +[\\w\\-]+)*)(?: +"(.*?)")? *(?:\\n|$)')
    RE_SPACES = re.compile('  +')

    def __init__(self, parser: blockparser.BlockParser):
        if False:
            return 10
        'Initialization.'
        super().__init__(parser)
        self.current_sibling: etree.Element | None = None
        self.content_indent = 0

    def parse_content(self, parent: etree.Element, block: str) -> tuple[etree.Element | None, str, str]:
        if False:
            i = 10
            return i + 15
        'Get sibling admonition.\n\n        Retrieve the appropriate sibling element. This can get tricky when\n        dealing with lists.\n\n        '
        old_block = block
        the_rest = ''
        if self.current_sibling is not None:
            sibling = self.current_sibling
            (block, the_rest) = self.detab(block, self.content_indent)
            self.current_sibling = None
            self.content_indent = 0
            return (sibling, block, the_rest)
        sibling = self.lastChild(parent)
        if sibling is None or sibling.tag != 'div' or sibling.get('class', '').find(self.CLASSNAME) == -1:
            sibling = None
        else:
            last_child = self.lastChild(sibling)
            indent = 0
            while last_child is not None:
                if sibling is not None and block.startswith(' ' * self.tab_length * 2) and (last_child is not None) and (last_child.tag in ('ul', 'ol', 'dl')):
                    sibling = self.lastChild(last_child)
                    last_child = self.lastChild(sibling) if sibling is not None else None
                    block = block[self.tab_length:]
                    indent += self.tab_length
                else:
                    last_child = None
            if not block.startswith(' ' * self.tab_length):
                sibling = None
            if sibling is not None:
                indent += self.tab_length
                (block, the_rest) = self.detab(old_block, indent)
                self.current_sibling = sibling
                self.content_indent = indent
        return (sibling, block, the_rest)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            return 10
        if self.RE.search(block):
            return True
        else:
            return self.parse_content(parent, block)[0] is not None

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            return 10
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            if m.start() > 0:
                self.parser.parseBlocks(parent, [block[:m.start()]])
            block = block[m.end():]
            (block, theRest) = self.detab(block)
        else:
            (sibling, block, theRest) = self.parse_content(parent, block)
        if m:
            (klass, title) = self.get_class_and_title(m)
            div = etree.SubElement(parent, 'div')
            div.set('class', '{} {}'.format(self.CLASSNAME, klass))
            if title:
                p = etree.SubElement(div, 'p')
                p.text = title
                p.set('class', self.CLASSNAME_TITLE)
        else:
            if sibling.tag in ('li', 'dd') and sibling.text:
                text = sibling.text
                sibling.text = ''
                p = etree.SubElement(sibling, 'p')
                p.text = text
            div = sibling
        self.parser.parseChunk(div, block)
        if theRest:
            blocks.insert(0, theRest)

    def get_class_and_title(self, match: re.Match[str]) -> tuple[str, str | None]:
        if False:
            while True:
                i = 10
        (klass, title) = (match.group(1).lower(), match.group(2))
        klass = self.RE_SPACES.sub(' ', klass)
        if title is None:
            title = klass.split(' ', 1)[0].capitalize()
        elif title == '':
            title = None
        return (klass, title)

def makeExtension(**kwargs):
    if False:
        return 10
    return AdmonitionExtension(**kwargs)