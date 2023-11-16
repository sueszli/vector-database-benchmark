"""
An implementation of [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)'s
parsing of Markdown syntax in raw HTML.

See the [documentation](https://Python-Markdown.github.io/extensions/raw_html)
for details.
"""
from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
if TYPE_CHECKING:
    from markdown import Markdown

class HTMLExtractorExtra(HTMLExtractor):
    """
    Override `HTMLExtractor` and create `etree` `Elements` for any elements which should have content parsed as
    Markdown.
    """

    def __init__(self, md: Markdown, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.block_level_tags = set(md.block_level_elements.copy())
        self.span_tags = set(['address', 'dd', 'dt', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'legend', 'li', 'p', 'summary', 'td', 'th'])
        self.raw_tags = set(['canvas', 'math', 'option', 'pre', 'script', 'style', 'textarea'])
        super().__init__(md, *args, **kwargs)
        self.block_tags = set(self.block_level_tags) - (self.span_tags | self.raw_tags | self.empty_tags)
        self.span_and_blocks_tags = self.block_tags | self.span_tags

    def reset(self):
        if False:
            return 10
        'Reset this instance.  Loses all unprocessed data.'
        self.mdstack: list[str] = []
        self.treebuilder = etree.TreeBuilder()
        self.mdstate: list[Literal['block', 'span', 'off', None]] = []
        super().reset()

    def close(self):
        if False:
            print('Hello World!')
        'Handle any buffered data.'
        super().close()
        if self.mdstack:
            self.handle_endtag(self.mdstack[0])

    def get_element(self) -> etree.Element:
        if False:
            for i in range(10):
                print('nop')
        ' Return element from `treebuilder` and reset `treebuilder` for later use. '
        element = self.treebuilder.close()
        self.treebuilder = etree.TreeBuilder()
        return element

    def get_state(self, tag, attrs: Mapping[str, str]) -> Literal['block', 'span', 'off', None]:
        if False:
            while True:
                i = 10
        " Return state from tag and `markdown` attribute. One of 'block', 'span', or 'off'. "
        md_attr = attrs.get('markdown', '0')
        if md_attr == 'markdown':
            md_attr = '1'
        parent_state = self.mdstate[-1] if self.mdstate else None
        if parent_state == 'off' or (parent_state == 'span' and md_attr != '0'):
            md_attr = parent_state
        if md_attr == '1' and tag in self.block_tags or (md_attr == 'block' and tag in self.span_and_blocks_tags):
            return 'block'
        elif md_attr == '1' and tag in self.span_tags or (md_attr == 'span' and tag in self.span_and_blocks_tags):
            return 'span'
        elif tag in self.block_level_tags:
            return 'off'
        else:
            return None

    def handle_starttag(self, tag, attrs):
        if False:
            while True:
                i = 10
        if tag in self.empty_tags and (self.at_line_start() or self.intail):
            attrs = {key: value if value is not None else key for (key, value) in attrs}
            if 'markdown' in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
            self.handle_empty_tag(data, True)
            return
        if tag in self.block_level_tags and (self.at_line_start() or self.intail):
            attrs = {key: value if value is not None else key for (key, value) in attrs}
            state = self.get_state(tag, attrs)
            if self.inraw or (state in [None, 'off'] and (not self.mdstack)):
                attrs.pop('markdown', None)
                super().handle_starttag(tag, attrs)
            else:
                if 'p' in self.mdstack and tag in self.block_level_tags:
                    self.handle_endtag('p')
                self.mdstate.append(state)
                self.mdstack.append(tag)
                attrs['markdown'] = state
                self.treebuilder.start(tag, attrs)
        elif self.inraw:
            super().handle_starttag(tag, attrs)
        else:
            text = self.get_starttag_text()
            if self.mdstate and self.mdstate[-1] == 'off':
                self.handle_data(self.md.htmlStash.store(text))
            else:
                self.handle_data(text)
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.clear_cdata_mode()

    def handle_endtag(self, tag):
        if False:
            return 10
        if tag in self.block_level_tags:
            if self.inraw:
                super().handle_endtag(tag)
            elif tag in self.mdstack:
                while self.mdstack:
                    item = self.mdstack.pop()
                    self.mdstate.pop()
                    self.treebuilder.end(item)
                    if item == tag:
                        break
                if not self.mdstack:
                    element = self.get_element()
                    item = self.cleandoc[-1] if self.cleandoc else ''
                    if not item.endswith('\n\n') and item.endswith('\n'):
                        self.cleandoc.append('\n')
                    self.cleandoc.append(self.md.htmlStash.store(element))
                    self.cleandoc.append('\n\n')
                    self.state = []
                    if not blank_line_re.match(self.rawdata[self.line_offset + self.offset + len(self.get_endtag_text(tag)):]):
                        self.intail = True
            else:
                text = self.get_endtag_text(tag)
                if self.mdstate and self.mdstate[-1] == 'off':
                    self.handle_data(self.md.htmlStash.store(text))
                else:
                    self.handle_data(text)
        elif self.inraw:
            super().handle_endtag(tag)
        else:
            text = self.get_endtag_text(tag)
            if self.mdstate and self.mdstate[-1] == 'off':
                self.handle_data(self.md.htmlStash.store(text))
            else:
                self.handle_data(text)

    def handle_startendtag(self, tag, attrs):
        if False:
            return 10
        if tag in self.empty_tags:
            attrs = {key: value if value is not None else key for (key, value) in attrs}
            if 'markdown' in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
        else:
            data = self.get_starttag_text()
        self.handle_empty_tag(data, is_block=self.md.is_block_level(tag))

    def handle_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.intail and '\n' in data:
            self.intail = False
        if self.inraw or not self.mdstack:
            super().handle_data(data)
        else:
            self.treebuilder.data(data)

    def handle_empty_tag(self, data, is_block):
        if False:
            while True:
                i = 10
        if self.inraw or not self.mdstack:
            super().handle_empty_tag(data, is_block)
        elif self.at_line_start() and is_block:
            self.handle_data('\n' + self.md.htmlStash.store(data) + '\n\n')
        else:
            self.handle_data(self.md.htmlStash.store(data))

    def parse_pi(self, i: int) -> int:
        if False:
            return 10
        if self.at_line_start() or self.intail or self.mdstack:
            return super(HTMLExtractor, self).parse_pi(i)
        self.handle_data('<?')
        return i + 2

    def parse_html_declaration(self, i: int) -> int:
        if False:
            return 10
        if self.at_line_start() or self.intail or self.mdstack:
            return super(HTMLExtractor, self).parse_html_declaration(i)
        self.handle_data('<!')
        return i + 2

class HtmlBlockPreprocessor(Preprocessor):
    """Remove html blocks from the text and store them for later retrieval."""

    def run(self, lines: list[str]) -> list[str]:
        if False:
            return 10
        source = '\n'.join(lines)
        parser = HTMLExtractorExtra(self.md)
        parser.feed(source)
        parser.close()
        return ''.join(parser.cleandoc).split('\n')

class MarkdownInHtmlProcessor(BlockProcessor):
    """Process Markdown Inside HTML Blocks which have been stored in the `HtmlStash`."""

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            return 10
        return True

    def parse_element_content(self, element: etree.Element) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Recursively parse the text content of an `etree` Element as Markdown.\n\n        Any block level elements generated from the Markdown will be inserted as children of the element in place\n        of the text content. All `markdown` attributes are removed. For any elements in which Markdown parsing has\n        been disabled, the text content of it and its children are wrapped in an `AtomicString`.\n        '
        md_attr = element.attrib.pop('markdown', 'off')
        if md_attr == 'block':
            for child in list(element):
                self.parse_element_content(child)
            tails = []
            for (pos, child) in enumerate(element):
                if child.tail:
                    block = child.tail.rstrip('\n')
                    child.tail = ''
                    dummy = etree.Element('div')
                    self.parser.parseBlocks(dummy, block.split('\n\n'))
                    children = list(dummy)
                    children.reverse()
                    tails.append((pos + 1, children))
            tails.reverse()
            for (pos, tail) in tails:
                for item in tail:
                    element.insert(pos, item)
            if element.text:
                block = element.text.rstrip('\n')
                element.text = ''
                dummy = etree.Element('div')
                self.parser.parseBlocks(dummy, block.split('\n\n'))
                children = list(dummy)
                children.reverse()
                for child in children:
                    element.insert(0, child)
        elif md_attr == 'span':
            for child in list(element):
                self.parse_element_content(child)
        else:
            if element.text is None:
                element.text = ''
            element.text = util.AtomicString(element.text)
            for child in list(element):
                self.parse_element_content(child)
                if child.tail:
                    child.tail = util.AtomicString(child.tail)

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        if False:
            return 10
        m = util.HTML_PLACEHOLDER_RE.match(blocks[0])
        if m:
            index = int(m.group(1))
            element = self.parser.md.htmlStash.rawHtmlBlocks[index]
            if isinstance(element, etree.Element):
                blocks.pop(0)
                self.parse_element_content(element)
                parent.append(element)
                self.parser.md.htmlStash.rawHtmlBlocks.pop(index)
                self.parser.md.htmlStash.rawHtmlBlocks.insert(index, '')
                return True
        return False

class MarkdownInHTMLPostprocessor(RawHtmlPostprocessor):

    def stash_to_string(self, text: str | etree.Element) -> str:
        if False:
            print('Hello World!')
        ' Override default to handle any `etree` elements still in the stash. '
        if isinstance(text, etree.Element):
            return self.md.serializer(text)
        else:
            return str(text)

class MarkdownInHtmlExtension(Extension):
    """Add Markdown parsing in HTML to Markdown class."""

    def extendMarkdown(self, md):
        if False:
            for i in range(10):
                print('nop')
        ' Register extension instances. '
        md.preprocessors.register(HtmlBlockPreprocessor(md), 'html_block', 20)
        md.parser.blockprocessors.register(MarkdownInHtmlProcessor(md.parser), 'markdown_block', 105)
        md.postprocessors.register(MarkdownInHTMLPostprocessor(md), 'raw_html', 30)

def makeExtension(**kwargs):
    if False:
        i = 10
        return i + 15
    return MarkdownInHtmlExtension(**kwargs)