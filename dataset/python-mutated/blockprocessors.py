"""
A block processor parses blocks of text and adds new elements to the ElementTree. Blocks of text,
separated from other text by blank lines, may have a different syntax and produce a differently
structured tree than other Markdown. Block processors excel at handling code formatting, equation
layouts, tables, etc.
"""
from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
if TYPE_CHECKING:
    from markdown import Markdown
logger = logging.getLogger('MARKDOWN')

def build_block_parser(md: Markdown, **kwargs: Any) -> BlockParser:
    if False:
        return 10
    ' Build the default block parser used by Markdown. '
    parser = BlockParser(md)
    parser.blockprocessors.register(EmptyBlockProcessor(parser), 'empty', 100)
    parser.blockprocessors.register(ListIndentProcessor(parser), 'indent', 90)
    parser.blockprocessors.register(CodeBlockProcessor(parser), 'code', 80)
    parser.blockprocessors.register(HashHeaderProcessor(parser), 'hashheader', 70)
    parser.blockprocessors.register(SetextHeaderProcessor(parser), 'setextheader', 60)
    parser.blockprocessors.register(HRProcessor(parser), 'hr', 50)
    parser.blockprocessors.register(OListProcessor(parser), 'olist', 40)
    parser.blockprocessors.register(UListProcessor(parser), 'ulist', 30)
    parser.blockprocessors.register(BlockQuoteProcessor(parser), 'quote', 20)
    parser.blockprocessors.register(ReferenceProcessor(parser), 'reference', 15)
    parser.blockprocessors.register(ParagraphProcessor(parser), 'paragraph', 10)
    return parser

class BlockProcessor:
    """ Base class for block processors.

    Each subclass will provide the methods below to work with the source and
    tree. Each processor will need to define it's own `test` and `run`
    methods. The `test` method should return True or False, to indicate
    whether the current block should be processed by this processor. If the
    test passes, the parser will call the processors `run` method.

    Attributes:
        BlockProcessor.parser (BlockParser): The `BlockParser` instance this is attached to.
        BlockProcessor.tab_length (int): The tab length set on the `Markdown` instance.

    """

    def __init__(self, parser: BlockParser):
        if False:
            print('Hello World!')
        self.parser = parser
        self.tab_length = parser.md.tab_length

    def lastChild(self, parent: etree.Element) -> etree.Element | None:
        if False:
            print('Hello World!')
        ' Return the last child of an `etree` element. '
        if len(parent):
            return parent[-1]
        else:
            return None

    def detab(self, text: str, length: int | None=None) -> tuple[str, str]:
        if False:
            print('Hello World!')
        ' Remove a tab from the front of each line of the given text. '
        if length is None:
            length = self.tab_length
        newtext = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith(' ' * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append('')
            else:
                break
        return ('\n'.join(newtext), '\n'.join(lines[len(newtext):]))

    def looseDetab(self, text: str, level: int=1) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Remove a tab from front of lines but allowing dedented lines. '
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' ' * self.tab_length * level):
                lines[i] = lines[i][self.tab_length * level:]
        return '\n'.join(lines)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            i = 10
            return i + 15
        ' Test for block type. Must be overridden by subclasses.\n\n        As the parser loops through processors, it will call the `test`\n        method on each to determine if the given block of text is of that\n        type. This method must return a boolean `True` or `False`. The\n        actual method of testing is left to the needs of that particular\n        block type. It could be as simple as `block.startswith(some_string)`\n        or a complex regular expression. As the block type may be different\n        depending on the parent of the block (i.e. inside a list), the parent\n        `etree` element is also provided and may be used as part of the test.\n\n        Keyword arguments:\n            parent: An `etree` element which will be the parent of the block.\n            block: A block of text from the source which has been split at blank lines.\n        '
        pass

    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:
        if False:
            for i in range(10):
                print('nop')
        " Run processor. Must be overridden by subclasses.\n\n        When the parser determines the appropriate type of a block, the parser\n        will call the corresponding processor's `run` method. This method\n        should parse the individual lines of the block and append them to\n        the `etree`.\n\n        Note that both the `parent` and `etree` keywords are pointers\n        to instances of the objects which should be edited in place. Each\n        processor must make changes to the existing objects as there is no\n        mechanism to return new/different objects to replace them.\n\n        This means that this method should be adding `SubElements` or adding text\n        to the parent, and should remove (`pop`) or add (`insert`) items to\n        the list of blocks.\n\n        If `False` is returned, this will have the same effect as returning `False`\n        from the `test` method.\n\n        Keyword arguments:\n            parent: An `etree` element which is the parent of the current block.\n            blocks: A list of all remaining blocks of the document.\n        "
        pass

class ListIndentProcessor(BlockProcessor):
    """ Process children of list items.

    Example

        * a list item
            process this part

            or this part

    """
    ITEM_TYPES = ['li']
    ' List of tags used for list items. '
    LIST_TYPES = ['ul', 'ol']
    ' Types of lists this processor can operate on. '

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self.INDENT_RE = re.compile('^(([ ]{%s})+)' % self.tab_length)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return block.startswith(' ' * self.tab_length) and (not self.parser.state.isstate('detabbed')) and (parent.tag in self.ITEM_TYPES or (len(parent) and parent[-1] is not None and (parent[-1].tag in self.LIST_TYPES)))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            return 10
        block = blocks.pop(0)
        (level, sibling) = self.get_level(parent, block)
        block = self.looseDetab(block, level)
        self.parser.state.set('detabbed')
        if parent.tag in self.ITEM_TYPES:
            if len(parent) and parent[-1].tag in self.LIST_TYPES:
                self.parser.parseBlocks(parent[-1], [block])
            else:
                self.parser.parseBlocks(parent, [block])
        elif sibling.tag in self.ITEM_TYPES:
            self.parser.parseBlocks(sibling, [block])
        elif len(sibling) and sibling[-1].tag in self.ITEM_TYPES:
            if sibling[-1].text:
                p = etree.Element('p')
                p.text = sibling[-1].text
                sibling[-1].text = ''
                sibling[-1].insert(0, p)
            self.parser.parseChunk(sibling[-1], block)
        else:
            self.create_item(sibling, block)
        self.parser.state.reset()

    def create_item(self, parent: etree.Element, block: str) -> None:
        if False:
            return 10
        ' Create a new `li` and parse the block with it as the parent. '
        li = etree.SubElement(parent, 'li')
        self.parser.parseBlocks(li, [block])

    def get_level(self, parent: etree.Element, block: str) -> tuple[int, etree.Element]:
        if False:
            for i in range(10):
                print('nop')
        ' Get level of indentation based on list level. '
        m = self.INDENT_RE.match(block)
        if m:
            indent_level = len(m.group(1)) / self.tab_length
        else:
            indent_level = 0
        if self.parser.state.isstate('list'):
            level = 1
        else:
            level = 0
        while indent_level > level:
            child = self.lastChild(parent)
            if child is not None and (child.tag in self.LIST_TYPES or child.tag in self.ITEM_TYPES):
                if child.tag in self.LIST_TYPES:
                    level += 1
                parent = child
            else:
                break
        return (level, parent)

class CodeBlockProcessor(BlockProcessor):
    """ Process code blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            print('Hello World!')
        return block.startswith(' ' * self.tab_length)

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            i = 10
            return i + 15
        sibling = self.lastChild(parent)
        block = blocks.pop(0)
        theRest = ''
        if sibling is not None and sibling.tag == 'pre' and len(sibling) and (sibling[0].tag == 'code'):
            code = sibling[0]
            (block, theRest) = self.detab(block)
            code.text = util.AtomicString('{}\n{}\n'.format(code.text, util.code_escape(block.rstrip())))
        else:
            pre = etree.SubElement(parent, 'pre')
            code = etree.SubElement(pre, 'code')
            (block, theRest) = self.detab(block)
            code.text = util.AtomicString('%s\n' % util.code_escape(block.rstrip()))
        if theRest:
            blocks.insert(0, theRest)

class BlockQuoteProcessor(BlockProcessor):
    """ Process blockquotes. """
    RE = re.compile('(^|\\n)[ ]{0,3}>[ ]?(.*)')

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.RE.search(block)) and (not util.nearing_recursion_limit())

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]
            self.parser.parseBlocks(parent, [before])
            block = '\n'.join([self.clean(line) for line in block[m.start():].split('\n')])
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag == 'blockquote':
            quote = sibling
        else:
            quote = etree.SubElement(parent, 'blockquote')
        self.parser.state.set('blockquote')
        self.parser.parseChunk(quote, block)
        self.parser.state.reset()

    def clean(self, line: str) -> str:
        if False:
            return 10
        ' Remove `>` from beginning of a line. '
        m = self.RE.match(line)
        if line.strip() == '>':
            return ''
        elif m:
            return m.group(2)
        else:
            return line

class OListProcessor(BlockProcessor):
    """ Process ordered list blocks. """
    TAG: str = 'ol'
    ' The tag used for the the wrapping element. '
    STARTSWITH: str = '1'
    '\n    The integer (as a string ) with which the list starts. For example, if a list is initialized as\n    `3. Item`, then the `ol` tag will be assigned an HTML attribute of `starts="3"`. Default: `"1"`.\n    '
    LAZY_OL: bool = True
    ' Ignore `STARTSWITH` if `True`. '
    SIBLING_TAGS: list[str] = ['ol', 'ul']
    '\n    Markdown does not require the type of a new list item match the previous list item type.\n    This is the list of types which can be mixed.\n    '

    def __init__(self, parser: BlockParser):
        if False:
            print('Hello World!')
        super().__init__(parser)
        self.RE = re.compile('^[ ]{0,%d}\\d+\\.[ ]+(.*)' % (self.tab_length - 1))
        self.CHILD_RE = re.compile('^[ ]{0,%d}((\\d+\\.)|[*+-])[ ]+(.*)' % (self.tab_length - 1))
        self.INDENT_RE = re.compile('^[ ]{%d,%d}((\\d+\\.)|[*+-])[ ]+.*' % (self.tab_length, self.tab_length * 2 - 1))

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            return 10
        items = self.get_items(blocks.pop(0))
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag in self.SIBLING_TAGS:
            lst = sibling
            if lst[-1].text:
                p = etree.Element('p')
                p.text = lst[-1].text
                lst[-1].text = ''
                lst[-1].insert(0, p)
            lch = self.lastChild(lst[-1])
            if lch is not None and lch.tail:
                p = etree.SubElement(lst[-1], 'p')
                p.text = lch.tail.lstrip()
                lch.tail = ''
            li = etree.SubElement(lst, 'li')
            self.parser.state.set('looselist')
            firstitem = items.pop(0)
            self.parser.parseBlocks(li, [firstitem])
            self.parser.state.reset()
        elif parent.tag in ['ol', 'ul']:
            lst = parent
        else:
            lst = etree.SubElement(parent, self.TAG)
            if not self.LAZY_OL and self.STARTSWITH != '1':
                lst.attrib['start'] = self.STARTSWITH
        self.parser.state.set('list')
        for item in items:
            if item.startswith(' ' * self.tab_length):
                self.parser.parseBlocks(lst[-1], [item])
            else:
                li = etree.SubElement(lst, 'li')
                self.parser.parseBlocks(li, [item])
        self.parser.state.reset()

    def get_items(self, block: str) -> list[str]:
        if False:
            i = 10
            return i + 15
        ' Break a block into list items. '
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                if not items and self.TAG == 'ol':
                    INTEGER_RE = re.compile('(\\d+)')
                    self.STARTSWITH = INTEGER_RE.match(m.group(1)).group()
                items.append(m.group(3))
            elif self.INDENT_RE.match(line):
                if items[-1].startswith(' ' * self.tab_length):
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items

class UListProcessor(OListProcessor):
    """ Process unordered list blocks. """
    TAG: str = 'ul'
    ' The tag used for the the wrapping element. '

    def __init__(self, parser: BlockParser):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parser)
        self.RE = re.compile('^[ ]{0,%d}[*+-][ ]+(.*)' % (self.tab_length - 1))

class HashHeaderProcessor(BlockProcessor):
    """ Process Hash Headers. """
    RE = re.compile('(?:^|\\n)(?P<level>#{1,6})(?P<header>(?:\\\\.|[^\\\\])*?)#*(?:\\n|$)')

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return bool(self.RE.search(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            print('Hello World!')
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]
            after = block[m.end():]
            if before:
                self.parser.parseBlocks(parent, [before])
            h = etree.SubElement(parent, 'h%d' % len(m.group('level')))
            h.text = m.group('header').strip()
            if after:
                blocks.insert(0, after)
        else:
            logger.warn("We've got a problem header: %r" % block)

class SetextHeaderProcessor(BlockProcessor):
    """ Process Setext-style Headers. """
    RE = re.compile('^.*?\\n[=-]+[ ]*(\\n|$)', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            i = 10
            return i + 15
        lines = blocks.pop(0).split('\n')
        if lines[1].startswith('='):
            level = 1
        else:
            level = 2
        h = etree.SubElement(parent, 'h%d' % level)
        h.text = lines[0].strip()
        if len(lines) > 2:
            blocks.insert(0, '\n'.join(lines[2:]))

class HRProcessor(BlockProcessor):
    """ Process Horizontal Rules. """
    RE = '^[ ]{0,3}(?=(?P<atomicgroup>(-+[ ]{0,2}){3,}|(_+[ ]{0,2}){3,}|(\\*+[ ]{0,2}){3,}))(?P=atomicgroup)[ ]*$'
    SEARCH_RE = re.compile(RE, re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            i = 10
            return i + 15
        m = self.SEARCH_RE.search(block)
        if m:
            self.match = m
            return True
        return False

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            print('Hello World!')
        block = blocks.pop(0)
        match = self.match
        prelines = block[:match.start()].rstrip('\n')
        if prelines:
            self.parser.parseBlocks(parent, [prelines])
        etree.SubElement(parent, 'hr')
        postlines = block[match.end():].lstrip('\n')
        if postlines:
            blocks.insert(0, postlines)

class EmptyBlockProcessor(BlockProcessor):
    """ Process blocks that are empty or start with an empty line. """

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return not block or block.startswith('\n')

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        block = blocks.pop(0)
        filler = '\n\n'
        if block:
            filler = '\n'
            theRest = block[1:]
            if theRest:
                blocks.insert(0, theRest)
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag == 'pre' and len(sibling) and (sibling[0].tag == 'code'):
            sibling[0].text = util.AtomicString('{}{}'.format(sibling[0].text, filler))

class ReferenceProcessor(BlockProcessor):
    """ Process link references. """
    RE = re.compile('^[ ]{0,3}\\[([^\\[\\]]*)\\]:[ ]*\\n?[ ]*([^\\s]+)[ ]*(?:\\n[ ]*)?((["\\\'])(.*)\\4[ ]*|\\((.*)\\)[ ]*)?$', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            id = m.group(1).strip().lower()
            link = m.group(2).lstrip('<').rstrip('>')
            title = m.group(5) or m.group(6)
            self.parser.md.references[id] = (link, title)
            if block[m.end():].strip():
                blocks.insert(0, block[m.end():].lstrip('\n'))
            if block[:m.start()].strip():
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        blocks.insert(0, block)
        return False

class ParagraphProcessor(BlockProcessor):
    """ Process Paragraph blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            while True:
                i = 10
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        block = blocks.pop(0)
        if block.strip():
            if self.parser.state.isstate('list'):
                sibling = self.lastChild(parent)
                if sibling is not None:
                    if sibling.tail:
                        sibling.tail = '{}\n{}'.format(sibling.tail, block)
                    else:
                        sibling.tail = '\n%s' % block
                elif parent.text:
                    parent.text = '{}\n{}'.format(parent.text, block)
                else:
                    parent.text = block.lstrip()
            else:
                p = etree.SubElement(parent, 'p')
                p.text = block.lstrip()