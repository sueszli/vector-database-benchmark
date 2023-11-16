"""
The block parser handles basic parsing of Markdown blocks.  It doesn't concern
itself with inline elements such as `**bold**` or `*italics*`, but rather just
catches blocks, lists, quotes, etc.

The `BlockParser` is made up of a bunch of `BlockProcessors`, each handling a
different type of block. Extensions may add/replace/remove `BlockProcessors`
as they need to alter how Markdown blocks are parsed.
"""
from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
if TYPE_CHECKING:
    from markdown import Markdown
    from .blockprocessors import BlockProcessor

class State(list):
    """ Track the current and nested state of the parser.

    This utility class is used to track the state of the `BlockParser` and
    support multiple levels if nesting. It's just a simple API wrapped around
    a list. Each time a state is set, that state is appended to the end of the
    list. Each time a state is reset, that state is removed from the end of
    the list.

    Therefore, each time a state is set for a nested block, that state must be
    reset when we back out of that level of nesting or the state could be
    corrupted.

    While all the methods of a list object are available, only the three
    defined below need be used.

    """

    def set(self, state: Any):
        if False:
            while True:
                i = 10
        ' Set a new state. '
        self.append(state)

    def reset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Step back one step in nested state. '
        self.pop()

    def isstate(self, state: Any) -> bool:
        if False:
            i = 10
            return i + 15
        ' Test that top (current) level is of given state. '
        if len(self):
            return self[-1] == state
        else:
            return False

class BlockParser:
    """ Parse Markdown blocks into an `ElementTree` object.

    A wrapper class that stitches the various `BlockProcessors` together,
    looping through them and creating an `ElementTree` object.

    """

    def __init__(self, md: Markdown):
        if False:
            return 10
        ' Initialize the block parser.\n\n        Arguments:\n            md: A Markdown instance.\n\n        Attributes:\n            BlockParser.md (Markdown): A Markdown instance.\n            BlockParser.state (State): Tracks the nesting level of current location in document being parsed.\n            BlockParser.blockprocessors (util.Registry): A collection of\n                [`blockprocessors`][markdown.blockprocessors].\n\n        '
        self.blockprocessors: util.Registry[BlockProcessor] = util.Registry()
        self.state = State()
        self.md = md

    def parseDocument(self, lines: Iterable[str]) -> etree.ElementTree:
        if False:
            for i in range(10):
                print('nop')
        ' Parse a Markdown document into an `ElementTree`.\n\n        Given a list of lines, an `ElementTree` object (not just a parent\n        `Element`) is created and the root element is passed to the parser\n        as the parent. The `ElementTree` object is returned.\n\n        This should only be called on an entire document, not pieces.\n\n        Arguments:\n            lines: A list of lines (strings).\n\n        Returns:\n            An element tree.\n        '
        self.root = etree.Element(self.md.doc_tag)
        self.parseChunk(self.root, '\n'.join(lines))
        return etree.ElementTree(self.root)

    def parseChunk(self, parent: etree.Element, text: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Parse a chunk of Markdown text and attach to given `etree` node.\n\n        While the `text` argument is generally assumed to contain multiple\n        blocks which will be split on blank lines, it could contain only one\n        block. Generally, this method would be called by extensions when\n        block parsing is required.\n\n        The `parent` `etree` Element passed in is altered in place.\n        Nothing is returned.\n\n        Arguments:\n            parent: The parent element.\n            text: The text to parse.\n\n        '
        self.parseBlocks(parent, text.split('\n\n'))

    def parseBlocks(self, parent: etree.Element, blocks: list[str]) -> None:
        if False:
            return 10
        " Process blocks of Markdown text and attach to given `etree` node.\n\n        Given a list of `blocks`, each `blockprocessor` is stepped through\n        until there are no blocks left. While an extension could potentially\n        call this method directly, it's generally expected to be used\n        internally.\n\n        This is a public method as an extension may need to add/alter\n        additional `BlockProcessors` which call this method to recursively\n        parse a nested block.\n\n        Arguments:\n            parent: The parent element.\n            blocks: The blocks of text to parse.\n\n        "
        while blocks:
            for processor in self.blockprocessors:
                if processor.test(parent, blocks[0]):
                    if processor.run(parent, blocks) is not False:
                        break