"""
Modify the behavior of Lists in Python-Markdown to act in a sane manor.

See [documentation](https://Python-Markdown.github.io/extensions/sane_lists)
for details.
"""
from __future__ import annotations
from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .. import blockparser

class SaneOListProcessor(OListProcessor):
    """ Override `SIBLING_TAGS` to not include `ul` and set `LAZY_OL` to `False`. """
    SIBLING_TAGS = ['ol']
    ' Exclude `ul` from list of siblings. '
    LAZY_OL = False
    ' Disable lazy list behavior. '

    def __init__(self, parser: blockparser.BlockParser):
        if False:
            i = 10
            return i + 15
        super().__init__(parser)
        self.CHILD_RE = re.compile('^[ ]{0,%d}((\\d+\\.))[ ]+(.*)' % (self.tab_length - 1))

class SaneUListProcessor(UListProcessor):
    """ Override `SIBLING_TAGS` to not include `ol`. """
    SIBLING_TAGS = ['ul']
    ' Exclude `ol` from list of siblings. '

    def __init__(self, parser: blockparser.BlockParser):
        if False:
            return 10
        super().__init__(parser)
        self.CHILD_RE = re.compile('^[ ]{0,%d}(([*+-]))[ ]+(.*)' % (self.tab_length - 1))

class SaneListExtension(Extension):
    """ Add sane lists to Markdown. """

    def extendMarkdown(self, md):
        if False:
            i = 10
            return i + 15
        ' Override existing Processors. '
        md.parser.blockprocessors.register(SaneOListProcessor(md.parser), 'olist', 40)
        md.parser.blockprocessors.register(SaneUListProcessor(md.parser), 'ulist', 30)

def makeExtension(**kwargs):
    if False:
        print('Hello World!')
    return SaneListExtension(**kwargs)