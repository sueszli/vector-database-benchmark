"""
This extension adds abbreviation handling to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/abbreviations)
for details.
"""
from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..util import AtomicString
import re
import xml.etree.ElementTree as etree

class AbbrExtension(Extension):
    """ Abbreviation Extension for Python-Markdown. """

    def extendMarkdown(self, md):
        if False:
            return 10
        ' Insert `AbbrPreprocessor` before `ReferencePreprocessor`. '
        md.parser.blockprocessors.register(AbbrPreprocessor(md.parser), 'abbr', 16)

class AbbrPreprocessor(BlockProcessor):
    """ Abbreviation Preprocessor - parse text for abbr references. """
    RE = re.compile('^[*]\\[(?P<abbr>[^\\]]*)\\][ ]?:[ ]*\\n?[ ]*(?P<title>.*)$', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        if False:
            return 10
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        if False:
            while True:
                i = 10
        '\n        Find and remove all Abbreviation references from the text.\n        Each reference is set as a new `AbbrPattern` in the markdown instance.\n\n        '
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            abbr = m.group('abbr').strip()
            title = m.group('title').strip()
            self.parser.md.inlinePatterns.register(AbbrInlineProcessor(self._generate_pattern(abbr), title), 'abbr-%s' % abbr, 2)
            if block[m.end():].strip():
                blocks.insert(0, block[m.end():].lstrip('\n'))
            if block[:m.start()].strip():
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        blocks.insert(0, block)
        return False

    def _generate_pattern(self, text: str) -> str:
        if False:
            return 10
        "\n        Given a string, returns an regex pattern to match that string.\n\n        'HTML' -> r'(?P<abbr>[H][T][M][L])'\n\n        Note: we force each char as a literal match (in brackets) as we don't\n        know what they will be beforehand.\n\n        "
        chars = list(text)
        for i in range(len(chars)):
            chars[i] = '[%s]' % chars[i]
        return '(?P<abbr>\\b%s\\b)' % ''.join(chars)

class AbbrInlineProcessor(InlineProcessor):
    """ Abbreviation inline pattern. """

    def __init__(self, pattern: str, title: str):
        if False:
            i = 10
            return i + 15
        super().__init__(pattern)
        self.title = title

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            i = 10
            return i + 15
        abbr = etree.Element('abbr')
        abbr.text = AtomicString(m.group('abbr'))
        abbr.set('title', self.title)
        return (abbr, m.start(0), m.end(0))

def makeExtension(**kwargs):
    if False:
        return 10
    return AbbrExtension(**kwargs)