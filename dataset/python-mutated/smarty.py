"""
Adds conversion of ASCII dashes, quotes and ellipses to their HTML
entity equivalents.

See the [documentation](https://Python-Markdown.github.io/extensions/smarty)
for details.
"""
from __future__ import annotations
from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
from ..util import Registry
from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from markdown import Markdown
    from .. import inlinepatterns
    import re
    import xml.etree.ElementTree as etree
punctClass = '[!"#\\$\\%\'()*+,-.\\/:;<=>?\\@\\[\\\\\\]\\^_`{|}~]'
endOfWordClass = '[\\s.,;:!?)]'
closeClass = '[^\\ \\t\\r\\n\\[\\{\\(\\-\\u0002\\u0003]'
openingQuotesBase = '(\\s|&nbsp;|--|–|—|&[mn]dash;|&#8211;|&#8212;)'
substitutions = {'mdash': '&mdash;', 'ndash': '&ndash;', 'ellipsis': '&hellip;', 'left-angle-quote': '&laquo;', 'right-angle-quote': '&raquo;', 'left-single-quote': '&lsquo;', 'right-single-quote': '&rsquo;', 'left-double-quote': '&ldquo;', 'right-double-quote': '&rdquo;'}
singleQuoteStartRe = "^'(?=%s\\B)" % punctClass
doubleQuoteStartRe = '^"(?=%s\\B)' % punctClass
doubleQuoteSetsRe = '"\'(?=\\w)'
singleQuoteSetsRe = '\'"(?=\\w)'
decadeAbbrRe = "(?<!\\w)'(?=\\d{2}s)"
openingDoubleQuotesRegex = '%s"(?=\\w)' % openingQuotesBase
closingDoubleQuotesRegex = '"(?=\\s)'
closingDoubleQuotesRegex2 = '(?<=%s)"' % closeClass
openingSingleQuotesRegex = "%s'(?=\\w)" % openingQuotesBase
closingSingleQuotesRegex = "(?<=%s)'(?!\\s|s\\b|\\d)" % closeClass
closingSingleQuotesRegex2 = "'(\\s|s\\b)"
remainingSingleQuotesRegex = "'"
remainingDoubleQuotesRegex = '"'
HTML_STRICT_RE = HTML_RE + '(?!\\>)'

class SubstituteTextPattern(HtmlInlineProcessor):

    def __init__(self, pattern: str, replace: Sequence[int | str | etree.Element], md: Markdown):
        if False:
            print('Hello World!')
        ' Replaces matches with some text. '
        HtmlInlineProcessor.__init__(self, pattern)
        self.replace = replace
        self.md = md

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        if False:
            while True:
                i = 10
        result = ''
        for part in self.replace:
            if isinstance(part, int):
                result += m.group(part)
            else:
                result += self.md.htmlStash.store(part)
        return (result, m.start(0), m.end(0))

class SmartyExtension(Extension):
    """ Add Smarty to Markdown. """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.config = {'smart_quotes': [True, 'Educate quotes'], 'smart_angled_quotes': [False, 'Educate angled quotes'], 'smart_dashes': [True, 'Educate dashes'], 'smart_ellipses': [True, 'Educate ellipses'], 'substitutions': [{}, 'Overwrite default substitutions']}
        ' Default configuration options. '
        super().__init__(**kwargs)
        self.substitutions: dict[str, str] = dict(substitutions)
        self.substitutions.update(self.getConfig('substitutions', default={}))

    def _addPatterns(self, md: Markdown, patterns: Sequence[tuple[str, Sequence[int | str | etree.Element]]], serie: str, priority: int):
        if False:
            i = 10
            return i + 15
        for (ind, pattern) in enumerate(patterns):
            pattern += (md,)
            pattern = SubstituteTextPattern(*pattern)
            name = 'smarty-%s-%d' % (serie, ind)
            self.inlinePatterns.register(pattern, name, priority - ind)

    def educateDashes(self, md: Markdown) -> None:
        if False:
            i = 10
            return i + 15
        emDashesPattern = SubstituteTextPattern('(?<!-)---(?!-)', (self.substitutions['mdash'],), md)
        enDashesPattern = SubstituteTextPattern('(?<!-)--(?!-)', (self.substitutions['ndash'],), md)
        self.inlinePatterns.register(emDashesPattern, 'smarty-em-dashes', 50)
        self.inlinePatterns.register(enDashesPattern, 'smarty-en-dashes', 45)

    def educateEllipses(self, md: Markdown) -> None:
        if False:
            return 10
        ellipsesPattern = SubstituteTextPattern('(?<!\\.)\\.{3}(?!\\.)', (self.substitutions['ellipsis'],), md)
        self.inlinePatterns.register(ellipsesPattern, 'smarty-ellipses', 10)

    def educateAngledQuotes(self, md: Markdown) -> None:
        if False:
            for i in range(10):
                print('nop')
        leftAngledQuotePattern = SubstituteTextPattern('\\<\\<', (self.substitutions['left-angle-quote'],), md)
        rightAngledQuotePattern = SubstituteTextPattern('\\>\\>', (self.substitutions['right-angle-quote'],), md)
        self.inlinePatterns.register(leftAngledQuotePattern, 'smarty-left-angle-quotes', 40)
        self.inlinePatterns.register(rightAngledQuotePattern, 'smarty-right-angle-quotes', 35)

    def educateQuotes(self, md: Markdown) -> None:
        if False:
            print('Hello World!')
        lsquo = self.substitutions['left-single-quote']
        rsquo = self.substitutions['right-single-quote']
        ldquo = self.substitutions['left-double-quote']
        rdquo = self.substitutions['right-double-quote']
        patterns = ((singleQuoteStartRe, (rsquo,)), (doubleQuoteStartRe, (rdquo,)), (doubleQuoteSetsRe, (ldquo + lsquo,)), (singleQuoteSetsRe, (lsquo + ldquo,)), (decadeAbbrRe, (rsquo,)), (openingSingleQuotesRegex, (1, lsquo)), (closingSingleQuotesRegex, (rsquo,)), (closingSingleQuotesRegex2, (rsquo, 1)), (remainingSingleQuotesRegex, (lsquo,)), (openingDoubleQuotesRegex, (1, ldquo)), (closingDoubleQuotesRegex, (rdquo,)), (closingDoubleQuotesRegex2, (rdquo,)), (remainingDoubleQuotesRegex, (ldquo,)))
        self._addPatterns(md, patterns, 'quotes', 30)

    def extendMarkdown(self, md):
        if False:
            while True:
                i = 10
        configs = self.getConfigs()
        self.inlinePatterns: Registry[inlinepatterns.InlineProcessor] = Registry()
        if configs['smart_ellipses']:
            self.educateEllipses(md)
        if configs['smart_quotes']:
            self.educateQuotes(md)
        if configs['smart_angled_quotes']:
            self.educateAngledQuotes(md)
            md.inlinePatterns.register(HtmlInlineProcessor(HTML_STRICT_RE, md), 'html', 90)
        if configs['smart_dashes']:
            self.educateDashes(md)
        inlineProcessor = InlineProcessor(md)
        inlineProcessor.inlinePatterns = self.inlinePatterns
        md.treeprocessors.register(inlineProcessor, 'smarty', 2)
        md.ESCAPED_CHARS.extend(['"', "'"])

def makeExtension(**kwargs):
    if False:
        while True:
            i = 10
    return SmartyExtension(**kwargs)