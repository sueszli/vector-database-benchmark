"""
In version 3.0, a new, more flexible inline processor was added, [`markdown.inlinepatterns.InlineProcessor`][].   The
original inline patterns, which inherit from [`markdown.inlinepatterns.Pattern`][] or one of its children are still
supported, though users are encouraged to migrate.

The new `InlineProcessor` provides two major enhancements to `Patterns`:

1. Inline Processors no longer need to match the entire block, so regular expressions no longer need to start with
  `r'^(.*?)'` and end with `r'(.*?)%'`. This runs faster. The returned [`Match`][re.Match] object will only contain
   what is explicitly matched in the pattern, and extension pattern groups now start with `m.group(1)`.

2.  The `handleMatch` method now takes an additional input called `data`, which is the entire block under analysis,
    not just what is matched with the specified pattern. The method now returns the element *and* the indexes relative
    to `data` that the return element is replacing (usually `m.start(0)` and `m.end(0)`).  If the boundaries are
    returned as `None`, it is assumed that the match did not take place, and nothing will be altered in `data`.

    This allows handling of more complex constructs than regular expressions can handle, e.g., matching nested
    brackets, and explicit control of the span "consumed" by the processor.

"""
from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
if TYPE_CHECKING:
    from markdown import Markdown

def build_inlinepatterns(md: Markdown, **kwargs: Any) -> util.Registry[InlineProcessor]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Build the default set of inline patterns for Markdown.\n\n    The order in which processors and/or patterns are applied is very important - e.g. if we first replace\n    `http://.../` links with `<a>` tags and _then_ try to replace inline HTML, we would end up with a mess. So, we\n    apply the expressions in the following order:\n\n    * backticks and escaped characters have to be handled before everything else so that we can preempt any markdown\n      patterns by escaping them;\n\n    * then we handle the various types of links (auto-links must be handled before inline HTML);\n\n    * then we handle inline HTML.  At this point we will simply replace all inline HTML strings with a placeholder\n      and add the actual HTML to a stash;\n\n    * finally we apply strong, emphasis, etc.\n\n    '
    inlinePatterns = util.Registry()
    inlinePatterns.register(BacktickInlineProcessor(BACKTICK_RE), 'backtick', 190)
    inlinePatterns.register(EscapeInlineProcessor(ESCAPE_RE, md), 'escape', 180)
    inlinePatterns.register(ReferenceInlineProcessor(REFERENCE_RE, md), 'reference', 170)
    inlinePatterns.register(LinkInlineProcessor(LINK_RE, md), 'link', 160)
    inlinePatterns.register(ImageInlineProcessor(IMAGE_LINK_RE, md), 'image_link', 150)
    inlinePatterns.register(ImageReferenceInlineProcessor(IMAGE_REFERENCE_RE, md), 'image_reference', 140)
    inlinePatterns.register(ShortReferenceInlineProcessor(REFERENCE_RE, md), 'short_reference', 130)
    inlinePatterns.register(ShortImageReferenceInlineProcessor(IMAGE_REFERENCE_RE, md), 'short_image_ref', 125)
    inlinePatterns.register(AutolinkInlineProcessor(AUTOLINK_RE, md), 'autolink', 120)
    inlinePatterns.register(AutomailInlineProcessor(AUTOMAIL_RE, md), 'automail', 110)
    inlinePatterns.register(SubstituteTagInlineProcessor(LINE_BREAK_RE, 'br'), 'linebreak', 100)
    inlinePatterns.register(HtmlInlineProcessor(HTML_RE, md), 'html', 90)
    inlinePatterns.register(HtmlInlineProcessor(ENTITY_RE, md), 'entity', 80)
    inlinePatterns.register(SimpleTextInlineProcessor(NOT_STRONG_RE), 'not_strong', 70)
    inlinePatterns.register(AsteriskProcessor('\\*'), 'em_strong', 60)
    inlinePatterns.register(UnderscoreProcessor('_'), 'em_strong2', 50)
    return inlinePatterns
NOIMG = '(?<!\\!)'
' Match not an image. Partial regular expression which matches if not preceded by `!`. '
BACKTICK_RE = '(?:(?<!\\\\)((?:\\\\{2})+)(?=`+)|(?<!\\\\)(`+)(.+?)(?<!`)\\2(?!`))'
' Match backtick quoted string (`` `e=f()` `` or ``` ``e=f("`")`` ```). '
ESCAPE_RE = '\\\\(.)'
' Match a backslash escaped character (`\\<` or `\\*`). '
EMPHASIS_RE = '(\\*)([^\\*]+)\\1'
' Match emphasis with an asterisk (`*emphasis*`). '
STRONG_RE = '(\\*{2})(.+?)\\1'
' Match strong with an asterisk (`**strong**`). '
SMART_STRONG_RE = '(?<!\\w)(_{2})(?!_)(.+?)(?<!_)\\1(?!\\w)'
' Match strong with underscore while ignoring middle word underscores (`__smart__strong__`). '
SMART_EMPHASIS_RE = '(?<!\\w)(_)(?!_)(.+?)(?<!_)\\1(?!\\w)'
' Match emphasis with underscore while ignoring middle word underscores (`_smart_emphasis_`). '
SMART_STRONG_EM_RE = '(?<!\\w)(\\_)\\1(?!\\1)(.+?)(?<!\\w)\\1(?!\\1)(.+?)\\1{3}(?!\\w)'
' Match strong emphasis with underscores (`__strong _em__`). '
EM_STRONG_RE = '(\\*)\\1{2}(.+?)\\1(.*?)\\1{2}'
' Match emphasis strong with asterisk (`***strongem***` or `***em*strong**`). '
EM_STRONG2_RE = '(_)\\1{2}(.+?)\\1(.*?)\\1{2}'
' Match emphasis strong with underscores (`___emstrong___` or `___em_strong__`). '
STRONG_EM_RE = '(\\*)\\1{2}(.+?)\\1{2}(.*?)\\1'
' Match strong emphasis with asterisk (`***strong**em*`). '
STRONG_EM2_RE = '(_)\\1{2}(.+?)\\1{2}(.*?)\\1'
' Match strong emphasis with underscores (`___strong__em_`). '
STRONG_EM3_RE = '(\\*)\\1(?!\\1)([^*]+?)\\1(?!\\1)(.+?)\\1{3}'
' Match strong emphasis with asterisk (`**strong*em***`). '
LINK_RE = NOIMG + '\\['
' Match start of in-line link (`[text](url)` or `[text](<url>)` or `[text](url "title")`). '
IMAGE_LINK_RE = '\\!\\['
' Match start of in-line image link (`![alttxt](url)` or `![alttxt](<url>)`). '
REFERENCE_RE = LINK_RE
' Match start of reference link (`[Label][3]`). '
IMAGE_REFERENCE_RE = IMAGE_LINK_RE
' Match start of image reference (`![alt text][2]`). '
NOT_STRONG_RE = '((^|(?<=\\s))(\\*{1,3}|_{1,3})(?=\\s|$))'
' Match a stand-alone `*` or `_`. '
AUTOLINK_RE = '<((?:[Ff]|[Hh][Tt])[Tt][Pp][Ss]?://[^<>]*)>'
' Match an automatic link (`<http://www.example.com>`). '
AUTOMAIL_RE = '<([^<> !]+@[^@<> ]+)>'
' Match an automatic email link (`<me@example.com>`). '
HTML_RE = '(<(\\/?[a-zA-Z][^<>@ ]*( [^<>]*)?|!--(?:(?!<!--|-->).)*--)>)'
' Match an HTML tag (`<...>`). '
ENTITY_RE = '(&(?:\\#[0-9]+|\\#x[0-9a-fA-F]+|[a-zA-Z0-9]+);)'
' Match an HTML entity (`&#38;` (decimal) or `&#x26;` (hex) or `&amp;` (named)). '
LINE_BREAK_RE = '  \\n'
' Match two spaces at end of line. '

def dequote(string: str) -> str:
    if False:
        print('Hello World!')
    'Remove quotes from around a string.'
    if string.startswith('"') and string.endswith('"') or (string.startswith("'") and string.endswith("'")):
        return string[1:-1]
    else:
        return string

class EmStrongItem(NamedTuple):
    """Emphasis/strong pattern item."""
    pattern: re.Pattern[str]
    builder: str
    tags: str

class Pattern:
    """
    Base class that inline patterns subclass.

    Inline patterns are handled by means of `Pattern` subclasses, one per regular expression.
    Each pattern object uses a single regular expression and must support the following methods:
    [`getCompiledRegExp`][markdown.inlinepatterns.Pattern.getCompiledRegExp] and
    [`handleMatch`][markdown.inlinepatterns.Pattern.handleMatch].

    All the regular expressions used by `Pattern` subclasses must capture the whole block.  For this
    reason, they all start with `^(.*)` and end with `(.*)!`.  When passing a regular expression on
    class initialization, the `^(.*)` and `(.*)!` are added automatically and the regular expression
    is pre-compiled.

    It is strongly suggested that the newer style [`markdown.inlinepatterns.InlineProcessor`][] that
    use a more efficient and flexible search approach be used instead. However, the older style
    `Pattern` remains for backward compatibility with many existing third-party extensions.

    """
    ANCESTOR_EXCLUDES: Collection[str] = tuple()
    '\n    A collection of elements which are undesirable ancestors. The processor will be skipped if it\n    would cause the content to be a descendant of one of the listed tag names.\n    '
    compiled_re: re.Pattern[str]
    md: Markdown | None

    def __init__(self, pattern: str, md: Markdown | None=None):
        if False:
            while True:
                i = 10
        '\n        Create an instant of an inline pattern.\n\n        Arguments:\n            pattern: A regular expression that matches a pattern.\n            md: An optional pointer to the instance of `markdown.Markdown` and is available as\n                `self.md` on the class instance.\n\n\n        '
        self.pattern = pattern
        self.compiled_re = re.compile('^(.*?)%s(.*)$' % pattern, re.DOTALL | re.UNICODE)
        self.md = md

    def getCompiledRegExp(self) -> re.Pattern:
        if False:
            return 10
        ' Return a compiled regular expression. '
        return self.compiled_re

    def handleMatch(self, m: re.Match[str]) -> etree.Element | str:
        if False:
            return 10
        'Return a ElementTree element from the given match.\n\n        Subclasses should override this method.\n\n        Arguments:\n            m: A match object containing a match of the pattern.\n\n        Returns: An ElementTree Element object.\n\n        '
        pass

    def type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Return class name, to define pattern type '
        return self.__class__.__name__

    def unescape(self, text: str) -> str:
        if False:
            while True:
                i = 10
        ' Return unescaped text given text with an inline placeholder. '
        try:
            stash = self.md.treeprocessors['inline'].stashed_nodes
        except KeyError:
            return text

        def get_stash(m):
            if False:
                for i in range(10):
                    print('nop')
            id = m.group(1)
            if id in stash:
                value = stash.get(id)
                if isinstance(value, str):
                    return value
                else:
                    return ''.join(value.itertext())
        return util.INLINE_PLACEHOLDER_RE.sub(get_stash, text)

class InlineProcessor(Pattern):
    """
    Base class that inline processors subclass.

    This is the newer style inline processor that uses a more
    efficient and flexible search approach.

    """

    def __init__(self, pattern: str, md: Markdown | None=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an instant of an inline processor.\n\n        Arguments:\n            pattern: A regular expression that matches a pattern.\n            md: An optional pointer to the instance of `markdown.Markdown` and is available as\n                `self.md` on the class instance.\n\n        '
        self.pattern = pattern
        self.compiled_re = re.compile(pattern, re.DOTALL | re.UNICODE)
        self.safe_mode = False
        self.md = md

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str | None, int | None, int | None]:
        if False:
            print('Hello World!')
        'Return a ElementTree element from the given match and the\n        start and end index of the matched text.\n\n        If `start` and/or `end` are returned as `None`, it will be\n        assumed that the processor did not find a valid region of text.\n\n        Subclasses should override this method.\n\n        Arguments:\n            m: A re match object containing a match of the pattern.\n            data: The buffer currently under analysis.\n\n        Returns:\n            el: The ElementTree element, text or None.\n            start: The start of the region that has been matched or None.\n            end: The end of the region that has been matched or None.\n\n        '
        pass

class SimpleTextPattern(Pattern):
    """ Return a simple text of `group(2)` of a Pattern. """

    def handleMatch(self, m: re.Match[str]) -> str:
        if False:
            print('Hello World!')
        ' Return string content of `group(2)` of a matching pattern. '
        return m.group(2)

class SimpleTextInlineProcessor(InlineProcessor):
    """ Return a simple text of `group(1)` of a Pattern. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        if False:
            for i in range(10):
                print('nop')
        ' Return string content of `group(1)` of a matching pattern. '
        return (m.group(1), m.start(0), m.end(0))

class EscapeInlineProcessor(InlineProcessor):
    """ Return an escaped character. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str | None, int, int]:
        if False:
            return 10
        "\n        If the character matched by `group(1)` of a pattern is in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS]\n        then return the integer representing the character's Unicode code point (as returned by [`ord`][]) wrapped\n        in [`util.STX`][markdown.util.STX] and [`util.ETX`][markdown.util.ETX].\n\n        If the matched character is not in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS], then return `None`.\n        "
        char = m.group(1)
        if char in self.md.ESCAPED_CHARS:
            return ('{}{}{}'.format(util.STX, ord(char), util.ETX), m.start(0), m.end(0))
        else:
            return (None, m.start(0), m.end(0))

class SimpleTagPattern(Pattern):
    """
    Return element of type `tag` with a text attribute of `group(3)`
    of a Pattern.

    """

    def __init__(self, pattern: str, tag: str):
        if False:
            i = 10
            return i + 15
        '\n        Create an instant of an simple tag pattern.\n\n        Arguments:\n            pattern: A regular expression that matches a pattern.\n            tag: Tag of element.\n\n        '
        Pattern.__init__(self, pattern)
        self.tag = tag
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        if False:
            i = 10
            return i + 15
        "\n        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(3)` of a\n        matching pattern as the Element's text.\n        "
        el = etree.Element(self.tag)
        el.text = m.group(3)
        return el

class SimpleTagInlineProcessor(InlineProcessor):
    """
    Return element of type `tag` with a text attribute of `group(2)`
    of a Pattern.

    """

    def __init__(self, pattern: str, tag: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an instant of an simple tag processor.\n\n        Arguments:\n            pattern: A regular expression that matches a pattern.\n            tag: Tag of element.\n\n        '
        InlineProcessor.__init__(self, pattern)
        self.tag = tag
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            while True:
                i = 10
        "\n        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(2)` of a\n        matching pattern as the Element's text.\n        "
        el = etree.Element(self.tag)
        el.text = m.group(2)
        return (el, m.start(0), m.end(0))

class SubstituteTagPattern(SimpleTagPattern):
    """ Return an element of type `tag` with no children. """

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        if False:
            for i in range(10):
                print('nop')
        ' Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. '
        return etree.Element(self.tag)

class SubstituteTagInlineProcessor(SimpleTagInlineProcessor):
    """ Return an element of type `tag` with no children. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            for i in range(10):
                print('nop')
        ' Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. '
        return (etree.Element(self.tag), m.start(0), m.end(0))

class BacktickInlineProcessor(InlineProcessor):
    """ Return a `<code>` element containing the escaped matching text. """

    def __init__(self, pattern: str):
        if False:
            i = 10
            return i + 15
        InlineProcessor.__init__(self, pattern)
        self.ESCAPED_BSLASH = '{}{}{}'.format(util.STX, ord('\\'), util.ETX)
        self.tag = 'code'
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str, int, int]:
        if False:
            i = 10
            return i + 15
        '\n        If the match contains `group(3)` of a pattern, then return a `code`\n        [`Element`][xml.etree.ElementTree.Element] which contains HTML escaped text (with\n        [`code_escape`][markdown.util.code_escape]) as an [`AtomicString`][markdown.util.AtomicString].\n\n        If the match does not contain `group(3)` then return the text of `group(1)` backslash escaped.\n\n        '
        if m.group(3):
            el = etree.Element(self.tag)
            el.text = util.AtomicString(util.code_escape(m.group(3).strip()))
            return (el, m.start(0), m.end(0))
        else:
            return (m.group(1).replace('\\\\', self.ESCAPED_BSLASH), m.start(0), m.end(0))

class DoubleTagPattern(SimpleTagPattern):
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return [`Element`][xml.etree.ElementTree.Element] in following format:\n        `<tag1><tag2>group(3)</tag2>group(4)</tag2>` where `group(4)` is optional.\n\n        '
        (tag1, tag2) = self.tag.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(3)
        if len(m.groups()) == 5:
            el2.tail = m.group(4)
        return el1

class DoubleTagInlineProcessor(SimpleTagInlineProcessor):
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            return 10
        '\n        Return [`Element`][xml.etree.ElementTree.Element] in following format:\n        `<tag1><tag2>group(2)</tag2>group(3)</tag2>` where `group(3)` is optional.\n\n        '
        (tag1, tag2) = self.tag.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(2)
        if len(m.groups()) == 3:
            el2.tail = m.group(3)
        return (el1, m.start(0), m.end(0))

class HtmlInlineProcessor(InlineProcessor):
    """ Store raw inline html and return a placeholder. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        if False:
            for i in range(10):
                print('nop')
        ' Store the text of `group(1)` of a pattern and return a placeholder string. '
        rawhtml = self.backslash_unescape(self.unescape(m.group(1)))
        place_holder = self.md.htmlStash.store(rawhtml)
        return (place_holder, m.start(0), m.end(0))

    def unescape(self, text: str) -> str:
        if False:
            while True:
                i = 10
        ' Return unescaped text given text with an inline placeholder. '
        try:
            stash = self.md.treeprocessors['inline'].stashed_nodes
        except KeyError:
            return text

        def get_stash(m: re.Match[str]) -> str:
            if False:
                return 10
            id = m.group(1)
            value = stash.get(id)
            if value is not None:
                try:
                    return self.md.serializer(value)
                except Exception:
                    return '\\%s' % value
        return util.INLINE_PLACEHOLDER_RE.sub(get_stash, text)

    def backslash_unescape(self, text: str) -> str:
        if False:
            print('Hello World!')
        ' Return text with backslash escapes undone (backslashes are restored). '
        try:
            RE = self.md.treeprocessors['unescape'].RE
        except KeyError:
            return text

        def _unescape(m: re.Match[str]) -> str:
            if False:
                i = 10
                return i + 15
            return chr(int(m.group(1)))
        return RE.sub(_unescape, text)

class AsteriskProcessor(InlineProcessor):
    """Emphasis processor for handling strong and em matches inside asterisks."""
    PATTERNS = [EmStrongItem(re.compile(EM_STRONG_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'), EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'), EmStrongItem(re.compile(STRONG_EM3_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'), EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'), EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')]
    ' The various strong and emphasis patterns handled by this processor. '

    def build_single(self, m: re.Match[str], tag: str, idx: int) -> etree.Element:
        if False:
            print('Hello World!')
        'Return single tag.'
        el1 = etree.Element(tag)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        return el1

    def build_double(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        if False:
            while True:
                i = 10
        'Return double tag.'
        (tag1, tag2) = tags.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el2, None, idx)
        el1.append(el2)
        if len(m.groups()) == 3:
            text = m.group(3)
            self.parse_sub_patterns(text, el1, el2, idx)
        return el1

    def build_double2(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        if False:
            print('Hello World!')
        'Return double tags (variant 2): `<strong>text <em>text</em></strong>`.'
        (tag1, tag2) = tags.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        text = m.group(3)
        el1.append(el2)
        self.parse_sub_patterns(text, el2, None, idx)
        return el1

    def parse_sub_patterns(self, data: str, parent: etree.Element, last: etree.Element | None, idx: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses sub patterns.\n\n        `data`: text to evaluate.\n\n        `parent`: Parent to attach text and sub elements to.\n\n        `last`: Last appended child to parent. Can also be None if parent has no children.\n\n        `idx`: Current pattern index that was used to evaluate the parent.\n        '
        offset = 0
        pos = 0
        length = len(data)
        while pos < length:
            if self.compiled_re.match(data, pos):
                matched = False
                for (index, item) in enumerate(self.PATTERNS):
                    if index <= idx:
                        continue
                    m = item.pattern.match(data, pos)
                    if m:
                        text = data[offset:m.start(0)]
                        if text:
                            if last is not None:
                                last.tail = text
                            else:
                                parent.text = text
                        el = self.build_element(m, item.builder, item.tags, index)
                        parent.append(el)
                        last = el
                        offset = pos = m.end(0)
                        matched = True
                if not matched:
                    pos += 1
            else:
                pos += 1
        text = data[offset:]
        if text:
            if last is not None:
                last.tail = text
            else:
                parent.text = text

    def build_element(self, m: re.Match[str], builder: str, tags: str, index: int) -> etree.Element:
        if False:
            i = 10
            return i + 15
        'Element builder.'
        if builder == 'double2':
            return self.build_double2(m, tags, index)
        elif builder == 'double':
            return self.build_double(m, tags, index)
        else:
            return self.build_single(m, tags, index)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        if False:
            for i in range(10):
                print('nop')
        'Parse patterns.'
        el = None
        start = None
        end = None
        for (index, item) in enumerate(self.PATTERNS):
            m1 = item.pattern.match(data, m.start(0))
            if m1:
                start = m1.start(0)
                end = m1.end(0)
                el = self.build_element(m1, item.builder, item.tags, index)
                break
        return (el, start, end)

class UnderscoreProcessor(AsteriskProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""
    PATTERNS = [EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'), EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'), EmStrongItem(re.compile(SMART_STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'), EmStrongItem(re.compile(SMART_STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'), EmStrongItem(re.compile(SMART_EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')]
    ' The various strong and emphasis patterns handled by this processor. '

class LinkInlineProcessor(InlineProcessor):
    """ Return a link element from the given match. """
    RE_LINK = re.compile('\\(\\s*(?:(<[^<>]*>)\\s*(?:(\'[^\']*\'|"[^"]*")\\s*)?\\))?', re.DOTALL | re.UNICODE)
    RE_TITLE_CLEAN = re.compile('\\s')

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        if False:
            while True:
                i = 10
        ' Return an `a` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. '
        (text, index, handled) = self.getText(data, m.end(0))
        if not handled:
            return (None, None, None)
        (href, title, index, handled) = self.getLink(data, index)
        if not handled:
            return (None, None, None)
        el = etree.Element('a')
        el.text = text
        el.set('href', href)
        if title is not None:
            el.set('title', title)
        return (el, m.start(0), index)

    def getLink(self, data: str, index: int) -> tuple[str, str | None, int, bool]:
        if False:
            return 10
        'Parse data between `()` of `[Text]()` allowing recursive `()`. '
        href = ''
        title: str | None = None
        handled = False
        m = self.RE_LINK.match(data, pos=index)
        if m and m.group(1):
            href = m.group(1)[1:-1].strip()
            if m.group(2):
                title = m.group(2)[1:-1]
            index = m.end(0)
            handled = True
        elif m:
            bracket_count = 1
            backtrack_count = 1
            start_index = m.end()
            index = start_index
            last_bracket = -1
            quote: str | None = None
            start_quote = -1
            exit_quote = -1
            ignore_matches = False
            alt_quote = None
            start_alt_quote = -1
            exit_alt_quote = -1
            last = ''
            for pos in range(index, len(data)):
                c = data[pos]
                if c == '(':
                    if not ignore_matches:
                        bracket_count += 1
                    elif backtrack_count > 0:
                        backtrack_count -= 1
                elif c == ')':
                    if exit_quote != -1 and quote == last or (exit_alt_quote != -1 and alt_quote == last):
                        bracket_count = 0
                    elif not ignore_matches:
                        bracket_count -= 1
                    elif backtrack_count > 0:
                        backtrack_count -= 1
                        if backtrack_count == 0:
                            last_bracket = index + 1
                elif c in ("'", '"'):
                    if not quote:
                        ignore_matches = True
                        backtrack_count = bracket_count
                        bracket_count = 1
                        start_quote = index + 1
                        quote = c
                    elif c != quote and (not alt_quote):
                        start_alt_quote = index + 1
                        alt_quote = c
                    elif c == quote:
                        exit_quote = index + 1
                    elif alt_quote and c == alt_quote:
                        exit_alt_quote = index + 1
                index += 1
                if bracket_count == 0:
                    if exit_quote >= 0 and quote == last:
                        href = data[start_index:start_quote - 1]
                        title = ''.join(data[start_quote:exit_quote - 1])
                    elif exit_alt_quote >= 0 and alt_quote == last:
                        href = data[start_index:start_alt_quote - 1]
                        title = ''.join(data[start_alt_quote:exit_alt_quote - 1])
                    else:
                        href = data[start_index:index - 1]
                    break
                if c != ' ':
                    last = c
            if bracket_count != 0 and backtrack_count == 0:
                href = data[start_index:last_bracket - 1]
                index = last_bracket
                bracket_count = 0
            handled = bracket_count == 0
        if title is not None:
            title = self.RE_TITLE_CLEAN.sub(' ', dequote(self.unescape(title.strip())))
        href = self.unescape(href).strip()
        return (href, title, index, handled)

    def getText(self, data: str, index: int) -> tuple[str, int, bool]:
        if False:
            i = 10
            return i + 15
        'Parse the content between `[]` of the start of an image or link\n        resolving nested square brackets.\n\n        '
        bracket_count = 1
        text = []
        for pos in range(index, len(data)):
            c = data[pos]
            if c == ']':
                bracket_count -= 1
            elif c == '[':
                bracket_count += 1
            index += 1
            if bracket_count == 0:
                break
            text.append(c)
        return (''.join(text), index, bracket_count == 0)

class ImageInlineProcessor(LinkInlineProcessor):
    """ Return a `img` element from the given match. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        if False:
            i = 10
            return i + 15
        ' Return an `img` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. '
        (text, index, handled) = self.getText(data, m.end(0))
        if not handled:
            return (None, None, None)
        (src, title, index, handled) = self.getLink(data, index)
        if not handled:
            return (None, None, None)
        el = etree.Element('img')
        el.set('src', src)
        if title is not None:
            el.set('title', title)
        el.set('alt', self.unescape(text))
        return (el, m.start(0), index)

class ReferenceInlineProcessor(LinkInlineProcessor):
    """ Match to a stored reference and return link element. """
    NEWLINE_CLEANUP_RE = re.compile('\\s+', re.MULTILINE)
    RE_LINK = re.compile('\\s?\\[([^\\]]*)\\]', re.DOTALL | re.UNICODE)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        if False:
            while True:
                i = 10
        '\n        Return [`Element`][xml.etree.ElementTree.Element] returned by `makeTag` method or `(None, None, None)`.\n\n        '
        (text, index, handled) = self.getText(data, m.end(0))
        if not handled:
            return (None, None, None)
        (id, end, handled) = self.evalId(data, index, text)
        if not handled:
            return (None, None, None)
        id = self.NEWLINE_CLEANUP_RE.sub(' ', id)
        if id not in self.md.references:
            return (None, m.start(0), end)
        (href, title) = self.md.references[id]
        return (self.makeTag(href, title, text), m.start(0), end)

    def evalId(self, data: str, index: int, text: str) -> tuple[str | None, int, bool]:
        if False:
            print('Hello World!')
        '\n        Evaluate the id portion of `[ref][id]`.\n\n        If `[ref][]` use `[ref]`.\n        '
        m = self.RE_LINK.match(data, pos=index)
        if not m:
            return (None, index, False)
        else:
            id = m.group(1).lower()
            end = m.end(0)
            if not id:
                id = text.lower()
        return (id, end, True)

    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        if False:
            while True:
                i = 10
        ' Return an `a` [`Element`][xml.etree.ElementTree.Element]. '
        el = etree.Element('a')
        el.set('href', href)
        if title:
            el.set('title', title)
        el.text = text
        return el

class ShortReferenceInlineProcessor(ReferenceInlineProcessor):
    """Short form of reference: `[google]`. """

    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Evaluate the id of `[ref]`.  '
        return (text.lower(), index, True)

class ImageReferenceInlineProcessor(ReferenceInlineProcessor):
    """ Match to a stored reference and return `img` element. """

    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        if False:
            for i in range(10):
                print('nop')
        ' Return an `img` [`Element`][xml.etree.ElementTree.Element]. '
        el = etree.Element('img')
        el.set('src', href)
        if title:
            el.set('title', title)
        el.set('alt', self.unescape(text))
        return el

class ShortImageReferenceInlineProcessor(ImageReferenceInlineProcessor):
    """ Short form of image reference: `![ref]`. """

    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        if False:
            while True:
                i = 10
        'Evaluate the id of `[ref]`.  '
        return (text.lower(), index, True)

class AutolinkInlineProcessor(InlineProcessor):
    """ Return a link Element given an auto-link (`<http://example/com>`). """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            print('Hello World!')
        ' Return an `a` [`Element`][xml.etree.ElementTree.Element] of `group(1)`. '
        el = etree.Element('a')
        el.set('href', self.unescape(m.group(1)))
        el.text = util.AtomicString(m.group(1))
        return (el, m.start(0), m.end(0))

class AutomailInlineProcessor(InlineProcessor):
    """
    Return a `mailto` link Element given an auto-mail link (`<foo@example.com>`).
    """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        if False:
            print('Hello World!')
        ' Return an [`Element`][xml.etree.ElementTree.Element] containing a `mailto` link  of `group(1)`. '
        el = etree.Element('a')
        email = self.unescape(m.group(1))
        if email.startswith('mailto:'):
            email = email[len('mailto:'):]

        def codepoint2name(code: int) -> str:
            if False:
                while True:
                    i = 10
            'Return entity definition by code, or the code if not defined.'
            entity = entities.codepoint2name.get(code)
            if entity:
                return '{}{};'.format(util.AMP_SUBSTITUTE, entity)
            else:
                return '%s#%d;' % (util.AMP_SUBSTITUTE, code)
        letters = [codepoint2name(ord(letter)) for letter in email]
        el.text = util.AtomicString(''.join(letters))
        mailto = 'mailto:' + email
        mailto = ''.join([util.AMP_SUBSTITUTE + '#%d;' % ord(letter) for letter in mailto])
        el.set('href', mailto)
        return (el, m.start(0), m.end(0))