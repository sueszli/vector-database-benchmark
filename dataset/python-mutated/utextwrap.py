from __future__ import absolute_import
import sys
import textwrap
from unicodedata import east_asian_width as _eawidth
from bzrlib import osutils
__all__ = ['UTextWrapper', 'fill', 'wrap']

class UTextWrapper(textwrap.TextWrapper):
    """
    Extend TextWrapper for Unicode.

    This textwrapper handles east asian double width and split word
    even if !break_long_words when word contains double width
    characters.

    :param ambiguous_width: (keyword argument) width for character when
                            unicodedata.east_asian_width(c) == 'A'
                            (default: 1)

    Limitations:
    * expand_tabs doesn't fixed. It uses len() for calculating width
      of string on left of TAB.
    * Handles one codeunit as a single character having 1 or 2 width.
      This is not correct when there are surrogate pairs, combined
      characters or zero-width characters.
    * Treats all asian character are line breakable. But it is not
      true because line breaking is prohibited around some characters.
      (For example, breaking before punctation mark is prohibited.)
      See UAX # 14 "UNICODE LINE BREAKING ALGORITHM"
    """

    def __init__(self, width=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if width is None:
            width = (osutils.terminal_width() or osutils.default_terminal_width) - 1
        ambi_width = kwargs.pop('ambiguous_width', 1)
        if ambi_width == 1:
            self._east_asian_doublewidth = 'FW'
        elif ambi_width == 2:
            self._east_asian_doublewidth = 'FWA'
        else:
            raise ValueError('ambiguous_width should be 1 or 2')
        if sys.version_info < (2, 6):
            self.drop_whitespace = kwargs.pop('drop_whitespace', True)
            if not self.drop_whitespace:
                raise ValueError('TextWrapper version must drop whitespace')
        textwrap.TextWrapper.__init__(self, width, **kwargs)

    def _unicode_char_width(self, uc):
        if False:
            while True:
                i = 10
        'Return width of character `uc`.\n\n        :param:     uc      Single unicode character.\n        '
        return _eawidth(uc) in self._east_asian_doublewidth and 2 or 1

    def _width(self, s):
        if False:
            while True:
                i = 10
        'Returns width for s.\n\n        When s is unicode, take care of east asian width.\n        When s is bytes, treat all byte is single width character.\n        '
        charwidth = self._unicode_char_width
        return sum((charwidth(c) for c in s))

    def _cut(self, s, width):
        if False:
            i = 10
            return i + 15
        'Returns head and rest of s. (head+rest == s)\n\n        Head is large as long as _width(head) <= width.\n        '
        w = 0
        charwidth = self._unicode_char_width
        for (pos, c) in enumerate(s):
            w += charwidth(c)
            if w > width:
                return (s[:pos], s[pos:])
        return (s, u'')

    def _fix_sentence_endings(self, chunks):
        if False:
            for i in range(10):
                print('nop')
        '_fix_sentence_endings(chunks : [string])\n\n        Correct for sentence endings buried in \'chunks\'.  Eg. when the\n        original text contains "... foo.\nBar ...", munge_whitespace()\n        and split() will convert that to [..., "foo.", " ", "Bar", ...]\n        which has one too few spaces; this method simply changes the one\n        space to two.\n\n        Note: This function is copied from textwrap.TextWrap and modified\n        to use unicode always.\n        '
        i = 0
        L = len(chunks) - 1
        patsearch = self.sentence_end_re.search
        while i < L:
            if chunks[i + 1] == u' ' and patsearch(chunks[i]):
                chunks[i + 1] = u'  '
                i += 2
            else:
                i += 1

    def _handle_long_word(self, chunks, cur_line, cur_len, width):
        if False:
            i = 10
            return i + 15
        if width < 2:
            space_left = chunks[-1] and self._width(chunks[-1][0]) or 1
        else:
            space_left = width - cur_len
        if self.break_long_words:
            (head, rest) = self._cut(chunks[-1], space_left)
            cur_line.append(head)
            if rest:
                chunks[-1] = rest
            else:
                del chunks[-1]
        elif not cur_line:
            cur_line.append(chunks.pop())

    def _wrap_chunks(self, chunks):
        if False:
            for i in range(10):
                print('nop')
        lines = []
        if self.width <= 0:
            raise ValueError('invalid width %r (must be > 0)' % self.width)
        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            if lines:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            width = self.width - len(indent)
            if self.drop_whitespace and chunks[-1].strip() == '' and lines:
                del chunks[-1]
            while chunks:
                l = self._width(chunks[-1])
                if cur_len + l <= width:
                    cur_line.append(chunks.pop())
                    cur_len += l
                else:
                    break
            if chunks and self._width(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
            if self.drop_whitespace and cur_line and (not cur_line[-1].strip()):
                del cur_line[-1]
            if cur_line:
                lines.append(indent + u''.join(cur_line))
        return lines

    def _split(self, text):
        if False:
            i = 10
            return i + 15
        chunks = textwrap.TextWrapper._split(self, unicode(text))
        cjk_split_chunks = []
        for chunk in chunks:
            prev_pos = 0
            for (pos, char) in enumerate(chunk):
                if self._unicode_char_width(char) == 2:
                    if prev_pos < pos:
                        cjk_split_chunks.append(chunk[prev_pos:pos])
                    cjk_split_chunks.append(char)
                    prev_pos = pos + 1
            if prev_pos < len(chunk):
                cjk_split_chunks.append(chunk[prev_pos:])
        return cjk_split_chunks

    def wrap(self, text):
        if False:
            for i in range(10):
                print('nop')
        return textwrap.TextWrapper.wrap(self, unicode(text))

def wrap(text, width=None, **kwargs):
    if False:
        while True:
            i = 10
    "Wrap a single paragraph of text, returning a list of wrapped lines.\n\n    Reformat the single paragraph in 'text' so it fits in lines of no\n    more than 'width' columns, and return a list of wrapped lines.  By\n    default, tabs in 'text' are expanded with string.expandtabs(), and\n    all other whitespace characters (including newline) are converted to\n    space.  See TextWrapper class for available keyword args to customize\n    wrapping behaviour.\n    "
    return UTextWrapper(width=width, **kwargs).wrap(text)

def fill(text, width=None, **kwargs):
    if False:
        while True:
            i = 10
    "Fill a single paragraph of text, returning a new string.\n\n    Reformat the single paragraph in 'text' to fit in lines of no more\n    than 'width' columns, and return a new string containing the entire\n    wrapped paragraph.  As with wrap(), tabs are expanded and other\n    whitespace characters converted to space.  See TextWrapper class for\n    available keyword args to customize wrapping behaviour.\n    "
    return UTextWrapper(width=width, **kwargs).fill(text)