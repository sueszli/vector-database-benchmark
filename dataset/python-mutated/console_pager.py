"""Simple console pager."""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from fire.console import console_attr

class Pager(object):
    """A simple console text pager.

  This pager requires the entire contents to be available. The contents are
  written one page of lines at a time. The prompt is written after each page of
  lines. A one character response is expected. See HELP_TEXT below for more
  info.

  The contents are written as is. For example, ANSI control codes will be in
  effect. This is different from pagers like more(1) which is ANSI control code
  agnostic and miscalculates line lengths, and less(1) which displays control
  character names by default.

  Attributes:
    _attr: The current ConsoleAttr handle.
    _clear: A string that clears the prompt when written to _out.
    _contents: The entire contents of the text lines to page.
    _height: The terminal height in characters.
    _out: The output stream, log.out (effectively) if None.
    _prompt: The page break prompt.
    _search_direction: The search direction command, n:forward, N:reverse.
    _search_pattern: The current forward/reverse search compiled RE.
    _width: The termonal width in characters.
  """
    HELP_TEXT = '\n  Simple pager commands:\n\n    b, ^B, <PAGE-UP>, <LEFT-ARROW>\n      Back one page.\n    f, ^F, <SPACE>, <PAGE-DOWN>, <RIGHT-ARROW>\n      Forward one page. Does not quit if there are no more lines.\n    g, <HOME>\n      Back to the first page.\n    <number>g\n      Go to <number> lines from the top.\n    G, <END>\n      Forward to the last page.\n    <number>G\n      Go to <number> lines from the bottom.\n    h\n      Print pager command help.\n    j, +, <DOWN-ARROW>\n      Forward one line.\n    k, -, <UP-ARROW>\n      Back one line.\n    /pattern\n      Forward search for pattern.\n    ?pattern\n      Backward search for pattern.\n    n\n      Repeat current search.\n    N\n      Repeat current search in the opposite direction.\n    q, Q, ^C, ^D, ^Z\n      Quit return to the caller.\n    any other character\n      Prompt again.\n\n  Hit any key to continue:'
    PREV_POS_NXT_REPRINT = (-1, -1)

    def __init__(self, contents, out=None, prompt=None):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      contents: The entire contents of the text lines to page.\n      out: The output stream, log.out (effectively) if None.\n      prompt: The page break prompt, a default prompt is used if None..\n    '
        self._contents = contents
        self._out = out or sys.stdout
        self._search_pattern = None
        self._search_direction = None
        (self.prev_pos, self.prev_nxt) = self.PREV_POS_NXT_REPRINT
        self._attr = console_attr.GetConsoleAttr()
        (self._width, self._height) = self._attr.GetTermSize()
        if not prompt:
            prompt = '{bold}--({{percent}}%)--{normal}'.format(bold=self._attr.GetFontCode(bold=True), normal=self._attr.GetFontCode())
        self._clear = '\r{0}\r'.format(' ' * (self._attr.DisplayWidth(prompt) - 6))
        self._prompt = prompt
        self._lines = []
        for line in contents.splitlines():
            self._lines += self._attr.SplitLine(line, self._width)

    def _Write(self, s):
        if False:
            print('Hello World!')
        'Mockable helper that writes s to self._out.'
        self._out.write(s)

    def _GetSearchCommand(self, c):
        if False:
            print('Hello World!')
        'Consumes a search command and returns the equivalent pager command.\n\n    The search pattern is an RE that is pre-compiled and cached for subsequent\n    /<newline>, ?<newline>, n, or N commands.\n\n    Args:\n      c: The search command char.\n\n    Returns:\n      The pager command char.\n    '
        self._Write(c)
        buf = ''
        while True:
            p = self._attr.GetRawKey()
            if p in (None, '\n', '\r') or len(p) != 1:
                break
            self._Write(p)
            buf += p
        self._Write('\r' + ' ' * len(buf) + '\r')
        if buf:
            try:
                self._search_pattern = re.compile(buf)
            except re.error:
                self._search_pattern = None
                return ''
        self._search_direction = 'n' if c == '/' else 'N'
        return 'n'

    def _Help(self):
        if False:
            print('Hello World!')
        'Print command help and wait for any character to continue.'
        clear = self._height - (len(self.HELP_TEXT) - len(self.HELP_TEXT.replace('\n', '')))
        if clear > 0:
            self._Write('\n' * clear)
        self._Write(self.HELP_TEXT)
        self._attr.GetRawKey()
        self._Write('\n')

    def Run(self):
        if False:
            return 10
        'Run the pager.'
        if len(self._lines) <= self._height:
            self._Write(self._contents)
            return
        reset_prev_values = True
        self._height -= 1
        pos = 0
        while pos < len(self._lines):
            nxt = pos + self._height
            if nxt > len(self._lines):
                nxt = len(self._lines)
                pos = nxt - self._height
            if self.prev_pos < pos < self.prev_nxt:
                self._Write('\n'.join(self._lines[self.prev_nxt:nxt]) + '\n')
            elif pos != self.prev_pos and nxt != self.prev_nxt:
                self._Write('\n'.join(self._lines[pos:nxt]) + '\n')
            percent = self._prompt.format(percent=100 * nxt // len(self._lines))
            digits = ''
            while True:
                if reset_prev_values:
                    (self.prev_pos, self.prev_nxt) = (pos, nxt)
                    reset_prev_values = False
                self._Write(percent)
                c = self._attr.GetRawKey()
                self._Write(self._clear)
                if c in (None, 'q', 'Q', '\x03', '\x1b'):
                    return
                elif c in ('/', '?'):
                    c = self._GetSearchCommand(c)
                elif c.isdigit():
                    digits += c
                    continue
                if digits:
                    count = int(digits)
                    digits = ''
                else:
                    count = 0
                if c in ('<PAGE-UP>', '<LEFT-ARROW>', 'b', '\x02'):
                    nxt = pos - self._height
                    if nxt < 0:
                        nxt = 0
                elif c in ('<PAGE-DOWN>', '<RIGHT-ARROW>', 'f', '\x06', ' '):
                    if nxt >= len(self._lines):
                        continue
                    nxt = pos + self._height
                    if nxt >= len(self._lines):
                        nxt = pos
                elif c in ('<HOME>', 'g'):
                    nxt = count - 1
                    if nxt > len(self._lines) - self._height:
                        nxt = len(self._lines) - self._height
                    if nxt < 0:
                        nxt = 0
                elif c in ('<END>', 'G'):
                    nxt = len(self._lines) - count
                    if nxt > len(self._lines) - self._height:
                        nxt = len(self._lines) - self._height
                    if nxt < 0:
                        nxt = 0
                elif c == 'h':
                    self._Help()
                    (self.prev_pos, self.prev_nxt) = self.PREV_POS_NXT_REPRINT
                    nxt = pos
                    break
                elif c in ('<DOWN-ARROW>', 'j', '+', '\n', '\r'):
                    if nxt >= len(self._lines):
                        continue
                    nxt = pos + 1
                    if nxt >= len(self._lines):
                        nxt = pos
                elif c in ('<UP-ARROW>', 'k', '-'):
                    nxt = pos - 1
                    if nxt < 0:
                        nxt = 0
                elif c in ('n', 'N'):
                    if not self._search_pattern:
                        continue
                    nxt = pos
                    i = pos
                    direction = 1 if c == self._search_direction else -1
                    while True:
                        i += direction
                        if i < 0 or i >= len(self._lines):
                            break
                        if self._search_pattern.search(self._lines[i]):
                            nxt = i
                            break
                else:
                    continue
                if nxt != pos:
                    reset_prev_values = True
                    break
            pos = nxt