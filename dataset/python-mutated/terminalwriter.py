"""Helper functions for writing to terminals and files."""
import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth

def get_terminal_width() -> int:
    if False:
        print('Hello World!')
    (width, _) = shutil.get_terminal_size(fallback=(80, 24))
    if width < 40:
        width = 80
    return width

def should_do_markup(file: TextIO) -> bool:
    if False:
        while True:
            i = 10
    if os.environ.get('PY_COLORS') == '1':
        return True
    if os.environ.get('PY_COLORS') == '0':
        return False
    if 'NO_COLOR' in os.environ:
        return False
    if 'FORCE_COLOR' in os.environ:
        return True
    return hasattr(file, 'isatty') and file.isatty() and (os.environ.get('TERM') != 'dumb')

@final
class TerminalWriter:
    _esctable = dict(black=30, red=31, green=32, yellow=33, blue=34, purple=35, cyan=36, white=37, Black=40, Red=41, Green=42, Yellow=43, Blue=44, Purple=45, Cyan=46, White=47, bold=1, light=2, blink=5, invert=7)

    def __init__(self, file: Optional[TextIO]=None) -> None:
        if False:
            return 10
        if file is None:
            file = sys.stdout
        if hasattr(file, 'isatty') and file.isatty() and (sys.platform == 'win32'):
            try:
                import colorama
            except ImportError:
                pass
            else:
                file = colorama.AnsiToWin32(file).stream
                assert file is not None
        self._file = file
        self.hasmarkup = should_do_markup(file)
        self._current_line = ''
        self._terminal_width: Optional[int] = None
        self.code_highlight = True

    @property
    def fullwidth(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        if self._terminal_width is not None:
            return self._terminal_width
        return get_terminal_width()

    @fullwidth.setter
    def fullwidth(self, value: int) -> None:
        if False:
            while True:
                i = 10
        self._terminal_width = value

    @property
    def width_of_current_line(self) -> int:
        if False:
            while True:
                i = 10
        'Return an estimate of the width so far in the current line.'
        return wcswidth(self._current_line)

    def markup(self, text: str, **markup: bool) -> str:
        if False:
            i = 10
            return i + 15
        for name in markup:
            if name not in self._esctable:
                raise ValueError(f'unknown markup: {name!r}')
        if self.hasmarkup:
            esc = [self._esctable[name] for (name, on) in markup.items() if on]
            if esc:
                text = ''.join(('\x1b[%sm' % cod for cod in esc)) + text + '\x1b[0m'
        return text

    def sep(self, sepchar: str, title: Optional[str]=None, fullwidth: Optional[int]=None, **markup: bool) -> None:
        if False:
            return 10
        if fullwidth is None:
            fullwidth = self.fullwidth
        if sys.platform == 'win32':
            fullwidth -= 1
        if title is not None:
            N = max((fullwidth - len(title) - 2) // (2 * len(sepchar)), 1)
            fill = sepchar * N
            line = f'{fill} {title} {fill}'
        else:
            line = sepchar * (fullwidth // len(sepchar))
        if len(line) + len(sepchar.rstrip()) <= fullwidth:
            line += sepchar.rstrip()
        self.line(line, **markup)

    def write(self, msg: str, *, flush: bool=False, **markup: bool) -> None:
        if False:
            print('Hello World!')
        if msg:
            current_line = msg.rsplit('\n', 1)[-1]
            if '\n' in msg:
                self._current_line = current_line
            else:
                self._current_line += current_line
            msg = self.markup(msg, **markup)
            try:
                self._file.write(msg)
            except UnicodeEncodeError:
                msg = msg.encode('unicode-escape').decode('ascii')
                self._file.write(msg)
            if flush:
                self.flush()

    def line(self, s: str='', **markup: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.write(s, **markup)
        self.write('\n')

    def flush(self) -> None:
        if False:
            return 10
        self._file.flush()

    def _write_source(self, lines: Sequence[str], indents: Sequence[str]=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write lines of source code possibly highlighted.\n\n        Keeping this private for now because the API is clunky. We should discuss how\n        to evolve the terminal writer so we can have more precise color support, for example\n        being able to write part of a line in one color and the rest in another, and so on.\n        '
        if indents and len(indents) != len(lines):
            raise ValueError('indents size ({}) should have same size as lines ({})'.format(len(indents), len(lines)))
        if not indents:
            indents = [''] * len(lines)
        source = '\n'.join(lines)
        new_lines = self._highlight(source).splitlines()
        for (indent, new_line) in zip(indents, new_lines):
            self.line(indent + new_line)

    def _highlight(self, source: str, lexer: Literal['diff', 'python']='python') -> str:
        if False:
            print('Hello World!')
        'Highlight the given source if we have markup support.'
        from _pytest.config.exceptions import UsageError
        if not self.hasmarkup or not self.code_highlight:
            return source
        try:
            from pygments.formatters.terminal import TerminalFormatter
            if lexer == 'python':
                from pygments.lexers.python import PythonLexer as Lexer
            elif lexer == 'diff':
                from pygments.lexers.diff import DiffLexer as Lexer
            from pygments import highlight
            import pygments.util
        except ImportError:
            return source
        else:
            try:
                highlighted: str = highlight(source, Lexer(), TerminalFormatter(bg=os.getenv('PYTEST_THEME_MODE', 'dark'), style=os.getenv('PYTEST_THEME')))
                return highlighted
            except pygments.util.ClassNotFound:
                raise UsageError("PYTEST_THEME environment variable had an invalid value: '{}'. Only valid pygment styles are allowed.".format(os.getenv('PYTEST_THEME')))
            except pygments.util.OptionError:
                raise UsageError("PYTEST_THEME_MODE environment variable had an invalid value: '{}'. The only allowed values are 'dark' and 'light'.".format(os.getenv('PYTEST_THEME_MODE')))