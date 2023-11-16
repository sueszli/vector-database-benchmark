import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style
from .winterm import WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
winterm = None
if windll is not None:
    winterm = WinTerm()

def is_stream_closed(stream):
    if False:
        for i in range(10):
            print('nop')
    return not hasattr(stream, 'closed') or stream.closed

def is_a_tty(stream):
    if False:
        while True:
            i = 10
    return hasattr(stream, 'isatty') and stream.isatty()

class StreamWrapper(object):
    """
    Wraps a stream (such as stdout), acting as a transparent proxy for all
    attribute access apart from method 'write()', which is delegated to our
    Converter instance.
    """

    def __init__(self, wrapped, converter):
        if False:
            for i in range(10):
                print('nop')
        self.__wrapped = wrapped
        self.__convertor = converter

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self.__wrapped, name)

    def write(self, text):
        if False:
            while True:
                i = 10
        self.__convertor.write(text)

class AnsiToWin32(object):
    """
    Implements a 'write()' method which, on Windows, will strip ANSI character
    sequences from the text, and if outputting to a tty, will convert them into
    win32 function calls.
    """
    ANSI_CSI_RE = re.compile('\x01?\x1b\\[((?:\\d|;)*)([a-zA-Z])\x02?')
    ANSI_OSC_RE = re.compile('\x01?\x1b\\]([^\x07]*)(\x07)\x02?')

    def __init__(self, wrapped, convert=None, strip=None, autoreset=False):
        if False:
            while True:
                i = 10
        self.wrapped = wrapped
        self.autoreset = autoreset
        self.stream = StreamWrapper(wrapped, self)
        on_windows = os.name == 'nt'
        conversion_supported = on_windows and winapi_test()
        if strip is None:
            strip = conversion_supported or (not is_stream_closed(wrapped) and (not is_a_tty(wrapped)))
        self.strip = strip
        if convert is None:
            convert = conversion_supported and (not is_stream_closed(wrapped)) and is_a_tty(wrapped)
        self.convert = convert
        self.win32_calls = self.get_win32_calls()
        self.on_stderr = self.wrapped is sys.stderr

    def should_wrap(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        True if this class is actually needed. If false, then the output\n        stream will not be affected, nor will win32 calls be issued, so\n        wrapping stdout is not actually required. This will generally be\n        False on non-Windows platforms, unless optional functionality like\n        autoreset has been requested using kwargs to init()\n        '
        return self.convert or self.strip or self.autoreset

    def get_win32_calls(self):
        if False:
            print('Hello World!')
        if self.convert and winterm:
            return {AnsiStyle.RESET_ALL: (winterm.reset_all,), AnsiStyle.BRIGHT: (winterm.style, WinStyle.BRIGHT), AnsiStyle.DIM: (winterm.style, WinStyle.NORMAL), AnsiStyle.NORMAL: (winterm.style, WinStyle.NORMAL), AnsiFore.BLACK: (winterm.fore, WinColor.BLACK), AnsiFore.RED: (winterm.fore, WinColor.RED), AnsiFore.GREEN: (winterm.fore, WinColor.GREEN), AnsiFore.YELLOW: (winterm.fore, WinColor.YELLOW), AnsiFore.BLUE: (winterm.fore, WinColor.BLUE), AnsiFore.MAGENTA: (winterm.fore, WinColor.MAGENTA), AnsiFore.CYAN: (winterm.fore, WinColor.CYAN), AnsiFore.WHITE: (winterm.fore, WinColor.GREY), AnsiFore.RESET: (winterm.fore,), AnsiFore.LIGHTBLACK_EX: (winterm.fore, WinColor.BLACK, True), AnsiFore.LIGHTRED_EX: (winterm.fore, WinColor.RED, True), AnsiFore.LIGHTGREEN_EX: (winterm.fore, WinColor.GREEN, True), AnsiFore.LIGHTYELLOW_EX: (winterm.fore, WinColor.YELLOW, True), AnsiFore.LIGHTBLUE_EX: (winterm.fore, WinColor.BLUE, True), AnsiFore.LIGHTMAGENTA_EX: (winterm.fore, WinColor.MAGENTA, True), AnsiFore.LIGHTCYAN_EX: (winterm.fore, WinColor.CYAN, True), AnsiFore.LIGHTWHITE_EX: (winterm.fore, WinColor.GREY, True), AnsiBack.BLACK: (winterm.back, WinColor.BLACK), AnsiBack.RED: (winterm.back, WinColor.RED), AnsiBack.GREEN: (winterm.back, WinColor.GREEN), AnsiBack.YELLOW: (winterm.back, WinColor.YELLOW), AnsiBack.BLUE: (winterm.back, WinColor.BLUE), AnsiBack.MAGENTA: (winterm.back, WinColor.MAGENTA), AnsiBack.CYAN: (winterm.back, WinColor.CYAN), AnsiBack.WHITE: (winterm.back, WinColor.GREY), AnsiBack.RESET: (winterm.back,), AnsiBack.LIGHTBLACK_EX: (winterm.back, WinColor.BLACK, True), AnsiBack.LIGHTRED_EX: (winterm.back, WinColor.RED, True), AnsiBack.LIGHTGREEN_EX: (winterm.back, WinColor.GREEN, True), AnsiBack.LIGHTYELLOW_EX: (winterm.back, WinColor.YELLOW, True), AnsiBack.LIGHTBLUE_EX: (winterm.back, WinColor.BLUE, True), AnsiBack.LIGHTMAGENTA_EX: (winterm.back, WinColor.MAGENTA, True), AnsiBack.LIGHTCYAN_EX: (winterm.back, WinColor.CYAN, True), AnsiBack.LIGHTWHITE_EX: (winterm.back, WinColor.GREY, True)}
        return dict()

    def write(self, text):
        if False:
            return 10
        if self.strip or self.convert:
            self.write_and_convert(text)
        else:
            self.wrapped.write(text)
            self.wrapped.flush()
        if self.autoreset:
            self.reset_all()

    def reset_all(self):
        if False:
            for i in range(10):
                print('nop')
        if self.convert:
            self.call_win32('m', (0,))
        elif not self.strip and (not is_stream_closed(self.wrapped)):
            self.wrapped.write(Style.RESET_ALL)

    def write_and_convert(self, text):
        if False:
            print('Hello World!')
        '\n        Write the given text to our wrapped stream, stripping any ANSI\n        sequences from the text, and optionally converting them into win32\n        calls.\n        '
        cursor = 0
        text = self.convert_osc(text)
        for match in self.ANSI_CSI_RE.finditer(text):
            (start, end) = match.span()
            self.write_plain_text(text, cursor, start)
            self.convert_ansi(*match.groups())
            cursor = end
        self.write_plain_text(text, cursor, len(text))

    def write_plain_text(self, text, start, end):
        if False:
            for i in range(10):
                print('nop')
        if start < end:
            self._write(text[start:end])
            self.wrapped.flush()

    def _write(self, text, retry=5):
        if False:
            return 10
        try:
            self.wrapped.write(text)
        except IOError as err:
            if not (err.errno == 0 and retry > 0):
                raise
            self._write(text, retry - 1)
        except UnicodeError:
            self.wrapped.write('?')

    def convert_ansi(self, paramstring, command):
        if False:
            print('Hello World!')
        if self.convert:
            params = self.extract_params(command, paramstring)
            self.call_win32(command, params)

    def extract_params(self, command, paramstring):
        if False:
            while True:
                i = 10
        if command in 'Hf':
            params = tuple((int(p) if len(p) != 0 else 1 for p in paramstring.split(';')))
            while len(params) < 2:
                params = params + (1,)
        else:
            params = tuple((int(p) for p in paramstring.split(';') if len(p) != 0))
            if len(params) == 0:
                if command in 'JKm':
                    params = (0,)
                elif command in 'ABCD':
                    params = (1,)
        return params

    def call_win32(self, command, params):
        if False:
            print('Hello World!')
        if command == 'm':
            for param in params:
                if param in self.win32_calls:
                    func_args = self.win32_calls[param]
                    func = func_args[0]
                    args = func_args[1:]
                    kwargs = dict(on_stderr=self.on_stderr)
                    func(*args, **kwargs)
        elif command in 'J':
            winterm.erase_screen(params[0], on_stderr=self.on_stderr)
        elif command in 'K':
            winterm.erase_line(params[0], on_stderr=self.on_stderr)
        elif command in 'Hf':
            winterm.set_cursor_position(params, on_stderr=self.on_stderr)
        elif command in 'ABCD':
            n = params[0]
            (x, y) = {'A': (0, -n), 'B': (0, n), 'C': (n, 0), 'D': (-n, 0)}[command]
            winterm.cursor_adjust(x, y, on_stderr=self.on_stderr)

    def convert_osc(self, text):
        if False:
            for i in range(10):
                print('nop')
        for match in self.ANSI_OSC_RE.finditer(text):
            (start, end) = match.span()
            text = text[:start] + text[end:]
            (paramstring, command) = match.groups()
            if command in '\x07':
                params = paramstring.split(';')
                if params[0] in '02':
                    winterm.set_title(params[1])
        return text