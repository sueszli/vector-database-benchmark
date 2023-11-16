""" Outputs to the user.

Printing with intends or plain, mostly a compensation for the print strangeness.

We want to avoid "from __future__ import print_function" in every file out
there, which makes adding another debug print rather tedious. This should
cover all calls/uses of "print" we have to do, and the make it easy to simply
to "print for_debug" without much hassle (braces).

"""
from __future__ import print_function
import os
import struct
import sys
import textwrap
import traceback
from nuitka.utils.Utils import isWin32Windows
is_quiet = False
progress = None

def setQuiet():
    if False:
        print('Hello World!')
    global is_quiet
    is_quiet = True

def printIndented(level, *what):
    if False:
        for i in range(10):
            print('nop')
    print('    ' * level, *what)

def printSeparator(level=0):
    if False:
        i = 10
        return i + 15
    print('    ' * level, '*' * 10)

def printLine(*what, **kwargs):
    if False:
        while True:
            i = 10
    is_atty = sys.stdout.isatty()
    if progress and is_atty:
        progress.close()
    print(*what, **kwargs)

def printError(message):
    if False:
        while True:
            i = 10
    my_print(message, file=sys.stderr)

def flushStandardOutputs():
    if False:
        for i in range(10):
            print('nop')
    sys.stdout.flush()
    sys.stderr.flush()

def _getEnableStyleCode(style):
    if False:
        for i in range(10):
            print('nop')
    style = _aliasStyle(style)
    if style == 'pink':
        style = '\x1b[95m'
    elif style == 'blue':
        style = '\x1b[94m'
    elif style == 'green':
        style = '\x1b[92m'
    elif style == 'yellow':
        style = '\x1b[93m'
    elif style == 'red':
        style = '\x1b[91m'
    elif style == 'bold':
        style = '\x1b[1m'
    elif style == 'underline':
        style = '\x1b[4m'
    else:
        style = None
    return style
_enabled_ansi = False

def _enableAnsi():
    if False:
        i = 10
        return i + 15
    global _enabled_ansi
    if not _enabled_ansi:
        if os.name == 'nt':
            os.system('')
        _enabled_ansi = True

def _getDisableStyleCode():
    if False:
        i = 10
        return i + 15
    return '\x1b[0m'

def _getIoctlGWINSZ(fd):
    if False:
        return 10
    try:
        import fcntl
        import termios
        return struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
    except BaseException:
        return None

def _getTerminalSizeWin32():
    if False:
        print('Hello World!')
    try:
        from ctypes import create_string_buffer, windll
        h = windll.kernel32.GetStdHandle(-12)
        buffer = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, buffer)
        if res:
            (_, _, _, _, _, left, top, right, bottom, _, _) = struct.unpack('hhhhHhhhhhh', buffer.raw)
            columns = right - left + 1
            lines = bottom - top + 1
            return (lines, columns)
    except BaseException:
        return None

def _getTerminalSize():
    if False:
        for i in range(10):
            print('nop')
    if str is bytes:
        if isWin32Windows():
            columns = _getTerminalSizeWin32()
        else:
            columns = _getIoctlGWINSZ(0) or _getIoctlGWINSZ(1) or _getIoctlGWINSZ(2)
        if columns:
            return columns[1]
        try:
            return int(os.environ.get('COLUMNS', '1000'))
        except ValueError:
            return 1000
    else:
        try:
            return os.get_terminal_size()[0]
        except OSError:
            return 1000

def _aliasStyle(style):
    if False:
        return 10
    if style == 'test-prepare':
        return 'pink'
    if style == 'test-progress':
        return 'blue'
    if style == 'test-debug':
        return 'bold'
    if style == 'link':
        return 'blue'
    else:
        return style

def _my_print(file_output, is_atty, args, kwargs):
    if False:
        return 10
    if 'style' in kwargs:
        style = kwargs['style']
        del kwargs['style']
        if 'end' in kwargs:
            end = kwargs['end']
            del kwargs['end']
        else:
            end = '\n'
        if style is not None and is_atty:
            enable_style = _getEnableStyleCode(style)
            if enable_style is None:
                raise ValueError('%r is an invalid value for keyword argument style' % style)
            _enableAnsi()
            print(enable_style, end='', **kwargs)
        print(*args, end=end, **kwargs)
        if style is not None and is_atty:
            print(_getDisableStyleCode(), end='', **kwargs)
    else:
        print(*args, **kwargs)
    file_output.flush()

def my_print(*args, **kwargs):
    if False:
        print('Hello World!')
    'Make sure we flush after every print.\n\n    Not even the "-u" option does more than that and this is easy enough.\n\n    Use kwarg style=[option] to print in a style listed below\n    '
    file_output = kwargs.get('file', sys.stdout)
    is_atty = file_output.isatty()
    if progress and is_atty:
        with progress.withExternalWritingPause():
            _my_print(file_output, is_atty, args, kwargs)
    else:
        _my_print(file_output, is_atty, args, kwargs)

class ReportingSystemExit(SystemExit):
    """Our own system exit, after which a report should be written."""

    def __init__(self, exit_code, exit_message):
        if False:
            return 10
        SystemExit.__init__(self, exit_code)
        self.exit_message = exit_message

class OurLogger(object):

    def __init__(self, name, quiet=False, base_style=None):
        if False:
            while True:
                i = 10
        self.name = name
        self.base_style = base_style
        self.is_quiet = quiet

    def my_print(self, message, **kwargs):
        if False:
            return 10
        my_print(message, **kwargs)

    @staticmethod
    def _warnMnemonic(mnemonic, style, output_function):
        if False:
            return 10
        if mnemonic.startswith('http'):
            url = mnemonic
            extra_prefix = ''
        else:
            url = 'https://nuitka.net/info/%s.html' % mnemonic
            extra_prefix = 'Complex topic! '
        output_function('    %sMore information can be found at %s%s' % (extra_prefix, _getEnableStyleCode('link'), url), style=style)

    def warning(self, message, style='red', mnemonic=None):
        if False:
            while True:
                i = 10
        if mnemonic is not None:
            from .Options import shallDisplayWarningMnemonic
            if not shallDisplayWarningMnemonic(mnemonic):
                return
        if self.name:
            prefix = '%s:WARNING: ' % self.name
        else:
            prefix = 'WARNING: '
        style = style or self.base_style
        if sys.stderr.isatty():
            width = _getTerminalSize() or 10000
        else:
            width = 10000
        formatted_message = textwrap.fill(message, width=width, initial_indent=prefix, subsequent_indent=prefix, break_on_hyphens=False, break_long_words=False, expand_tabs=False, replace_whitespace=False)
        self.my_print(formatted_message, style=style, file=sys.stderr)
        if mnemonic is not None:
            self._warnMnemonic(mnemonic, style=style, output_function=self.warning)

    def sysexit(self, message='', style=None, mnemonic=None, exit_code=1, reporting=False):
        if False:
            i = 10
            return i + 15
        from nuitka.Progress import closeProgressBar
        closeProgressBar()
        if exit_code != 0 and style is None:
            style = 'red'
        if message:
            if exit_code != 0:
                self.my_print('FATAL: %s' % message, style=style, file=sys.stderr)
            else:
                self.my_print(message, style=style, file=sys.stderr)
        if mnemonic is not None:
            self._warnMnemonic(mnemonic, style=style, output_function=self.warning)
        if reporting:
            raise ReportingSystemExit(exit_code=exit_code, exit_message=message)
        sys.exit(exit_code)

    def sysexit_exception(self, message, exception, exit_code=1):
        if False:
            return 10
        self.my_print('FATAL: %s' % message, style='red', file=sys.stderr)
        traceback.print_exc()
        self.sysexit('FATAL:' + repr(exception), exit_code=exit_code, reporting=True)

    def isQuiet(self):
        if False:
            return 10
        return is_quiet or self.is_quiet

    def info(self, message, style=None, mnemonic=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.isQuiet():
            if self.name:
                message = '%s:INFO: %s' % (self.name, message)
            style = style or self.base_style
            self.my_print(message, style=style)
            if mnemonic is not None:
                self._warnMnemonic(mnemonic, style=style, output_function=self.info)

class FileLogger(OurLogger):

    def __init__(self, name, quiet=False, base_style=None, file_handle=None):
        if False:
            return 10
        OurLogger.__init__(self, name=name, quiet=quiet, base_style=base_style)
        self.file_handle = file_handle

    def my_print(self, message, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'file' not in kwargs:
            kwargs['file'] = self.file_handle or sys.stdout
        my_print(message, **kwargs)
        kwargs['file'].flush()

    def setFileHandle(self, file_handle):
        if False:
            return 10
        self.file_handle = file_handle

    def isFileOutput(self):
        if False:
            print('Hello World!')
        return self.file_handle is not None

    def info(self, message, style=None, mnemonic=None):
        if False:
            print('Hello World!')
        if not self.isQuiet() or self.file_handle:
            message = '%s:INFO: %s' % (self.name, message)
            style = style or self.base_style
            self.my_print(message, style=style)
            if mnemonic is not None:
                self._warnMnemonic(mnemonic, style=style, output_function=self.my_print)

    def debug(self, message, style=None):
        if False:
            for i in range(10):
                print('nop')
        if self.file_handle:
            message = '%s:DEBUG: %s' % (self.name, message)
            style = style or self.base_style
            self.my_print(message, style=style)

    def info_to_file_only(self, message, style=None):
        if False:
            print('Hello World!')
        if self.file_handle:
            self.info(message, style=style)

    def info_if_file(self, message, other_logger, style=None):
        if False:
            for i in range(10):
                print('nop')
        if self.file_handle:
            self.info(message, style=style)
        else:
            other_logger.info(message, style=style)
general = OurLogger('Nuitka')
plugins_logger = OurLogger('Nuitka-Plugins')
recursion_logger = OurLogger('Nuitka-Inclusion')
progress_logger = OurLogger('Nuitka-Progress', quiet=True)
memory_logger = OurLogger('Nuitka-Memory')
dependencies_logger = OurLogger('Nuitka-Dependencies')
optimization_logger = FileLogger('Nuitka-Optimization')
pgo_logger = FileLogger('Nuitka-PGO')
code_generation_logger = OurLogger('Nuitka-CodeGen')
inclusion_logger = FileLogger('Nuitka-Inclusion')
scons_logger = OurLogger('Nuitka-Scons')
scons_details_logger = OurLogger('Nuitka-Scons')
postprocessing_logger = OurLogger('Nuitka-Postprocessing')
options_logger = OurLogger('Nuitka-Options')
unusual_logger = OurLogger('Nuitka-Unusual')
data_composer_logger = OurLogger('Nuitka-DataComposer')
onefile_logger = OurLogger('Nuitka-Onefile')
tools_logger = OurLogger('Nuitka-Tools')
wheel_logger = OurLogger('Nuitka-Wheel', base_style='blue')
cache_logger = OurLogger('Nuitka-Cache')
reports_logger = OurLogger('Nuitka-Reports')