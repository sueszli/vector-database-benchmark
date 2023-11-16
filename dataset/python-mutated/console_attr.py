"""A module for console attributes, special characters and functions.

The target architectures {linux, macos, windows} support inline encoding for
all attributes except color. Windows requires win32 calls to manipulate the
console color state.

Usage:

  # Get the console attribute state.
  out = log.out
  con = console_attr.GetConsoleAttr(out=out)

  # Get the ISO 8879:1986//ENTITIES Box and Line Drawing characters.
  box = con.GetBoxLineCharacters()
  # Print an X inside a box.
  out.write(box.dr)
  out.write(box.h)
  out.write(box.dl)
  out.write('\\n')
  out.write(box.v)
  out.write('X')
  out.write(box.v)
  out.write('\\n')
  out.write(box.ur)
  out.write(box.h)
  out.write(box.ul)
  out.write('\\n')

  # Print the bullet characters.
  for c in con.GetBullets():
    out.write(c)
  out.write('\\n')

  # Print FAIL in red.
  out.write('Epic ')
  con.Colorize('FAIL', 'red')
  out.write(', my first.')

  # Print italic and bold text.
  bold = con.GetFontCode(bold=True)
  italic = con.GetFontCode(italic=True)
  normal = con.GetFontCode()
  out.write('This is {bold}bold{normal}, this is {italic}italic{normal},'
            ' and this is normal.\\n'.format(bold=bold, italic=italic,
                                            normal=normal))

  # Read one character from stdin with echo disabled.
  c = con.GetRawKey()
  if c is None:
    print 'EOF\\n'

  # Return the display width of a string that may contain FontCode() chars.
  display_width = con.DisplayWidth(string)

  # Reset the memoized state.
  con = console_attr.ResetConsoleAttr()

  # Print the console width and height in characters.
  width, height = con.GetTermSize()
  print 'width={width}, height={height}'.format(width=width, height=height)

  # Colorize table data cells.
  fail = console_attr.Colorizer('FAIL', 'red')
  pass = console_attr.Colorizer('PASS', 'green')
  cells = ['label', fail, 'more text', pass, 'end']
  for cell in cells;
    if isinstance(cell, console_attr.Colorizer):
      cell.Render()
    else:
      out.write(cell)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import unicodedata
from fire.console import console_attr_os
from fire.console import encoding as encoding_util
from fire.console import text
import six

class BoxLineCharacters(object):
    """Box/line drawing characters.

  The element names are from ISO 8879:1986//ENTITIES Box and Line Drawing//EN:
    http://www.w3.org/2003/entities/iso8879doc/isobox.html
  """

class BoxLineCharactersUnicode(BoxLineCharacters):
    """unicode Box/line drawing characters (cp437 compatible unicode)."""
    dl = '┐'
    dr = '┌'
    h = '─'
    hd = '┬'
    hu = '┴'
    ul = '┘'
    ur = '└'
    v = '│'
    vh = '┼'
    vl = '┤'
    vr = '├'
    d_dl = '╗'
    d_dr = '╔'
    d_h = '═'
    d_hd = '╦'
    d_hu = '╩'
    d_ul = '╝'
    d_ur = '╚'
    d_v = '║'
    d_vh = '╬'
    d_vl = '╣'
    d_vr = '╠'

class BoxLineCharactersAscii(BoxLineCharacters):
    """ASCII Box/line drawing characters."""
    dl = '+'
    dr = '+'
    h = '-'
    hd = '+'
    hu = '+'
    ul = '+'
    ur = '+'
    v = '|'
    vh = '+'
    vl = '+'
    vr = '+'
    d_dl = '#'
    d_dr = '#'
    d_h = '='
    d_hd = '#'
    d_hu = '#'
    d_ul = '#'
    d_ur = '#'
    d_v = '#'
    d_vh = '#'
    d_vl = '#'
    d_vr = '#'

class BoxLineCharactersScreenReader(BoxLineCharactersAscii):
    dl = ' '
    dr = ' '
    hd = ' '
    hu = ' '
    ul = ' '
    ur = ' '
    vh = ' '
    vl = ' '
    vr = ' '

class ProgressTrackerSymbols(object):
    """Characters used by progress trackers."""

class ProgressTrackerSymbolsUnicode(ProgressTrackerSymbols):
    """Characters used by progress trackers."""

    @property
    def spin_marks(self):
        if False:
            return 10
        return ['⠏', '⠛', '⠹', '⠼', '⠶', '⠧']
    success = text.TypedText(['✓'], text_type=text.TextTypes.PT_SUCCESS)
    failed = text.TypedText(['X'], text_type=text.TextTypes.PT_FAILURE)
    interrupted = '-'
    not_started = '.'
    prefix_length = 2

class ProgressTrackerSymbolsAscii(ProgressTrackerSymbols):
    """Characters used by progress trackers."""

    @property
    def spin_marks(self):
        if False:
            for i in range(10):
                print('nop')
        return ['|', '/', '-', '\\']
    success = 'OK'
    failed = 'X'
    interrupted = '-'
    not_started = '.'
    prefix_length = 3

class ConsoleAttr(object):
    """Console attribute and special drawing characters and functions accessor.

  Use GetConsoleAttr() to get a global ConsoleAttr object shared by all callers.
  Use ConsoleAttr() for abstracting multiple consoles.

  If _out is not associated with a console, or if the console properties cannot
  be determined, the default behavior is ASCII art with no attributes.

  Attributes:
    _ANSI_COLOR: The ANSI color control sequence dict.
    _ANSI_COLOR_RESET: The ANSI color reset control sequence string.
    _csi: The ANSI Control Sequence indicator string, '' if not supported.
    _encoding: The character encoding.
        ascii: ASCII art. This is the default.
        utf8: UTF-8 unicode.
        win: Windows code page 437.
    _font_bold: The ANSI bold font embellishment code string.
    _font_italic: The ANSI italic font embellishment code string.
    _get_raw_key: A function that reads one keypress from stdin with no echo.
    _out: The console output file stream.
    _term: TERM environment variable value.
    _term_size: The terminal (x, y) dimensions in characters.
  """
    _CONSOLE_ATTR_STATE = None
    _ANSI_COLOR = {'red': '31;1m', 'yellow': '33;1m', 'green': '32m', 'blue': '34;1m'}
    _ANSI_COLOR_RESET = '39;0m'
    _BULLETS_UNICODE = ('▪', '◆', '▸', '▫', '◇', '▹')
    _BULLETS_WINDOWS = ('■', '≡', '∞', 'Φ', '·')
    _BULLETS_ASCII = ('o', '*', '+', '-')

    def __init__(self, encoding=None, suppress_output=False):
        if False:
            print('Hello World!')
        "Constructor.\n\n    Args:\n      encoding: Encoding override.\n        ascii -- ASCII art. This is the default.\n        utf8 -- UTF-8 unicode.\n        win -- Windows code page 437.\n      suppress_output: True to create a ConsoleAttr that doesn't want to output\n        anything.\n    "
        if not encoding:
            encoding = self._GetConsoleEncoding()
        elif encoding == 'win':
            encoding = 'cp437'
        self._encoding = encoding or 'ascii'
        self._term = '' if suppress_output else os.getenv('TERM', '').lower()
        if self.SupportsAnsi():
            self._csi = '\x1b['
            self._font_bold = '1'
            self._font_italic = '4'
        else:
            self._csi = None
            self._font_bold = ''
            self._font_italic = ''
        is_screen_reader = False
        if self._encoding == 'utf8' and (not is_screen_reader):
            self._box_line_characters = BoxLineCharactersUnicode()
            self._bullets = self._BULLETS_UNICODE
            self._progress_tracker_symbols = ProgressTrackerSymbolsUnicode()
        elif self._encoding == 'cp437' and (not is_screen_reader):
            self._box_line_characters = BoxLineCharactersUnicode()
            self._bullets = self._BULLETS_WINDOWS
            self._progress_tracker_symbols = ProgressTrackerSymbolsAscii()
        else:
            self._box_line_characters = BoxLineCharactersAscii()
            if is_screen_reader:
                self._box_line_characters = BoxLineCharactersScreenReader()
            self._bullets = self._BULLETS_ASCII
            self._progress_tracker_symbols = ProgressTrackerSymbolsAscii()
        self._get_raw_key = [console_attr_os.GetRawKeyFunction()]
        self._term_size = (0, 0) if suppress_output else console_attr_os.GetTermSize()
        self._display_width_cache = {}

    def _GetConsoleEncoding(self):
        if False:
            i = 10
            return i + 15
        'Gets the encoding as declared by the stdout stream.\n\n    Returns:\n      str, The encoding name or None if it could not be determined.\n    '
        console_encoding = getattr(sys.stdout, 'encoding', None)
        if not console_encoding:
            return None
        console_encoding = console_encoding.lower()
        if 'utf-8' in console_encoding:
            return 'utf8'
        elif 'cp437' in console_encoding:
            return 'cp437'
        return None

    def Colorize(self, string, color, justify=None):
        if False:
            return 10
        'Generates a colorized string, optionally justified.\n\n    Args:\n      string: The string to write.\n      color: The color name -- must be in _ANSI_COLOR.\n      justify: The justification function, no justification if None. For\n        example, justify=lambda s: s.center(10)\n\n    Returns:\n      str, The colorized string that can be printed to the console.\n    '
        if justify:
            string = justify(string)
        if self._csi and color in self._ANSI_COLOR:
            return '{csi}{color_code}{string}{csi}{reset_code}'.format(csi=self._csi, color_code=self._ANSI_COLOR[color], reset_code=self._ANSI_COLOR_RESET, string=string)
        return string

    def ConvertOutputToUnicode(self, buf):
        if False:
            while True:
                i = 10
        'Converts a console output string buf to unicode.\n\n    Mainly used for testing. Allows test comparisons in unicode while ensuring\n    that unicode => encoding => unicode works.\n\n    Args:\n      buf: The console output string to convert.\n\n    Returns:\n      The console output string buf converted to unicode.\n    '
        if isinstance(buf, six.text_type):
            buf = buf.encode(self._encoding)
        return six.text_type(buf, self._encoding, 'replace')

    def GetBoxLineCharacters(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the box/line drawing characters object.\n\n    The element names are from ISO 8879:1986//ENTITIES Box and Line Drawing//EN:\n      http://www.w3.org/2003/entities/iso8879doc/isobox.html\n\n    Returns:\n      A BoxLineCharacters object for the console output device.\n    '
        return self._box_line_characters

    def GetBullets(self):
        if False:
            return 10
        'Returns the bullet characters list.\n\n    Use the list elements in order for best appearance in nested bullet lists,\n    wrapping back to the first element for deep nesting. The list size depends\n    on the console implementation.\n\n    Returns:\n      A tuple of bullet characters.\n    '
        return self._bullets

    def GetProgressTrackerSymbols(self):
        if False:
            while True:
                i = 10
        'Returns the progress tracker characters object.\n\n    Returns:\n      A ProgressTrackerSymbols object for the console output device.\n    '
        return self._progress_tracker_symbols

    def GetControlSequenceIndicator(self):
        if False:
            print('Hello World!')
        'Returns the control sequence indicator string.\n\n    Returns:\n      The control sequence indicator string or None if control sequences are not\n      supported.\n    '
        return self._csi

    def GetControlSequenceLen(self, buf):
        if False:
            print('Hello World!')
        'Returns the control sequence length at the beginning of buf.\n\n    Used in display width computations. Control sequences have display width 0.\n\n    Args:\n      buf: The string to check for a control sequence.\n\n    Returns:\n      The control sequence length at the beginning of buf or 0 if buf does not\n      start with a control sequence.\n    '
        if not self._csi or not buf.startswith(self._csi):
            return 0
        n = 0
        for c in buf:
            n += 1
            if c.isalpha():
                break
        return n

    def GetEncoding(self):
        if False:
            i = 10
            return i + 15
        'Returns the current encoding.'
        return self._encoding

    def GetFontCode(self, bold=False, italic=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns a font code string for 0 or more embellishments.\n\n    GetFontCode() with no args returns the default font code string.\n\n    Args:\n      bold: True for bold embellishment.\n      italic: True for italic embellishment.\n\n    Returns:\n      The font code string for the requested embellishments. Write this string\n        to the console output to control the font settings.\n    '
        if not self._csi:
            return ''
        codes = []
        if bold:
            codes.append(self._font_bold)
        if italic:
            codes.append(self._font_italic)
        return '{csi}{codes}m'.format(csi=self._csi, codes=';'.join(codes))

    def GetRawKey(self):
        if False:
            i = 10
            return i + 15
        'Reads one key press from stdin with no echo.\n\n    Returns:\n      The key name, None for EOF, <KEY-*> for function keys, otherwise a\n      character.\n    '
        return self._get_raw_key[0]()

    def GetTermIdentifier(self):
        if False:
            while True:
                i = 10
        "Returns the TERM environment variable for the console.\n\n    Returns:\n      str: A str that describes the console's text capabilities\n    "
        return self._term

    def GetTermSize(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the terminal (x, y) dimensions in characters.\n\n    Returns:\n      (x, y): A tuple of the terminal x and y dimensions.\n    '
        return self._term_size

    def DisplayWidth(self, buf):
        if False:
            return 10
        'Returns the display width of buf, handling unicode and ANSI controls.\n\n    Args:\n      buf: The string to count from.\n\n    Returns:\n      The display width of buf, handling unicode and ANSI controls.\n    '
        if not isinstance(buf, six.string_types):
            return len(buf)
        cached = self._display_width_cache.get(buf, None)
        if cached is not None:
            return cached
        width = 0
        max_width = 0
        i = 0
        while i < len(buf):
            if self._csi and buf[i:].startswith(self._csi):
                i += self.GetControlSequenceLen(buf[i:])
            elif buf[i] == '\n':
                max_width = max(width, max_width)
                width = 0
                i += 1
            else:
                width += GetCharacterDisplayWidth(buf[i])
                i += 1
        max_width = max(width, max_width)
        self._display_width_cache[buf] = max_width
        return max_width

    def SplitIntoNormalAndControl(self, buf):
        if False:
            return 10
        'Returns a list of (normal_string, control_sequence) tuples from buf.\n\n    Args:\n      buf: The input string containing one or more control sequences\n        interspersed with normal strings.\n\n    Returns:\n      A list of (normal_string, control_sequence) tuples.\n    '
        if not self._csi or not buf:
            return [(buf, '')]
        seq = []
        i = 0
        while i < len(buf):
            c = buf.find(self._csi, i)
            if c < 0:
                seq.append((buf[i:], ''))
                break
            normal = buf[i:c]
            i = c + self.GetControlSequenceLen(buf[c:])
            seq.append((normal, buf[c:i]))
        return seq

    def SplitLine(self, line, width):
        if False:
            return 10
        'Splits line into width length chunks.\n\n    Args:\n      line: The line to split.\n      width: The width of each chunk except the last which could be smaller than\n        width.\n\n    Returns:\n      A list of chunks, all but the last with display width == width.\n    '
        lines = []
        chunk = ''
        w = 0
        keep = False
        for (normal, control) in self.SplitIntoNormalAndControl(line):
            keep = True
            while True:
                n = width - w
                w += len(normal)
                if w <= width:
                    break
                lines.append(chunk + normal[:n])
                chunk = ''
                keep = False
                w = 0
                normal = normal[n:]
            chunk += normal + control
        if chunk or keep:
            lines.append(chunk)
        return lines

    def SupportsAnsi(self):
        if False:
            while True:
                i = 10
        return self._encoding != 'ascii' and ('screen' in self._term or 'xterm' in self._term)

class Colorizer(object):
    """Resource string colorizer.

  Attributes:
    _con: ConsoleAttr object.
    _color: Color name.
    _string: The string to colorize.
    _justify: The justification function, no justification if None. For example,
      justify=lambda s: s.center(10)
  """

    def __init__(self, string, color, justify=None):
        if False:
            print('Hello World!')
        'Constructor.\n\n    Args:\n      string: The string to colorize.\n      color: Color name used to index ConsoleAttr._ANSI_COLOR.\n      justify: The justification function, no justification if None. For\n        example, justify=lambda s: s.center(10)\n    '
        self._con = GetConsoleAttr()
        self._color = color
        self._string = string
        self._justify = justify

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._string == six.text_type(other)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __gt__(self, other):
        if False:
            return 10
        return self._string > six.text_type(other)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self._string < six.text_type(other)

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        return not self < other

    def __le__(self, other):
        if False:
            print('Hello World!')
        return not self > other

    def __len__(self):
        if False:
            print('Hello World!')
        return self._con.DisplayWidth(self._string)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._string

    def Render(self, stream, justify=None):
        if False:
            while True:
                i = 10
        'Renders the string as self._color on the console.\n\n    Args:\n      stream: The stream to render the string to. The stream given here *must*\n        have the same encoding as sys.stdout for this to work properly.\n      justify: The justification function, self._justify if None.\n    '
        stream.write(self._con.Colorize(self._string, self._color, justify or self._justify))

def GetConsoleAttr(encoding=None, reset=False):
    if False:
        while True:
            i = 10
    'Gets the console attribute state.\n\n  If this is the first call or reset is True or encoding is not None and does\n  not match the current encoding or out is not None and does not match the\n  current out then the state is (re)initialized. Otherwise the current state\n  is returned.\n\n  This call associates the out file stream with the console. All console related\n  output should go to the same stream.\n\n  Args:\n    encoding: Encoding override.\n      ascii -- ASCII. This is the default.\n      utf8 -- UTF-8 unicode.\n      win -- Windows code page 437.\n    reset: Force re-initialization if True.\n\n  Returns:\n    The global ConsoleAttr state object.\n  '
    attr = ConsoleAttr._CONSOLE_ATTR_STATE
    if not reset:
        if not attr:
            reset = True
        elif encoding and encoding != attr.GetEncoding():
            reset = True
    if reset:
        attr = ConsoleAttr(encoding=encoding)
        ConsoleAttr._CONSOLE_ATTR_STATE = attr
    return attr

def ResetConsoleAttr(encoding=None):
    if False:
        i = 10
        return i + 15
    'Resets the console attribute state to the console default.\n\n  Args:\n    encoding: Reset to this encoding instead of the default.\n      ascii -- ASCII. This is the default.\n      utf8 -- UTF-8 unicode.\n      win -- Windows code page 437.\n\n  Returns:\n    The global ConsoleAttr state object.\n  '
    return GetConsoleAttr(encoding=encoding, reset=True)

def GetCharacterDisplayWidth(char):
    if False:
        while True:
            i = 10
    "Returns the monospaced terminal display width of char.\n\n  Assumptions:\n    - monospaced display\n    - ambiguous or unknown chars default to width 1\n    - ASCII control char width is 1 => don't use this for control chars\n\n  Args:\n    char: The character to determine the display width of.\n\n  Returns:\n    The monospaced terminal display width of char: either 0, 1, or 2.\n  "
    if not isinstance(char, six.text_type):
        return 1
    char = unicodedata.normalize('NFC', char)
    if unicodedata.combining(char) != 0:
        return 0
    elif unicodedata.category(char) == 'Cf':
        return 0
    elif unicodedata.east_asian_width(char) in 'FW':
        return 2
    else:
        return 1

def SafeText(data, encoding=None, escape=True):
    if False:
        for i in range(10):
            print('nop')
    b"Converts the data to a text string compatible with the given encoding.\n\n  This works the same way as Decode() below except it guarantees that any\n  characters in the resulting text string can be re-encoded using the given\n  encoding (or GetConsoleAttr().GetEncoding() if None is given). This means\n  that the string will be safe to print to sys.stdout (for example) without\n  getting codec exceptions if the user's terminal doesn't support the encoding\n  used by the source of the text.\n\n  Args:\n    data: Any bytes, string, or object that has str() or unicode() methods.\n    encoding: The encoding name to ensure compatibility with. Defaults to\n      GetConsoleAttr().GetEncoding().\n    escape: Replace unencodable characters with a \\uXXXX or \\xXX equivalent if\n      True. Otherwise replace unencodable characters with an appropriate unknown\n      character, '?' for ASCII, and the unicode unknown replacement character\n      \\uFFFE for unicode.\n\n  Returns:\n    A text string representation of the data, but modified to remove any\n    characters that would result in an encoding exception with the target\n    encoding. In the worst case, with escape=False, it will contain only ?\n    characters.\n  "
    if data is None:
        return 'None'
    encoding = encoding or GetConsoleAttr().GetEncoding()
    string = encoding_util.Decode(data, encoding=encoding)
    try:
        string.encode(encoding)
        return string
    except UnicodeError:
        return string.encode(encoding, 'backslashreplace' if escape else 'replace').decode(encoding)

def EncodeToBytes(data):
    if False:
        for i in range(10):
            print('nop')
    'Encode data to bytes.\n\n  The primary use case is for base64/mime style 7-bit ascii encoding where the\n  encoder input must be bytes. "safe" means that the conversion always returns\n  bytes and will not raise codec exceptions.\n\n  If data is text then an 8-bit ascii encoding is attempted, then the console\n  encoding, and finally utf-8.\n\n  Args:\n    data: Any bytes, string, or object that has str() or unicode() methods.\n\n  Returns:\n    A bytes string representation of the data.\n  '
    if data is None:
        return b''
    if isinstance(data, bytes):
        return data
    s = six.text_type(data)
    try:
        return s.encode('iso-8859-1')
    except UnicodeEncodeError:
        pass
    try:
        return s.encode(GetConsoleAttr().GetEncoding())
    except UnicodeEncodeError:
        pass
    return s.encode('utf-8')

def Decode(data, encoding=None):
    if False:
        while True:
            i = 10
    "Converts the given string, bytes, or object to a text string.\n\n  Args:\n    data: Any bytes, string, or object that has str() or unicode() methods.\n    encoding: A suggesting encoding used to decode. If this encoding doesn't\n      work, other defaults are tried. Defaults to\n      GetConsoleAttr().GetEncoding().\n\n  Returns:\n    A text string representation of the data.\n  "
    encoding = encoding or GetConsoleAttr().GetEncoding()
    return encoding_util.Decode(data, encoding=encoding)