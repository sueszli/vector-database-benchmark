""" Utilities for processing ANSI escape codes and special ASCII characters.
"""
from collections import namedtuple
import re
from qtpy import QtGui
from qtconsole.styles import dark_style
EraseAction = namedtuple('EraseAction', ['action', 'area', 'erase_to'])
MoveAction = namedtuple('MoveAction', ['action', 'dir', 'unit', 'count'])
ScrollAction = namedtuple('ScrollAction', ['action', 'dir', 'unit', 'count'])
CarriageReturnAction = namedtuple('CarriageReturnAction', ['action'])
NewLineAction = namedtuple('NewLineAction', ['action'])
BeepAction = namedtuple('BeepAction', ['action'])
BackSpaceAction = namedtuple('BackSpaceAction', ['action'])
CSI_COMMANDS = 'ABCDEFGHJKSTfmnsu'
CSI_SUBPATTERN = '\\[(.*?)([%s])' % CSI_COMMANDS
OSC_SUBPATTERN = '\\](.*?)[\x07\x1b]'
ANSI_PATTERN = '\x01?\x1b(%s|%s)\x02?' % (CSI_SUBPATTERN, OSC_SUBPATTERN)
ANSI_OR_SPECIAL_PATTERN = re.compile('(\x07|\x08|\r(?!\n)|\r?\n)|(?:%s)' % ANSI_PATTERN)
SPECIAL_PATTERN = re.compile('([\x0c])')

class AnsiCodeProcessor(object):
    """ Translates special ASCII characters and ANSI escape codes into readable
        attributes. It also supports a few non-standard, xterm-specific codes.
    """
    bold_text_enabled = False
    default_color_map = {}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.actions = []
        self.color_map = self.default_color_map.copy()
        self.reset_sgr()

    def reset_sgr(self):
        if False:
            i = 10
            return i + 15
        ' Reset graphics attributs to their default values.\n        '
        self.intensity = 0
        self.italic = False
        self.bold = False
        self.underline = False
        self.foreground_color = None
        self.background_color = None

    def split_string(self, string):
        if False:
            print('Hello World!')
        ' Yields substrings for which the same escape code applies.\n        '
        self.actions = []
        start = 0
        last_char = '\n' if len(string) > 0 and string[-1] == '\n' else None
        string = string[:-1] if last_char is not None else string
        for match in ANSI_OR_SPECIAL_PATTERN.finditer(string):
            raw = string[start:match.start()]
            substring = SPECIAL_PATTERN.sub(self._replace_special, raw)
            if substring or self.actions:
                yield substring
                self.actions = []
            start = match.end()
            groups = [g for g in match.groups() if g is not None]
            g0 = groups[0]
            if g0 == '\x07':
                self.actions.append(BeepAction('beep'))
                yield None
                self.actions = []
            elif g0 == '\r':
                self.actions.append(CarriageReturnAction('carriage-return'))
                yield None
                self.actions = []
            elif g0 == '\x08':
                self.actions.append(BackSpaceAction('backspace'))
                yield None
                self.actions = []
            elif g0 == '\n' or g0 == '\r\n':
                self.actions.append(NewLineAction('newline'))
                yield g0
                self.actions = []
            else:
                params = [param for param in groups[1].split(';') if param]
                if g0.startswith('['):
                    try:
                        params = list(map(int, params))
                    except ValueError:
                        pass
                    else:
                        self.set_csi_code(groups[2], params)
                elif g0.startswith(']'):
                    self.set_osc_code(params)
        raw = string[start:]
        substring = SPECIAL_PATTERN.sub(self._replace_special, raw)
        if substring or self.actions:
            yield substring
        if last_char is not None:
            self.actions.append(NewLineAction('newline'))
            yield last_char

    def set_csi_code(self, command, params=[]):
        if False:
            while True:
                i = 10
        ' Set attributes based on CSI (Control Sequence Introducer) code.\n\n        Parameters\n        ----------\n        command : str\n            The code identifier, i.e. the final character in the sequence.\n\n        params : sequence of integers, optional\n            The parameter codes for the command.\n        '
        if command == 'm':
            if params:
                self.set_sgr_code(params)
            else:
                self.set_sgr_code([0])
        elif command == 'J' or command == 'K':
            code = params[0] if params else 0
            if 0 <= code <= 2:
                area = 'screen' if command == 'J' else 'line'
                if code == 0:
                    erase_to = 'end'
                elif code == 1:
                    erase_to = 'start'
                elif code == 2:
                    erase_to = 'all'
                self.actions.append(EraseAction('erase', area, erase_to))
        elif command == 'S' or command == 'T':
            dir = 'up' if command == 'S' else 'down'
            count = params[0] if params else 1
            self.actions.append(ScrollAction('scroll', dir, 'line', count))

    def set_osc_code(self, params):
        if False:
            while True:
                i = 10
        ' Set attributes based on OSC (Operating System Command) parameters.\n\n        Parameters\n        ----------\n        params : sequence of str\n            The parameters for the command.\n        '
        try:
            command = int(params.pop(0))
        except (IndexError, ValueError):
            return
        if command == 4:
            try:
                color = int(params.pop(0))
                spec = params.pop(0)
                self.color_map[color] = self._parse_xterm_color_spec(spec)
            except (IndexError, ValueError):
                pass

    def set_sgr_code(self, params):
        if False:
            i = 10
            return i + 15
        ' Set attributes based on SGR (Select Graphic Rendition) codes.\n\n        Parameters\n        ----------\n        params : sequence of ints\n            A list of SGR codes for one or more SGR commands. Usually this\n            sequence will have one element per command, although certain\n            xterm-specific commands requires multiple elements.\n        '
        if not params:
            return
        code = params.pop(0)
        if code == 0:
            self.reset_sgr()
        elif code == 1:
            if self.bold_text_enabled:
                self.bold = True
            else:
                self.intensity = 1
        elif code == 2:
            self.intensity = 0
        elif code == 3:
            self.italic = True
        elif code == 4:
            self.underline = True
        elif code == 22:
            self.intensity = 0
            self.bold = False
        elif code == 23:
            self.italic = False
        elif code == 24:
            self.underline = False
        elif code >= 30 and code <= 37:
            self.foreground_color = code - 30
        elif code == 38 and params:
            _color_type = params.pop(0)
            if _color_type == 5 and params:
                self.foreground_color = params.pop(0)
            elif _color_type == 2:
                self.foreground_color = params[:3]
                params[:3] = []
        elif code == 39:
            self.foreground_color = None
        elif code >= 40 and code <= 47:
            self.background_color = code - 40
        elif code == 48 and params:
            _color_type = params.pop(0)
            if _color_type == 5 and params:
                self.background_color = params.pop(0)
            elif _color_type == 2:
                self.background_color = params[:3]
                params[:3] = []
        elif code == 49:
            self.background_color = None
        self.set_sgr_code(params)

    def _parse_xterm_color_spec(self, spec):
        if False:
            print('Hello World!')
        if spec.startswith('rgb:'):
            return tuple(map(lambda x: int(x, 16), spec[4:].split('/')))
        elif spec.startswith('rgbi:'):
            return tuple(map(lambda x: int(float(x) * 255), spec[5:].split('/')))
        elif spec == '?':
            raise ValueError('Unsupported xterm color spec')
        return spec

    def _replace_special(self, match):
        if False:
            while True:
                i = 10
        special = match.group(1)
        if special == '\x0c':
            self.actions.append(ScrollAction('scroll', 'down', 'page', 1))
        return ''

class QtAnsiCodeProcessor(AnsiCodeProcessor):
    """ Translates ANSI escape codes into QTextCharFormats.
    """
    darkbg_color_map = {0: 'black', 1: 'darkred', 2: 'darkgreen', 3: 'brown', 4: 'darkblue', 5: 'darkviolet', 6: 'steelblue', 7: 'grey', 8: 'grey', 9: 'red', 10: 'lime', 11: 'yellow', 12: 'deepskyblue', 13: 'magenta', 14: 'cyan', 15: 'white'}
    default_color_map = darkbg_color_map.copy()

    def get_color(self, color, intensity=0):
        if False:
            print('Hello World!')
        ' Returns a QColor for a given color code or rgb list, or None if one\n            cannot be constructed.\n        '
        if isinstance(color, int):
            if color < 8 and intensity > 0:
                color += 8
            constructor = self.color_map.get(color, None)
        elif isinstance(color, (tuple, list)):
            constructor = color
        else:
            return None
        if isinstance(constructor, str):
            return QtGui.QColor(constructor)
        elif isinstance(constructor, (tuple, list)):
            return QtGui.QColor(*constructor)
        return None

    def get_format(self):
        if False:
            while True:
                i = 10
        ' Returns a QTextCharFormat that encodes the current style attributes.\n        '
        format = QtGui.QTextCharFormat()
        qcolor = self.get_color(self.foreground_color, self.intensity)
        if qcolor is not None:
            format.setForeground(qcolor)
        qcolor = self.get_color(self.background_color, self.intensity)
        if qcolor is not None:
            format.setBackground(qcolor)
        if self.bold:
            format.setFontWeight(QtGui.QFont.Bold)
        else:
            format.setFontWeight(QtGui.QFont.Normal)
        format.setFontItalic(self.italic)
        format.setFontUnderline(self.underline)
        return format

    def set_background_color(self, style):
        if False:
            return 10
        '\n        Given a syntax style, attempt to set a color map that will be\n        aesthetically pleasing.\n        '
        self.default_color_map = self.darkbg_color_map.copy()
        if not dark_style(style):
            for i in range(8):
                self.default_color_map[i + 8] = self.default_color_map[i]
            self.default_color_map[7] = self.default_color_map[15] = 'black'
        self.color_map.update(self.default_color_map)