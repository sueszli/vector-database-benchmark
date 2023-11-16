"""
    pygments.formatters.terminal256
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Formatter for 256-color terminal output with ANSI sequences.

    RGB-to-XTERM color conversion routines adapted from xterm256-conv
    tool (http://frexx.de/xterm-256-notes/data/xterm256-conv2.tar.bz2)
    by Wolfgang Frisch.

    Formatter version 1.

    :copyright: Copyright 2006-2023 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from pip._vendor.pygments.formatter import Formatter
from pip._vendor.pygments.console import codes
from pip._vendor.pygments.style import ansicolors
__all__ = ['Terminal256Formatter', 'TerminalTrueColorFormatter']

class EscapeSequence:

    def __init__(self, fg=None, bg=None, bold=False, underline=False, italic=False):
        if False:
            for i in range(10):
                print('nop')
        self.fg = fg
        self.bg = bg
        self.bold = bold
        self.underline = underline
        self.italic = italic

    def escape(self, attrs):
        if False:
            while True:
                i = 10
        if len(attrs):
            return '\x1b[' + ';'.join(attrs) + 'm'
        return ''

    def color_string(self):
        if False:
            print('Hello World!')
        attrs = []
        if self.fg is not None:
            if self.fg in ansicolors:
                esc = codes[self.fg.replace('ansi', '')]
                if ';01m' in esc:
                    self.bold = True
                attrs.append(esc[2:4])
            else:
                attrs.extend(('38', '5', '%i' % self.fg))
        if self.bg is not None:
            if self.bg in ansicolors:
                esc = codes[self.bg.replace('ansi', '')]
                attrs.append(str(int(esc[2:4]) + 10))
            else:
                attrs.extend(('48', '5', '%i' % self.bg))
        if self.bold:
            attrs.append('01')
        if self.underline:
            attrs.append('04')
        if self.italic:
            attrs.append('03')
        return self.escape(attrs)

    def true_color_string(self):
        if False:
            return 10
        attrs = []
        if self.fg:
            attrs.extend(('38', '2', str(self.fg[0]), str(self.fg[1]), str(self.fg[2])))
        if self.bg:
            attrs.extend(('48', '2', str(self.bg[0]), str(self.bg[1]), str(self.bg[2])))
        if self.bold:
            attrs.append('01')
        if self.underline:
            attrs.append('04')
        if self.italic:
            attrs.append('03')
        return self.escape(attrs)

    def reset_string(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = []
        if self.fg is not None:
            attrs.append('39')
        if self.bg is not None:
            attrs.append('49')
        if self.bold or self.underline or self.italic:
            attrs.append('00')
        return self.escape(attrs)

class Terminal256Formatter(Formatter):
    """
    Format tokens with ANSI color sequences, for output in a 256-color
    terminal or console.  Like in `TerminalFormatter` color sequences
    are terminated at newlines, so that paging the output works correctly.

    The formatter takes colors from a style defined by the `style` option
    and converts them to nearest ANSI 256-color escape sequences. Bold and
    underline attributes from the style are preserved (and displayed).

    .. versionadded:: 0.9

    .. versionchanged:: 2.2
       If the used style defines foreground colors in the form ``#ansi*``, then
       `Terminal256Formatter` will map these to non extended foreground color.
       See :ref:`AnsiTerminalStyle` for more information.

    .. versionchanged:: 2.4
       The ANSI color names have been updated with names that are easier to
       understand and align with colornames of other projects and terminals.
       See :ref:`this table <new-ansi-color-names>` for more information.


    Options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).

    `linenos`
        Set to ``True`` to have line numbers on the terminal output as well
        (default: ``False`` = no line numbers).
    """
    name = 'Terminal256'
    aliases = ['terminal256', 'console256', '256']
    filenames = []

    def __init__(self, **options):
        if False:
            i = 10
            return i + 15
        Formatter.__init__(self, **options)
        self.xterm_colors = []
        self.best_match = {}
        self.style_string = {}
        self.usebold = 'nobold' not in options
        self.useunderline = 'nounderline' not in options
        self.useitalic = 'noitalic' not in options
        self._build_color_table()
        self._setup_styles()
        self.linenos = options.get('linenos', False)
        self._lineno = 0

    def _build_color_table(self):
        if False:
            return 10
        self.xterm_colors.append((0, 0, 0))
        self.xterm_colors.append((205, 0, 0))
        self.xterm_colors.append((0, 205, 0))
        self.xterm_colors.append((205, 205, 0))
        self.xterm_colors.append((0, 0, 238))
        self.xterm_colors.append((205, 0, 205))
        self.xterm_colors.append((0, 205, 205))
        self.xterm_colors.append((229, 229, 229))
        self.xterm_colors.append((127, 127, 127))
        self.xterm_colors.append((255, 0, 0))
        self.xterm_colors.append((0, 255, 0))
        self.xterm_colors.append((255, 255, 0))
        self.xterm_colors.append((92, 92, 255))
        self.xterm_colors.append((255, 0, 255))
        self.xterm_colors.append((0, 255, 255))
        self.xterm_colors.append((255, 255, 255))
        valuerange = (0, 95, 135, 175, 215, 255)
        for i in range(217):
            r = valuerange[i // 36 % 6]
            g = valuerange[i // 6 % 6]
            b = valuerange[i % 6]
            self.xterm_colors.append((r, g, b))
        for i in range(1, 22):
            v = 8 + i * 10
            self.xterm_colors.append((v, v, v))

    def _closest_color(self, r, g, b):
        if False:
            while True:
                i = 10
        distance = 257 * 257 * 3
        match = 0
        for i in range(0, 254):
            values = self.xterm_colors[i]
            rd = r - values[0]
            gd = g - values[1]
            bd = b - values[2]
            d = rd * rd + gd * gd + bd * bd
            if d < distance:
                match = i
                distance = d
        return match

    def _color_index(self, color):
        if False:
            print('Hello World!')
        index = self.best_match.get(color, None)
        if color in ansicolors:
            index = color
            self.best_match[color] = index
        if index is None:
            try:
                rgb = int(str(color), 16)
            except ValueError:
                rgb = 0
            r = rgb >> 16 & 255
            g = rgb >> 8 & 255
            b = rgb & 255
            index = self._closest_color(r, g, b)
            self.best_match[color] = index
        return index

    def _setup_styles(self):
        if False:
            return 10
        for (ttype, ndef) in self.style:
            escape = EscapeSequence()
            if ndef['ansicolor']:
                escape.fg = self._color_index(ndef['ansicolor'])
            elif ndef['color']:
                escape.fg = self._color_index(ndef['color'])
            if ndef['bgansicolor']:
                escape.bg = self._color_index(ndef['bgansicolor'])
            elif ndef['bgcolor']:
                escape.bg = self._color_index(ndef['bgcolor'])
            if self.usebold and ndef['bold']:
                escape.bold = True
            if self.useunderline and ndef['underline']:
                escape.underline = True
            if self.useitalic and ndef['italic']:
                escape.italic = True
            self.style_string[str(ttype)] = (escape.color_string(), escape.reset_string())

    def _write_lineno(self, outfile):
        if False:
            i = 10
            return i + 15
        self._lineno += 1
        outfile.write('%s%04d: ' % (self._lineno != 1 and '\n' or '', self._lineno))

    def format(self, tokensource, outfile):
        if False:
            return 10
        return Formatter.format(self, tokensource, outfile)

    def format_unencoded(self, tokensource, outfile):
        if False:
            return 10
        if self.linenos:
            self._write_lineno(outfile)
        for (ttype, value) in tokensource:
            not_found = True
            while ttype and not_found:
                try:
                    (on, off) = self.style_string[str(ttype)]
                    spl = value.split('\n')
                    for line in spl[:-1]:
                        if line:
                            outfile.write(on + line + off)
                        if self.linenos:
                            self._write_lineno(outfile)
                        else:
                            outfile.write('\n')
                    if spl[-1]:
                        outfile.write(on + spl[-1] + off)
                    not_found = False
                except KeyError:
                    ttype = ttype.parent
            if not_found:
                outfile.write(value)
        if self.linenos:
            outfile.write('\n')

class TerminalTrueColorFormatter(Terminal256Formatter):
    """
    Format tokens with ANSI color sequences, for output in a true-color
    terminal or console.  Like in `TerminalFormatter` color sequences
    are terminated at newlines, so that paging the output works correctly.

    .. versionadded:: 2.1

    Options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).
    """
    name = 'TerminalTrueColor'
    aliases = ['terminal16m', 'console16m', '16m']
    filenames = []

    def _build_color_table(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _color_tuple(self, color):
        if False:
            i = 10
            return i + 15
        try:
            rgb = int(str(color), 16)
        except ValueError:
            return None
        r = rgb >> 16 & 255
        g = rgb >> 8 & 255
        b = rgb & 255
        return (r, g, b)

    def _setup_styles(self):
        if False:
            i = 10
            return i + 15
        for (ttype, ndef) in self.style:
            escape = EscapeSequence()
            if ndef['color']:
                escape.fg = self._color_tuple(ndef['color'])
            if ndef['bgcolor']:
                escape.bg = self._color_tuple(ndef['bgcolor'])
            if self.usebold and ndef['bold']:
                escape.bold = True
            if self.useunderline and ndef['underline']:
                escape.underline = True
            if self.useitalic and ndef['italic']:
                escape.italic = True
            self.style_string[str(ttype)] = (escape.true_color_string(), escape.reset_string())