"""
Class and program to colorize python source code for ANSI terminals.

Based on an HTML code highlighter by Jurgen Hermann found at:
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52298

Modifications by Fernando Perez (fperez@colorado.edu).

Information on the original HTML highlighter follows:

MoinMoin - Python Source Parser

Title: Colorize Python source using the built-in tokenizer

Submitter: Jurgen Hermann
Last Updated:2001/04/06

Version no:1.2

Description:

This code is part of MoinMoin (http://moin.sourceforge.net/) and converts
Python source code to HTML markup, rendering comments, keywords,
operators, numeric and string literals in different colors.

It shows how to use the built-in keyword, token and tokenize modules to
scan Python source code and re-emit it with no changes to its original
formatting (which is the hard part).
"""
__all__ = ['ANSICodeColors', 'Parser']
_scheme_default = 'Linux'
import keyword
import os
import sys
import token
import tokenize
generate_tokens = tokenize.generate_tokens
from IPython.utils.coloransi import TermColors, InputTermColors, ColorScheme, ColorSchemeTable
from .colorable import Colorable
from io import StringIO
_KEYWORD = token.NT_OFFSET + 1
_TEXT = token.NT_OFFSET + 2
Colors = TermColors
NoColor = ColorScheme('NoColor', {'header': Colors.NoColor, token.NUMBER: Colors.NoColor, token.OP: Colors.NoColor, token.STRING: Colors.NoColor, tokenize.COMMENT: Colors.NoColor, token.NAME: Colors.NoColor, token.ERRORTOKEN: Colors.NoColor, _KEYWORD: Colors.NoColor, _TEXT: Colors.NoColor, 'in_prompt': InputTermColors.NoColor, 'in_number': InputTermColors.NoColor, 'in_prompt2': InputTermColors.NoColor, 'in_normal': InputTermColors.NoColor, 'out_prompt': Colors.NoColor, 'out_number': Colors.NoColor, 'normal': Colors.NoColor})
LinuxColors = ColorScheme('Linux', {'header': Colors.LightRed, token.NUMBER: Colors.LightCyan, token.OP: Colors.Yellow, token.STRING: Colors.LightBlue, tokenize.COMMENT: Colors.LightRed, token.NAME: Colors.Normal, token.ERRORTOKEN: Colors.Red, _KEYWORD: Colors.LightGreen, _TEXT: Colors.Yellow, 'in_prompt': InputTermColors.Green, 'in_number': InputTermColors.LightGreen, 'in_prompt2': InputTermColors.Green, 'in_normal': InputTermColors.Normal, 'out_prompt': Colors.Red, 'out_number': Colors.LightRed, 'normal': Colors.Normal})
NeutralColors = ColorScheme('Neutral', {'header': Colors.Red, token.NUMBER: Colors.Cyan, token.OP: Colors.Blue, token.STRING: Colors.Blue, tokenize.COMMENT: Colors.Red, token.NAME: Colors.Normal, token.ERRORTOKEN: Colors.Red, _KEYWORD: Colors.Green, _TEXT: Colors.Blue, 'in_prompt': InputTermColors.Blue, 'in_number': InputTermColors.LightBlue, 'in_prompt2': InputTermColors.Blue, 'in_normal': InputTermColors.Normal, 'out_prompt': Colors.Red, 'out_number': Colors.LightRed, 'normal': Colors.Normal})
if os.name == 'nt':
    NeutralColors = LinuxColors.copy(name='Neutral')
LightBGColors = ColorScheme('LightBG', {'header': Colors.Red, token.NUMBER: Colors.Cyan, token.OP: Colors.Blue, token.STRING: Colors.Blue, tokenize.COMMENT: Colors.Red, token.NAME: Colors.Normal, token.ERRORTOKEN: Colors.Red, _KEYWORD: Colors.Green, _TEXT: Colors.Blue, 'in_prompt': InputTermColors.Blue, 'in_number': InputTermColors.LightBlue, 'in_prompt2': InputTermColors.Blue, 'in_normal': InputTermColors.Normal, 'out_prompt': Colors.Red, 'out_number': Colors.LightRed, 'normal': Colors.Normal})
ANSICodeColors = ColorSchemeTable([NoColor, LinuxColors, LightBGColors, NeutralColors], _scheme_default)
Undefined = object()

class Parser(Colorable):
    """ Format colored Python source.
    """

    def __init__(self, color_table=None, out=sys.stdout, parent=None, style=None):
        if False:
            for i in range(10):
                print('nop')
        ' Create a parser with a specified color table and output channel.\n\n        Call format() to process code.\n        '
        super(Parser, self).__init__(parent=parent)
        self.color_table = color_table if color_table else ANSICodeColors
        self.out = out
        self.pos = None
        self.lines = None
        self.raw = None
        if not style:
            self.style = self.default_style
        else:
            self.style = style

    def format(self, raw, out=None, scheme=Undefined):
        if False:
            print('Hello World!')
        import warnings
        if scheme is not Undefined:
            warnings.warn('The `scheme` argument of IPython.utils.PyColorize:Parser.format is deprecated since IPython 6.0.It will have no effect. Set the parser `style` directly.', stacklevel=2)
        return self.format2(raw, out)[0]

    def format2(self, raw, out=None):
        if False:
            for i in range(10):
                print('nop')
        " Parse and send the colored source.\n\n        If out and scheme are not specified, the defaults (given to\n        constructor) are used.\n\n        out should be a file-type object. Optionally, out can be given as the\n        string 'str' and the parser will automatically return the output in a\n        string."
        string_output = 0
        if out == 'str' or self.out == 'str' or isinstance(self.out, StringIO):
            out_old = self.out
            self.out = StringIO()
            string_output = 1
        elif out is not None:
            self.out = out
        else:
            raise ValueError('`out` or `self.out` should be file-like or the value `"str"`')
        if self.style == 'NoColor':
            error = False
            self.out.write(raw)
            if string_output:
                return (raw, error)
            return (None, error)
        colors = self.color_table[self.style].colors
        self.colors = colors
        self.raw = raw.expandtabs().rstrip()
        self.lines = [0, 0]
        pos = 0
        raw_find = self.raw.find
        lines_append = self.lines.append
        while True:
            pos = raw_find('\n', pos) + 1
            if not pos:
                break
            lines_append(pos)
        lines_append(len(self.raw))
        self.pos = 0
        text = StringIO(self.raw)
        error = False
        try:
            for atoken in generate_tokens(text.readline):
                self(*atoken)
        except tokenize.TokenError as ex:
            msg = ex.args[0]
            line = ex.args[1][0]
            self.out.write('%s\n\n*** ERROR: %s%s%s\n' % (colors[token.ERRORTOKEN], msg, self.raw[self.lines[line]:], colors.normal))
            error = True
        self.out.write(colors.normal + '\n')
        if string_output:
            output = self.out.getvalue()
            self.out = out_old
            return (output, error)
        return (None, error)

    def _inner_call_(self, toktype, toktext, start_pos):
        if False:
            i = 10
            return i + 15
        'like call but write to a temporary buffer'
        buff = StringIO()
        (srow, scol) = start_pos
        colors = self.colors
        owrite = buff.write
        linesep = os.linesep
        oldpos = self.pos
        newpos = self.lines[srow] + scol
        self.pos = newpos + len(toktext)
        if newpos > oldpos:
            owrite(self.raw[oldpos:newpos])
        if toktype in [token.INDENT, token.DEDENT]:
            self.pos = newpos
            buff.seek(0)
            return buff.read()
        if token.LPAR <= toktype <= token.OP:
            toktype = token.OP
        elif toktype == token.NAME and keyword.iskeyword(toktext):
            toktype = _KEYWORD
        color = colors.get(toktype, colors[_TEXT])
        if linesep in toktext:
            toktext = toktext.replace(linesep, '%s%s%s' % (colors.normal, linesep, color))
        owrite('%s%s%s' % (color, toktext, colors.normal))
        buff.seek(0)
        return buff.read()

    def __call__(self, toktype, toktext, start_pos, end_pos, line):
        if False:
            while True:
                i = 10
        ' Token handler, with syntax highlighting.'
        self.out.write(self._inner_call_(toktype, toktext, start_pos))