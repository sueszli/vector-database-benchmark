"""
    pygments.formatters.rtf
    ~~~~~~~~~~~~~~~~~~~~~~~

    A formatter that generates RTF files.

    :copyright: Copyright 2006-2023 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from pip._vendor.pygments.formatter import Formatter
from pip._vendor.pygments.util import get_int_opt, surrogatepair
__all__ = ['RtfFormatter']

class RtfFormatter(Formatter):
    """
    Format tokens as RTF markup. This formatter automatically outputs full RTF
    documents with color information and other useful stuff. Perfect for Copy and
    Paste into Microsoft(R) Word(R) documents.

    Please note that ``encoding`` and ``outencoding`` options are ignored.
    The RTF format is ASCII natively, but handles unicode characters correctly
    thanks to escape sequences.

    .. versionadded:: 0.6

    Additional options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).

    `fontface`
        The used font family, for example ``Bitstream Vera Sans``. Defaults to
        some generic font which is supposed to have fixed width.

    `fontsize`
        Size of the font used. Size is specified in half points. The
        default is 24 half-points, giving a size 12 font.

        .. versionadded:: 2.0
    """
    name = 'RTF'
    aliases = ['rtf']
    filenames = ['*.rtf']

    def __init__(self, **options):
        if False:
            while True:
                i = 10
        '\n        Additional options accepted:\n\n        ``fontface``\n            Name of the font used. Could for example be ``\'Courier New\'``\n            to further specify the default which is ``\'\\fmodern\'``. The RTF\n            specification claims that ``\\fmodern`` are "Fixed-pitch serif\n            and sans serif fonts". Hope every RTF implementation thinks\n            the same about modern...\n\n        '
        Formatter.__init__(self, **options)
        self.fontface = options.get('fontface') or ''
        self.fontsize = get_int_opt(options, 'fontsize', 0)

    def _escape(self, text):
        if False:
            print('Hello World!')
        return text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')

    def _escape_text(self, text):
        if False:
            while True:
                i = 10
        if not text:
            return ''
        text = self._escape(text)
        buf = []
        for c in text:
            cn = ord(c)
            if cn < 2 ** 7:
                buf.append(str(c))
            elif 2 ** 7 <= cn < 2 ** 16:
                buf.append('{\\u%d}' % cn)
            elif 2 ** 16 <= cn:
                buf.append('{\\u%d}{\\u%d}' % surrogatepair(cn))
        return ''.join(buf).replace('\n', '\\par\n')

    def format_unencoded(self, tokensource, outfile):
        if False:
            return 10
        outfile.write('{\\rtf1\\ansi\\uc0\\deff0{\\fonttbl{\\f0\\fmodern\\fprq1\\fcharset0%s;}}{\\colortbl;' % (self.fontface and ' ' + self._escape(self.fontface) or ''))
        color_mapping = {}
        offset = 1
        for (_, style) in self.style:
            for color in (style['color'], style['bgcolor'], style['border']):
                if color and color not in color_mapping:
                    color_mapping[color] = offset
                    outfile.write('\\red%d\\green%d\\blue%d;' % (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)))
                    offset += 1
        outfile.write('}\\f0 ')
        if self.fontsize:
            outfile.write('\\fs%d' % self.fontsize)
        for (ttype, value) in tokensource:
            while not self.style.styles_token(ttype) and ttype.parent:
                ttype = ttype.parent
            style = self.style.style_for_token(ttype)
            buf = []
            if style['bgcolor']:
                buf.append('\\cb%d' % color_mapping[style['bgcolor']])
            if style['color']:
                buf.append('\\cf%d' % color_mapping[style['color']])
            if style['bold']:
                buf.append('\\b')
            if style['italic']:
                buf.append('\\i')
            if style['underline']:
                buf.append('\\ul')
            if style['border']:
                buf.append('\\chbrdr\\chcfpat%d' % color_mapping[style['border']])
            start = ''.join(buf)
            if start:
                outfile.write('{%s ' % start)
            outfile.write(self._escape_text(value))
            if start:
                outfile.write('}')
        outfile.write('}')