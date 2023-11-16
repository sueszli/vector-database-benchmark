"""
This implementation of LayoutElement represents one uninterrupted block of text
"""
import typing
from decimal import Decimal
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.glyph_line import GlyphLine
from borb.pdf.canvas.font.simple_font.font_type_1 import StandardType1Font
from borb.pdf.canvas.font.simple_font.true_type_font import TrueTypeFont
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.layout_element import Alignment
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.page.page import Page

class ChunkOfText(LayoutElement):
    """
    This implementation of LayoutElement represents one uninterrupted block of text
    """

    def __init__(self, text: str, font: typing.Union[Font, str]='Helvetica', font_size: Decimal=Decimal(12), font_color: Color=HexColor('000000'), border_top: bool=False, border_right: bool=False, border_bottom: bool=False, border_left: bool=False, border_radius_top_left: Decimal=Decimal(0), border_radius_top_right: Decimal=Decimal(0), border_radius_bottom_right: Decimal=Decimal(0), border_radius_bottom_left: Decimal=Decimal(0), border_color: Color=HexColor('000000'), border_width: Decimal=Decimal(1), padding_top: Decimal=Decimal(0), padding_right: Decimal=Decimal(0), padding_bottom: Decimal=Decimal(0), padding_left: Decimal=Decimal(0), margin_top: typing.Optional[Decimal]=None, margin_right: typing.Optional[Decimal]=None, margin_bottom: typing.Optional[Decimal]=None, margin_left: typing.Optional[Decimal]=None, vertical_alignment: Alignment=Alignment.TOP, horizontal_alignment: Alignment=Alignment.LEFT, fixed_leading: typing.Optional[Decimal]=None, multiplied_leading: typing.Optional[Decimal]=None, background_color: typing.Optional[Color]=None):
        if False:
            while True:
                i = 10
        super().__init__(font_size=font_size, border_top=border_top, border_right=border_right, border_bottom=border_bottom, border_left=border_left, border_radius_top_left=border_radius_top_left, border_radius_top_right=border_radius_top_right, border_radius_bottom_right=border_radius_bottom_right, border_radius_bottom_left=border_radius_bottom_left, border_color=border_color, border_width=border_width, padding_top=padding_top, padding_right=padding_right, padding_bottom=padding_bottom, padding_left=padding_left, margin_top=margin_top or Decimal(0), margin_right=margin_right or Decimal(0), margin_bottom=margin_bottom or Decimal(0), margin_left=margin_left or Decimal(0), vertical_alignment=vertical_alignment, horizontal_alignment=horizontal_alignment, background_color=background_color)
        self._text: str = text
        self._is_tagged: bool = False
        if isinstance(font, str):
            self._font: Font = StandardType1Font(font)
            assert self._font
        else:
            self._font = font
        self._font_color = font_color
        if fixed_leading is None and multiplied_leading is None:
            multiplied_leading = Decimal(1.2)
        assert fixed_leading is not None or multiplied_leading is not None
        assert fixed_leading is None or fixed_leading > 0
        assert multiplied_leading is None or multiplied_leading > 0
        self._multiplied_leading: typing.Optional[Decimal] = multiplied_leading
        self._fixed_leading: typing.Optional[Decimal] = fixed_leading

    def _get_content_box(self, available_space: Rectangle) -> Rectangle:
        if False:
            for i in range(10):
                print('nop')
        assert self._font_size is not None
        line_height: Decimal = self._font_size
        if self._multiplied_leading is not None:
            line_height *= self._multiplied_leading
        if self._fixed_leading is not None:
            line_height += self._fixed_leading
        w: Decimal = GlyphLine.from_str(self._text, self._font, self._font_size).get_width_in_text_space()
        return Rectangle(available_space.get_x(), available_space.get_y() + available_space.get_height() - line_height, w, line_height)

    def _get_font_resource_name(self, font: Font, page: Page):
        if False:
            i = 10
            return i + 15
        if 'Resources' not in page:
            page[Name('Resources')] = Dictionary().set_parent(page)
        if 'Font' not in page['Resources']:
            page['Resources'][Name('Font')] = Dictionary()
        font_resource_name = [k for (k, v) in page['Resources']['Font'].items() if v == font]
        if len(font_resource_name) > 0:
            return font_resource_name[0]
        else:
            font_index = len(page['Resources']['Font']) + 1
            page['Resources']['Font'][Name('F%d' % font_index)] = font
            return Name('F%d' % font_index)

    def _pad_string_with_zeroes(self, s: str, n: int=2) -> str:
        if False:
            while True:
                i = 10
        while len(s) < n:
            s = '0' + s
        return s

    def _paint_content_box(self, page: 'Page', content_box: Rectangle) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self._font is not None
        rgb_color = self._font_color.to_rgb()
        assert self._font_size is not None
        line_height: Decimal = self._font_size
        if self._multiplied_leading is not None:
            line_height *= self._multiplied_leading
        if self._fixed_leading is not None:
            line_height += self._fixed_leading
        descent: Decimal = self._font.get_descent() / Decimal(1000) * self._font_size
        content = 'q\nBT\n%f %f %f rg\n/%s %f Tf\n%f 0 0 %f %f %f Tm\n%s\nET\nQ' % (float(rgb_color.red), float(rgb_color.green), float(rgb_color.blue), self._get_font_resource_name(self._font, page), float(1), float(self._font_size), float(self._font_size), float(content_box.get_x()), float(content_box.get_y() + content_box.get_height() - self._font_size - descent), self._write_text_bytes())
        page.append_to_content_stream(content)

    def _write_text_bytes(self) -> str:
        if False:
            while True:
                i = 10
        hex_mode: bool = False
        for c in self._text:
            if ord(c) != self._font.unicode_to_character_identifier(c):
                hex_mode = True
                break
        if hex_mode or isinstance(self._font, TrueTypeFont):
            return self._write_text_bytes_in_hex()
        else:
            return self._write_text_bytes_in_ascii()

    def _write_text_bytes_in_ascii(self) -> str:
        if False:
            while True:
                i = 10
        '\n        This function escapes certain reserved characters in PDF strings.\n        '
        sOut: str = ''
        for c in self._text:
            if c == '\r':
                sOut += '\\r'
            elif c == '\n':
                sOut += '\\n'
            elif c == '\t':
                sOut += '\\t'
            elif c == '\x08':
                sOut += '\\b'
            elif c == '\x0c':
                sOut += '\\f'
            elif c in ['(', ')', '\\']:
                sOut += '\\' + c
            elif 0 <= ord(c) < 8:
                sOut += '\\00' + oct(ord(c))[2:]
            elif 8 <= ord(c) < 32:
                sOut += '\\0' + oct(ord(c))[2:]
            else:
                sOut += c
        return ''.join(['(', sOut, ') Tj'])

    def _write_text_bytes_in_hex(self) -> str:
        if False:
            print('Hello World!')
        font: Font = self._font
        use_four_bytes: bool = False
        if 'Encoding' in font and font['Encoding'] in ['Identity-H', 'Identity-V']:
            use_four_bytes = True
        sOut: str = ''
        for c in self._text:
            cid: typing.Optional[int] = self._font.unicode_to_character_identifier(c)
            assert cid is not None, "Font %s can not represent '%s'" % (self._font.get_font_name(), c)
            hex_rep: str = hex(int(cid))[2:]
            hex_rep = self._pad_string_with_zeroes(hex_rep, 4 if use_four_bytes else 2)
            sOut += ''.join(['<', hex_rep, '>'])
        return ''.join(['[', sOut, '] TJ'])

    def get_font(self) -> Font:
        if False:
            return 10
        '\n        This function returns the Font of this LayoutElement\n        '
        return self._font

    def get_font_color(self) -> Color:
        if False:
            return 10
        '\n        This function returns the font Color of this LayoutElement\n        '
        return self._font_color

    def get_text(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the text of this LayoutElement\n        '
        return self._text