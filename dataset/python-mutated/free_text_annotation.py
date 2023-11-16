"""
A free text annotation (PDF 1.3) displays text directly on the page. Unlike an ordinary text annotation (see
12.5.6.4, “Text Annotations”), a free text annotation has no open or closed state; instead of being displayed in a
pop-up window, the text shall be always visible. Table 174 shows the annotation dictionary entries specific to
this type of annotation. 12.7.3.3, “Variable Text” describes the process of using these entries to generate the
appearance of the text in these annotations.
"""
import typing
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.simple_font.font_type_1 import StandardType1Font
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class FreeTextAnnotation(Annotation):
    """
    A free text annotation (PDF 1.3) displays text directly on the page. Unlike an ordinary text annotation (see
    12.5.6.4, “Text Annotations”), a free text annotation has no open or closed state; instead of being displayed in a
    pop-up window, the text shall be always visible. Table 174 shows the annotation dictionary entries specific to
    this type of annotation. 12.7.3.3, “Variable Text” describes the process of using these entries to generate the
    appearance of the text in these annotations.
    """

    def __init__(self, bounding_box: Rectangle, contents: str, background_color: typing.Optional[Color]=None, font: Font=StandardType1Font('Helvetica'), font_size: Decimal=Decimal(12), font_color: Color=HexColor('000000')):
        if False:
            i = 10
            return i + 15
        super(FreeTextAnnotation, self).__init__(bounding_box=bounding_box, contents=contents, color=background_color)
        self._font: Font = font
        self._font_color_rgb: 'RGBColor' = font_color.to_rgb()
        self._font_size: Decimal = font_size
        self._font_name: str = 'F0'
        self[Name('Subtype')] = Name('FreeText')
        self[Name('F')] = bDecimal(20)
        self[Name('DA')] = String('/%s %f Tf %f %f %f rg' % (self._font_name, self._font_size, self._font_color_rgb.red, self._font_color_rgb.green, self._font_color_rgb.blue))
        self[Name('Q')] = bDecimal(0)
        self[Name('IT')] = Name('FreeTextTypeWriter')

    def _embed_font_in_page(self, page: 'Page') -> None:
        if False:
            while True:
                i = 10
        if 'Resources' not in page:
            page[Name('Resources')] = Dictionary()
        if 'Font' not in page['Resources']:
            page['Resources'][Name('Font')] = Dictionary()
        font_number: int = len(page['Resources']['Font'])
        font_name: str = 'F%d' % font_number
        while font_name in page['Resources']['Font']:
            font_number += 1
            font_name = 'F%d' % font_number
        page['Resources']['Font'][Name(font_name)] = self._font
        self[Name('DA')] = String('/%s %f Tf %f %f %f rg' % (self._font_name, self._font_size, self._font_color_rgb.red, self._font_color_rgb.green, self._font_color_rgb.blue))