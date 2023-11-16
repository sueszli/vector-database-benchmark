"""
Square and circle annotations (PDF 1.3) shall display, respectively, a rectangle or an ellipse on the page. When
opened, they shall display a pop-up window containing the text of the associated note. The rectangle or ellipse
shall be inscribed within the annotation rectangle defined by the annotation dictionary’s Rect entry (see
Table 168).
"""
import typing
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class CircleAnnotation(Annotation):
    """
    Square and circle annotations (PDF 1.3) shall display, respectively, a rectangle or an ellipse on the page. When
    opened, they shall display a pop-up window containing the text of the associated note. The rectangle or ellipse
    shall be inscribed within the annotation rectangle defined by the annotation dictionary’s Rect entry (see
    Table 168).
    """

    def __init__(self, bounding_box: Rectangle, fill_color: Color, stroke_color: Color, rectangle_difference: typing.Optional[typing.Tuple[Decimal, Decimal, Decimal, Decimal]]=None):
        if False:
            i = 10
            return i + 15
        super(CircleAnnotation, self).__init__(bounding_box=bounding_box, color=stroke_color)
        self[Name('Subtype')] = Name('Circle')
        if fill_color is not None:
            self[Name('IC')] = List().set_is_inline(True)
            self['IC'].append(bDecimal(fill_color.to_rgb().red))
            self['IC'].append(bDecimal(fill_color.to_rgb().green))
            self['IC'].append(bDecimal(fill_color.to_rgb().blue))
        if rectangle_difference is not None:
            self[Name('RD')] = List().set_is_inline(True)
            self['RD'].append(bDecimal(rectangle_difference[0]))
            self['RD'].append(bDecimal(rectangle_difference[1]))
            self['RD'].append(bDecimal(rectangle_difference[2]))
            self['RD'].append(bDecimal(rectangle_difference[3]))