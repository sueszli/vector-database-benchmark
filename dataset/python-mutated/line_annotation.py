"""
The purpose of a line annotation (PDF 1.3) is to display a single straight line on the page. When opened, it shall
display a pop-up window containing the text of the associated note. Table 175 shows the annotation dictionary
entries specific to this type of annotation.
"""
import typing
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation
from borb.pdf.canvas.layout.annotation.polyline_annotation import LineEndStyleType

class LineAnnotation(Annotation):
    """
    The purpose of a line annotation (PDF 1.3) is to display a single straight line on the page. When opened, it shall
    display a pop-up window containing the text of the associated note. Table 175 shows the annotation dictionary
    entries specific to this type of annotation.
    """

    def __init__(self, start_point: typing.Tuple[Decimal, Decimal], end_point: typing.Tuple[Decimal, Decimal], left_line_end_style: LineEndStyleType=LineEndStyleType.NONE, right_line_end_style: LineEndStyleType=LineEndStyleType.NONE, stroke_color: Color=HexColor('000000')):
        if False:
            return 10
        x = min([start_point[0], end_point[0]])
        y = min([start_point[1], end_point[1]])
        w = max([start_point[0], end_point[0]]) - x
        h = max([start_point[1], end_point[1]]) - y
        super(LineAnnotation, self).__init__(bounding_box=Rectangle(x, y, w, h), color=stroke_color)
        self[Name('Subtype')] = Name('Line')
        self[Name('L')] = List().set_is_inline(True)
        self['L'].append(start_point[0])
        self['L'].append(start_point[1])
        self['L'].append(end_point[0])
        self['L'].append(end_point[1])
        self[Name('LE')] = List().set_is_inline(True)
        self['LE'].append(left_line_end_style.value)
        self['LE'].append(right_line_end_style)
        if stroke_color is not None:
            self[Name('IC')] = List().set_is_inline(True)
            self['IC'].append(bDecimal(stroke_color.to_rgb().red))
            self['IC'].append(bDecimal(stroke_color.to_rgb().green))
            self['IC'].append(bDecimal(stroke_color.to_rgb().blue))