"""
An ink annotation (PDF 1.3) represents a freehand “scribble” composed of one or more disjoint paths. When
opened, it shall display a pop-up window containing the text of the associated note. Table 182 shows the
annotation dictionary entries specific to this type of annotation.
"""
import typing
from _decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import List as bList
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class InkAnnotation(Annotation):
    """
    An ink annotation (PDF 1.3) represents a freehand “scribble” composed of one or more disjoint paths. When
    opened, it shall display a pop-up window containing the text of the associated note. Table 182 shows the
    annotation dictionary entries specific to this type of annotation.
    """

    def __init__(self, points: typing.List[typing.Tuple[Decimal, Decimal]], color: typing.Optional[Color]=None, line_width: typing.Optional[Decimal]=None):
        if False:
            while True:
                i = 10
        super(InkAnnotation, self).__init__(bounding_box=Rectangle(min([x for (x, y) in points]), min([y for (x, y) in points]), max([x for (x, y) in points]) - min([x for (x, y) in points]), max([y for (x, y) in points]) - min([y for (x, y) in points])), color=color)
        self[Name('Subtype')] = Name('Ink')
        self[Name('InkList')] = bList().set_is_inline(True)
        self['InkList'].append(bList().set_is_inline(True))
        for p in points:
            self['InkList'][0].append(bDecimal(p[0]))
            self['InkList'][0].append(bDecimal(p[1]))
        if line_width is not None:
            self[Name('Border')] = bList().set_is_inline(True)
            self['Border'].append(bDecimal(0))
            self['Border'].append(bDecimal(0))
            self['Border'].append(bDecimal(line_width))