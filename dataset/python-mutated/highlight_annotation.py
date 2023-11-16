"""
Text markup annotations shall appear as highlights, underlines, strikeouts (all PDF 1.3), or jagged (“squiggly”)
underlines (PDF 1.4) in the text of a document. When opened, they shall display a pop-up window containing
the text of the associated note. Table 179 shows the annotation dictionary entries specific to these types of
annotations.
"""
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class HighlightAnnotation(Annotation):
    """
    Text markup annotations shall appear as highlights, underlines, strikeouts (all PDF 1.3), or jagged (“squiggly”)
    underlines (PDF 1.4) in the text of a document. When opened, they shall display a pop-up window containing
    the text of the associated note. Table 179 shows the annotation dictionary entries specific to these types of
    annotations.
    """

    def __init__(self, bounding_box: Rectangle, color: Color=HexColor('faed27')):
        if False:
            for i in range(10):
                print('nop')
        super(HighlightAnnotation, self).__init__(bounding_box=bounding_box, color=color)
        self[Name('Subtype')] = Name('Highlight')
        self[Name('QuadPoints')] = List().set_is_inline(True)
        self['QuadPoints'].append(bDecimal(bounding_box.get_x()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_y()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_x()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_y() + bounding_box.get_height()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_x() + bounding_box.get_width()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_y()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_x() + bounding_box.get_width()))
        self['QuadPoints'].append(bDecimal(bounding_box.get_y() + bounding_box.get_height()))
        self[Name('Border')] = List().set_is_inline(True)
        self['Border'].append(bDecimal(0))
        self['Border'].append(bDecimal(0))
        self['Border'].append(bDecimal(1))