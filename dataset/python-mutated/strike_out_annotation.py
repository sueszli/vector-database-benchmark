"""
Text markup annotations shall appear as highlights, underlines, strikeouts (all PDF 1.3), or jagged (“squiggly”)
underlines (PDF 1.4) in the text of a document. When opened, they shall display a pop-up window containing
the text of the associated note. Table 179 shows the annotation dictionary entries specific to these types of
annotations.
"""
import zlib
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Stream
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class StrikeOutAnnotation(Annotation):
    """
    Text markup annotations shall appear as highlights, underlines, strikeouts (all PDF 1.3), or jagged (“squiggly”)
    underlines (PDF 1.4) in the text of a document. When opened, they shall display a pop-up window containing
    the text of the associated note. Table 179 shows the annotation dictionary entries specific to these types of
    annotations.
    """

    def __init__(self, bounding_box: Rectangle, stroke_color: Color=HexColor('ff0000'), stroke_width: Decimal=Decimal(1)):
        if False:
            return 10
        super(StrikeOutAnnotation, self).__init__(bounding_box=bounding_box)
        self[Name('Subtype')] = Name('StrikeOut')
        self[Name('AP')] = Dictionary()
        self['AP'][Name('N')] = Stream()
        self['AP']['N'][Name('Type')] = Name('XObject')
        self['AP']['N'][Name('Subtype')] = Name('Form')
        appearance_stream_content = 'q %f %f %f RG %f w 0 40 m %f 40 l S Q' % (stroke_color.to_rgb().red, stroke_color.to_rgb().green, stroke_color.to_rgb().blue, stroke_width, bounding_box.width)
        self['AP']['N'][Name('DecodedBytes')] = bytes(appearance_stream_content, 'latin1')
        self['AP']['N'][Name('Bytes')] = zlib.compress(self['AP']['N'][Name('DecodedBytes')])
        self['AP']['N'][Name('Length')] = bDecimal(len(self['AP']['N'][Name('Bytes')]))
        self['AP']['N'][Name('Filter')] = Name('FlateDecode')
        self['AP']['N'][Name('BBox')] = List().set_is_inline(True)
        self['AP']['N']['BBox'].append(bDecimal(0))
        self['AP']['N']['BBox'].append(bDecimal(0))
        self['AP']['N']['BBox'].append(bDecimal(bounding_box.width))
        self['AP']['N']['BBox'].append(bDecimal(100))