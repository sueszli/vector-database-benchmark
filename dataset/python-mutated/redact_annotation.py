"""
A redaction annotation (PDF 1.7) identifies content that is intended to be removed from the document. The
intent of redaction annotations is to enable the following process:

a) Content identification. A user applies redact annotations that specify the pieces or regions of content that
should be removed. Up until the next step is performed, the user can see, move and redefine these
annotations.

b) Content removal. The user instructs the viewer application to apply the redact annotations, after which the
content in the area specified by the redact annotations is removed. In the removed content’s place, some
marking appears to indicate the area has been redacted. Also, the redact annotations are removed from
the PDF document.

Redaction annotations provide a mechanism for the first step in the redaction process (content identification).
This allows content to be marked for redaction in a non-destructive way, thus enabling a review process for
evaluating potential redactions prior to removing the specified content.
"""
import typing
import zlib
from decimal import Decimal
from borb.io.read.types import Boolean
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Stream
from borb.io.read.types import String
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class RedactAnnotation(Annotation):
    """
    A redaction annotation (PDF 1.7) identifies content that is intended to be removed from the document. The
    intent of redaction annotations is to enable the following process:

    a) Content identification. A user applies redact annotations that specify the pieces or regions of content that
    should be removed. Up until the next step is performed, the user can see, move and redefine these
    annotations.

    b) Content removal. The user instructs the viewer application to apply the redact annotations, after which the
    content in the area specified by the redact annotations is removed. In the removed content’s place, some
    marking appears to indicate the area has been redacted. Also, the redact annotations are removed from
    the PDF document.

    Redaction annotations provide a mechanism for the first step in the redaction process (content identification).
    This allows content to be marked for redaction in a non-destructive way, thus enabling a review process for
    evaluating potential redactions prior to removing the specified content.
    """

    def __init__(self, bounding_box: Rectangle, fill_color: typing.Optional[Color]=None, stroke_color: typing.Optional[Color]=None, stroke_width: Decimal=Decimal(1), overlay_text: typing.Optional[str]=None, repeat_overlay_text: typing.Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        super(RedactAnnotation, self).__init__(bounding_box=bounding_box)
        self[Name('Subtype')] = Name('Redact')
        if fill_color is not None:
            self[Name('IC')] = List().set_is_inline(True)
            self['IC'].append(bDecimal(fill_color.to_rgb().red))
            self['IC'].append(bDecimal(fill_color.to_rgb().green))
            self['IC'].append(bDecimal(fill_color.to_rgb().blue))
        if overlay_text is not None:
            self[Name('OverlayText')] = String(overlay_text)
        if repeat_overlay_text is not None:
            assert overlay_text is not None
            self[Name('Repeat')] = Boolean(repeat_overlay_text)
        self[Name('AP')] = Dictionary()
        self['AP'][Name('N')] = Stream()
        self['AP']['N'][Name('Type')] = Name('XObject')
        self['AP']['N'][Name('Subtype')] = Name('Form')
        appearance_stream_content = 'q'
        if stroke_color is not None:
            appearance_stream_content += ' %f %f %f RG' % (stroke_color.to_rgb().red, stroke_color.to_rgb().green, stroke_color.to_rgb().blue)
        if fill_color is not None:
            appearance_stream_content += ' %f %f %f rg' % (fill_color.to_rgb().red, fill_color.to_rgb().green, fill_color.to_rgb().blue)
        if stroke_color is not None and fill_color is not None:
            appearance_stream_content += ' %f w 0 0 100 100 re b' % stroke_width
        elif stroke_color is not None:
            appearance_stream_content += ' %f w 0 0 100 100 re s' % stroke_width
        elif fill_color is not None:
            appearance_stream_content += ' %f w 0 0 100 100 re f' % stroke_width
        appearance_stream_content += ' Q'
        self['AP']['N'][Name('DecodedBytes')] = bytes(appearance_stream_content, 'latin1')
        self['AP']['N'][Name('Bytes')] = zlib.compress(self['AP']['N'][Name('DecodedBytes')])
        self['AP']['N'][Name('Length')] = bDecimal(len(self['AP']['N'][Name('Bytes')]))
        self['AP']['N'][Name('Filter')] = Name('FlateDecode')
        self['AP']['N'][Name('BBox')] = List().set_is_inline(True)
        self['AP']['N']['BBox'].append(bDecimal(0))
        self['AP']['N']['BBox'].append(bDecimal(0))
        self['AP']['N']['BBox'].append(bDecimal(100))
        self['AP']['N']['BBox'].append(bDecimal(100))