"""
A rubber stamp annotation (PDF 1.3) displays text or graphics intended to look as if they were stamped on the
page with a rubber stamp. When opened, it shall display a pop-up window containing the text of the associated
note. Table 181 shows the annotation dictionary entries specific to this type of annotation.
"""
import enum
import typing
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Name
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class RubberStampAnnotationIconType(enum.Enum):
    """
    This Enum represents all possible rubber stamp annotation icons
    """
    APPROVED = Name('Approved')
    AS_IS = Name('AsIs')
    CONFIDENTIAL = Name('Confidential')
    DEPARTMENTAL = Name('Departmental')
    DRAFT = Name('Draft')
    EXPERIMENTAL = Name('Experimental')
    EXPIRED = Name('Expired')
    FINAL = Name('Final')
    FOR_COMMENT = Name('ForComment')
    FOR_PUBLIC_RELEASE = Name('ForPublicRelease')
    NOT_APPROVED = Name('NotApproved')
    NOT_FOR_PUBLIC_RELEASE = Name('NotForPublicRelease')
    SOLD = Name('Sold')
    TOP_SECRET = Name('TopSecret')

class RubberStampAnnotation(Annotation):
    """
    A rubber stamp annotation (PDF 1.3) displays text or graphics intended to look as if they were stamped on the
    page with a rubber stamp. When opened, it shall display a pop-up window containing the text of the associated
    note. Table 181 shows the annotation dictionary entries specific to this type of annotation.
    """

    def __init__(self, bounding_box: Rectangle, name: RubberStampAnnotationIconType=RubberStampAnnotationIconType.DRAFT, contents: typing.Optional[str]=None, color: typing.Optional[Color]=None):
        if False:
            i = 10
            return i + 15
        super(RubberStampAnnotation, self).__init__(bounding_box=bounding_box, contents=contents, color=color)
        self[Name('Subtype')] = Name('Stamp')
        self[Name('Name')] = name.value
        self[Name('CA')] = bDecimal(1)