"""
A link annotation represents either a hypertext link to a destination elsewhere in the document (see 12.3.2,
“Destinations”) or an action to be performed (12.6, “Actions”). Table 173 shows the annotation dictionary
entries specific to this type of annotation.
"""
import enum
import typing
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class DestinationType(enum.Enum):
    """
    This Enum represents all possible destination types (when adding a link annotation)
    """
    FIT = Name('Fit')
    FIT_B = Name('FitB')
    FIT_B_H = Name('FitBH')
    FIT_B_V = Name('FitBV')
    FIT_H = Name('FitH')
    FIT_R = Name('FitR')
    FIT_V = Name('FitV')
    X_Y_Z = Name('XYZ')

class LinkAnnotation(Annotation):
    """
    A link annotation represents either a hypertext link to a destination elsewhere in the document (see 12.3.2,
    “Destinations”) or an action to be performed (12.6, “Actions”). Table 173 shows the annotation dictionary
    entries specific to this type of annotation.
    """

    def __init__(self, bounding_box: Rectangle, page: Decimal, destination_type: DestinationType, color: typing.Optional[Color]=None, top: typing.Optional[Decimal]=None, right: typing.Optional[Decimal]=None, bottom: typing.Optional[Decimal]=None, left: typing.Optional[Decimal]=None, zoom: typing.Optional[Decimal]=None, highlighting_mode: typing.Optional[str]=None):
        if False:
            print('Hello World!')
        super(LinkAnnotation, self).__init__(bounding_box=bounding_box, color=color)
        self[Name('Subtype')] = Name('Link')
        destination = List()
        destination.set_is_inline(True)
        destination.append(bDecimal(page))
        destination.append(destination_type.value)
        if destination_type == DestinationType.X_Y_Z:
            assert left is not None and bottom is None and (right is None) and (top is not None) and (zoom is not None)
            destination.append(bDecimal(left))
            destination.append(bDecimal(top))
            destination.append(bDecimal(zoom))
        if destination_type == DestinationType.FIT:
            assert left is None and bottom is None and (right is None) and (top is None) and (zoom is None)
        if destination_type == DestinationType.FIT_H:
            assert left is None and bottom is None and (right is None) and (top is not None) and (zoom is None)
            destination.append(bDecimal(top))
        if destination_type == DestinationType.FIT_V:
            assert left is not None and bottom is None and (right is None) and (top is None) and (zoom is None)
            destination.append(bDecimal(left))
        if destination_type == DestinationType.FIT_R:
            assert left is not None and bottom is not None and (right is not None) and (top is not None) and (zoom is None)
            destination.append(bDecimal(left))
            destination.append(bDecimal(bottom))
            destination.append(bDecimal(right))
            destination.append(bDecimal(top))
        if destination_type == DestinationType.FIT_B_H:
            assert left is None and bottom is None and (right is None) and (top is not None) and (zoom is None)
            destination.append(bDecimal(top))
        if destination_type == DestinationType.FIT_B_V:
            assert left is not None and bottom is None and (right is None) and (top is None) and (zoom is None)
            destination.append(bDecimal(left))
        self[Name('Dest')] = destination
        if highlighting_mode is not None:
            assert highlighting_mode in ['N', 'I', 'O', 'P']
            self[Name('H')] = String(highlighting_mode)