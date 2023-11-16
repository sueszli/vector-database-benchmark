"""
Set the stroking colour space to DeviceRGB (or the DefaultRGB colour
space; see 8.6.5.6, "Default Colour Spaces") and set the colour to use for
stroking operations. Each operand shall be a number between 0.0
(minimum intensity) and 1.0 (maximum intensity).
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.color.color import RGBColor
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetRGBStroking(CanvasOperator):
    """
    Set the stroking colour space to DeviceRGB (or the DefaultRGB colour
    space; see 8.6.5.6, "Default Colour Spaces") and set the colour to use for
    stroking operations. Each operand shall be a number between 0.0
    (minimum intensity) and 1.0 (maximum intensity).
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('RG', 3)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invoke the RG operator\n        '
        assert isinstance(operands[0], Decimal), 'operand 0 of rg operator must be of type Decimal'
        assert isinstance(operands[1], Decimal), 'operand 1 of rg operator must be of type Decimal'
        assert isinstance(operands[2], Decimal), 'operand 2 of rg operator must be of type Decimal'
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.stroke_color = RGBColor(operands[0], operands[1], operands[2])