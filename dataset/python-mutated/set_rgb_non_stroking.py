"""
Same as RG but used for nonstroking operations.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.color.color import RGBColor
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetRGBNonStroking(CanvasOperator):
    """
    Same as RG but used for nonstroking operations.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('rg', 3)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invoke the rg operator\n        '
        assert isinstance(operands[0], Decimal), 'operand 0 of rg operator must be of type Decimal'
        assert isinstance(operands[1], Decimal), 'operand 1 of rg operator must be of type Decimal'
        assert isinstance(operands[2], Decimal), 'operand 2 of rg operator must be of type Decimal'
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.non_stroke_color = RGBColor(operands[0], operands[1], operands[2])