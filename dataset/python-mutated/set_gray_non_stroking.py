"""
Same as G but used for nonstroking operations.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.color.color import GrayColor
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetGrayNonStroking(CanvasOperator):
    """
    Same as G but used for nonstroking operations.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('g', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            print('Hello World!')
        '\n        Invoke the g operator\n        '
        assert isinstance(operands[0], Decimal), 'Operand 0 of g must be a Decimal'
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.non_stroke_color = GrayColor(operands[0])