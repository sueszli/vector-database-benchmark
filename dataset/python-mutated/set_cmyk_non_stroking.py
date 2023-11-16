"""
Same as K but used for nonstroking operations.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.color.color import CMYKColor
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetCMYKNonStroking(CanvasOperator):
    """
    Same as K but used for nonstroking operations.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('k', 4)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Invoke the k operator\n        '
        assert isinstance(operands[0], Decimal), 'Operand 0 of k must be a Decimal'
        assert isinstance(operands[1], Decimal), 'Operand 1 of k must be a Decimal'
        assert isinstance(operands[2], Decimal), 'Operand 2 of k must be a Decimal'
        assert isinstance(operands[3], Decimal), 'Operand 3 of k must be a Decimal'
        c = operands[0]
        m = operands[1]
        y = operands[2]
        k = operands[3]
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.non_stroke_color = CMYKColor(c, m, y, k)