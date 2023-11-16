"""
Append a straight line segment from the current point to the
point (x, y). The new current point shall be (x, y).
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.geometry.line_segment import LineSegment
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class AppendLineSegment(CanvasOperator):
    """
    Append a straight line segment from the current point to the
    point (x, y). The new current point shall be (x, y).
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('l', 2)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Invokes the l operator\n        '
        assert isinstance(operands[0], Decimal), 'operand 0 of l operator must be of type Decimal'
        assert isinstance(operands[1], Decimal), 'operand 1 of l operator must be of type Decimal'
        canvas = canvas_stream_processor.get_canvas()
        gs = canvas.graphics_state
        assert len(gs.path) > 0
        x0 = gs.path[-1].x1
        y0 = gs.path[-1].y1
        gs.path.append(LineSegment(x0, y0, operands[0], operands[1]))