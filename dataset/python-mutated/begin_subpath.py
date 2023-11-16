"""
Begin a new subpath by moving the current point to
coordinates (x, y), omitting any connecting line segment. If
the previous path construction operator in the current path
was also m, the new m overrides it; no vestige of the
previous m operation remains in the path.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.geometry.line_segment import LineSegment
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class BeginSubpath(CanvasOperator):
    """
    Begin a new subpath by moving the current point to
    coordinates (x, y), omitting any connecting line segment. If
    the previous path construction operator in the current path
    was also m, the new m overrides it; no vestige of the
    previous m operation remains in the path.
    """

    def __init__(self):
        if False:
            return 10
        super().__init__('m', 2)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            print('Hello World!')
        '\n        Invoke the m operator\n        '
        assert isinstance(operands[0], Decimal), 'operand 0 of m operator must be of type Decimal'
        assert isinstance(operands[1], Decimal), 'operand 1 of m operator must be of type Decimal'
        canvas = canvas_stream_processor.get_canvas()
        gs = canvas.graphics_state
        gs.path.append(LineSegment(operands[0], operands[1], operands[0], operands[1]))