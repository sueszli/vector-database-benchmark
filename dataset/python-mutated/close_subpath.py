"""
Close the current subpath by appending a straight line
segment from the current point to the starting point of the
subpath. If the current subpath is already closed, h shall do
nothing.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.geometry.line_segment import LineSegment
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class CloseSubpath(CanvasOperator):
    """
    Close the current subpath by appending a straight line
    segment from the current point to the starting point of the
    subpath. If the current subpath is already closed, h shall do
    nothing.

    This operator terminates the current subpath. Appending
    another segment to the current path shall begin a new
    subpath, even if the new segment begins at the endpoint
    reached by the h operation.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('h', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the h operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        gs = canvas.graphics_state
        if len(gs.path) == 0:
            return
        x0 = gs.path[0].x0
        y0 = gs.path[0].y0
        xn = gs.path[-1].x1
        yn = gs.path[-1].y1
        gs.path.append(LineSegment(x0, y0, xn, yn))