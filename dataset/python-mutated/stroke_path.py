"""
Stroke the path.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.event.line_render_event import LineRenderEvent
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class StrokePath(CanvasOperator):
    """
    Stroke the path.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__('S', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the S operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        gs = canvas.graphics_state
        for el in event_listeners:
            for l in gs.path:
                el._event_occurred(LineRenderEvent(gs, l))
        gs.path = []