"""
Save the current graphics state on the graphics state stack (see
8.4.2, "Graphics State Stack").
"""
import copy
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class PushGraphicsState(CanvasOperator):
    """
    Save the current graphics state on the graphics state stack (see
    8.4.2, "Graphics State Stack").
    """

    def __init__(self):
        if False:
            return 10
        super().__init__('q', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the q operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state_stack.append(copy.deepcopy(canvas.graphics_state))