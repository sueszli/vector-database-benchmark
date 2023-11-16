"""
Restore the graphics state by removing the most recently saved
state from the stack and making it the current state (see 8.4.2,
"Graphics State Stack").
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class PopGraphicsState(CanvasOperator):
    """
    Restore the graphics state by removing the most recently saved
    state from the stack and making it the current state (see 8.4.2,
    "Graphics State Stack").
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Q', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            print('Hello World!')
        '\n        Invoke the Q operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        assert len(canvas.graphics_state_stack) > 0, 'Stack underflow. Q operator was applied to an empty stack.'
        canvas.graphics_state = canvas.graphics_state_stack.pop(-1)