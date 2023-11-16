"""
Show a text string.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.event.chunk_of_text_render_event import ChunkOfTextRenderEvent
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class ShowText(CanvasOperator):
    """
    Show a text string.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Tj', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the Tj operator\n        '
        assert isinstance(operands[0], String), 'Operand 0 of Tj must be a String'
        canvas = canvas_stream_processor.get_canvas()
        assert canvas.graphics_state.font is not None
        font_name: typing.Optional[Name] = None
        if isinstance(canvas.graphics_state.font, Name):
            font_name = canvas.graphics_state.font
            canvas.graphics_state.font = canvas_stream_processor.get_resource('Font', canvas.graphics_state.font)
        tri = ChunkOfTextRenderEvent(canvas.graphics_state, operands[0])
        for l in event_listeners:
            l._event_occurred(tri)
        canvas.graphics_state.text_matrix[2][0] += tri.get_baseline().width
        if font_name is not None:
            canvas.graphics_state.font = font_name