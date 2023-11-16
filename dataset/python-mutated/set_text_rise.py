"""
Set the text rise, T rise , to rise, which shall be a number expressed in
unscaled text space units. Initial value: 0.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetTextRise(CanvasOperator):
    """
    Set the text rise, T rise , to rise, which shall be a number expressed in
    unscaled text space units. Initial value: 0.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('Ts', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke the Ts operator\n        '
        assert isinstance(operands[0], Decimal)
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.text_rise = operands[0]