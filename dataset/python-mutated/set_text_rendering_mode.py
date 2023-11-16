"""
Set the text rendering mode, T mode , to render, which shall be an integer.
Initial value: 0.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetTextRenderingMode(CanvasOperator):
    """
    Set the text rendering mode, T mode , to render, which shall be an integer.
    Initial value: 0.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Tr', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Invoke the Tr operator\n        '
        pass