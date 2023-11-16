"""
(PDF 1.1) Begin a compatibility section. Unrecognized operators (along with
their operands) shall be ignored without error until the balancing EX operator
is encountered.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class EndCompatibilitySection(CanvasOperator):
    """
    (PDF 1.1) Begin a compatibility section. Unrecognized operators (along with
    their operands) shall be ignored without error until the balancing EX operator
    is encountered.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__('EX', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Invoke the EX operator\n        '
        canvas_stream_processor.get_canvas().in_compatibility_section = False