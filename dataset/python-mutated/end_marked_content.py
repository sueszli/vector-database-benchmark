"""
End a marked-content sequence begun by a BMC or BDC operator.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class EndMarkedContent(CanvasOperator):
    """
    End a marked-content sequence begun by a BMC or BDC operator.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('EMC', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invoke the EMC operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        assert len(canvas.marked_content_stack) > 0
        canvas.marked_content_stack.pop(-1)