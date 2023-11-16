"""
Close and stroke the path. This operator shall have the same effect as the
sequence h S.
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class CloseAndStrokePath(CanvasOperator):
    """
    Close and stroke the path. This operator shall have the same effect as the
    sequence h S.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('s', 0)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            print('Hello World!')
        '\n        Invoke the s operator\n        '
        close_subpath_op: typing.Optional[CanvasOperator] = canvas_stream_processor.get_operator('h')
        assert close_subpath_op
        close_subpath_op.invoke(canvas_stream_processor, [], event_listeners)
        stroke_path_op: typing.Optional[CanvasOperator] = canvas_stream_processor.get_operator('S')
        assert stroke_path_op
        stroke_path_op.invoke(canvas_stream_processor, [], event_listeners)