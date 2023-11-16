"""
This CanvasOperator copies an existing operator and writes its bytes to a content stream of the canvas.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.io.read.types import HexadecimalString
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class CopyCommandOperator(CanvasOperator):
    """
    This CanvasOperator copies an existing operator and writes its bytes to a content stream of the canvas.
    """

    def __init__(self, operator_to_copy: CanvasOperator, output_content_stream: bytearray):
        if False:
            while True:
                i = 10
        super().__init__('', 0)
        self._operator_to_copy = operator_to_copy
        self._output_content_stream: bytearray = output_content_stream

    def _operand_to_str(self, op: AnyPDFType) -> str:
        if False:
            i = 10
            return i + 15
        if isinstance(op, Decimal):
            return str(op)
        if isinstance(op, HexadecimalString):
            return '<' + str(op) + '>'
        if isinstance(op, String):
            return '(' + str(op) + ')'
        if isinstance(op, Name):
            return '/' + str(op)
        if isinstance(op, list):
            return '[' + ''.join([self._operand_to_str(x) + ' ' for x in op])[:-1] + ']'
        return ''

    def get_number_of_operands(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return the number of operands for this CanvasOperator\n        '
        return self._operator_to_copy.get_number_of_operands()

    def get_text(self) -> str:
        if False:
            return 10
        '\n        Return the str that invokes this CanvasOperator\n        '
        return self._operator_to_copy.get_text()

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invokes this CanvasOperator\n        '
        self._operator_to_copy.invoke(canvas_stream_processor, operands)
        canvas = canvas_stream_processor.get_canvas()
        self._output_content_stream += b'\n'
        self._output_content_stream += b''.join([bytes(self._operand_to_str(s), encoding='utf8') + b' ' for s in operands])
        self._output_content_stream += bytes(self.get_text(), encoding='utf8')