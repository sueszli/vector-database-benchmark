"""
An operator in a programming language is a symbol that tells the compiler or interpreter to perform specific mathematical,
relational or logical operation and produce final result.
CanvasOperator defines an interface to work on Canvas objects. Typically these operators involve drawing graphics, text,
setting the active color and so on
"""
import typing
from borb.io.read.types import AnyPDFType

class CanvasOperator:
    """
    An operator in a programming language is a symbol that tells the compiler or interpreter to perform specific mathematical,
    relational or logical operation and produce final result.
    CanvasOperator defines an interface to work on Canvas objects. Typically these operators involve drawing graphics, text,
    setting the active color and so on
    """

    def __init__(self, text: str, number_of_operands: int):
        if False:
            while True:
                i = 10
        self._text: str = text
        self._number_of_operands: int = number_of_operands

    def get_number_of_operands(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return the number of operands for this CanvasOperator\n        '
        return self._number_of_operands

    def get_text(self) -> str:
        if False:
            return 10
        '\n        Return the str that invokes this CanvasOperator\n        '
        return self._text

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Invokes this CanvasOperator\n        '
        pass