"""
Set the character spacing, Tc , to charSpace, which shall be a number
expressed in unscaled text space units. Character spacing shall be used
by the Tj, TJ, and ' operators. Initial value: 0.
"""
import typing
from decimal import Decimal
from borb.io.read.types import AnyPDFType
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class SetCharacterSpacing(CanvasOperator):
    """
    Set the character spacing, Tc , to charSpace, which shall be a number
    expressed in unscaled text space units. Character spacing shall be used
    by the Tj, TJ, and ' operators. Initial value: 0.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('Tc', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Invoke the Tc operator\n        '
        assert isinstance(operands[0], Decimal), 'Operand 0 of Tc must be a Decimal'
        canvas = canvas_stream_processor.get_canvas()
        canvas.graphics_state.character_spacing = operands[0]