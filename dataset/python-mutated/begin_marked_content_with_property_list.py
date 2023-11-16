"""
Begin a marked-content sequence with an associated property list,
terminated by a balancing EMC operator. tag shall be a name object
indicating the role or significance of the sequence. properties shall be
either an inline dictionary containing the property list or a name object
associated with it in the Properties subdictionary of the current resource
dictionary (see 14.6.2, “Property Lists”).
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Name
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class BeginMarkedContentWithPropertyList(CanvasOperator):
    """
    Begin a marked-content sequence with an associated property list,
    terminated by a balancing EMC operator. tag shall be a name object
    indicating the role or significance of the sequence. properties shall be
    either an inline dictionary containing the property list or a name object
    associated with it in the Properties subdictionary of the current resource
    dictionary (see 14.6.2, “Property Lists”).
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('BDC', 2)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invoke the BDC operator\n        '
        assert isinstance(operands[0], Name), 'Operand 0 of BDC must be a Name'
        canvas = canvas_stream_processor.get_canvas()
        canvas.marked_content_stack.append(operands[0])