"""
This implementation of ReadBaseTransformer is responsible for reading Decimal objects
"""
import io
import typing
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Decimal as bDecimal
from borb.pdf.canvas.event.event_listener import EventListener

class NumberTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading Decimal objects
    """

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            while True:
                i = 10
        '\n        This function returns True if the object to be transformed is a Decimal object\n        '
        return isinstance(object, bDecimal)

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            print('Hello World!')
        '\n        This function reads a Decimal from a byte stream\n        '
        assert isinstance(object_to_transform, bDecimal), 'object_to_transform must be of type Decimal'
        return bDecimal(object_to_transform).set_parent(parent_object)