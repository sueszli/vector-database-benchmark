"""
This implementation of ReadBaseTransformer is responsible for reading String objects
"""
import io
import typing
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import HexadecimalString
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.event.event_listener import EventListener

class StringTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading String objects
    """

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            return 10
        '\n        This function returns True if the object to be converted represents a String or Hexadecimal String or a Name\n        '
        return isinstance(object, String) or isinstance(object, HexadecimalString) or isinstance(object, Name)

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            while True:
                i = 10
        '\n        This function reads a String from a byte stream\n        '
        assert isinstance(object_to_transform, String) or isinstance(object_to_transform, HexadecimalString) or isinstance(object_to_transform, Name)
        object_to_transform.set_parent(parent_object)
        return object_to_transform