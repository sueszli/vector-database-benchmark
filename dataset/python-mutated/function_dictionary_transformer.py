"""
This implementation of ReadBaseTransformer is responsible for reading a Function Dictionary
"""
import io
import typing
from decimal import Decimal
from borb.io.filter.stream_decode_util import decode_stream
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Dictionary
from borb.io.read.types import Function
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.io.read.types import Stream
from borb.pdf.canvas.event.event_listener import EventListener

class FunctionDictionaryTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading a Function Dictionary
    """

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            return 10
        '\n        This function returns True if the object to be transformed is a Dictionary with /FunctionType key\n        '
        return isinstance(object, dict) and 'FunctionType' in object and isinstance(object['FunctionType'], Decimal) and (int(object['FunctionType']) in [0, 2, 3, 4])

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function reads a Dictionary with /FunctionType key from a byte stream.\n        '
        assert isinstance(object_to_transform, Dictionary), 'object_to_transform must be of type Dictionary.'
        assert 'FunctionType' in object_to_transform, 'object_to_transform Dictionary must be FunctionType.'
        assert isinstance(object_to_transform['FunctionType'], Decimal), 'object_to_transform must contain a valid /FunctionType entry.'
        function_type: int = int(object_to_transform['FunctionType'])
        assert function_type in [0, 2, 3, 4], 'FunctionType must be in [0, 2, 3, 4]'
        transformed_object: Function = Function()
        if isinstance(object_to_transform, Stream):
            decode_stream(object_to_transform)
            transformed_object[Name('Bytes')] = object_to_transform['Bytes']
            transformed_object[Name('DecodedBytes')] = object_to_transform['DecodedBytes']
        assert context is not None, 'context must be defined to read (Function) Dictionary objects'
        assert context.tokenizer is not None, 'context.tokenizer must be defined to read (Function) Dictionary objects'
        xref = parent_object.get_root().get('XRef')
        for (k, v) in object_to_transform.items():
            if isinstance(v, Reference):
                v = xref.get_object(v, context.source, context.tokenizer)
                transformed_object[k] = v
        for (k, v) in object_to_transform.items():
            if not isinstance(v, Reference):
                v = self.get_root_transformer().transform(v, transformed_object, context, [])
                if v is not None:
                    transformed_object[k] = v
        transformed_object.set_parent(parent_object)
        return transformed_object