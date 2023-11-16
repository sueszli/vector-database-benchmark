"""
This implementation of WriteBaseTransformer is responsible for writing booleans
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Boolean
from borb.io.write.transformer import Transformer
from borb.io.write.transformer import WriteTransformerState

class BooleanTransformer(Transformer):
    """
    This implementation of WriteBaseTransformer is responsible for writing booleans
    """

    def can_be_transformed(self, any: AnyPDFType):
        if False:
            while True:
                i = 10
        '\n        This function returns True if the object to be converted represents a Boolean object\n        '
        return isinstance(any, Boolean)

    def transform(self, object_to_transform: AnyPDFType, context: typing.Optional[WriteTransformerState]=None):
        if False:
            i = 10
            return i + 15
        '\n        This method writes a Boolean to a byte stream\n        '
        assert context is not None, 'context must be defined to write bool objects'
        assert context.destination is not None, 'context.destination must be defined to write bool objects'
        assert isinstance(object_to_transform, Boolean), 'object_to_transform must be of type Boolean'
        if bool(object_to_transform):
            context.destination.write(bytes('true', 'latin1'))
        else:
            context.destination.write(bytes('false', 'latin1'))