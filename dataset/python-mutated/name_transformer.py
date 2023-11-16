"""
This implementation of WriteBaseTransformer is responsible for writing Name objects
"""
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Name
from borb.io.write.transformer import Transformer
from borb.io.write.transformer import WriteTransformerState

class NameTransformer(Transformer):
    """
    This implementation of WriteBaseTransformer is responsible for writing Name objects
    """

    def can_be_transformed(self, any: AnyPDFType):
        if False:
            print('Hello World!')
        '\n        This function returns True if the object to be converted represents a Name object\n        '
        return isinstance(any, Name)

    def transform(self, object_to_transform: AnyPDFType, context: typing.Optional[WriteTransformerState]=None):
        if False:
            while True:
                i = 10
        '\n        This method writes a Name to a byte stream\n        '
        assert context is not None
        assert context.destination is not None
        assert isinstance(object_to_transform, Name)
        context.destination.write(bytes('/' + str(object_to_transform), 'latin1'))