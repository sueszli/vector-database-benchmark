"""
This implementation of WriteBaseTransformer is responsible
for writing Dictionary objects of /Type /Pages
"""
import logging
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.write.object.dictionary_transformer import DictionaryTransformer
from borb.io.write.transformer import WriteTransformerState
logger = logging.getLogger(__name__)

class PagesTransformer(DictionaryTransformer):
    """
    This implementation of WriteBaseTransformer is responsible
    for writing Dictionary objects of /Type /Pages
    """

    def can_be_transformed(self, any: AnyPDFType):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns True if the object to be converted represents a /Pages Dictionary\n        '
        return isinstance(any, Dictionary) and 'Type' in any and (any['Type'] == 'Pages')

    def transform(self, object_to_transform: AnyPDFType, context: typing.Optional[WriteTransformerState]=None):
        if False:
            while True:
                i = 10
        '\n        This method writes a /Pages Dictionary to a byte stream\n        '
        assert isinstance(object_to_transform, Dictionary)
        assert context is not None, 'A WriteTransformerState must be defined in order to write Pages Dictionary objects.'
        object_to_transform[Name('Kids')].set_is_inline(True)
        queue: typing.List[AnyPDFType] = []
        for (i, k) in enumerate(object_to_transform['Kids']):
            queue.append(k)
            object_to_transform['Kids'][i] = self.get_reference(k, context)
        super(PagesTransformer, self).transform(object_to_transform, context)
        for p in queue:
            self.get_root_transformer().transform(p, context)
        for (i, k) in enumerate(queue):
            object_to_transform['Kids'][i] = k