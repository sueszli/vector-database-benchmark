"""
This implementation of WriteBaseTransformer is responsible
for writing Dictionary objects of /Type /Page
"""
import logging
import typing
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.write.font.subsetter import Subsetter
from borb.io.write.object.dictionary_transformer import DictionaryTransformer
from borb.io.write.transformer import WriteTransformerState
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
logger = logging.getLogger(__name__)

class PageTransformer(DictionaryTransformer):
    """
    This implementation of WriteBaseTransformer is responsible
    for writing Dictionary objects of /Type /Page
    """

    def can_be_transformed(self, any: AnyPDFType):
        if False:
            return 10
        '\n        This function returns True if the object to be converted represents an /Page Dictionary\n        '
        return isinstance(any, Dictionary) and 'Type' in any and (any['Type'] == 'Page')

    def transform(self, object_to_transform: AnyPDFType, context: typing.Optional[WriteTransformerState]=None):
        if False:
            while True:
                i = 10
        '\n        This method writes a /Page Dictionary to a byte stream\n        '
        assert isinstance(object_to_transform, Dictionary)
        assert isinstance(object_to_transform, Page)
        assert context is not None, 'context must be defined in order to write Page objects.'
        assert context.root_object is not None, 'context.root_object must be defined in order to write Page objects.'
        assert isinstance(context.root_object, Document), 'context.root_object must be of type Document in order to write Page objects.'
        pages_dict = context.root_object['XRef']['Trailer']['Root']['Pages']
        object_to_transform[Name('Parent')] = self.get_reference(pages_dict, context)
        for k in ['ArtBox', 'BleedBox', 'CropBox', 'MediaBox', 'TrimBox']:
            if k in object_to_transform:
                object_to_transform[k].set_is_inline(True)
        if context.apply_font_subsetting:
            page = Subsetter.apply(object_to_transform)
        super(PageTransformer, self).transform(object_to_transform, context)