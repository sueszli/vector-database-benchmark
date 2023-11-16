"""
This implementation of ReadBaseTransformer is responsible for reading the /Catalog object
"""
import io
import typing
from borb.io.read.object.dictionary_transformer import DictionaryTransformer
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List as bList
from borb.io.read.types import Name
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.page.page import Page

class RootDictionaryTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading the /Catalog object
    """

    def _re_order_pages(self, root_dictionary: dict) -> None:
        if False:
            while True:
                i = 10
        pages_in_order: typing.List[Page] = []
        stack_to_handle: typing.List[AnyPDFType] = []
        stack_to_handle.append(root_dictionary['Pages'])
        while len(stack_to_handle) > 0:
            obj = stack_to_handle.pop(0)
            if isinstance(obj, Page):
                pages_in_order.append(obj)
            if isinstance(obj, Dictionary) and 'Type' in obj and (obj['Type'] == 'Pages') and ('Kids' in obj) and isinstance(obj['Kids'], typing.List):
                for k in obj['Kids']:
                    stack_to_handle.append(k)
        root_dictionary['Pages'][Name('Kids')] = bList()
        for p in pages_in_order:
            root_dictionary['Pages']['Kids'].append(p)
        root_dictionary['Pages'][Name('Count')] = bDecimal(len(pages_in_order))

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        This function returns True if the object to be converted represents a /Catalog Dictionary\n        '
        return isinstance(object, typing.Dict) and 'Type' in object and (object['Type'] == 'Catalog')

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        This function reads a /Catalog Dictionary from a byte stream\n        '
        assert isinstance(object_to_transform, Dictionary), 'object_to_transform must be of type Dictionary'
        transformed_root_dictionary: typing.Optional[Dictionary] = None
        for t in self.get_root_transformer().get_children():
            if isinstance(t, DictionaryTransformer):
                transformed_root_dictionary = t.transform(object_to_transform, parent_object, context, event_listeners)
                break
        assert transformed_root_dictionary is not None
        assert isinstance(transformed_root_dictionary, Dictionary)
        self._re_order_pages(transformed_root_dictionary)
        return transformed_root_dictionary