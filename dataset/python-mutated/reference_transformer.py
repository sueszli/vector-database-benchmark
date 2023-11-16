"""
This implementation of ReadBaseTransformer is responsible for reading Reference objects
e.g. 97 0 R
"""
import io
import logging
import typing
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Reference
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.xref.xref import XREF
logger = logging.getLogger(__name__)

class ReferenceTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading Reference objects
    e.g. 97 0 R
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super(ReferenceTransformer, self).__init__()
        self._cache: typing.Dict[Reference, AnyPDFType] = {}
        self._cache_hits: int = 0
        self._cache_fails: int = 0

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns True if the object to be converted represents a Reference\n        '
        return isinstance(object, Reference)

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            while True:
                i = 10
        '\n        This function reads a Reference from a byte stream\n        '
        assert isinstance(object_to_transform, Reference), 'object_to_transform must be of type Reference'
        assert context is not None
        if object_to_transform in context.indirect_reference_chain:
            return None
        ref_from_cache = self._cache.get(object_to_transform, None)
        if ref_from_cache is not None:
            self._cache_hits += 1
            if ref_from_cache.get_parent() is None:
                ref_from_cache.set_parent(parent_object)
                return ref_from_cache
            if ref_from_cache.get_parent() != parent_object:
                ref_from_cache_copy = ref_from_cache
                ref_from_cache_copy.set_parent(parent_object)
                return ref_from_cache_copy
        self._cache_fails += 1
        logger.debug('ref. cache hits: %d, fails: %d, ratio %f' % (self._cache_hits, self._cache_fails, self._cache_hits / (self._cache_hits + self._cache_fails)))
        assert context.root_object is not None, 'context.root_object must be defined to read Reference objects'
        assert context.root_object['XRef'] is not None, 'XREF must be defined to read Reference objects'
        assert isinstance(context.root_object['XRef'], XREF), 'XREF must be defined to read Reference objects'
        assert context.tokenizer is not None, 'context.tokenizer must be defined to read Reference objects'
        assert context.source is not None, 'context.source must be defined to read Reference objects'
        xref = context.root_object['XRef']
        src = context.source
        tok = context.tokenizer
        referenced_object = xref.get_object(object_to_transform, src, tok)
        if referenced_object is None:
            return None
        assert referenced_object is not None
        context.indirect_reference_chain.add(object_to_transform)
        transformed_referenced_object = self.get_root_transformer().transform(referenced_object, parent_object, context, event_listeners)
        context.indirect_reference_chain.remove(object_to_transform)
        if transformed_referenced_object is not None:
            self._cache[object_to_transform] = transformed_referenced_object
        try:
            transformed_referenced_object.set_reference(object_to_transform)
        except:
            logger.debug('Unable to set reference on object %s' % str(transformed_referenced_object))
            pass
        return transformed_referenced_object