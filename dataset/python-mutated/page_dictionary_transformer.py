"""
This implementation of ReadBaseTransformer is responsible for reading Page objects
"""
import io
import typing
import zlib
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Stream
from borb.pdf.canvas.canvas import Canvas
from borb.pdf.canvas.canvas_stream_processor import CanvasStreamProcessor
from borb.pdf.canvas.event.begin_page_event import BeginPageEvent
from borb.pdf.canvas.event.end_page_event import EndPageEvent
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.page.page import Page

class PageDictionaryTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading Page objects
    """

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            return 10
        '\n        This function returns True if the object to be converted represents a /Page Dictionary\n        '
        return isinstance(object, typing.Dict) and 'Type' in object and (object['Type'] == 'Page')

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            while True:
                i = 10
        '\n        This function reads a /Page Dictionary from a byte stream\n        '
        if isinstance(object_to_transform, Page):
            return object_to_transform
        page_out = Page()
        page_out.set_parent(parent_object)
        assert isinstance(object_to_transform, Dictionary), 'object_to_transform must be of type Dictionary'
        for (k, v) in object_to_transform.items():
            if k == 'Parent':
                continue
            v = self.get_root_transformer().transform(v, page_out, context, event_listeners)
            if v is not None:
                page_out[k] = v
        for l in event_listeners:
            l._event_occurred(BeginPageEvent(page_out))
        if 'Contents' not in page_out:
            return
        if not isinstance(page_out['Contents'], List) and (not isinstance(page_out['Contents'], Stream)):
            return
        contents = page_out['Contents']
        if isinstance(contents, List):
            bts = b''.join([x['DecodedBytes'] + b' ' for x in contents])
            page_out[Name('Contents')] = Stream()
            assert isinstance(page_out['Contents'], Stream)
            page_out['Contents'][Name('DecodedBytes')] = bts
            page_out['Contents'][Name('Bytes')] = zlib.compress(bts, 9)
            page_out['Contents'][Name('Filter')] = Name('FlateDecode')
            page_out['Contents'][Name('Length')] = bDecimal(len(bts))
            contents = page_out['Contents']
            contents.set_parent(page_out)
        canvas = Canvas().set_parent(page_out)
        if len(event_listeners) > 0:
            CanvasStreamProcessor(page_out, canvas, []).read(io.BytesIO(contents['DecodedBytes']), event_listeners)
        for l in event_listeners:
            l._event_occurred(EndPageEvent(page_out))
        return page_out