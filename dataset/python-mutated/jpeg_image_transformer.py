"""
This implementation of ReadBaseTransformer is responsible for reading a jpeg image object
"""
import io
import logging
import typing
from PIL import Image
from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import PDFObject
from borb.io.read.types import Stream
from borb.pdf.canvas.event.event_listener import EventListener
logger = logging.getLogger(__name__)

class JPEGImageTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading a jpeg image object
    """

    def can_be_transformed(self, object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType]) -> bool:
        if False:
            while True:
                i = 10
        '\n        This function returns True if the object to be transformed is a JPEG object\n        '
        return isinstance(object, Stream) and object.get('Type', None) in ['XObject', None] and (object.get('Subtype', None) in ['Image', None]) and ('Filter' in object) and (object['Filter'] == 'DCTDecode' or (isinstance(object['Filter'], list) and object['Filter'][0] == 'DCTDecode'))

    def transform(self, object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType], parent_object: typing.Any, context: typing.Optional[ReadTransformerState]=None, event_listeners: typing.List[EventListener]=[]) -> typing.Any:
        if False:
            while True:
                i = 10
        '\n        This function reads a JPEG Image from a byte stream\n        '
        assert isinstance(object_to_transform, Stream), 'object_to_transform must be of type Stream'
        if object_to_transform.get('Type', None) is None:
            logger.debug('Image object did not specify /Type /XObject.')
        if object_to_transform.get('Subtype', None) is None:
            logger.debug('Image object did not specify /Subtype /Image.')
        try:
            tmp = Image.open(io.BytesIO(object_to_transform['Bytes']))
            tmp.getpixel((0, 0))
        except:
            logger.debug('Unable to read jbig2 image. Constructing empty image of same dimensions.')
            w = int(object_to_transform['Width'])
            h = int(object_to_transform['Height'])
            tmp = Image.new('RGB', (w, h), (128, 128, 128))
        PDFObject.add_pdf_object_methods(tmp)
        tmp.set_parent(parent_object)
        return tmp