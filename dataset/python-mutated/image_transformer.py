"""
This implementation of WriteBaseTransformer is responsible for writing Image objects
"""
import io
import typing
from PIL import Image as PILImage
from borb.io.read.pdf_object import PDFObject
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.io.read.types import Stream
from borb.io.write.transformer import Transformer
from borb.io.write.transformer import WriteTransformerState

class ImageTransformer(Transformer):
    """
    This implementation of WriteBaseTransformer is responsible for writing Image objects
    """

    def _convert_to_rgb_mode(self, image: PILImage.Image) -> PILImage.Image:
        if False:
            i = 10
            return i + 15
        image_out: PILImage.Image = image
        if image_out.mode == 'P':
            image_out = image_out.convert('RGBA')
        if image_out.mode == 'LA':
            image_out = image_out.convert('RGBA')
        if image_out.mode == 'RGBA':
            fill_color = (255, 255, 255)
            non_alpha_mode: str = image_out.mode[:-1]
            background = PILImage.new(non_alpha_mode, image_out.size, fill_color)
            background.paste(image_out, mask=image_out.split()[-1])
            image_out = background
        image_out = image_out.convert('RGB')
        PDFObject.add_pdf_object_methods(image_out)
        image_out.set_reference(image.get_reference())
        return image_out

    def can_be_transformed(self, any: AnyPDFType):
        if False:
            return 10
        '\n        This function returns True if the object to be converted represents an Image object\n        '
        return isinstance(any, PILImage.Image)

    def transform(self, object_to_transform: AnyPDFType, context: typing.Optional[WriteTransformerState]=None):
        if False:
            i = 10
            return i + 15
        '\n        This method writes an Image to a byte stream\n        '
        assert context is not None, 'context must be defined in order to write Image objects.'
        assert context.destination is not None, 'context.destination must be defined in order to write Image objects.'
        assert isinstance(object_to_transform, PILImage.Image), 'object_to_transform must be of type PILImage.Image'
        contents: typing.Optional[bytes] = None
        filter_name: typing.Optional[Name] = None
        try:
            with io.BytesIO() as output:
                assert isinstance(object_to_transform, PILImage.Image)
                object_to_transform = self._convert_to_rgb_mode(object_to_transform)
                assert isinstance(object_to_transform, PILImage.Image)
                object_to_transform.save(output, format='JPEG')
                contents = output.getvalue()
            filter_name = Name('DCTDecode')
        except Exception as e:
            pass
        assert contents is not None
        out_value = Stream()
        out_value[Name('Type')] = Name('XObject')
        out_value[Name('Subtype')] = Name('Image')
        out_value[Name('Width')] = bDecimal(object_to_transform.width)
        out_value[Name('Height')] = bDecimal(object_to_transform.height)
        out_value[Name('Length')] = bDecimal(len(contents))
        out_value[Name('Filter')] = filter_name
        out_value[Name('BitsPerComponent')] = bDecimal(8)
        out_value[Name('ColorSpace')] = Name('DeviceRGB')
        out_value[Name('Bytes')] = contents
        out_value.set_reference(object_to_transform.get_reference())
        started_object = False
        ref = out_value.get_reference()
        if ref is not None:
            assert isinstance(ref, Reference)
            if ref.object_number is not None and ref.byte_offset is None:
                started_object = True
                self._start_object(out_value, context)
        cl = context.compression_level
        context.compression_level = 9
        self.get_root_transformer().transform(out_value, context)
        context.compression_level = cl
        if started_object:
            self._end_object(out_value, context)