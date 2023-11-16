import logging
from io import BytesIO
from types import TracebackType
from typing import Optional, Tuple, Type
from PIL import Image
from synapse.logging.opentracing import trace
logger = logging.getLogger(__name__)
EXIF_ORIENTATION_TAG = 274
EXIF_TRANSPOSE_MAPPINGS = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE, 6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}

class ThumbnailError(Exception):
    """An error occurred generating a thumbnail."""

class Thumbnailer:
    FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG'}

    @staticmethod
    def set_limits(max_image_pixels: int) -> None:
        if False:
            while True:
                i = 10
        Image.MAX_IMAGE_PIXELS = max_image_pixels

    def __init__(self, input_path: str):
        if False:
            print('Hello World!')
        self._closed = False
        try:
            self.image = Image.open(input_path)
        except OSError as e:
            raise ThumbnailError from e
        except Image.DecompressionBombError as e:
            raise ThumbnailError from e
        (self.width, self.height) = self.image.size
        self.transpose_method = None
        try:
            image_exif = self.image._getexif()
            if image_exif is not None:
                image_orientation = image_exif.get(EXIF_ORIENTATION_TAG)
                assert type(image_orientation) is int
                self.transpose_method = EXIF_TRANSPOSE_MAPPINGS.get(image_orientation)
        except Exception as e:
            logger.info('Error parsing image EXIF information: %s', e)

    @trace
    def transpose(self) -> Tuple[int, int]:
        if False:
            while True:
                i = 10
        'Transpose the image using its EXIF Orientation tag\n\n        Returns:\n            A tuple containing the new image size in pixels as (width, height).\n        '
        if self.transpose_method is not None:
            with self.image:
                self.image = self.image.transpose(self.transpose_method)
            (self.width, self.height) = self.image.size
            self.transpose_method = None
            self.image.info['exif'] = None
        return self.image.size

    def aspect(self, max_width: int, max_height: int) -> Tuple[int, int]:
        if False:
            print('Hello World!')
        'Calculate the largest size that preserves aspect ratio which\n        fits within the given rectangle::\n\n            (w_in / h_in) = (w_out / h_out)\n            w_out = max(min(w_max, h_max * (w_in / h_in)), 1)\n            h_out = max(min(h_max, w_max * (h_in / w_in)), 1)\n\n        Args:\n            max_width: The largest possible width.\n            max_height: The largest possible height.\n        '
        if max_width * self.height < max_height * self.width:
            return (max_width, max(max_width * self.height // self.width, 1))
        else:
            return (max(max_height * self.width // self.height, 1), max_height)

    def _resize(self, width: int, height: int) -> Image.Image:
        if False:
            while True:
                i = 10
        if self.image.mode in ['1', 'L', 'P']:
            if self.image.info.get('transparency', None) is not None:
                with self.image:
                    self.image = self.image.convert('RGBA')
            else:
                with self.image:
                    self.image = self.image.convert('RGB')
        return self.image.resize((width, height), Image.LANCZOS)

    @trace
    def scale(self, width: int, height: int, output_type: str) -> BytesIO:
        if False:
            print('Hello World!')
        'Rescales the image to the given dimensions.\n\n        Returns:\n            The bytes of the encoded image ready to be written to disk\n        '
        with self._resize(width, height) as scaled:
            return self._encode_image(scaled, output_type)

    @trace
    def crop(self, width: int, height: int, output_type: str) -> BytesIO:
        if False:
            i = 10
            return i + 15
        'Rescales and crops the image to the given dimensions preserving\n        aspect::\n            (w_in / h_in) = (w_scaled / h_scaled)\n            w_scaled = max(w_out, h_out * (w_in / h_in))\n            h_scaled = max(h_out, w_out * (h_in / w_in))\n\n        Args:\n            max_width: The largest possible width.\n            max_height: The largest possible height.\n\n        Returns:\n            The bytes of the encoded image ready to be written to disk\n        '
        if width * self.height > height * self.width:
            scaled_width = width
            scaled_height = width * self.height // self.width
            crop_top = (scaled_height - height) // 2
            crop_bottom = height + crop_top
            crop = (0, crop_top, width, crop_bottom)
        else:
            scaled_width = height * self.width // self.height
            scaled_height = height
            crop_left = (scaled_width - width) // 2
            crop_right = width + crop_left
            crop = (crop_left, 0, crop_right, height)
        with self._resize(scaled_width, scaled_height) as scaled_image:
            with scaled_image.crop(crop) as cropped:
                return self._encode_image(cropped, output_type)

    def _encode_image(self, output_image: Image.Image, output_type: str) -> BytesIO:
        if False:
            return 10
        output_bytes_io = BytesIO()
        fmt = self.FORMATS[output_type]
        if fmt == 'JPEG':
            output_image = output_image.convert('RGB')
        output_image.save(output_bytes_io, fmt, quality=80)
        return output_bytes_io

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Closes the underlying image file.\n\n        Once closed no other functions can be called.\n\n        Can be called multiple times.\n        '
        if self._closed:
            return
        self._closed = True
        image = getattr(self, 'image', None)
        if image is None:
            return
        image.close()

    def __enter__(self) -> 'Thumbnailer':
        if False:
            i = 10
            return i + 15
        'Make `Thumbnailer` a context manager that calls `close` on\n        `__exit__`.\n        '
        return self

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            return 10
        self.close()

    def __del__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.close()