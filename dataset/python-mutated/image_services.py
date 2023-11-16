"""Provides app identity services."""
from __future__ import annotations
import io
from PIL import Image
from typing import Tuple

def _get_pil_image_dimensions(pil_image: Image) -> Tuple[int, int]:
    if False:
        return 10
    'Gets the dimensions of the Pillow Image.\n\n    Args:\n        pil_image: Image. A file in the Pillow Image format.\n\n    Returns:\n        tuple(int, int). Returns height and width of the image.\n    '
    (width, height) = pil_image.size
    return (height, width)

def get_image_dimensions(file_content: bytes) -> Tuple[int, int]:
    if False:
        i = 10
        return i + 15
    'Gets the dimensions of the image with the given file_content.\n\n    Args:\n        file_content: bytes. The content of the file.\n\n    Returns:\n        tuple(int). Returns height and width of the image.\n    '
    image = Image.open(io.BytesIO(file_content))
    return _get_pil_image_dimensions(image)

def compress_image(image_content: bytes, scaling_factor: float) -> bytes:
    if False:
        i = 10
        return i + 15
    'Compresses the image by resizing the image with the scaling factor.\n\n    Args:\n        image_content: bytes. Content of the file to be compressed.\n        scaling_factor: float. The number by which the dimensions of the image\n            will be scaled. This is expected to be in the interval (0, 1].\n\n    Returns:\n        bytes. Returns the content of the compressed image.\n\n    Raises:\n        ValueError. Scaling factor is not in the interval (0, 1].\n    '
    if scaling_factor > 1 or scaling_factor <= 0:
        raise ValueError('Scaling factor should be in the interval (0, 1], received %f.' % scaling_factor)
    image = Image.open(io.BytesIO(image_content))
    image_format = image.format
    (height, width) = _get_pil_image_dimensions(image)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    new_image_dimensions = (new_width, new_height)
    image.thumbnail(new_image_dimensions, Image.ANTIALIAS)
    with io.BytesIO() as output:
        image.save(output, format=image_format)
        new_image_content = output.getvalue()
    return new_image_content