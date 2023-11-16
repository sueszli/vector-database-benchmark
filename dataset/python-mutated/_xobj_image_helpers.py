"""Code in here is only used by pypdf.filters._xobj_to_image"""
import sys
from io import BytesIO
from typing import Any, List, Tuple, Union, cast
from ._utils import logger_warning
from .constants import ColorSpaces
from .errors import PdfReadError
from .generic import ArrayObject, DecodedStreamObject, EncodedStreamObject, IndirectObject, NullObject
if sys.version_info[:2] >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
if sys.version_info[:2] >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
try:
    from PIL import Image
except ImportError:
    raise ImportError("pillow is required to do image extraction. It can be installed via 'pip install pypdf[image]'")
mode_str_type: TypeAlias = Literal['', '1', 'RGB', '2bits', '4bits', 'P', 'L', 'RGBA', 'CMYK']
MAX_IMAGE_MODE_NESTING_DEPTH: int = 10

def _get_imagemode(color_space: Union[str, List[Any], Any], color_components: int, prev_mode: mode_str_type, depth: int=0) -> Tuple[mode_str_type, bool]:
    if False:
        return 10
    '\n    Returns\n        Image mode not taking into account mask(transparency)\n        ColorInversion is required (like for some DeviceCMYK)\n    '
    if depth > MAX_IMAGE_MODE_NESTING_DEPTH:
        raise PdfReadError('Color spaces nested too deep. If required, consider increasing MAX_IMAGE_MODE_NESTING_DEPTH.')
    if isinstance(color_space, NullObject):
        return ('', False)
    if isinstance(color_space, str):
        pass
    elif not isinstance(color_space, list):
        raise PdfReadError('can not interprete colorspace', color_space)
    elif color_space[0].startswith('/Cal'):
        color_space = '/Device' + color_space[0][4:]
    elif color_space[0] == '/ICCBased':
        icc_profile = color_space[1].get_object()
        color_components = cast(int, icc_profile['/N'])
        color_space = icc_profile.get('/Alternate', '')
    elif color_space[0] == '/Indexed':
        color_space = color_space[1]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        (mode2, invert_color) = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        if mode2 in ('RGB', 'CMYK'):
            mode2 = 'P'
        return (mode2, invert_color)
    elif color_space[0] == '/Separation':
        color_space = color_space[2]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        (mode2, invert_color) = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        return (mode2, True)
    elif color_space[0] == '/DeviceN':
        color_components = len(color_space[1])
        color_space = color_space[2]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        (mode2, invert_color) = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        return (mode2, invert_color)
    mode_map = {'1bit': '1', '/DeviceGray': 'L', 'palette': 'P', '/DeviceRGB': 'RGB', '/DeviceCMYK': 'CMYK', '2bit': '2bits', '4bit': '4bits'}
    mode: mode_str_type = mode_map.get(color_space) or list(mode_map.values())[color_components] or prev_mode
    return (mode, mode == 'CMYK')

def _handle_flate(size: Tuple[int, int], data: bytes, mode: mode_str_type, color_space: str, colors: int, obj_as_text: str) -> Tuple[Image.Image, str, str, bool]:
    if False:
        print('Hello World!')
    '\n    Process image encoded in flateEncode\n    Returns img, image_format, extension, color inversion\n    '

    def bits2byte(data: bytes, size: Tuple[int, int], bits: int) -> bytes:
        if False:
            print('Hello World!')
        mask = (2 << bits) - 1
        nbuff = bytearray(size[0] * size[1])
        by = 0
        bit = 8 - bits
        for y in range(size[1]):
            if bit != 0 and bit != 8 - bits:
                by += 1
                bit = 8 - bits
            for x in range(size[0]):
                nbuff[y * size[0] + x] = data[by] >> bit & mask
                bit -= bits
                if bit < 0:
                    by += 1
                    bit = 8 - bits
        return bytes(nbuff)
    extension = '.png'
    image_format = 'PNG'
    lookup: Any
    base: Any
    hival: Any
    if isinstance(color_space, ArrayObject) and color_space[0] == '/Indexed':
        (color_space, base, hival, lookup) = (value.get_object() for value in color_space)
    if mode == '2bits':
        mode = 'P'
        data = bits2byte(data, size, 2)
    elif mode == '4bits':
        mode = 'P'
        data = bits2byte(data, size, 4)
    img = Image.frombytes(mode, size, data)
    if color_space == '/Indexed':
        from .generic import TextStringObject
        if isinstance(lookup, (EncodedStreamObject, DecodedStreamObject)):
            lookup = lookup.get_data()
        if isinstance(lookup, TextStringObject):
            lookup = lookup.original_bytes
        if isinstance(lookup, str):
            lookup = lookup.encode()
        try:
            (nb, conv, mode) = {'1': (0, '', ''), 'L': (1, 'P', 'L'), 'P': (0, '', ''), 'RGB': (3, 'P', 'RGB'), 'CMYK': (4, 'P', 'CMYK')}[_get_imagemode(base, 0, '')[0]]
        except KeyError:
            logger_warning(f'Base {base} not coded please share the pdf file with pypdf dev team', __name__)
            lookup = None
        else:
            if img.mode == '1':
                assert len(lookup) == 2 * nb, len(lookup)
                colors_arr = [lookup[:nb], lookup[nb:]]
                arr = b''.join([b''.join([colors_arr[1 if img.getpixel((x, y)) > 127 else 0] for x in range(img.size[0])]) for y in range(img.size[1])])
                img = Image.frombytes(mode, img.size, arr)
            else:
                img = img.convert(conv)
                if len(lookup) != (hival + 1) * nb:
                    logger_warning(f'Invalid Lookup Table in {obj_as_text}', __name__)
                    lookup = None
                elif mode == 'L':
                    lookup = b''.join([bytes([b, b, b]) for b in lookup])
                    mode = 'RGB'
                elif mode == 'CMYK':
                    _rgb = []
                    for (_c, _m, _y, _k) in (lookup[n:n + 4] for n in range(0, 4 * (len(lookup) // 4), 4)):
                        _r = int(255 * (1 - _c / 255) * (1 - _k / 255))
                        _g = int(255 * (1 - _m / 255) * (1 - _k / 255))
                        _b = int(255 * (1 - _y / 255) * (1 - _k / 255))
                        _rgb.append(bytes((_r, _g, _b)))
                    lookup = b''.join(_rgb)
                    mode = 'RGB'
                if lookup is not None:
                    img.putpalette(lookup, rawmode=mode)
            img = img.convert('L' if base == ColorSpaces.DEVICE_GRAY else 'RGB')
    elif not isinstance(color_space, NullObject) and color_space[0] == '/ICCBased':
        mode2 = _get_imagemode(color_space, colors, mode)[0]
        if mode != mode2:
            img = Image.frombytes(mode2, size, data)
    if mode == 'CMYK':
        extension = '.tif'
        image_format = 'TIFF'
    return (img, image_format, extension, False)

def _handle_jpx(size: Tuple[int, int], data: bytes, mode: mode_str_type, color_space: str, colors: int) -> Tuple[Image.Image, str, str, bool]:
    if False:
        while True:
            i = 10
    '\n    Process image encoded in flateEncode\n    Returns img, image_format, extension, inversion\n    '
    extension = '.jp2'
    img1 = Image.open(BytesIO(data), formats=('JPEG2000',))
    (mode, invert_color) = _get_imagemode(color_space, colors, mode)
    if mode == '':
        mode = cast(mode_str_type, img1.mode)
        invert_color = mode in ('CMYK',)
    if img1.mode == 'RGBA' and mode == 'RGB':
        mode = 'RGBA'
    try:
        if img1.mode != mode:
            img = Image.frombytes(mode, img1.size, img1.tobytes())
        else:
            img = img1
    except OSError:
        img = Image.frombytes(mode, img1.size, img1.tobytes())
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    image_format = 'JPEG2000'
    return (img, image_format, extension, invert_color)