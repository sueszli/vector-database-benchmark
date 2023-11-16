import functools
import operator
import re
from . import ExifTags, Image, ImagePalette

def _border(border):
    if False:
        return 10
    if isinstance(border, tuple):
        if len(border) == 2:
            (left, top) = (right, bottom) = border
        elif len(border) == 4:
            (left, top, right, bottom) = border
    else:
        left = top = right = bottom = border
    return (left, top, right, bottom)

def _color(color, mode):
    if False:
        return 10
    if isinstance(color, str):
        from . import ImageColor
        color = ImageColor.getcolor(color, mode)
    return color

def _lut(image, lut):
    if False:
        while True:
            i = 10
    if image.mode == 'P':
        msg = 'mode P support coming soon'
        raise NotImplementedError(msg)
    elif image.mode in ('L', 'RGB'):
        if image.mode == 'RGB' and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        msg = f'not supported for mode {image.mode}'
        raise OSError(msg)

def autocontrast(image, cutoff=0, ignore=None, mask=None, preserve_tone=False):
    if False:
        while True:
            i = 10
    '\n    Maximize (normalize) image contrast. This function calculates a\n    histogram of the input image (or mask region), removes ``cutoff`` percent of the\n    lightest and darkest pixels from the histogram, and remaps the image\n    so that the darkest pixel becomes black (0), and the lightest\n    becomes white (255).\n\n    :param image: The image to process.\n    :param cutoff: The percent to cut off from the histogram on the low and\n                   high ends. Either a tuple of (low, high), or a single\n                   number for both.\n    :param ignore: The background pixel value (use None for no background).\n    :param mask: Histogram used in contrast operation is computed using pixels\n                 within the mask. If no mask is given the entire image is used\n                 for histogram computation.\n    :param preserve_tone: Preserve image tone in Photoshop-like style autocontrast.\n\n                          .. versionadded:: 8.2.0\n\n    :return: An image.\n    '
    if preserve_tone:
        histogram = image.convert('L').histogram(mask)
    else:
        histogram = image.histogram(mask)
    lut = []
    for layer in range(0, len(histogram), 256):
        h = histogram[layer:layer + 256]
        if ignore is not None:
            try:
                h[ignore] = 0
            except TypeError:
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            n = 0
            for ix in range(256):
                n = n + h[ix]
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            lut.extend(list(range(256)))
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                ix = int(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut.append(ix)
    return _lut(image, lut)

def colorize(image, black, white, mid=None, blackpoint=0, whitepoint=255, midpoint=127):
    if False:
        i = 10
        return i + 15
    '\n    Colorize grayscale image.\n    This function calculates a color wedge which maps all black pixels in\n    the source image to the first color and all white pixels to the\n    second color. If ``mid`` is specified, it uses three-color mapping.\n    The ``black`` and ``white`` arguments should be RGB tuples or color names;\n    optionally you can use three-color mapping by also specifying ``mid``.\n    Mapping positions for any of the colors can be specified\n    (e.g. ``blackpoint``), where these parameters are the integer\n    value corresponding to where the corresponding color should be mapped.\n    These parameters must have logical order, such that\n    ``blackpoint <= midpoint <= whitepoint`` (if ``mid`` is specified).\n\n    :param image: The image to colorize.\n    :param black: The color to use for black input pixels.\n    :param white: The color to use for white input pixels.\n    :param mid: The color to use for midtone input pixels.\n    :param blackpoint: an int value [0, 255] for the black mapping.\n    :param whitepoint: an int value [0, 255] for the white mapping.\n    :param midpoint: an int value [0, 255] for the midtone mapping.\n    :return: An image.\n    '
    assert image.mode == 'L'
    if mid is None:
        assert 0 <= blackpoint <= whitepoint <= 255
    else:
        assert 0 <= blackpoint <= midpoint <= whitepoint <= 255
    black = _color(black, 'RGB')
    white = _color(white, 'RGB')
    if mid is not None:
        mid = _color(mid, 'RGB')
    red = []
    green = []
    blue = []
    for i in range(0, blackpoint):
        red.append(black[0])
        green.append(black[1])
        blue.append(black[2])
    if mid is None:
        range_map = range(0, whitepoint - blackpoint)
        for i in range_map:
            red.append(black[0] + i * (white[0] - black[0]) // len(range_map))
            green.append(black[1] + i * (white[1] - black[1]) // len(range_map))
            blue.append(black[2] + i * (white[2] - black[2]) // len(range_map))
    else:
        range_map1 = range(0, midpoint - blackpoint)
        range_map2 = range(0, whitepoint - midpoint)
        for i in range_map1:
            red.append(black[0] + i * (mid[0] - black[0]) // len(range_map1))
            green.append(black[1] + i * (mid[1] - black[1]) // len(range_map1))
            blue.append(black[2] + i * (mid[2] - black[2]) // len(range_map1))
        for i in range_map2:
            red.append(mid[0] + i * (white[0] - mid[0]) // len(range_map2))
            green.append(mid[1] + i * (white[1] - mid[1]) // len(range_map2))
            blue.append(mid[2] + i * (white[2] - mid[2]) // len(range_map2))
    for i in range(0, 256 - whitepoint):
        red.append(white[0])
        green.append(white[1])
        blue.append(white[2])
    image = image.convert('RGB')
    return _lut(image, red + green + blue)

def contain(image, size, method=Image.Resampling.BICUBIC):
    if False:
        i = 10
        return i + 15
    '\n    Returns a resized version of the image, set to the maximum width and height\n    within the requested size, while maintaining the original aspect ratio.\n\n    :param image: The image to resize.\n    :param size: The requested output size in pixels, given as a\n                 (width, height) tuple.\n    :param method: Resampling method to use. Default is\n                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.\n                   See :ref:`concept-filters`.\n    :return: An image.\n    '
    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]
    if im_ratio != dest_ratio:
        if im_ratio > dest_ratio:
            new_height = round(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = round(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)

def cover(image, size, method=Image.Resampling.BICUBIC):
    if False:
        return 10
    '\n    Returns a resized version of the image, so that the requested size is\n    covered, while maintaining the original aspect ratio.\n\n    :param image: The image to resize.\n    :param size: The requested output size in pixels, given as a\n                 (width, height) tuple.\n    :param method: Resampling method to use. Default is\n                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.\n                   See :ref:`concept-filters`.\n    :return: An image.\n    '
    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]
    if im_ratio != dest_ratio:
        if im_ratio < dest_ratio:
            new_height = round(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = round(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)

def pad(image, size, method=Image.Resampling.BICUBIC, color=None, centering=(0.5, 0.5)):
    if False:
        i = 10
        return i + 15
    '\n    Returns a resized and padded version of the image, expanded to fill the\n    requested aspect ratio and size.\n\n    :param image: The image to resize and crop.\n    :param size: The requested output size in pixels, given as a\n                 (width, height) tuple.\n    :param method: Resampling method to use. Default is\n                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.\n                   See :ref:`concept-filters`.\n    :param color: The background color of the padded image.\n    :param centering: Control the position of the original image within the\n                      padded version.\n\n                          (0.5, 0.5) will keep the image centered\n                          (0, 0) will keep the image aligned to the top left\n                          (1, 1) will keep the image aligned to the bottom\n                          right\n    :return: An image.\n    '
    resized = contain(image, size, method)
    if resized.size == size:
        out = resized
    else:
        out = Image.new(image.mode, size, color)
        if resized.palette:
            out.putpalette(resized.getpalette())
        if resized.width != size[0]:
            x = round((size[0] - resized.width) * max(0, min(centering[0], 1)))
            out.paste(resized, (x, 0))
        else:
            y = round((size[1] - resized.height) * max(0, min(centering[1], 1)))
            out.paste(resized, (0, y))
    return out

def crop(image, border=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove border from image.  The same amount of pixels are removed\n    from all four sides.  This function works on all image modes.\n\n    .. seealso:: :py:meth:`~PIL.Image.Image.crop`\n\n    :param image: The image to crop.\n    :param border: The number of pixels to remove.\n    :return: An image.\n    '
    (left, top, right, bottom) = _border(border)
    return image.crop((left, top, image.size[0] - right, image.size[1] - bottom))

def scale(image, factor, resample=Image.Resampling.BICUBIC):
    if False:
        print('Hello World!')
    '\n    Returns a rescaled image by a specific factor given in parameter.\n    A factor greater than 1 expands the image, between 0 and 1 contracts the\n    image.\n\n    :param image: The image to rescale.\n    :param factor: The expansion factor, as a float.\n    :param resample: Resampling method to use. Default is\n                     :py:attr:`~PIL.Image.Resampling.BICUBIC`.\n                     See :ref:`concept-filters`.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    if factor == 1:
        return image.copy()
    elif factor <= 0:
        msg = 'the factor must be greater than 0'
        raise ValueError(msg)
    else:
        size = (round(factor * image.width), round(factor * image.height))
        return image.resize(size, resample)

def deform(image, deformer, resample=Image.Resampling.BILINEAR):
    if False:
        return 10
    '\n    Deform the image.\n\n    :param image: The image to deform.\n    :param deformer: A deformer object.  Any object that implements a\n                    ``getmesh`` method can be used.\n    :param resample: An optional resampling filter. Same values possible as\n       in the PIL.Image.transform function.\n    :return: An image.\n    '
    return image.transform(image.size, Image.Transform.MESH, deformer.getmesh(image), resample)

def equalize(image, mask=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Equalize the image histogram. This function applies a non-linear\n    mapping to the input image, in order to create a uniform\n    distribution of grayscale values in the output image.\n\n    :param image: The image to equalize.\n    :param mask: An optional mask.  If given, only the pixels selected by\n                 the mask are included in the analysis.\n    :return: An image.\n    '
    if image.mode == 'P':
        image = image.convert('RGB')
    h = image.histogram(mask)
    lut = []
    for b in range(0, len(h), 256):
        histo = [_f for _f in h[b:b + 256] if _f]
        if len(histo) <= 1:
            lut.extend(list(range(256)))
        else:
            step = (functools.reduce(operator.add, histo) - histo[-1]) // 255
            if not step:
                lut.extend(list(range(256)))
            else:
                n = step // 2
                for i in range(256):
                    lut.append(n // step)
                    n = n + h[i + b]
    return _lut(image, lut)

def expand(image, border=0, fill=0):
    if False:
        i = 10
        return i + 15
    '\n    Add border to the image\n\n    :param image: The image to expand.\n    :param border: Border width, in pixels.\n    :param fill: Pixel fill value (a color value).  Default is 0 (black).\n    :return: An image.\n    '
    (left, top, right, bottom) = _border(border)
    width = left + image.size[0] + right
    height = top + image.size[1] + bottom
    color = _color(fill, image.mode)
    if image.palette:
        palette = ImagePalette.ImagePalette(palette=image.getpalette())
        if isinstance(color, tuple):
            color = palette.getcolor(color)
    else:
        palette = None
    out = Image.new(image.mode, (width, height), color)
    if palette:
        out.putpalette(palette.palette)
    out.paste(image, (left, top))
    return out

def fit(image, size, method=Image.Resampling.BICUBIC, bleed=0.0, centering=(0.5, 0.5)):
    if False:
        while True:
            i = 10
    '\n    Returns a resized and cropped version of the image, cropped to the\n    requested aspect ratio and size.\n\n    This function was contributed by Kevin Cazabon.\n\n    :param image: The image to resize and crop.\n    :param size: The requested output size in pixels, given as a\n                 (width, height) tuple.\n    :param method: Resampling method to use. Default is\n                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.\n                   See :ref:`concept-filters`.\n    :param bleed: Remove a border around the outside of the image from all\n                  four edges. The value is a decimal percentage (use 0.01 for\n                  one percent). The default value is 0 (no border).\n                  Cannot be greater than or equal to 0.5.\n    :param centering: Control the cropping position.  Use (0.5, 0.5) for\n                      center cropping (e.g. if cropping the width, take 50% off\n                      of the left side, and therefore 50% off the right side).\n                      (0.0, 0.0) will crop from the top left corner (i.e. if\n                      cropping the width, take all of the crop off of the right\n                      side, and if cropping the height, take all of it off the\n                      bottom).  (1.0, 0.0) will crop from the bottom left\n                      corner, etc. (i.e. if cropping the width, take all of the\n                      crop off the left side, and if cropping the height take\n                      none from the top, and therefore all off the bottom).\n    :return: An image.\n    '
    centering = list(centering)
    if not 0.0 <= centering[0] <= 1.0:
        centering[0] = 0.5
    if not 0.0 <= centering[1] <= 1.0:
        centering[1] = 0.5
    if not 0.0 <= bleed < 0.5:
        bleed = 0.0
    bleed_pixels = (bleed * image.size[0], bleed * image.size[1])
    live_size = (image.size[0] - bleed_pixels[0] * 2, image.size[1] - bleed_pixels[1] * 2)
    live_size_ratio = live_size[0] / live_size[1]
    output_ratio = size[0] / size[1]
    if live_size_ratio == output_ratio:
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio
    crop_left = bleed_pixels[0] + (live_size[0] - crop_width) * centering[0]
    crop_top = bleed_pixels[1] + (live_size[1] - crop_height) * centering[1]
    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    return image.resize(size, method, box=crop)

def flip(image):
    if False:
        while True:
            i = 10
    '\n    Flip the image vertically (top to bottom).\n\n    :param image: The image to flip.\n    :return: An image.\n    '
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

def grayscale(image):
    if False:
        while True:
            i = 10
    '\n    Convert the image to grayscale.\n\n    :param image: The image to convert.\n    :return: An image.\n    '
    return image.convert('L')

def invert(image):
    if False:
        return 10
    '\n    Invert (negate) the image.\n\n    :param image: The image to invert.\n    :return: An image.\n    '
    lut = []
    for i in range(256):
        lut.append(255 - i)
    return image.point(lut) if image.mode == '1' else _lut(image, lut)

def mirror(image):
    if False:
        while True:
            i = 10
    '\n    Flip image horizontally (left to right).\n\n    :param image: The image to mirror.\n    :return: An image.\n    '
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

def posterize(image, bits):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reduce the number of bits for each color channel.\n\n    :param image: The image to posterize.\n    :param bits: The number of bits to keep for each channel (1-8).\n    :return: An image.\n    '
    lut = []
    mask = ~(2 ** (8 - bits) - 1)
    for i in range(256):
        lut.append(i & mask)
    return _lut(image, lut)

def solarize(image, threshold=128):
    if False:
        while True:
            i = 10
    '\n    Invert all pixel values above a threshold.\n\n    :param image: The image to solarize.\n    :param threshold: All pixels above this grayscale level are inverted.\n    :return: An image.\n    '
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(i)
        else:
            lut.append(255 - i)
    return _lut(image, lut)

def exif_transpose(image, *, in_place=False):
    if False:
        print('Hello World!')
    '\n    If an image has an EXIF Orientation tag, other than 1, transpose the image\n    accordingly, and remove the orientation data.\n\n    :param image: The image to transpose.\n    :param in_place: Boolean. Keyword-only argument.\n        If ``True``, the original image is modified in-place, and ``None`` is returned.\n        If ``False`` (default), a new :py:class:`~PIL.Image.Image` object is returned\n        with the transposition applied. If there is no transposition, a copy of the\n        image will be returned.\n    '
    image.load()
    image_exif = image.getexif()
    orientation = image_exif.get(ExifTags.Base.Orientation)
    method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}.get(orientation)
    if method is not None:
        transposed_image = image.transpose(method)
        if in_place:
            image.im = transposed_image.im
            image.pyaccess = None
            image._size = transposed_image._size
        exif_image = image if in_place else transposed_image
        exif = exif_image.getexif()
        if ExifTags.Base.Orientation in exif:
            del exif[ExifTags.Base.Orientation]
            if 'exif' in exif_image.info:
                exif_image.info['exif'] = exif.tobytes()
            elif 'Raw profile type exif' in exif_image.info:
                exif_image.info['Raw profile type exif'] = exif.tobytes().hex()
            elif 'XML:com.adobe.xmp' in exif_image.info:
                for pattern in ('tiff:Orientation="([0-9])"', '<tiff:Orientation>([0-9])</tiff:Orientation>'):
                    exif_image.info['XML:com.adobe.xmp'] = re.sub(pattern, '', exif_image.info['XML:com.adobe.xmp'])
        if not in_place:
            return transposed_image
    elif not in_place:
        return image.copy()