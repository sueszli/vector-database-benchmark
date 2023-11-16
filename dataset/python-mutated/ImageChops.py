from . import Image

def constant(image, value):
    if False:
        return 10
    'Fill a channel with a given gray level.\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    return Image.new('L', image.size, value)

def duplicate(image):
    if False:
        for i in range(10):
            print('nop')
    'Copy a channel. Alias for :py:meth:`PIL.Image.Image.copy`.\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    return image.copy()

def invert(image):
    if False:
        i = 10
        return i + 15
    '\n    Invert an image (channel). ::\n\n        out = MAX - image\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image.load()
    return image._new(image.im.chop_invert())

def lighter(image1, image2):
    if False:
        print('Hello World!')
    '\n    Compares the two images, pixel by pixel, and returns a new image containing\n    the lighter values. ::\n\n        out = max(image1, image2)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_lighter(image2.im))

def darker(image1, image2):
    if False:
        return 10
    '\n    Compares the two images, pixel by pixel, and returns a new image containing\n    the darker values. ::\n\n        out = min(image1, image2)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_darker(image2.im))

def difference(image1, image2):
    if False:
        print('Hello World!')
    '\n    Returns the absolute value of the pixel-by-pixel difference between the two\n    images. ::\n\n        out = abs(image1 - image2)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_difference(image2.im))

def multiply(image1, image2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Superimposes two images on top of each other.\n\n    If you multiply an image with a solid black image, the result is black. If\n    you multiply with a solid white image, the image is unaffected. ::\n\n        out = image1 * image2 / MAX\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_multiply(image2.im))

def screen(image1, image2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Superimposes two inverted images on top of each other. ::\n\n        out = MAX - ((MAX - image1) * (MAX - image2) / MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_screen(image2.im))

def soft_light(image1, image2):
    if False:
        return 10
    '\n    Superimposes two images on top of each other using the Soft Light algorithm\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_soft_light(image2.im))

def hard_light(image1, image2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Superimposes two images on top of each other using the Hard Light algorithm\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_hard_light(image2.im))

def overlay(image1, image2):
    if False:
        while True:
            i = 10
    '\n    Superimposes two images on top of each other using the Overlay algorithm\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_overlay(image2.im))

def add(image1, image2, scale=1.0, offset=0):
    if False:
        while True:
            i = 10
    '\n    Adds two images, dividing the result by scale and adding the\n    offset. If omitted, scale defaults to 1.0, and offset to 0.0. ::\n\n        out = ((image1 + image2) / scale + offset)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_add(image2.im, scale, offset))

def subtract(image1, image2, scale=1.0, offset=0):
    if False:
        while True:
            i = 10
    '\n    Subtracts two images, dividing the result by scale and adding the offset.\n    If omitted, scale defaults to 1.0, and offset to 0.0. ::\n\n        out = ((image1 - image2) / scale + offset)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_subtract(image2.im, scale, offset))

def add_modulo(image1, image2):
    if False:
        i = 10
        return i + 15
    'Add two images, without clipping the result. ::\n\n        out = ((image1 + image2) % MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_add_modulo(image2.im))

def subtract_modulo(image1, image2):
    if False:
        return 10
    'Subtract two images, without clipping the result. ::\n\n        out = ((image1 - image2) % MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_subtract_modulo(image2.im))

def logical_and(image1, image2):
    if False:
        return 10
    'Logical AND between two images.\n\n    Both of the images must have mode "1". If you would like to perform a\n    logical AND on an image with a mode other than "1", try\n    :py:meth:`~PIL.ImageChops.multiply` instead, using a black-and-white mask\n    as the second image. ::\n\n        out = ((image1 and image2) % MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_and(image2.im))

def logical_or(image1, image2):
    if False:
        print('Hello World!')
    'Logical OR between two images.\n\n    Both of the images must have mode "1". ::\n\n        out = ((image1 or image2) % MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_or(image2.im))

def logical_xor(image1, image2):
    if False:
        print('Hello World!')
    'Logical XOR between two images.\n\n    Both of the images must have mode "1". ::\n\n        out = ((bool(image1) != bool(image2)) % MAX)\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_xor(image2.im))

def blend(image1, image2, alpha):
    if False:
        i = 10
        return i + 15
    'Blend images using constant transparency weight. Alias for\n    :py:func:`PIL.Image.blend`.\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    return Image.blend(image1, image2, alpha)

def composite(image1, image2, mask):
    if False:
        while True:
            i = 10
    'Create composite using transparency mask. Alias for\n    :py:func:`PIL.Image.composite`.\n\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    return Image.composite(image1, image2, mask)

def offset(image, xoffset, yoffset=None):
    if False:
        return 10
    'Returns a copy of the image where data has been offset by the given\n    distances. Data wraps around the edges. If ``yoffset`` is omitted, it\n    is assumed to be equal to ``xoffset``.\n\n    :param image: Input image.\n    :param xoffset: The horizontal distance.\n    :param yoffset: The vertical distance.  If omitted, both\n        distances are set to the same value.\n    :rtype: :py:class:`~PIL.Image.Image`\n    '
    if yoffset is None:
        yoffset = xoffset
    image.load()
    return image._new(image.im.offset(xoffset, yoffset))