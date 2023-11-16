"""
Author: RedFantom
License: GNU GPLv3
Copyright (c) 2017-2018 RedFantom
"""

def shift_hue(image, hue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shifts the hue of an image in HSV format.\n    :param image: PIL Image to perform operation on\n    :param hue: value between 0 and 2.0\n    '
    hue = (hue - 1.0) * 180
    img = image.copy().convert('HSV')
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            (h, s, v) = pixels[i, j]
            h = abs(int(h + hue))
            if h > 255:
                h -= 255
            pixels[i, j] = (h, s, v)
    return img.convert('RGBA')

def make_transparent(image):
    if False:
        for i in range(10):
            print('nop')
    'Turn all black pixels in an image into transparent ones'
    data = image.copy().getdata()
    modified = []
    for item in data:
        if _check_pixel(item) is True:
            modified.append((255, 255, 255, 255))
            continue
        modified.append(item)
    image.putdata(modified)
    return image

def _check_pixel(tup):
    if False:
        print('Hello World!')
    'Check if a pixel is black, supports RGBA'
    return tup[0] == 0 and tup[1] == 0 and (tup[2] == 0)