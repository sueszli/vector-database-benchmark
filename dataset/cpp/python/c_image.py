# -*- coding: utf-8 -*-
from PIL import Image
from os.path import join, dirname, getmtime, exists, expanduser
from .config import *
from .py3patch import *

import ctypes
import sys
import os


def call_c():
    """
    Call the C program for converting RGB to Ansi colors
    """
    library = expanduser('~/.image.so')
    sauce = join(dirname(__file__), 'image.c')
    if not exists(library) or getmtime(sauce) > getmtime(library):
        build = "cc -fPIC -shared -o %s %s" % (library, sauce)
        os.system(build + " >/dev/null 2>&1")
    image_c = ctypes.cdll.LoadLibrary(library)
    image_c.init()
    return image_c.rgb_to_ansi

rgb2short = call_c()


def pixel_print(pixel):
    """
    Print a pixel with given Ansi color
    """
    r, g, b = pixel[:3]

    if c['24BIT'] is True:
        sys.stdout.write('\033[48;2;%d;%d;%dm \033[0m'
                         % (r, g, b))
    else:
        ansicolor = rgb2short(r, g, b)
        sys.stdout.write('\033[48;5;%sm \033[0m' % (ansicolor))


def block_print(higher, lower):
    """
    Print two pixels arranged above each other with Ansi color.
    Abuses Unicode to print two pixels in the space of one terminal block.
    """
    r0, g0, b0 = lower[:3]
    r1, g1, b1 = higher[:3]

    if c['24BIT'] is True:
        sys.stdout.write('\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▄\033[0m'
                         % (r1, g1, b1, r0, g0, b0))
    else:
        i0 = rgb2short(r0, g0, b0)
        i1 = rgb2short(r1, g1, b1)
        sys.stdout.write('\033[38;5;%sm\033[48;5;%sm▄\033[0m' % (i1, i0))


def image_to_display(path, start=None, length=None):
    """
    Display an image
    """
    rows, columns = os.popen('stty size', 'r').read().split()
    if not start:
        start = c['IMAGE_SHIFT']
    if not length:
        length = int(columns) - 2 * start
    i = Image.open(path)
    i = i.convert('RGBA')
    w, h = i.size
    i.load()
    width = min(w, length)
    height = int(float(h) * (float(width) / float(w)))

    if c['IMAGE_RESIZE_TO_FIT'] is True:
        # If it image won't fit in the terminal without scrolling shrink it
        # Subtract 3 from rows so the tweet message fits in too.
        h = 2 * (int(rows) - 3)
        if height >= h:
            width = int(float(width) * (float(h) / float(height)))
            height = h
    if (height <= 0) or (width <= 0):
        raise ValueError("image has negative dimensions")

    height = min(height, c['IMAGE_MAX_HEIGHT'])

    # Sixel
    if c['IMAGE_ON_TERM'] == 'sixel':
        import fcntl, struct, termios
        from io import BytesIO
        from libsixel import sixel_dither_new, sixel_dither_initialize, sixel_encode, sixel_output_new, SIXEL_PIXELFORMAT_RGBA8888
        from resizeimage import resizeimage

        # FIXME: rows and columns are gotten a second time. Maybe use this at 
        # the begining of function instead of the call to stty size
        farg = struct.pack("HHHH", 0, 0, 0, 0)
        fd_stdout = sys.stdout.fileno()
        fretint = fcntl.ioctl(fd_stdout, termios.TIOCGWINSZ, farg)
        rows, columns, xpixels, ypixels = struct.unpack("HHHH", fretint)
        max_width_pixels = width * (xpixels // columns)
        max_height_pixels = height * (ypixels // rows)

        # FIXME: This way is preferable to avoid an addition dependency, but it doesn't work correctly
        # i = i.resize((max_width_pixels, max_height_pixels), Image.ANTIALIAS)
        i = resizeimage.resize_thumbnail(i, [max_width_pixels, max_height_pixels])

        sixel = BytesIO()
        dither = sixel_dither_new(256)
        sixel_dither_initialize(dither, i.tobytes(), i.width, i.height, SIXEL_PIXELFORMAT_RGBA8888)
        sixel_encode(i.tobytes(), i.width, i.height, 1, dither, 
                sixel_output_new(lambda imgdata, sixel: sixel.write(imgdata), sixel))
        sys.stdout.write('%s%s' % (' ' * start, sixel.getvalue().decode('ascii')))

    # SGR
    else:
        i = i.resize((width, height), Image.ANTIALIAS)

        for real_y in xrange(height // 2):
            sys.stdout.write(' ' * start)
            for x in xrange(width):
                y = real_y * 2
                p0 = i.getpixel((x, y))
                p1 = i.getpixel((x, y + 1))
                block_print(p1, p0)
            sys.stdout.write('\n')

"""
For direct using purpose
"""
if __name__ == '__main__':
    image_to_display(sys.argv[1])
