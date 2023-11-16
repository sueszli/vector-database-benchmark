import sys
from PIL import Image, features
try:
    Image.core.ping
except ImportError as v:
    print('***', v)
    sys.exit()
except AttributeError:
    pass

def testimage():
    if False:
        while True:
            i = 10
    '\n    PIL lets you create in-memory images with various pixel types:\n\n    >>> from PIL import Image, ImageDraw, ImageFilter, ImageMath\n    >>> im = Image.new("1", (128, 128)) # monochrome\n    >>> def _info(im): return im.format, im.mode, im.size\n    >>> _info(im)\n    (None, \'1\', (128, 128))\n    >>> _info(Image.new("L", (128, 128))) # grayscale (luminance)\n    (None, \'L\', (128, 128))\n    >>> _info(Image.new("P", (128, 128))) # palette\n    (None, \'P\', (128, 128))\n    >>> _info(Image.new("RGB", (128, 128))) # truecolor\n    (None, \'RGB\', (128, 128))\n    >>> _info(Image.new("I", (128, 128))) # 32-bit integer\n    (None, \'I\', (128, 128))\n    >>> _info(Image.new("F", (128, 128))) # 32-bit floating point\n    (None, \'F\', (128, 128))\n\n    Or open existing files:\n\n    >>> with Image.open("Tests/images/hopper.gif") as im:\n    ...     _info(im)\n    (\'GIF\', \'P\', (128, 128))\n    >>> _info(Image.open("Tests/images/hopper.ppm"))\n    (\'PPM\', \'RGB\', (128, 128))\n    >>> try:\n    ...  _info(Image.open("Tests/images/hopper.jpg"))\n    ... except OSError as v:\n    ...  print(v)\n    (\'JPEG\', \'RGB\', (128, 128))\n\n    PIL doesn\'t actually load the image data until it\'s needed,\n    or you call the "load" method:\n\n    >>> im = Image.open("Tests/images/hopper.ppm")\n    >>> print(im.im) # internal image attribute\n    None\n    >>> a = im.load()\n    >>> type(im.im) # doctest: +ELLIPSIS\n    <... \'...ImagingCore\'>\n\n    You can apply many different operations on images.  Most\n    operations return a new image:\n\n    >>> im = Image.open("Tests/images/hopper.ppm")\n    >>> _info(im.convert("L"))\n    (None, \'L\', (128, 128))\n    >>> _info(im.copy())\n    (None, \'RGB\', (128, 128))\n    >>> _info(im.crop((32, 32, 96, 96)))\n    (None, \'RGB\', (64, 64))\n    >>> _info(im.filter(ImageFilter.BLUR))\n    (None, \'RGB\', (128, 128))\n    >>> im.getbands()\n    (\'R\', \'G\', \'B\')\n    >>> im.getbbox()\n    (0, 0, 128, 128)\n    >>> len(im.getdata())\n    16384\n    >>> im.getextrema()\n    ((0, 255), (0, 255), (0, 255))\n    >>> im.getpixel((0, 0))\n    (20, 20, 70)\n    >>> len(im.getprojection())\n    2\n    >>> len(im.histogram())\n    768\n    >>> \'%.7f\' % im.entropy()\n    \'8.8212866\'\n    >>> _info(im.point(list(range(256))*3))\n    (None, \'RGB\', (128, 128))\n    >>> _info(im.resize((64, 64)))\n    (None, \'RGB\', (64, 64))\n    >>> _info(im.rotate(45))\n    (None, \'RGB\', (128, 128))\n    >>> [_info(ch) for ch in im.split()]\n    [(None, \'L\', (128, 128)), (None, \'L\', (128, 128)), (None, \'L\', (128, 128))]\n    >>> len(im.convert("1").tobitmap())\n    10456\n    >>> len(im.tobytes())\n    49152\n    >>> _info(im.transform((512, 512), Image.Transform.AFFINE, (1,0,0,0,1,0)))\n    (None, \'RGB\', (512, 512))\n    >>> _info(im.transform((512, 512), Image.Transform.EXTENT, (32,32,96,96)))\n    (None, \'RGB\', (512, 512))\n\n    The ImageDraw module lets you draw stuff in raster images:\n\n    >>> im = Image.new("L", (128, 128), 64)\n    >>> d = ImageDraw.ImageDraw(im)\n    >>> d.line((0, 0, 128, 128), fill=128)\n    >>> d.line((0, 128, 128, 0), fill=128)\n    >>> im.getextrema()\n    (64, 128)\n\n    In 1.1.4, you can specify colors in a number of ways:\n\n    >>> xy = 0, 0, 128, 128\n    >>> im = Image.new("RGB", (128, 128), 0)\n    >>> d = ImageDraw.ImageDraw(im)\n    >>> d.rectangle(xy, "#f00")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n    >>> d.rectangle(xy, "#ff0000")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n    >>> d.rectangle(xy, "rgb(255,0,0)")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n    >>> d.rectangle(xy, "rgb(100%,0%,0%)")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n    >>> d.rectangle(xy, "hsl(0, 100%, 50%)")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n    >>> d.rectangle(xy, "red")\n    >>> im.getpixel((0, 0))\n    (255, 0, 0)\n\n    In 1.1.6, you can use the ImageMath module to do image\n    calculations.\n\n    >>> im = ImageMath.eval("float(im + 20)", im=im.convert("L"))\n    >>> im.mode, im.size\n    (\'F\', (128, 128))\n\n    PIL can do many other things, but I\'ll leave that for another\n    day.\n\n    Cheers /F\n    '
if __name__ == '__main__':
    exit_status = 0
    features.pilinfo(sys.stdout, False)
    import doctest
    print('Running selftest:')
    status = doctest.testmod(sys.modules[__name__])
    if status[0]:
        print('*** %s tests of %d failed.' % status)
        exit_status = 1
    else:
        print('--- %s tests passed.' % status[1])
    sys.exit(exit_status)