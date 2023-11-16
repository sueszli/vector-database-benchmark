"""
(Experimental) WCK-style drawing interface operations

.. seealso:: :py:mod:`PIL.ImageDraw`
"""
from . import Image, ImageColor, ImageDraw, ImageFont, ImagePath

class Pen:
    """Stores an outline color and width."""

    def __init__(self, color, width=1, opacity=255):
        if False:
            print('Hello World!')
        self.color = ImageColor.getrgb(color)
        self.width = width

class Brush:
    """Stores a fill color"""

    def __init__(self, color, opacity=255):
        if False:
            return 10
        self.color = ImageColor.getrgb(color)

class Font:
    """Stores a TrueType font and color"""

    def __init__(self, color, file, size=12):
        if False:
            return 10
        self.color = ImageColor.getrgb(color)
        self.font = ImageFont.truetype(file, size)

class Draw:
    """
    (Experimental) WCK-style drawing interface
    """

    def __init__(self, image, size=None, color=None):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(image, 'im'):
            image = Image.new(image, size, color)
        self.draw = ImageDraw.Draw(image)
        self.image = image
        self.transform = None

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        return self.image

    def render(self, op, xy, pen, brush=None):
        if False:
            for i in range(10):
                print('nop')
        outline = fill = None
        width = 1
        if isinstance(pen, Pen):
            outline = pen.color
            width = pen.width
        elif isinstance(brush, Pen):
            outline = brush.color
            width = brush.width
        if isinstance(brush, Brush):
            fill = brush.color
        elif isinstance(pen, Brush):
            fill = pen.color
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        if op == 'line':
            self.draw.line(xy, fill=outline, width=width)
        else:
            getattr(self.draw, op)(xy, fill=fill, outline=outline)

    def settransform(self, offset):
        if False:
            i = 10
            return i + 15
        'Sets a transformation offset.'
        (xoffset, yoffset) = offset
        self.transform = (1, 0, xoffset, 0, 1, yoffset)

    def arc(self, xy, start, end, *options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draws an arc (a portion of a circle outline) between the start and end\n        angles, inside the given bounding box.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.arc`\n        '
        self.render('arc', xy, start, end, *options)

    def chord(self, xy, start, end, *options):
        if False:
            print('Hello World!')
        '\n        Same as :py:meth:`~PIL.ImageDraw2.Draw.arc`, but connects the end points\n        with a straight line.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.chord`\n        '
        self.render('chord', xy, start, end, *options)

    def ellipse(self, xy, *options):
        if False:
            print('Hello World!')
        '\n        Draws an ellipse inside the given bounding box.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.ellipse`\n        '
        self.render('ellipse', xy, *options)

    def line(self, xy, *options):
        if False:
            i = 10
            return i + 15
        '\n        Draws a line between the coordinates in the ``xy`` list.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.line`\n        '
        self.render('line', xy, *options)

    def pieslice(self, xy, start, end, *options):
        if False:
            i = 10
            return i + 15
        '\n        Same as arc, but also draws straight lines between the end points and the\n        center of the bounding box.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.pieslice`\n        '
        self.render('pieslice', xy, start, end, *options)

    def polygon(self, xy, *options):
        if False:
            i = 10
            return i + 15
        '\n        Draws a polygon.\n\n        The polygon outline consists of straight lines between the given\n        coordinates, plus a straight line between the last and the first\n        coordinate.\n\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.polygon`\n        '
        self.render('polygon', xy, *options)

    def rectangle(self, xy, *options):
        if False:
            i = 10
            return i + 15
        '\n        Draws a rectangle.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.rectangle`\n        '
        self.render('rectangle', xy, *options)

    def text(self, xy, text, font):
        if False:
            print('Hello World!')
        '\n        Draws the string at the given position.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.text`\n        '
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        self.draw.text(xy, text, font=font.font, fill=font.color)

    def textbbox(self, xy, text, font):
        if False:
            while True:
                i = 10
        '\n        Returns bounding box (in pixels) of given text.\n\n        :return: ``(left, top, right, bottom)`` bounding box\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textbbox`\n        '
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        return self.draw.textbbox(xy, text, font=font.font)

    def textlength(self, text, font):
        if False:
            i = 10
            return i + 15
        '\n        Returns length (in pixels) of given text.\n        This is the amount by which following text should be offset.\n\n        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textlength`\n        '
        return self.draw.textlength(text, font=font.font)