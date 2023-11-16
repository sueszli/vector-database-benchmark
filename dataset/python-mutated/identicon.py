"""
This module provides a class Identicon that can be used to generate identicons
from strings.

It provides a (slighltly modified) version of https://github.com/evuez/identicons
which has been released under the MIT license, as described in attributions.md.
"""
from base64 import b64encode
from hashlib import md5
from io import BytesIO
from PIL import Image, ImageDraw
GRID_SIZE = 5
BORDER_SIZE = 20
SQUARE_SIZE = 40

class Identicon:

    def __init__(self, str_, background='#fafbfc'):
        if False:
            i = 10
            return i + 15
        '\n\t\t`str_` is the string used to generate the identicon.\n\t\t`background` is the background of the identicon.\n\t\t'
        w = h = BORDER_SIZE * 2 + SQUARE_SIZE * GRID_SIZE
        self.image = Image.new('RGB', (w, h), background)
        self.draw = ImageDraw.Draw(self.image)
        self.hash = self.digest(str_)

    def digest(self, str_):
        if False:
            return 10
        '\n\t\tReturns a md5 numeric hash\n\t\t'
        return int(md5(str_.encode('utf-8')).hexdigest(), 16)

    def calculate(self):
        if False:
            return 10
        '\n\t\tCreates the identicon.\n\t\tFirst three bytes are used to generate the color,\n\t\tremaining bytes are used to create the drawing\n\t\t'
        color = (self.hash & 255, self.hash >> 8 & 255, self.hash >> 16 & 255)
        self.hash >>= 24
        square_x = square_y = 0
        for x in range(GRID_SIZE * (GRID_SIZE + 1) // 2):
            if self.hash & 1:
                x = BORDER_SIZE + square_x * SQUARE_SIZE
                y = BORDER_SIZE + square_y * SQUARE_SIZE
                self.draw.rectangle((x, y, x + SQUARE_SIZE, y + SQUARE_SIZE), fill=color, outline=color)
                x = BORDER_SIZE + (GRID_SIZE - 1 - square_x) * SQUARE_SIZE
                self.draw.rectangle((x, y, x + SQUARE_SIZE, y + SQUARE_SIZE), fill=color, outline=color)
            self.hash >>= 1
            square_y += 1
            if square_y == GRID_SIZE:
                square_y = 0
                square_x += 1

    def generate(self):
        if False:
            i = 10
            return i + 15
        '\n\t\tSave and show calculated identicon\n\t\t'
        self.calculate()
        with open('identicon.png', 'wb') as out:
            self.image.save(out, 'PNG')
        self.image.show()

    def base64(self, format='PNG'):
        if False:
            i = 10
            return i + 15
        "\n\t\tReturn the identicon's base64\n\n\t\tCreated by: liuzheng712\n\t\tBug report: https://github.com/liuzheng712/identicons/issues\n\t\t"
        self.calculate()
        self.image.encoderinfo = {}
        self.image.encoderconfig = ()
        buff = BytesIO()
        self.image.save(buff, format=format.upper())
        return f'data:image/png;base64,{b64encode(buff.getvalue()).decode()}'