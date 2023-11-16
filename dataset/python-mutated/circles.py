"""
Example that shows animated circles. Note that it would probably be more
efficient to use a canvas for this sort of thing.
"""
from time import time
from flexx import flx

class Circle(flx.Label):
    CSS = '\n    .flx-Circle {\n        background: #f00;\n        border-radius: 10px;\n        width: 10px;\n        height: 10px;\n    }\n    '

class Circles(flx.Widget):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        with flx.PinboardLayout():
            self._circles = [Circle() for i in range(32)]
        self.tick()

    def tick(self):
        if False:
            for i in range(10):
                print('nop')
        global Math, window
        t = time()
        for (i, circle) in enumerate(self._circles):
            x = Math.sin(i * 0.2 + t) * 30 + 50
            y = Math.cos(i * 0.2 + t) * 30 + 50
            circle.apply_style(dict(left=x + '%', top=y + '%'))
        window.setTimeout(self.tick, 30)
if __name__ == '__main__':
    m = flx.App(Circles).launch('app')
    flx.run()