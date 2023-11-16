"""
This example demonstrates what things from Python land can be used in JS.

Flexx detects what names are used in the transpiled JS of a JsComponent
(a widget, in this case), and tries to look these up in the module,
converting the used objects if possible.

Check out the source of the generated page to see what Flexx did.

Note that once running, there is no interaction with the Python side, so this
example can be exported to standalone HTML.
"""
from flexx import flx
info = dict(name='John', age=42)
from sys import version

def poly(x, *coefs):
    if False:
        while True:
            i = 10
    degree = len(coefs) - 1
    y = 0
    for coef in coefs:
        y += coef * x ** degree
        degree -= 1
    return y
from html import escape

class UsingPython(flx.Widget):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        lines = []
        lines.append('This JS was generated from Python ' + version)
        lines.append('Person %s is %i years old' % (info['name'], info['age']))
        lines.append('Evaling 4*x**2 + 5*x + 6 with x=4: ' + poly(4, 4, 5, 6))
        lines.append('... and with x=12: ' + poly(12, 4, 5, 6))
        lines.append('String with escaped html: ' + escape('html <tags>!'))
        lines.append('String with escaped html: ' + escape('Woezel & Pip'))
        self.label = flx.Label(wrap=0, html='<br />'.join(lines))
if __name__ == '__main__':
    m = flx.launch(UsingPython, 'browser')
    flx.run()