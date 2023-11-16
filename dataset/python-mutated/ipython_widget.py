import ipywidgets as widgets
from IPython import get_ipython

class PrinterX:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.w = w = widgets.HTML()

    def show(self):
        if False:
            print('Hello World!')
        return self.w

    def write(self, s):
        if False:
            return 10
        self.w.value = s
print('Running from within ipython?', get_ipython() is not None)
p = PrinterX()
p.show()
p.write('ffffffffff')