"""
Piet Mondriaan was a Dutch painter who is famous for his style that looks
a little like this. Best viewed in a square window.
"""
from flexx import flx

class MyVBox(flx.VFix):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['spacing'] = kwargs.get('spacing', 15)
        kwargs['padding'] = 0
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

class MyHBox(flx.HFix):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['spacing'] = kwargs.get('spacing', 15)
        kwargs['padding'] = 0
        super().__init__(**kwargs)

class Mondriaan(flx.Widget):
    CSS = '\n    .flx-Mondriaan {background: #000;}\n    .flx-Mondriaan .edge {background:none;}\n    .flx-Mondriaan .white {background:#fff;}\n    .flx-Mondriaan .red {background:#f23;}\n    .flx-Mondriaan .blue {background:#249;}\n    .flx-Mondriaan .yellow {background:#ff7;}\n    '

    def init(self):
        if False:
            while True:
                i = 10
        with MyHBox():
            with MyVBox(flex=2):
                with MyVBox(flex=4, spacing=30):
                    flx.Widget(flex=1, css_class='white')
                    flx.Widget(flex=1, css_class='white')
                with MyVBox(flex=2, css_class='blue'):
                    flx.Widget(flex=1, css_class='edge')
                    flx.Widget(flex=1, css_class='edge')
            with MyVBox(flex=6):
                with MyVBox(flex=4, spacing=30, css_class='red'):
                    flx.Widget(flex=1, css_class='edge')
                    flx.Widget(flex=1, css_class='edge')
                with MyHBox(flex=2):
                    flx.Widget(flex=6, css_class='white')
                    with MyVBox(flex=1):
                        flx.Widget(flex=1, css_class='white')
                        flx.Widget(flex=1, css_class='yellow')
if __name__ == '__main__':
    m = flx.launch(Mondriaan, 'app')
    flx.run()