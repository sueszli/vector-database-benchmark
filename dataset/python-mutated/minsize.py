""" Test minsize property.
"""
from flexx import flx

class Tester(flx.Widget):

    def init(self):
        if False:
            i = 10
            return i + 15
        super().init()
        with flx.VBox():
            flx.Label(text='You should see 5 pairs of buttons')
            with flx.HFix():
                with flx.GroupWidget(title='asdas'):
                    with flx.HFix():
                        flx.Button(text='foo')
                        flx.Button(text='bar')
            with flx.HFix(minsize=50):
                flx.Button(text='foo')
                flx.Button(text='bar')
            with flx.HFix():
                flx.Button(text='foo', minsize=50)
                flx.Button(text='bar')
            with flx.HFix():
                flx.Button(text='foo', style='min-height:50px;')
                flx.Button(text='bar')
            with flx.Widget():
                with flx.HFix():
                    flx.Button(text='foo')
                    flx.Button(text='bar')
            flx.Widget(flex=1, style='background:#f99;')
if __name__ == '__main__':
    m = flx.launch(Tester, 'firefox')
    flx.run()