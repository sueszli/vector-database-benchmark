"""
Simple use of a dropdown containing a tree widget
"""
from flexx import flx

class Example(flx.Widget):
    CSS = '\n        .flx-DropdownContainer > .flx-TreeWidget {\n            min-height: 150px;\n        }\n    '

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        with flx.DropdownContainer(text='Scene graph'):
            with flx.TreeWidget(max_selected=1):
                for i in range(20):
                    flx.TreeItem(text='foo %i' % i, checked=False)
if __name__ == '__main__':
    m = flx.launch(Example)
    flx.run()