"""
Example that shows how to make the content of a widget scrollable.
It comes down to setting a style attribute: "overflow-y: auto;".
"""
from flexx import flx

class ScrollExample(flx.Widget):
    CSS = '\n    .flx-ScrollExample {\n        overflow-y: scroll;  // scroll or auto\n    }\n    '

    def init(self):
        if False:
            while True:
                i = 10
        with flx.Widget():
            for i in range(100):
                flx.Button(text='button ' + str(i))
if __name__ == '__main__':
    m = flx.launch(ScrollExample)
    flx.run()