"""
Import apps from other example modules, and host these as widgets in a
single app.
"""
from flexx import flx
from flexxamples.demos.drawing import Drawing
from flexxamples.howtos.splitters import Split
from flexxamples.demos.twente import Twente

class MultiApp(flx.TabLayout):

    def init(self):
        if False:
            while True:
                i = 10
        Drawing(title='Drawing')
        Split(title='Split')
        Twente(title='Twente')
if __name__ == '__main__':
    flx.launch(MultiApp)
    flx.run()