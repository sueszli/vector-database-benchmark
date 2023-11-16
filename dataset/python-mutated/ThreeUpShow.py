"""ThreeUpShow is a variant of ShowBase that defines three cameras covering
different parts of the window."""
__all__ = ['ThreeUpShow']
from .ShowBase import ShowBase

class ThreeUpShow(ShowBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        ShowBase.__init__(self)

    def makeCamera(self, win, sort=0, scene=None, displayRegion=(0, 1, 0, 1), stereo=None, aspectRatio=None, clearDepth=0, clearColor=None, lens=None, camName='cam', mask=None, useCamera=None):
        if False:
            for i in range(10):
                print('nop')
        self.camRS = ShowBase.makeCamera(self, win, displayRegion=(0.5, 1, 0, 1), aspectRatio=0.67, camName='camRS')
        self.camLL = ShowBase.makeCamera(self, win, displayRegion=(0, 0.5, 0, 0.5), camName='camLL')
        self.camUR = ShowBase.makeCamera(self, win, displayRegion=(0, 0.5, 0.5, 1), camName='camUR')
        return self.camUR