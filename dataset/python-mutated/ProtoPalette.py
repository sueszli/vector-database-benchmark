"""
Palette for Prototyping
"""
from .ProtoPaletteBase import ProtoPaletteBase
import os

class ProtoPalette(ProtoPaletteBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.dirname = os.path.dirname(__file__)
        ProtoPaletteBase.__init__(self)