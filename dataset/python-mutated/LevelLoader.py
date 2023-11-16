"""
This is just a sample code.

LevelLoader should be rewritten
to be game specific.

You need to define which ObjectMgr, ObjectHandler,
ObjectPalette, ProtoPalette would be used by importing section.
Then declare them in initLoader function.
You also need to define defaultPath in initLoader function, too.
"""
import os
from direct.leveleditor.LevelLoaderBase import LevelLoaderBase
from direct.leveleditor.ObjectMgr import ObjectMgr
from direct.leveleditor.ProtoPalette import ProtoPalette
from .ObjectHandler import ObjectHandler
from .ObjectPalette import ObjectPalette

class LevelLoader(LevelLoaderBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LevelLoaderBase.__init__(self)

    def initLoader(self):
        if False:
            for i in range(10):
                print('nop')
        self.defaultPath = os.path.dirname(__file__)
        from direct.showbase import ShowBaseGlobal
        base = ShowBaseGlobal.base
        base.objectPalette = ObjectPalette()
        base.protoPalette = ProtoPalette()
        base.objectHandler = ObjectHandler(None)
        base.objectMgr = ObjectMgr(None)