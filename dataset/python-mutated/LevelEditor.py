"""
This is just a sample code.

LevelEditor, ObjectHandler, ObjectPalette should be rewritten
to be game specific.
"""
import os
import wx
from .LevelEditorUI import LevelEditorUI
from .LevelEditorBase import LevelEditorBase
from .ObjectMgr import ObjectMgr
from .AnimMgr import AnimMgr
from .ObjectHandler import ObjectHandler
from .ObjectPalette import ObjectPalette
from .ProtoPalette import ProtoPalette

class LevelEditor(LevelEditorBase):
    """ Class for Panda3D LevelEditor """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LevelEditorBase.__init__(self)
        self.settingsFile = os.path.dirname(__file__) + '/LevelEditor.cfg'
        self.objectMgr = ObjectMgr(self)
        self.animMgr = AnimMgr(self)
        self.objectPalette = ObjectPalette()
        self.objectHandler = ObjectHandler(self)
        self.protoPalette = ProtoPalette()
        self.ui = LevelEditorUI(self)
        self.ui.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        self.objectPalette.populate()
        self.protoPalette.populate()
        self.ui.objectPaletteUI.populate()
        self.ui.protoPaletteUI.populate()
        self.initialize()
        self.ui.SetCursor(wx.Cursor(wx.CURSOR_ARROW))