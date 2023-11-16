from .LevelEditorUIBase import LevelEditorUIBase

class LevelEditorUI(LevelEditorUIBase):
    """ Class for Panda3D LevelEditor """
    appversion = '0.1'
    appname = 'Panda3D Level Editor'

    def __init__(self, editor):
        if False:
            for i in range(10):
                print('nop')
        LevelEditorUIBase.__init__(self, editor)