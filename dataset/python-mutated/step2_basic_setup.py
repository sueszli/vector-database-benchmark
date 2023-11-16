from direct.showbase.ShowBase import ShowBase
base = ShowBase()
from panda3d.core import *
from direct.gui.DirectGui import *
import sys

class World(object):

    def __init__(self):
        if False:
            return 10
        self.title = OnscreenText(text='Panda3D: Tutorial 1 - Solar System', parent=base.a2dBottomRight, align=TextNode.A_right, style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=0.07)
        base.setBackgroundColor(0, 0, 0)
        base.disableMouse()
        camera.setPos(0, 0, 45)
        camera.setHpr(0, -90, 0)
w = World()
base.run()