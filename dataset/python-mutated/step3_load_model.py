from direct.showbase.ShowBase import ShowBase
base = ShowBase()
from panda3d.core import NodePath, TextNode
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
        self.sizescale = 0.6
        self.loadPlanets()

    def loadPlanets(self):
        if False:
            while True:
                i = 10
        self.sky = loader.loadModel('models/solar_sky_sphere')
        self.sky.reparentTo(render)
        self.sky.setScale(40)
        self.sky_tex = loader.loadTexture('models/stars_1k_tex.jpg')
        self.sky.setTexture(self.sky_tex, 1)
        self.sun = loader.loadModel('models/planet_sphere')
        self.sun.reparentTo(render)
        self.sun_tex = loader.loadTexture('models/sun_1k_tex.jpg')
        self.sun.setTexture(self.sun_tex, 1)
        self.sun.setScale(2 * self.sizescale)
w = World()
base.run()