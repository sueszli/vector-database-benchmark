from direct.showbase.ShowBase import ShowBase
base = ShowBase()
from direct.gui.DirectGui import *
from panda3d.core import TextNode
import sys

class World(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.title = OnscreenText(text='Panda3D: Tutorial 1 - Solar System', parent=base.a2dBottomRight, align=TextNode.A_right, style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=0.07)
        base.setBackgroundColor(0, 0, 0)
        base.disableMouse()
        camera.setPos(0, 0, 45)
        camera.setHpr(0, -90, 0)
        self.yearscale = 60
        self.dayscale = self.yearscale / 365.0 * 5
        self.orbitscale = 10
        self.sizescale = 0.6
        self.loadPlanets()
        self.rotatePlanets()

    def loadPlanets(self):
        if False:
            print('Hello World!')
        self.orbit_root_mercury = render.attachNewNode('orbit_root_mercury')
        self.orbit_root_venus = render.attachNewNode('orbit_root_venus')
        self.orbit_root_mars = render.attachNewNode('orbit_root_mars')
        self.orbit_root_earth = render.attachNewNode('orbit_root_earth')
        self.orbit_root_moon = self.orbit_root_earth.attachNewNode('orbit_root_moon')
        self.sky = loader.loadModel('models/solar_sky_sphere')
        self.sky_tex = loader.loadTexture('models/stars_1k_tex.jpg')
        self.sky.setTexture(self.sky_tex, 1)
        self.sky.reparentTo(render)
        self.sky.setScale(40)
        self.sun = loader.loadModel('models/planet_sphere')
        self.sun_tex = loader.loadTexture('models/sun_1k_tex.jpg')
        self.sun.setTexture(self.sun_tex, 1)
        self.sun.reparentTo(render)
        self.sun.setScale(2 * self.sizescale)
        self.mercury = loader.loadModel('models/planet_sphere')
        self.mercury_tex = loader.loadTexture('models/mercury_1k_tex.jpg')
        self.mercury.setTexture(self.mercury_tex, 1)
        self.mercury.reparentTo(self.orbit_root_mercury)
        self.mercury.setPos(0.38 * self.orbitscale, 0, 0)
        self.mercury.setScale(0.385 * self.sizescale)
        self.venus = loader.loadModel('models/planet_sphere')
        self.venus_tex = loader.loadTexture('models/venus_1k_tex.jpg')
        self.venus.setTexture(self.venus_tex, 1)
        self.venus.reparentTo(self.orbit_root_venus)
        self.venus.setPos(0.72 * self.orbitscale, 0, 0)
        self.venus.setScale(0.923 * self.sizescale)
        self.mars = loader.loadModel('models/planet_sphere')
        self.mars_tex = loader.loadTexture('models/mars_1k_tex.jpg')
        self.mars.setTexture(self.mars_tex, 1)
        self.mars.reparentTo(self.orbit_root_mars)
        self.mars.setPos(1.52 * self.orbitscale, 0, 0)
        self.mars.setScale(0.515 * self.sizescale)
        self.earth = loader.loadModel('models/planet_sphere')
        self.earth_tex = loader.loadTexture('models/earth_1k_tex.jpg')
        self.earth.setTexture(self.earth_tex, 1)
        self.earth.reparentTo(self.orbit_root_earth)
        self.earth.setScale(self.sizescale)
        self.earth.setPos(self.orbitscale, 0, 0)
        self.orbit_root_moon.setPos(self.orbitscale, 0, 0)
        self.moon = loader.loadModel('models/planet_sphere')
        self.moon_tex = loader.loadTexture('models/moon_1k_tex.jpg')
        self.moon.setTexture(self.moon_tex, 1)
        self.moon.reparentTo(self.orbit_root_moon)
        self.moon.setScale(0.1 * self.sizescale)
        self.moon.setPos(0.1 * self.orbitscale, 0, 0)

    def rotatePlanets(self):
        if False:
            for i in range(10):
                print('nop')
        self.day_period_sun = self.sun.hprInterval(20, (360, 0, 0))
        self.orbit_period_mercury = self.orbit_root_mercury.hprInterval(0.241 * self.yearscale, (360, 0, 0))
        self.day_period_mercury = self.mercury.hprInterval(59 * self.dayscale, (360, 0, 0))
        self.orbit_period_venus = self.orbit_root_venus.hprInterval(0.615 * self.yearscale, (360, 0, 0))
        self.day_period_venus = self.venus.hprInterval(243 * self.dayscale, (360, 0, 0))
        self.orbit_period_earth = self.orbit_root_earth.hprInterval(self.yearscale, (360, 0, 0))
        self.day_period_earth = self.earth.hprInterval(self.dayscale, (360, 0, 0))
        self.orbit_period_moon = self.orbit_root_moon.hprInterval(0.0749 * self.yearscale, (360, 0, 0))
        self.day_period_moon = self.moon.hprInterval(0.0749 * self.yearscale, (360, 0, 0))
        self.orbit_period_mars = self.orbit_root_mars.hprInterval(1.881 * self.yearscale, (360, 0, 0))
        self.day_period_mars = self.mars.hprInterval(1.03 * self.dayscale, (360, 0, 0))
        self.day_period_sun.loop()
        self.orbit_period_mercury.loop()
        self.day_period_mercury.loop()
        self.orbit_period_venus.loop()
        self.day_period_venus.loop()
        self.orbit_period_earth.loop()
        self.day_period_earth.loop()
        self.orbit_period_moon.loop()
        self.day_period_moon.loop()
        self.orbit_period_mars.loop()
        self.day_period_mars.loop()
w = World()
base.run()