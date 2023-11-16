from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import LPoint3, LVector3
from panda3d.core import Filename
from panda3d.physics import BaseParticleEmitter, BaseParticleRenderer
from panda3d.physics import PointParticleFactory, SpriteParticleRenderer
from panda3d.physics import LinearNoiseForce, DiscEmitter
from direct.particles.Particles import Particles
from direct.particles.ParticleEffect import ParticleEffect
from direct.particles.ForceGroup import ForceGroup
from direct.gui.OnscreenText import OnscreenText
import sys
HELP_TEXT = '\n1: Load Steam\n2: Load Dust\n3: Load Fountain\n4: Load Smoke\n5: Load Smokering\n6: Load Fireish\nESC: Quit\n'

class ParticleDemo(ShowBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ShowBase.__init__(self)
        self.title = OnscreenText(text='Panda3D: Tutorial - Particles', parent=base.a2dBottomCenter, style=1, fg=(1, 1, 1, 1), pos=(0, 0.1), scale=0.08)
        self.escapeEvent = OnscreenText(text=HELP_TEXT, parent=base.a2dTopLeft, style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.06), align=TextNode.ALeft, scale=0.05)
        self.accept('escape', sys.exit)
        self.accept('1', self.loadParticleConfig, ['steam.ptf'])
        self.accept('2', self.loadParticleConfig, ['dust.ptf'])
        self.accept('3', self.loadParticleConfig, ['fountain.ptf'])
        self.accept('4', self.loadParticleConfig, ['smoke.ptf'])
        self.accept('5', self.loadParticleConfig, ['smokering.ptf'])
        self.accept('6', self.loadParticleConfig, ['fireish.ptf'])
        self.accept('escape', sys.exit)
        base.disableMouse()
        base.camera.setPos(0, -20, 2)
        base.camLens.setFov(25)
        base.setBackgroundColor(0, 0, 0)
        base.enableParticles()
        self.t = loader.loadModel('teapot')
        self.t.setPos(0, 10, 0)
        self.t.reparentTo(render)
        self.setupLights()
        self.p = ParticleEffect()
        self.loadParticleConfig('steam.ptf')

    def loadParticleConfig(self, filename):
        if False:
            return 10
        self.p.cleanup()
        self.p = ParticleEffect()
        self.p.loadConfig(Filename(filename))
        self.p.start(self.t)
        self.p.setPos(3.0, 0.0, 2.25)

    def setupLights(self):
        if False:
            for i in range(10):
                print('nop')
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.4, 0.4, 0.35, 1))
        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setDirection(LVector3(0, 8, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        self.t.setLight(self.t.attachNewNode(directionalLight))
        self.t.setLight(self.t.attachNewNode(ambientLight))
demo = ParticleDemo()
demo.run()