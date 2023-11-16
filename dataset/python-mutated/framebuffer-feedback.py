from direct.showbase.ShowBase import ShowBase
from panda3d.core import GraphicsOutput
from panda3d.core import Filename, Texture
from panda3d.core import CardMaker
from panda3d.core import NodePath, TextNode
from panda3d.core import AmbientLight, DirectionalLight
from direct.showbase.DirectObject import DirectObject
from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task
from direct.actor.Actor import Actor
from random import uniform
import sys
import os

def addInstructions(pos, msg):
    if False:
        return 10
    return OnscreenText(text=msg, parent=base.a2dTopLeft, style=1, fg=(1, 1, 1, 1), pos=(0.06, -pos - 0.03), align=TextNode.ALeft, scale=0.05)

class MotionTrails(ShowBase):

    def __init__(self):
        if False:
            print('Hello World!')
        ShowBase.__init__(self)
        self.disableMouse()
        self.camera.setPos(0, -26, 4)
        self.setBackgroundColor(0, 0, 0)
        self.tex = Texture()
        self.tex.setMinfilter(Texture.FTLinear)
        self.win.addRenderTexture(self.tex, GraphicsOutput.RTMTriggeredCopyTexture)
        self.tex.setClearColor((0, 0, 0, 1))
        self.tex.clearImage()
        self.backcam = self.makeCamera2d(self.win, sort=-10)
        self.background = NodePath('background')
        self.backcam.reparentTo(self.background)
        self.background.setDepthTest(0)
        self.background.setDepthWrite(0)
        self.backcam.node().getDisplayRegion(0).setClearDepthActive(0)
        self.bcard = self.win.getTextureCard()
        self.bcard.reparentTo(self.background)
        self.bcard.setTransparency(1)
        self.fcard = self.win.getTextureCard()
        self.fcard.reparentTo(self.render2d)
        self.fcard.setTransparency(1)
        self.chooseEffectGhost()
        taskMgr.add(self.takeSnapShot, 'takeSnapShot')
        blackmaker = CardMaker('blackmaker')
        blackmaker.setColor(0, 0, 0, 1)
        blackmaker.setFrame(-1.0, -0.5, 0.65, 1.0)
        instcard = NodePath(blackmaker.generate())
        instcard.reparentTo(self.render2d)
        blackmaker.setFrame(-0.5, 0.5, -1.0, -0.85)
        titlecard = NodePath(blackmaker.generate())
        titlecard.reparentTo(self.render2d)
        if self.win.getGsg().getCopyTextureInverted():
            print('Copy texture is inverted.')
            self.bcard.setScale(1, 1, -1)
            self.fcard.setScale(1, 1, -1)
        title = OnscreenText(text='Panda3D: Tutorial - Motion Trails', fg=(1, 1, 1, 1), parent=base.a2dBottomCenter, pos=(0, 0.1), scale=0.08)
        instr0 = addInstructions(0.06, 'Press ESC to exit')
        instr1 = addInstructions(0.12, 'Press 1: Ghost effect')
        instr2 = addInstructions(0.18, 'Press 2: PaintBrush effect')
        instr3 = addInstructions(0.24, 'Press 3: Double Vision effect')
        instr4 = addInstructions(0.3, 'Press 4: Wings of Blue effect')
        instr5 = addInstructions(0.36, 'Press 5: Whirlpool effect')
        self.accept('escape', sys.exit, [0])
        self.accept('1', self.chooseEffectGhost)
        self.accept('2', self.chooseEffectPaintBrush)
        self.accept('3', self.chooseEffectDoubleVision)
        self.accept('4', self.chooseEffectWingsOfBlue)
        self.accept('5', self.chooseEffectWhirlpool)

    def takeSnapShot(self, task):
        if False:
            for i in range(10):
                print('nop')
        if task.time > self.nextclick:
            self.nextclick += 1.0 / self.clickrate
            if self.nextclick < task.time:
                self.nextclick = task.time
            self.win.triggerCopy()
        return Task.cont

    def chooseEffectGhost(self):
        if False:
            for i in range(10):
                print('nop')
        self.setBackgroundColor(0, 0, 0, 1)
        self.bcard.hide()
        self.fcard.show()
        self.fcard.setColor(1.0, 1.0, 1.0, 0.99)
        self.fcard.setScale(1.0)
        self.fcard.setPos(0, 0, 0)
        self.fcard.setR(0)
        self.clickrate = 30
        self.nextclick = 0

    def chooseEffectPaintBrush(self):
        if False:
            print('Hello World!')
        self.setBackgroundColor(0, 0, 0, 1)
        self.bcard.show()
        self.fcard.hide()
        self.bcard.setColor(1, 1, 1, 1)
        self.bcard.setScale(1.0)
        self.bcard.setPos(0, 0, 0)
        self.bcard.setR(0)
        self.clickrate = 10000
        self.nextclick = 0

    def chooseEffectDoubleVision(self):
        if False:
            return 10
        self.setBackgroundColor(0, 0, 0, 1)
        self.bcard.show()
        self.bcard.setColor(1, 1, 1, 1)
        self.bcard.setScale(1.0)
        self.bcard.setPos(-0.05, 0, 0)
        self.bcard.setR(0)
        self.fcard.show()
        self.fcard.setColor(1, 1, 1, 0.6)
        self.fcard.setScale(1.0)
        self.fcard.setPos(0.05, 0, 0)
        self.fcard.setR(0)
        self.clickrate = 10000
        self.nextclick = 0

    def chooseEffectWingsOfBlue(self):
        if False:
            i = 10
            return i + 15
        self.setBackgroundColor(0, 0, 0, 1)
        self.fcard.hide()
        self.bcard.show()
        self.bcard.setColor(1.0, 0.9, 1.0, 254.0 / 255.0)
        self.bcard.setScale(1.1, 1, 0.95)
        self.bcard.setPos(0, 0, 0.05)
        self.bcard.setR(0)
        self.clickrate = 30
        self.nextclick = 0

    def chooseEffectWhirlpool(self):
        if False:
            for i in range(10):
                print('nop')
        self.setBackgroundColor(0, 0, 0, 1)
        self.bcard.show()
        self.fcard.hide()
        self.bcard.setColor(1, 1, 1, 1)
        self.bcard.setScale(0.999)
        self.bcard.setPos(0, 0, 0)
        self.bcard.setR(1)
        self.clickrate = 10000
        self.nextclick = 0
t = MotionTrails()
character = Actor()
character.loadModel('models/dancer')
character.reparentTo(t.render)
character.loadAnims({'win': 'models/dancer'})
character.loop('win')
dlight = DirectionalLight('dlight')
alight = AmbientLight('alight')
dlnp = t.render.attachNewNode(dlight)
alnp = t.render.attachNewNode(alight)
dlight.setColor((1.0, 0.9, 0.8, 1))
alight.setColor((0.2, 0.3, 0.4, 1))
dlnp.setHpr(0, -60, 0)
t.render.setLight(dlnp)
t.render.setLight(alnp)
t.run()