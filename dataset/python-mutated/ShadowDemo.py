"""Create a cheesy shadow effect by rendering the view of an
object (e.g. the local avatar) from a special camera as seen from
above (as if from the sun), using a solid gray foreground and a
solid white background, and then multitexturing that view onto the
world.

This is meant primarily as a demonstration of multipass and
multitexture rendering techniques.  It's not a particularly great
way to do shadows.
"""
__all__ = ['ShadowCaster', 'avatarShadow', 'piratesAvatarShadow', 'arbitraryShadow']
from panda3d.core import Camera, NodePath, OrthographicLens, Texture, TextureStage, VBase4
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
sc = None

class ShadowCaster:
    texXSize = 128
    texYSize = 128

    def __init__(self, lightPath, objectPath, filmX, filmY):
        if False:
            print('Hello World!')
        self.lightPath = lightPath
        self.objectPath = objectPath
        self.groundPath = None
        self.buffer = base.win.makeTextureBuffer('shadowBuffer', self.texXSize, self.texYSize)
        clearColor = VBase4(1, 1, 1, 1)
        self.buffer.setClearColor(clearColor)
        self.tex = self.buffer.getTexture()
        self.tex.setBorderColor(clearColor)
        self.tex.setWrapU(Texture.WMBorderColor)
        self.tex.setWrapV(Texture.WMBorderColor)
        dr = self.buffer.makeDisplayRegion()
        self.camera = Camera('shadowCamera')
        self.cameraPath = self.lightPath.attachNewNode(self.camera)
        self.camera.setScene(self.objectPath)
        dr.setCamera(self.cameraPath)
        initial = NodePath('initial')
        initial.setColor(0.6, 0.6, 0.6, 1, 1)
        initial.setTextureOff(2)
        initial.setLightOff(2)
        self.camera.setInitialState(initial.getState())
        self.lens = OrthographicLens()
        self.lens.setFilmSize(filmX, filmY)
        self.camera.setLens(self.lens)
        self.stage = TextureStage('shadow')
        self.objectPath.setTextureOff(self.stage)

    def setGround(self, groundPath):
        if False:
            print('Hello World!')
        ' Specifies the part of the world that is to be considered\n        the ground: this is the part onto which the rendered texture\n        will be applied. '
        if self.groundPath:
            self.groundPath.clearProjectTexture(self.stage)
        self.groundPath = groundPath
        self.groundPath.projectTexture(self.stage, self.tex, self.cameraPath)

    def clear(self):
        if False:
            return 10
        ' Undoes the effect of the ShadowCaster. '
        if self.groundPath:
            self.groundPath.clearProjectTexture(self.stage)
            self.groundPath = None
        if self.lightPath:
            self.lightPath.detachNode()
            self.lightPath = None
        if self.cameraPath:
            self.cameraPath.detachNode()
            self.cameraPath = None
            self.camera = None
            self.lens = None
        if self.buffer:
            base.graphicsEngine.removeWindow(self.buffer)
            self.tex = None
            self.buffer = None

def avatarShadow():
    if False:
        return 10
    base.localAvatar.dropShadow.stash()
    objectPath = base.localAvatar.getGeomNode()
    shadowCamera = objectPath.attachNewNode('shadowCamera')
    lightPath = shadowCamera.attachNewNode('lightPath')
    lightPath.setPos(5, 0, 7)

    def shadowCameraRotate(task, shadowCamera=shadowCamera):
        if False:
            print('Hello World!')
        shadowCamera.setHpr(render, 0, 0, 0)
        lightPath.lookAt(shadowCamera, 0, 0, 3)
        return Task.cont
    taskMgr.remove('shadowCamera')
    taskMgr.add(shadowCameraRotate, 'shadowCamera')
    global sc
    if sc is not None:
        sc.clear()
    sc = ShadowCaster(lightPath, objectPath, 4, 6)
    sc.setGround(render)
    return sc

def piratesAvatarShadow():
    if False:
        print('Hello World!')
    a = avatarShadow()
    base.localAvatar.getGeomNode().getChild(0).node().forceSwitch(0)
    return a

def arbitraryShadow(node):
    if False:
        while True:
            i = 10
    if hasattr(node, 'dropShadow'):
        node.dropShadow.stash()
    objectPath = node
    shadowCamera = objectPath.attachNewNode('shadowCamera')
    lightPath = shadowCamera.attachNewNode('lightPath')
    lightPath.setPos(50, 0, 50)

    def shadowCameraRotate(task, shadowCamera=shadowCamera):
        if False:
            for i in range(10):
                print('nop')
        shadowCamera.setHpr(render, 0, 0, 0)
        lightPath.lookAt(shadowCamera, 0, 0, 3)
        return Task.cont
    taskMgr.remove('shadowCamera')
    taskMgr.add(shadowCameraRotate, 'shadowCamera')
    global sc
    if sc is not None:
        sc.clear()
    sc = ShadowCaster(lightPath, objectPath, 100, 100)
    sc.setGround(render)
    return sc