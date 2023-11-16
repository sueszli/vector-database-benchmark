"""
Contains classes useful for 3D viewports.

Originally written by pro-rsoft,
Modified by gjeon.
Modified by Summer 2010 Carnegie Mellon University ETC PandaLE team: fixed a bug in Viewport.Close
"""
from __future__ import annotations
__all__ = ['Viewport', 'ViewportManager']
from panda3d.core import BitMask32, CollisionNode, CollisionPlane, NodePath, OrthographicLens, Plane, Point3
from direct.showbase.DirectObject import DirectObject
from direct.directtools.DirectGrid import DirectGrid
from direct.showbase.ShowBase import WindowControls
from direct.directtools.DirectGlobals import LE_CAM_MASKS, LE_showInOneCam
from .WxPandaWindow import WxPandaWindow
import wx
HORIZONTAL = wx.SPLIT_HORIZONTAL
VERTICAL = wx.SPLIT_VERTICAL
CREATENEW = 99
VPLEFT = 10
VPFRONT = 11
VPTOP = 12
VPPERSPECTIVE = 13

class ViewportManager:
    """Manages the global viewport stuff."""
    viewports: list[Viewport] = []
    gsg = None

    @staticmethod
    def initializeAll(*args, **kwargs):
        if False:
            while True:
                i = 10
        'Calls initialize() on all the viewports.'
        for v in ViewportManager.viewports:
            v.initialize(*args, **kwargs)

    @staticmethod
    def updateAll(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Calls Update() on all the viewports.'
        for v in ViewportManager.viewports:
            v.Update(*args, **kwargs)

    @staticmethod
    def layoutAll(*args, **kwargs):
        if False:
            return 10
        'Calls Layout() on all the viewports.'
        for v in ViewportManager.viewports:
            v.Layout(*args, **kwargs)

class Viewport(WxPandaWindow, DirectObject):
    """Class representing a 3D Viewport."""
    CREATENEW = CREATENEW
    VPLEFT = VPLEFT
    VPFRONT = VPFRONT
    VPTOP = VPTOP
    VPPERSPECTIVE = VPPERSPECTIVE

    def __init__(self, name, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.name = name
        DirectObject.__init__(self)
        kwargs['gsg'] = ViewportManager.gsg
        WxPandaWindow.__init__(self, *args, **kwargs)
        ViewportManager.viewports.append(self)
        if ViewportManager.gsg is None:
            ViewportManager.gsg = self.win.getGsg()
        self.camera = None
        self.lens = None
        self.camPos = None
        self.camLookAt = None
        self.initialized = False
        self.grid = None
        self.collPlane = None

    def initialize(self):
        if False:
            while True:
                i = 10
        self.Update()
        if self.win:
            self.cam2d = base.makeCamera2d(self.win)
            self.cam2d.node().setCameraMask(LE_CAM_MASKS[self.name])
        self.cam = base.camList[-1]
        self.camera = render.attachNewNode(self.name)
        self.cam.reparentTo(self.camera)
        self.camNode = self.cam.node()
        self.camNode.setCameraMask(LE_CAM_MASKS[self.name])
        self.bt = base.setupMouse(self.win, True)
        self.bt.node().setPrefix('_le_%s_' % self.name[:3])
        mw = self.bt.getParent()
        mk = mw.getParent()
        winCtrl = WindowControls(self.win, mouseWatcher=mw, cam=self.camera, camNode=self.camNode, cam2d=None, mouseKeyboard=mk, grid=self.grid)
        base.setupWindowControls(winCtrl)
        self.initialized = True
        if self.lens is not None:
            self.cam.node().setLens(self.lens)
        if self.camPos is not None:
            self.camera.setPos(self.camPos)
        if self.camLookAt is not None:
            self.camera.lookAt(self.camLookAt)
        self.camLens = self.camNode.getLens()
        if self.name in ['top', 'front', 'left']:
            x = self.ClientSize.GetWidth() * 0.1
            y = self.ClientSize.GetHeight() * 0.1
            self.camLens.setFilmSize(x, y)
        self.Bind(wx.EVT_SIZE, self.onSize)

    def Close(self):
        if False:
            for i in range(10):
                print('nop')
        'Closes the viewport.'
        if self.initialized:
            wx.Window.Close(self)
        ViewportManager.viewports.remove(self)

    def onSize(self, evt):
        if False:
            i = 10
            return i + 15
        'Invoked when the viewport is resized.'
        WxPandaWindow.onSize(self, evt)
        if self.win is not None:
            newWidth = self.ClientSize.GetWidth()
            newHeight = self.ClientSize.GetHeight()
            if hasattr(base, 'direct') and base.direct:
                for dr in base.direct.drList:
                    if dr.camNode == self.camNode:
                        dr.updateFilmSize(newWidth, newHeight)
                        break

    def onRightDown(self, evt=None):
        if False:
            print('Hello World!')
        'Invoked when the viewport is right-clicked.'
        if evt is None:
            mpos = wx.GetMouseState()
            mpos = self.ScreenToClient((mpos.x, mpos.y))
        else:
            mpos = evt.GetPosition()
        self.Update()

    @staticmethod
    def make(parent, vpType=None):
        if False:
            print('Hello World!')
        'Safe constructor that also takes CREATENEW, VPLEFT, VPTOP, etc.'
        if vpType is None or vpType == CREATENEW:
            return Viewport(parent)
        if isinstance(vpType, Viewport):
            return vpType
        if vpType == VPLEFT:
            return Viewport.makeLeft(parent)
        if vpType == VPFRONT:
            return Viewport.makeFront(parent)
        if vpType == VPTOP:
            return Viewport.makeTop(parent)
        if vpType == VPPERSPECTIVE:
            return Viewport.makePerspective(parent)
        raise TypeError('Unknown viewport type: %s' % vpType)

    @staticmethod
    def makeOrthographic(parent, name, campos):
        if False:
            return 10
        v = Viewport(name, parent)
        v.lens = OrthographicLens()
        v.lens.setFilmSize(30)
        v.camPos = campos
        v.camLookAt = Point3(0, 0, 0)
        v.grid = DirectGrid(parent=render)
        if name == 'left':
            v.grid.setHpr(0, 0, 90)
            collPlane = CollisionNode('LeftGridCol')
            collPlane.addSolid(CollisionPlane(Plane(1, 0, 0, 0)))
            collPlane.setIntoCollideMask(BitMask32.bit(21))
            v.collPlane = NodePath(collPlane)
            v.collPlane.wrtReparentTo(v.grid)
            LE_showInOneCam(v.grid, name)
        elif name == 'front':
            v.grid.setHpr(90, 0, 90)
            collPlane = CollisionNode('FrontGridCol')
            collPlane.addSolid(CollisionPlane(Plane(0, -1, 0, 0)))
            collPlane.setIntoCollideMask(BitMask32.bit(21))
            v.collPlane = NodePath(collPlane)
            v.collPlane.wrtReparentTo(v.grid)
            LE_showInOneCam(v.grid, name)
        else:
            collPlane = CollisionNode('TopGridCol')
            collPlane.addSolid(CollisionPlane(Plane(0, 0, 1, 0)))
            collPlane.setIntoCollideMask(BitMask32.bit(21))
            v.collPlane = NodePath(collPlane)
            v.collPlane.reparentTo(v.grid)
            LE_showInOneCam(v.grid, name)
        return v

    @staticmethod
    def makePerspective(parent):
        if False:
            print('Hello World!')
        v = Viewport('persp', parent)
        v.camPos = Point3(-19, -19, 19)
        v.camLookAt = Point3(0, 0, 0)
        v.grid = DirectGrid(parent=render)
        collPlane = CollisionNode('PerspGridCol')
        collPlane.addSolid(CollisionPlane(Plane(0, 0, 1, 0)))
        collPlane.setIntoCollideMask(BitMask32.bit(21))
        v.collPlane = NodePath(collPlane)
        v.collPlane.reparentTo(v.grid)
        collPlane2 = CollisionNode('PerspGridCol2')
        collPlane2.addSolid(CollisionPlane(Plane(0, 0, -1, 0)))
        collPlane2.setIntoCollideMask(BitMask32.bit(21))
        v.collPlane2 = NodePath(collPlane2)
        v.collPlane2.reparentTo(v.grid)
        LE_showInOneCam(v.grid, 'persp')
        return v

    @staticmethod
    def makeLeft(parent):
        if False:
            i = 10
            return i + 15
        return Viewport.makeOrthographic(parent, 'left', Point3(600, 0, 0))

    @staticmethod
    def makeFront(parent):
        if False:
            while True:
                i = 10
        return Viewport.makeOrthographic(parent, 'front', Point3(0, -600, 0))

    @staticmethod
    def makeTop(parent):
        if False:
            while True:
                i = 10
        return Viewport.makeOrthographic(parent, 'top', Point3(0, 0, 600))