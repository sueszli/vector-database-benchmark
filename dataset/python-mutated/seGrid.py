from direct.showbase.DirectObject import *
from direct.directtools.DirectUtil import *
from seGeometry import *
import math

class DirectGrid(NodePath, DirectObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        NodePath.__init__(self)
        self.assign(hidden.attachNewNode('DirectGrid'))
        useDirectRenderStyle(self)
        self.gridBack = loader.loadModel('models/misc/gridBack')
        self.gridBack.reparentTo(self)
        self.gridBack.setColor(0.5, 0.5, 0.5, 0.5)
        self.lines = self.attachNewNode('gridLines')
        self.minorLines = LineNodePath(self.lines)
        self.minorLines.lineNode.setName('minorLines')
        self.minorLines.setColor(VBase4(0.3, 0.55, 1, 1))
        self.minorLines.setThickness(1)
        self.majorLines = LineNodePath(self.lines)
        self.majorLines.lineNode.setName('majorLines')
        self.majorLines.setColor(VBase4(0.3, 0.55, 1, 1))
        self.majorLines.setThickness(5)
        self.centerLines = LineNodePath(self.lines)
        self.centerLines.lineNode.setName('centerLines')
        self.centerLines.setColor(VBase4(1, 0, 0, 0))
        self.centerLines.setThickness(3)
        self.snapMarker = loader.loadModel('models/misc/sphere')
        self.snapMarker.node().setName('gridSnapMarker')
        self.snapMarker.reparentTo(self)
        self.snapMarker.setColor(1, 0, 0, 1)
        self.snapMarker.setScale(0.3)
        self.snapPos = Point3(0)
        self.fXyzSnap = 1
        self.fHprSnap = 1
        self.gridSize = 100.0
        self.gridSpacing = 5.0
        self.snapAngle = 15.0
        self.enable()

    def enable(self):
        if False:
            i = 10
            return i + 15
        self.reparentTo(SEditor.group)
        self.updateGrid()
        self.fEnabled = 1

    def disable(self):
        if False:
            while True:
                i = 10
        self.reparentTo(hidden)
        self.fEnabled = 0

    def toggleGrid(self):
        if False:
            i = 10
            return i + 15
        if self.fEnabled:
            self.disable()
        else:
            self.enable()

    def isEnabled(self):
        if False:
            while True:
                i = 10
        return self.fEnabled

    def updateGrid(self):
        if False:
            return 10
        self.minorLines.reset()
        self.majorLines.reset()
        self.centerLines.reset()
        numLines = int(math.ceil(self.gridSize / self.gridSpacing))
        scaledSize = numLines * self.gridSpacing
        center = self.centerLines
        minor = self.minorLines
        major = self.majorLines
        for i in range(-numLines, numLines + 1):
            if i == 0:
                center.moveTo(i * self.gridSpacing, -scaledSize, 0)
                center.drawTo(i * self.gridSpacing, scaledSize, 0)
                center.moveTo(-scaledSize, i * self.gridSpacing, 0)
                center.drawTo(scaledSize, i * self.gridSpacing, 0)
            elif i % 5 == 0:
                major.moveTo(i * self.gridSpacing, -scaledSize, 0)
                major.drawTo(i * self.gridSpacing, scaledSize, 0)
                major.moveTo(-scaledSize, i * self.gridSpacing, 0)
                major.drawTo(scaledSize, i * self.gridSpacing, 0)
            else:
                minor.moveTo(i * self.gridSpacing, -scaledSize, 0)
                minor.drawTo(i * self.gridSpacing, scaledSize, 0)
                minor.moveTo(-scaledSize, i * self.gridSpacing, 0)
                minor.drawTo(scaledSize, i * self.gridSpacing, 0)
        center.create()
        minor.create()
        major.create()
        self.gridBack.setScale(scaledSize)

    def setXyzSnap(self, fSnap):
        if False:
            return 10
        self.fXyzSnap = fSnap

    def getXyzSnap(self):
        if False:
            i = 10
            return i + 15
        return self.fXyzSnap

    def setHprSnap(self, fSnap):
        if False:
            return 10
        self.fHprSnap = fSnap

    def getHprSnap(self):
        if False:
            while True:
                i = 10
        return self.fHprSnap

    def computeSnapPoint(self, point):
        if False:
            for i in range(10):
                print('nop')
        self.snapPos.assign(point)
        if self.fXyzSnap:
            self.snapPos.set(ROUND_TO(self.snapPos[0], self.gridSpacing), ROUND_TO(self.snapPos[1], self.gridSpacing), ROUND_TO(self.snapPos[2], self.gridSpacing))
        self.snapMarker.setPos(self.snapPos)
        return self.snapPos

    def computeSnapAngle(self, angle):
        if False:
            i = 10
            return i + 15
        return ROUND_TO(angle, self.snapAngle)

    def setSnapAngle(self, angle):
        if False:
            while True:
                i = 10
        self.snapAngle = angle

    def getSnapAngle(self):
        if False:
            while True:
                i = 10
        return self.snapAngle

    def setGridSpacing(self, spacing):
        if False:
            while True:
                i = 10
        self.gridSpacing = spacing
        self.updateGrid()

    def getGridSpacing(self):
        if False:
            i = 10
            return i + 15
        return self.gridSpacing

    def setGridSize(self, size):
        if False:
            while True:
                i = 10
        self.gridSize = size
        self.updateGrid()

    def getGridSize(self):
        if False:
            return 10
        return self.gridSize