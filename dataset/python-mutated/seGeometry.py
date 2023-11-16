from panda3d.core import *
from direct.directtools.DirectGlobals import *
from direct.directtools.DirectUtil import *
import math

class LineNodePath(NodePath):

    def __init__(self, parent=None, name=None, thickness=1.0, colorVec=VBase4(1)):
        if False:
            for i in range(10):
                print('nop')
        NodePath.__init__(self)
        if parent is None:
            parent = hidden
        self.lineNode = GeomNode('lineNode')
        self.assign(parent.attachNewNode(self.lineNode))
        if name:
            self.setName(name)
        ls = self.lineSegs = LineSegs()
        ls.setThickness(thickness)
        ls.setColor(colorVec)

    def moveTo(self, *_args):
        if False:
            return 10
        self.lineSegs.moveTo(*_args)

    def drawTo(self, *_args):
        if False:
            return 10
        self.lineSegs.drawTo(*_args)

    def create(self, frameAccurate=0):
        if False:
            return 10
        self.lineSegs.create(self.lineNode, frameAccurate)

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.lineSegs.reset()
        self.lineNode.removeAllGeoms()

    def isEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lineSegs.isEmpty()

    def setThickness(self, thickness):
        if False:
            while True:
                i = 10
        self.lineSegs.setThickness(thickness)

    def setColor(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        self.lineSegs.setColor(*_args)

    def setVertex(self, *_args):
        if False:
            return 10
        self.lineSegs.setVertex(*_args)

    def setVertexColor(self, vertex, *_args):
        if False:
            i = 10
            return i + 15
        self.lineSegs.setVertexColor(*(vertex,) + _args)

    def getCurrentPosition(self):
        if False:
            i = 10
            return i + 15
        return self.lineSegs.getCurrentPosition()

    def getNumVertices(self):
        if False:
            return 10
        return self.lineSegs.getNumVertices()

    def getVertex(self, index):
        if False:
            while True:
                i = 10
        return self.lineSegs.getVertex(index)

    def getVertexColor(self):
        if False:
            i = 10
            return i + 15
        return self.lineSegs.getVertexColor()

    def drawArrow(self, sv, ev, arrowAngle, arrowLength):
        if False:
            print('Hello World!')
        '\n        Do the work of moving the cursor around to draw an arrow from\n        sv to ev. Hack: the arrows take the z value of the end point\n        '
        self.moveTo(sv)
        self.drawTo(ev)
        v = sv - ev
        angle = math.atan2(v[1], v[0])
        a1 = angle + deg2Rad(arrowAngle)
        a2 = angle - deg2Rad(arrowAngle)
        a1x = arrowLength * math.cos(a1)
        a1y = arrowLength * math.sin(a1)
        a2x = arrowLength * math.cos(a2)
        a2y = arrowLength * math.sin(a2)
        z = ev[2]
        self.moveTo(ev)
        self.drawTo(Point3(ev + Point3(a1x, a1y, z)))
        self.moveTo(ev)
        self.drawTo(Point3(ev + Point3(a2x, a2y, z)))

    def drawArrow2d(self, sv, ev, arrowAngle, arrowLength):
        if False:
            print('Hello World!')
        '\n        Do the work of moving the cursor around to draw an arrow from\n        sv to ev. Hack: the arrows take the z value of the end point\n        '
        self.moveTo(sv)
        self.drawTo(ev)
        v = sv - ev
        angle = math.atan2(v[2], v[0])
        a1 = angle + deg2Rad(arrowAngle)
        a2 = angle - deg2Rad(arrowAngle)
        a1x = arrowLength * math.cos(a1)
        a1y = arrowLength * math.sin(a1)
        a2x = arrowLength * math.cos(a2)
        a2y = arrowLength * math.sin(a2)
        self.moveTo(ev)
        self.drawTo(Point3(ev + Point3(a1x, 0.0, a1y)))
        self.moveTo(ev)
        self.drawTo(Point3(ev + Point3(a2x, 0.0, a2y)))

    def drawLines(self, lineList):
        if False:
            while True:
                i = 10
        '\n        Given a list of lists of points, draw a separate line for each list\n        '
        for pointList in lineList:
            self.moveTo(*pointList[0])
            for point in pointList[1:]:
                self.drawTo(*point)

def planeIntersect(lineOrigin, lineDir, planeOrigin, normal):
    if False:
        print('Hello World!')
    t = 0
    offset = planeOrigin - lineOrigin
    t = offset.dot(normal) / lineDir.dot(normal)
    hitPt = lineDir * t
    return hitPt + lineOrigin

def getNearProjectionPoint(nodePath):
    if False:
        print('Hello World!')
    origin = nodePath.getPos(SEditor.camera)
    if origin[1] != 0.0:
        return origin * (SEditor.dr.near / origin[1])
    else:
        return Point3(0, SEditor.dr.near, 0)

def getScreenXY(nodePath):
    if False:
        for i in range(10):
            print('nop')
    nearVec = getNearProjectionPoint(nodePath)
    nearX = CLAMP(nearVec[0], SEditor.dr.left, SEditor.dr.right)
    nearY = CLAMP(nearVec[2], SEditor.dr.bottom, SEditor.dr.top)
    percentX = (nearX - SEditor.dr.left) / SEditor.dr.nearWidth
    percentY = (nearY - SEditor.dr.bottom) / SEditor.dr.nearHeight
    screenXY = Vec3(2 * percentX - 1.0, nearVec[1], 2 * percentY - 1.0)
    return screenXY

def getCrankAngle(center):
    if False:
        i = 10
        return i + 15
    x = SEditor.dr.mouseX - center[0]
    y = SEditor.dr.mouseY - center[2]
    return 180 + rad2Deg(math.atan2(y, x))

def relHpr(nodePath, base, h, p, r):
    if False:
        return 10
    mNodePath2Base = nodePath.getMat(base)
    mBase2NewBase = Mat4()
    composeMatrix(mBase2NewBase, UNIT_VEC, VBase3(h, p, r), ZERO_VEC, CSDefault)
    mBase2NodePath = base.getMat(nodePath)
    mNodePath2Parent = nodePath.getMat()
    resultMat = mNodePath2Base * mBase2NewBase
    resultMat = resultMat * mBase2NodePath
    resultMat = resultMat * mNodePath2Parent
    hpr = Vec3(0)
    decomposeMatrix(resultMat, VBase3(), hpr, VBase3(), CSDefault)
    nodePath.setHpr(hpr)

def qSlerp(startQuat, endQuat, t):
    if False:
        print('Hello World!')
    startQ = Quat(startQuat)
    destQuat = Quat.identQuat()
    cosOmega = startQ.getI() * endQuat.getI() + startQ.getJ() * endQuat.getJ() + startQ.getK() * endQuat.getK() + startQ.getR() * endQuat.getR()
    if cosOmega < 0.0:
        cosOmega *= -1
        startQ.setI(-1 * startQ.getI())
        startQ.setJ(-1 * startQ.getJ())
        startQ.setK(-1 * startQ.getK())
        startQ.setR(-1 * startQ.getR())
    if 1.0 + cosOmega > Q_EPSILON:
        if 1.0 - cosOmega > Q_EPSILON:
            omega = math.acos(cosOmega)
            sinOmega = math.sin(omega)
            startScale = math.sin((1.0 - t) * omega) / sinOmega
            endScale = math.sin(t * omega) / sinOmega
        else:
            startScale = 1.0 - t
            endScale = t
        destQuat.setI(startScale * startQ.getI() + endScale * endQuat.getI())
        destQuat.setJ(startScale * startQ.getJ() + endScale * endQuat.getJ())
        destQuat.setK(startScale * startQ.getK() + endScale * endQuat.getK())
        destQuat.setR(startScale * startQ.getR() + endScale * endQuat.getR())
    else:
        destQuat.setI(-startQ.getJ())
        destQuat.setJ(startQ.getI())
        destQuat.setK(-startQ.getR())
        destQuat.setR(startQ.getK())
        startScale = math.sin((0.5 - t) * math.pi)
        endScale = math.sin(t * math.pi)
        destQuat.setI(startScale * startQ.getI() + endScale * endQuat.getI())
        destQuat.setJ(startScale * startQ.getJ() + endScale * endQuat.getJ())
        destQuat.setK(startScale * startQ.getK() + endScale * endQuat.getK())
    return destQuat