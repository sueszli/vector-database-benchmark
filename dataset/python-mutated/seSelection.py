from panda3d.core import GeomNode
from direct.directtools.DirectGlobals import *
from direct.directtools.DirectUtil import *
from seGeometry import *
from direct.showbase.DirectObject import *
from quad import *
COA_ORIGIN = 0
COA_CENTER = 1

class DirectNodePath(NodePath):

    def __init__(self, nodePath):
        if False:
            while True:
                i = 10
        NodePath.__init__(self)
        self.assign(nodePath)
        self.bbox = DirectBoundingBox(self)
        center = self.bbox.getCenter()
        self.mCoa2Dnp = Mat4(Mat4.identMat())
        if SEditor.coaMode == COA_CENTER:
            self.mCoa2Dnp.setRow(3, Vec4(center[0], center[1], center[2], 1))
        self.tDnp2Widget = TransformState.makeIdentity()

    def highlight(self):
        if False:
            while True:
                i = 10
        self.bbox.show()

    def dehighlight(self):
        if False:
            i = 10
            return i + 15
        self.bbox.hide()

    def getCenter(self):
        if False:
            while True:
                i = 10
        return self.bbox.getCenter()

    def getRadius(self):
        if False:
            return 10
        return self.bbox.getRadius()

    def getMin(self):
        if False:
            return 10
        return self.bbox.getMin()

    def getMax(self):
        if False:
            print('Hello World!')
        return self.bbox.getMax()

class SelectedNodePaths(DirectObject):

    def __init__(self):
        if False:
            return 10
        self.reset()

    def reset(self):
        if False:
            print('Hello World!')
        self.selectedDict = {}
        self.deselectedDict = {}
        __builtins__['last'] = self.last = None

    def select(self, nodePath, fMultiSelect=0):
        if False:
            i = 10
            return i + 15
        ' Select the specified node path.  Multiselect as required '
        if not nodePath:
            print('Nothing selected!!')
            return None
        if not fMultiSelect:
            self.deselectAll()
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if not dnp:
            dnp = self.getDeselectedDict(id)
            if dnp:
                del self.deselectedDict[id]
                dnp.highlight()
            else:
                dnp = DirectNodePath(nodePath)
                dnp.highlight()
            self.selectedDict[dnp.get_key()] = dnp
        __builtins__['last'] = self.last = dnp
        return dnp

    def deselect(self, nodePath):
        if False:
            print('Hello World!')
        ' Deselect the specified node path '
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if dnp:
            dnp.dehighlight()
            del self.selectedDict[id]
            self.deselectedDict[id] = dnp
            messenger.send('DIRECT_deselectedNodePath', [dnp])
        return dnp

    def getSelectedAsList(self):
        if False:
            return 10
        '\n        Return a list of all selected node paths.  No verification of\n        connectivity is performed on the members of the list\n        '
        return list(self.selectedDict.values())

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.getSelectedAsList()[index]

    def getSelectedDict(self, id):
        if False:
            i = 10
            return i + 15
        '\n        Search selectedDict for node path, try to repair broken node paths.\n        '
        dnp = self.selectedDict.get(id, None)
        if dnp:
            return dnp
        else:
            return None

    def getDeselectedAsList(self):
        if False:
            return 10
        return list(self.deselectedDict.values())

    def getDeselectedDict(self, id):
        if False:
            return 10
        '\n        Search deselectedDict for node path, try to repair broken node paths.\n        '
        dnp = self.deselectedDict.get(id, None)
        if dnp:
            return dnp
        else:
            return None

    def forEachSelectedNodePathDo(self, func):
        if False:
            while True:
                i = 10
        '\n        Perform given func on selected node paths.  No node path\n        connectivity verification performed\n        '
        selectedNodePaths = self.getSelectedAsList()
        for nodePath in selectedNodePaths:
            func(nodePath)

    def forEachDeselectedNodePathDo(self, func):
        if False:
            while True:
                i = 10
        '\n        Perform given func on deselected node paths.  No node path\n        connectivity verification performed\n        '
        deselectedNodePaths = self.getDeselectedAsList()
        for nodePath in deselectedNodePaths:
            func(nodePath)

    def getWrtAll(self):
        if False:
            for i in range(10):
                print('nop')
        self.forEachSelectedNodePathDo(self.getWrt)

    def getWrt(self, nodePath):
        if False:
            return 10
        nodePath.tDnp2Widget = nodePath.getTransform(SEditor.widget)

    def moveWrtWidgetAll(self):
        if False:
            i = 10
            return i + 15
        self.forEachSelectedNodePathDo(self.moveWrtWidget)

    def moveWrtWidget(self, nodePath):
        if False:
            for i in range(10):
                print('nop')
        nodePath.setTransform(SEditor.widget, nodePath.tDnp2Widget)

    def deselectAll(self):
        if False:
            while True:
                i = 10
        self.forEachSelectedNodePathDo(self.deselect)

    def highlightAll(self):
        if False:
            return 10
        self.forEachSelectedNodePathDo(DirectNodePath.highlight)

    def dehighlightAll(self):
        if False:
            while True:
                i = 10
        self.forEachSelectedNodePathDo(DirectNodePath.dehighlight)

    def removeSelected(self):
        if False:
            for i in range(10):
                print('nop')
        selected = self.last
        if selected:
            selected.remove()
        __builtins__['last'] = self.last = None

    def removeAll(self):
        if False:
            print('Hello World!')
        self.forEachSelectedNodePathDo(NodePath.remove)

    def toggleVis(self, nodePath):
        if False:
            print('Hello World!')
        if nodePath.is_hidden():
            nodePath.show()
        else:
            nodePath.hide()

    def toggleVisSelected(self):
        if False:
            return 10
        selected = self.last
        if selected:
            if selected.is_hidden():
                selected.show()
            else:
                selected.hide()

    def toggleVisAll(self):
        if False:
            i = 10
            return i + 15
        self.forEachSelectedNodePathDo(self.toggleVis)

    def isolateSelected(self):
        if False:
            i = 10
            return i + 15
        selected = self.last
        if selected:
            selected.isolate()

    def getDirectNodePath(self, nodePath):
        if False:
            print('Hello World!')
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if dnp:
            return dnp
        return self.getDeselectedDict(id)

    def getNumSelected(self):
        if False:
            return 10
        return len(self.selectedDict.keys())

class DirectBoundingBox:

    def __init__(self, nodePath):
        if False:
            i = 10
            return i + 15
        self.nodePath = nodePath
        self.computeTightBounds()
        self.lines = self.createBBoxLines()

    def computeTightBounds(self):
        if False:
            for i in range(10):
                print('nop')
        tMat = Mat4()
        tMat.assign(self.nodePath.getMat())
        self.nodePath.clearMat()
        self.min = Point3(0)
        self.max = Point3(0)
        self.nodePath.calcTightBounds(self.min, self.max)
        self.center = Point3((self.min + self.max) / 2.0)
        self.radius = Vec3(self.max - self.min).length()
        self.nodePath.setMat(tMat)
        del tMat

    def computeBounds(self):
        if False:
            return 10
        self.bounds = self.getBounds()
        if self.bounds.isEmpty() or self.bounds.isInfinite():
            self.center = Point3(0)
            self.radius = 1.0
        else:
            self.center = self.bounds.getCenter()
            self.radius = self.bounds.getRadius()
        self.min = Point3(self.center - Point3(self.radius))
        self.max = Point3(self.center + Point3(self.radius))

    def createBBoxLines(self):
        if False:
            while True:
                i = 10
        lines = LineNodePath(hidden)
        lines.node().setName('bboxLines')
        lines.setColor(VBase4(1.0, 0.0, 0.0, 1.0))
        lines.setThickness(0.5)
        minX = self.min[0]
        minY = self.min[1]
        minZ = self.min[2]
        maxX = self.max[0]
        maxY = self.max[1]
        maxZ = self.max[2]
        lines.moveTo(minX, minY, minZ)
        lines.drawTo(maxX, minY, minZ)
        lines.drawTo(maxX, maxY, minZ)
        lines.drawTo(minX, maxY, minZ)
        lines.drawTo(minX, minY, minZ)
        lines.drawTo(minX, minY, maxZ)
        lines.drawTo(maxX, minY, maxZ)
        lines.drawTo(maxX, maxY, maxZ)
        lines.drawTo(minX, maxY, maxZ)
        lines.drawTo(minX, minY, maxZ)
        lines.moveTo(maxX, minY, minZ)
        lines.drawTo(maxX, minY, maxZ)
        lines.moveTo(maxX, maxY, minZ)
        lines.drawTo(maxX, maxY, maxZ)
        lines.moveTo(minX, maxY, minZ)
        lines.drawTo(minX, maxY, maxZ)
        lines.create()
        useDirectRenderStyle(lines)
        return lines

    def updateBBoxLines(self):
        if False:
            while True:
                i = 10
        ls = self.lines.lineSegs
        minX = self.min[0]
        minY = self.min[1]
        minZ = self.min[2]
        maxX = self.max[0]
        maxY = self.max[1]
        maxZ = self.max[2]
        ls.setVertex(0, minX, minY, minZ)
        ls.setVertex(1, maxX, minY, minZ)
        ls.setVertex(2, maxX, maxY, minZ)
        ls.setVertex(3, minX, maxY, minZ)
        ls.setVertex(4, minX, minY, minZ)
        ls.setVertex(5, minX, minY, maxZ)
        ls.setVertex(6, maxX, minY, maxZ)
        ls.setVertex(7, maxX, maxY, maxZ)
        ls.setVertex(8, minX, maxY, maxZ)
        ls.setVertex(9, minX, minY, maxZ)
        ls.setVertex(10, maxX, minY, minZ)
        ls.setVertex(11, maxX, minY, maxZ)
        ls.setVertex(12, maxX, maxY, minZ)
        ls.setVertex(13, maxX, maxY, maxZ)
        ls.setVertex(14, minX, maxY, minZ)
        ls.setVertex(15, minX, maxY, maxZ)

    def getBounds(self):
        if False:
            for i in range(10):
                print('nop')
        nodeBounds = BoundingSphere()
        nodeBounds.extendBy(self.nodePath.node().getInternalBound())
        for child in self.nodePath.getChildren():
            nodeBounds.extendBy(child.getBounds())
        return nodeBounds.makeCopy()

    def show(self):
        if False:
            while True:
                i = 10
        self.lines.reparentTo(self.nodePath)

    def hide(self):
        if False:
            while True:
                i = 10
        self.lines.reparentTo(hidden)

    def getCenter(self):
        if False:
            while True:
                i = 10
        return self.center

    def getRadius(self):
        if False:
            print('Hello World!')
        return self.radius

    def getMin(self):
        if False:
            while True:
                i = 10
        return self.min

    def getMax(self):
        if False:
            while True:
                i = 10
        return self.max

    def vecAsString(self, vec):
        if False:
            return 10
        return '%.2f %.2f %.2f' % (vec[0], vec[1], vec[2])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.__class__) + '\nNodePath:\t%s\n' % self.nodePath.getName() + 'Min:\t\t%s\n' % self.vecAsString(self.min) + 'Max:\t\t%s\n' % self.vecAsString(self.max) + 'Center:\t\t%s\n' % self.vecAsString(self.center) + 'Radius:\t\t%.2f' % self.radius

class SelectionQueue(CollisionHandlerQueue):

    def __init__(self, parentNP=render):
        if False:
            for i in range(10):
                print('nop')
        CollisionHandlerQueue.__init__(self)
        self.index = -1
        self.entry = None
        self.skipFlags = SKIP_NONE
        self.collisionNodePath = NodePath(CollisionNode('collisionNP'))
        self.setParentNP(parentNP)
        self.collisionNodePath.hide()
        self.collisionNode = self.collisionNodePath.node()
        self.collideWithGeom()
        self.ct = CollisionTraverser()
        self.ct.addCollider(self.collisionNodePath, self)
        self.unpickable = UNPICKABLE

    def setParentNP(self, parentNP):
        if False:
            return 10
        self.collisionNodePath.reparentTo(parentNP)

    def addCollider(self, collider):
        if False:
            print('Hello World!')
        self.collider = collider
        self.collisionNode.addSolid(self.collider)

    def collideWithBitMask(self, bitMask):
        if False:
            print('Hello World!')
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        self.collisionNode.setFromCollideMask(bitMask)

    def collideWithGeom(self):
        if False:
            print('Hello World!')
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        self.collisionNode.setFromCollideMask(GeomNode.getDefaultCollideMask())

    def collideWithWidget(self):
        if False:
            i = 10
            return i + 15
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        mask = BitMask32()
        mask.setBit(31)
        self.collisionNode.setFromCollideMask(mask)

    def addUnpickable(self, item):
        if False:
            return 10
        if item not in self.unpickable:
            self.unpickable.append(item)

    def removeUnpickable(self, item):
        if False:
            for i in range(10):
                print('nop')
        if item in self.unpickable:
            self.unpickable.remove(item)

    def setCurrentIndex(self, index):
        if False:
            i = 10
            return i + 15
        if index < 0 or index >= self.getNumEntries():
            self.index = -1
        else:
            self.index = index

    def setCurrentEntry(self, entry):
        if False:
            return 10
        self.entry = entry

    def getCurrentEntry(self):
        if False:
            print('Hello World!')
        return self.entry

    def isEntryBackfacing(self, entry):
        if False:
            return 10
        if not entry.hasSurfaceNormal():
            return 0
        fromNodePath = entry.getFromNodePath()
        v = Vec3(entry.getSurfacePoint(fromNodePath))
        n = entry.getSurfaceNormal(fromNodePath)
        if self.collisionNodePath.getParent() != base.cam:
            p2cam = self.collisionNodePath.getParent().getMat(base.cam)
            v = Vec3(p2cam.xformPoint(v))
            n = p2cam.xformVec(n)
        v.normalize()
        return v.dot(n) >= 0

    def findNextCollisionEntry(self, skipFlags=SKIP_NONE):
        if False:
            return 10
        return self.findCollisionEntry(skipFlags, self.index + 1)

    def findCollisionEntry(self, skipFlags=SKIP_NONE, startIndex=0):
        if False:
            print('Hello World!')
        self.setCurrentIndex(-1)
        self.setCurrentEntry(None)
        for i in range(startIndex, self.getNumEntries()):
            entry = self.getEntry(i)
            nodePath = entry.getIntoNodePath()
            if skipFlags & SKIP_HIDDEN and nodePath.isHidden():
                pass
            elif skipFlags & SKIP_BACKFACE and self.isEntryBackfacing(entry):
                pass
            elif skipFlags & SKIP_CAMERA and camera in nodePath.getAncestors():
                pass
            elif skipFlags & SKIP_UNPICKABLE and nodePath.getName() in self.unpickable:
                pass
            else:
                self.setCurrentIndex(i)
                self.setCurrentEntry(entry)
                break
        return self.getCurrentEntry()

class SelectionRay(SelectionQueue):

    def __init__(self, parentNP=render):
        if False:
            while True:
                i = 10
        SelectionQueue.__init__(self, parentNP)
        self.addCollider(CollisionRay())

    def pick(self, targetNodePath, xy=None):
        if False:
            print('Hello World!')
        if xy:
            mx = xy[0]
            my = xy[1]
        elif base.direct:
            mx = SEditor.dr.mouseX
            my = SEditor.dr.mouseY
        else:
            if not base.mouseWatcherNode.hasMouse():
                self.clearEntries()
                return
            mx = base.mouseWatcherNode.getMouseX()
            my = base.mouseWatcherNode.getMouseY()
        self.collider.setFromLens(base.camNode, mx, my)
        self.ct.traverse(targetNodePath)
        self.sortEntries()

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=render, skipFlags=SKIP_ALL):
        if False:
            print('Hello World!')
        self.collideWithBitMask(bitMask)
        self.pick(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pickGeom(self, targetNodePath=render, skipFlags=SKIP_ALL, xy=None):
        if False:
            for i in range(10):
                print('nop')
        self.collideWithGeom()
        self.pick(targetNodePath, xy=xy)
        return self.findCollisionEntry(skipFlags)

    def pickWidget(self, targetNodePath=render, skipFlags=SKIP_NONE):
        if False:
            i = 10
            return i + 15
        self.collideWithWidget()
        self.pick(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pick3D(self, targetNodePath, origin, dir):
        if False:
            return 10
        self.collider.setOrigin(origin)
        self.collider.setDirection(dir)
        self.ct.traverse(targetNodePath)
        self.sortEntries()

    def pickGeom3D(self, targetNodePath=render, origin=Point3(0), dir=Vec3(0, 0, -1), skipFlags=SKIP_HIDDEN | SKIP_CAMERA):
        if False:
            while True:
                i = 10
        self.collideWithGeom()
        self.pick3D(targetNodePath, origin, dir)
        return self.findCollisionEntry(skipFlags)

    def pickBitMask3D(self, bitMask=BitMask32.allOff(), targetNodePath=render, origin=Point3(0), dir=Vec3(0, 0, -1), skipFlags=SKIP_ALL):
        if False:
            i = 10
            return i + 15
        self.collideWithBitMask(bitMask)
        self.pick3D(targetNodePath, origin, dir)
        return self.findCollisionEntry(skipFlags)

class SelectionSegment(SelectionQueue):

    def __init__(self, parentNP=render, numSegments=1):
        if False:
            print('Hello World!')
        SelectionQueue.__init__(self, parentNP)
        self.colliders = []
        self.numColliders = 0
        for i in range(numSegments):
            self.addCollider(CollisionSegment())

    def addCollider(self, collider):
        if False:
            i = 10
            return i + 15
        self.colliders.append(collider)
        self.collisionNode.addSolid(collider)
        self.numColliders += 1

    def pickGeom(self, targetNodePath=render, endPointList=[], skipFlags=SKIP_HIDDEN | SKIP_CAMERA):
        if False:
            for i in range(10):
                print('nop')
        self.collideWithGeom()
        for i in range(min(len(endPointList), self.numColliders)):
            (pointA, pointB) = endPointList[i]
            collider = self.colliders[i]
            collider.setPointA(pointA)
            collider.setPointB(pointB)
        self.ct.traverse(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=render, endPointList=[], skipFlags=SKIP_HIDDEN | SKIP_CAMERA):
        if False:
            while True:
                i = 10
        self.collideWithBitMask(bitMask)
        for i in range(min(len(endPointList), self.numColliders)):
            (pointA, pointB) = endPointList[i]
            collider = self.colliders[i]
            collider.setPointA(pointA)
            collider.setPointB(pointB)
        self.ct.traverse(targetNodePath)
        return self.findCollisionEntry(skipFlags)

class SelectionSphere(SelectionQueue):

    def __init__(self, parentNP=render, numSpheres=1):
        if False:
            print('Hello World!')
        SelectionQueue.__init__(self, parentNP)
        self.colliders = []
        self.numColliders = 0
        for i in range(numSpheres):
            self.addCollider(CollisionSphere(Point3(0), 1))

    def addCollider(self, collider):
        if False:
            while True:
                i = 10
        self.colliders.append(collider)
        self.collisionNode.addSolid(collider)
        self.numColliders += 1

    def setCenter(self, i, center):
        if False:
            i = 10
            return i + 15
        c = self.colliders[i]
        c.setCenter(center)

    def setRadius(self, i, radius):
        if False:
            for i in range(10):
                print('nop')
        c = self.colliders[i]
        c.setRadius(radius)

    def setCenterRadius(self, i, center, radius):
        if False:
            while True:
                i = 10
        c = self.colliders[i]
        c.setCenter(center)
        c.setRadius(radius)

    def isEntryBackfacing(self, entry):
        if False:
            print('Hello World!')
        fromNodePath = entry.getFromNodePath()
        v = Vec3(entry.getSurfacePoint(fromNodePath) - entry.getFrom().getCenter())
        n = entry.getSurfaceNormal(fromNodePath)
        if v.length() < 0.05:
            return 1
        v.normalize()
        return v.dot(n) >= 0

    def pick(self, targetNodePath, skipFlags):
        if False:
            for i in range(10):
                print('nop')
        self.ct.traverse(targetNodePath)
        self.sortEntries()
        return self.findCollisionEntry(skipFlags)

    def pickGeom(self, targetNodePath=render, skipFlags=SKIP_HIDDEN | SKIP_CAMERA):
        if False:
            for i in range(10):
                print('nop')
        self.collideWithGeom()
        return self.pick(targetNodePath, skipFlags)

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=render, skipFlags=SKIP_HIDDEN | SKIP_CAMERA):
        if False:
            for i in range(10):
                print('nop')
        self.collideWithBitMask(bitMask)
        return self.pick(targetNodePath, skipFlags)