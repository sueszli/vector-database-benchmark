from panda3d.core import BitMask32, BoundingSphere, CollisionHandlerQueue, CollisionNode, CollisionRay, CollisionSegment, CollisionSphere, CollisionTraverser, GeomNode, Mat4, NodePath, Point3, TransformState, VBase4, Vec3, Vec4
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
from . import DirectGlobals as DG
from .DirectUtil import useDirectRenderStyle
from .DirectGeometry import LineNodePath
COA_ORIGIN = 0
COA_CENTER = 1

class DirectNodePath(NodePath):

    def __init__(self, nodePath, bboxColor=None):
        if False:
            for i in range(10):
                print('nop')
        NodePath.__init__(self)
        self.assign(nodePath)
        self.bbox = DirectBoundingBox(self, bboxColor)
        center = self.bbox.getCenter()
        self.mCoa2Dnp = Mat4(Mat4.identMat())
        if base.direct.coaMode == COA_CENTER:
            self.mCoa2Dnp.setRow(3, Vec4(center[0], center[1], center[2], 1))
        self.tDnp2Widget = TransformState.makeIdentity()

    def highlight(self, fRecompute=1):
        if False:
            for i in range(10):
                print('nop')
        if fRecompute:
            pass
        self.bbox.show()

    def dehighlight(self):
        if False:
            i = 10
            return i + 15
        self.bbox.hide()

    def getCenter(self):
        if False:
            i = 10
            return i + 15
        return self.bbox.getCenter()

    def getRadius(self):
        if False:
            while True:
                i = 10
        return self.bbox.getRadius()

    def getMin(self):
        if False:
            return 10
        return self.bbox.getMin()

    def getMax(self):
        if False:
            while True:
                i = 10
        return self.bbox.getMax()

class SelectedNodePaths(DirectObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reset()
        self.tagList = []

    def addTag(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if tag not in self.tagList:
            self.tagList.append(tag)

    def removeTag(self, tag):
        if False:
            while True:
                i = 10
        self.tagList.remove(tag)

    def reset(self):
        if False:
            print('Hello World!')
        self.selectedDict = {}
        self.selectedList = []
        self.deselectedDict = {}
        __builtins__['last'] = self.last = None

    def select(self, nodePath, fMultiSelect=0, fSelectTag=1):
        if False:
            print('Hello World!')
        ' Select the specified node path.  Multiselect as required '
        if not nodePath:
            print('Nothing selected!!')
            return None
        if not fMultiSelect:
            self.deselectAll()
        if fSelectTag:
            for tag in self.tagList:
                if nodePath.hasNetTag(tag):
                    nodePath = nodePath.findNetTag(tag)
                    break
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if dnp:
            self.deselect(nodePath)
            return None
        else:
            dnp = self.getDeselectedDict(id)
            if dnp:
                del self.deselectedDict[id]
                dnp.highlight()
            else:
                dnp = DirectNodePath(nodePath)
                dnp.highlight(fRecompute=0)
            self.selectedDict[dnp.get_key()] = dnp
            self.selectedList.append(dnp)
        __builtins__['last'] = self.last = dnp
        if base.direct.clusterMode == 'client':
            cluster.selectNodePath(dnp)
        return dnp

    def deselect(self, nodePath):
        if False:
            while True:
                i = 10
        ' Deselect the specified node path '
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if dnp:
            dnp.dehighlight()
            del self.selectedDict[id]
            if dnp in self.selectedList:
                self.selectedList.remove(dnp)
            self.deselectedDict[id] = dnp
            messenger.send('DIRECT_deselectedNodePath', [dnp])
            if base.direct.clusterMode == 'client':
                cluster.deselectNodePath(dnp)
        return dnp

    def getSelectedAsList(self):
        if False:
            return 10
        '\n        Return a list of all selected node paths.  No verification of\n        connectivity is performed on the members of the list\n        '
        return self.selectedList[:]

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self.getSelectedAsList()[index]

    def getSelectedDict(self, id):
        if False:
            while True:
                i = 10
        '\n        Search selectedDict for node path, try to repair broken node paths.\n        '
        dnp = self.selectedDict.get(id, None)
        if dnp:
            return dnp
        else:
            return None

    def getDeselectedAsList(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.deselectedDict.values())

    def getDeselectedDict(self, id):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        '\n        Perform given func on deselected node paths.  No node path\n        connectivity verification performed\n        '
        deselectedNodePaths = self.getDeselectedAsList()
        for nodePath in deselectedNodePaths:
            func(nodePath)

    def getWrtAll(self):
        if False:
            return 10
        self.forEachSelectedNodePathDo(self.getWrt)

    def getWrt(self, nodePath):
        if False:
            return 10
        nodePath.tDnp2Widget = nodePath.getTransform(base.direct.widget)

    def moveWrtWidgetAll(self):
        if False:
            return 10
        self.forEachSelectedNodePathDo(self.moveWrtWidget)

    def moveWrtWidget(self, nodePath):
        if False:
            return 10
        nodePath.setTransform(base.direct.widget, nodePath.tDnp2Widget)

    def deselectAll(self):
        if False:
            i = 10
            return i + 15
        self.forEachSelectedNodePathDo(self.deselect)

    def highlightAll(self):
        if False:
            for i in range(10):
                print('nop')
        self.forEachSelectedNodePathDo(DirectNodePath.highlight)

    def dehighlightAll(self):
        if False:
            return 10
        self.forEachSelectedNodePathDo(DirectNodePath.dehighlight)

    def removeSelected(self):
        if False:
            while True:
                i = 10
        selected = self.last
        if selected:
            selected.remove()
        __builtins__['last'] = self.last = None

    def removeAll(self):
        if False:
            print('Hello World!')
        self.forEachSelectedNodePathDo(NodePath.remove)

    def toggleVisSelected(self):
        if False:
            return 10
        selected = self.last
        if selected:
            if selected.isHidden():
                selected.show()
            else:
                selected.hide()

    def toggleVisAll(self):
        if False:
            while True:
                i = 10
        selectedNodePaths = self.getSelectedAsList()
        for nodePath in selectedNodePaths:
            if nodePath.isHidden():
                nodePath.show()
            else:
                nodePath.hide()

    def isolateSelected(self):
        if False:
            for i in range(10):
                print('nop')
        selected = self.last
        if selected:
            selected.showAllDescendents()
            for sib in selected.getParent().getChildren():
                if sib.node() != selected.node():
                    sib.hide()

    def getDirectNodePath(self, nodePath):
        if False:
            for i in range(10):
                print('nop')
        id = nodePath.get_key()
        dnp = self.getSelectedDict(id)
        if dnp:
            return dnp
        return self.getDeselectedDict(id)

    def getNumSelected(self):
        if False:
            while True:
                i = 10
        return len(self.selectedDict)

class DirectBoundingBox:

    def __init__(self, nodePath, bboxColor=None):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath = nodePath
        self.computeTightBounds()
        self.lines = self.createBBoxLines(bboxColor)

    def recompute(self):
        if False:
            i = 10
            return i + 15
        self.computeTightBounds()
        self.updateBBoxLines()

    def computeTightBounds(self):
        if False:
            print('Hello World!')
        tMat = Mat4(self.nodePath.getMat())
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
            for i in range(10):
                print('nop')
        self.bounds = self.getBounds()
        if self.bounds.isEmpty() or self.bounds.isInfinite():
            self.center = Point3(0)
            self.radius = 1.0
        else:
            self.center = self.bounds.getCenter()
            self.radius = self.bounds.getRadius()
        self.min = Point3(self.center - Point3(self.radius))
        self.max = Point3(self.center + Point3(self.radius))

    def createBBoxLines(self, bboxColor=None):
        if False:
            for i in range(10):
                print('nop')
        lines = LineNodePath(hidden)
        lines.node().setName('bboxLines')
        if bboxColor:
            lines.setColor(VBase4(*bboxColor))
        else:
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

    def setBoxColorScale(self, r, g, b, a):
        if False:
            return 10
        if self.lines:
            self.lines.reset()
            self.lines = None
        self.lines = self.createBBoxLines((r, g, b, a))
        self.show()

    def updateBBoxLines(self):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        self.lines.reparentTo(self.nodePath)

    def hide(self):
        if False:
            i = 10
            return i + 15
        self.lines.reparentTo(hidden)

    def getCenter(self):
        if False:
            for i in range(10):
                print('nop')
        return self.center

    def getRadius(self):
        if False:
            while True:
                i = 10
        return self.radius

    def getMin(self):
        if False:
            while True:
                i = 10
        return self.min

    def getMax(self):
        if False:
            for i in range(10):
                print('nop')
        return self.max

    def vecAsString(self, vec):
        if False:
            for i in range(10):
                print('nop')
        return '%.2f %.2f %.2f' % (vec[0], vec[1], vec[2])

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.__class__) + '\nNodePath:\t%s\n' % self.nodePath.getName() + 'Min:\t\t%s\n' % self.vecAsString(self.min) + 'Max:\t\t%s\n' % self.vecAsString(self.max) + 'Center:\t\t%s\n' % self.vecAsString(self.center) + 'Radius:\t\t%.2f' % self.radius

class SelectionQueue(CollisionHandlerQueue):

    def __init__(self, parentNP=None):
        if False:
            i = 10
            return i + 15
        if parentNP is None:
            parentNP = render
        CollisionHandlerQueue.__init__(self)
        self.index = -1
        self.entry = None
        self.skipFlags = DG.SKIP_NONE
        self.collisionNodePath = NodePath(CollisionNode('collisionNP'))
        self.setParentNP(parentNP)
        self.collisionNodePath.hide()
        self.collisionNode = self.collisionNodePath.node()
        self.collideWithGeom()
        self.ct = CollisionTraverser('DirectSelection')
        self.ct.setRespectPrevTransform(False)
        self.ct.addCollider(self.collisionNodePath, self)
        self.unpickable = DG.UNPICKABLE

    def setParentNP(self, parentNP):
        if False:
            while True:
                i = 10
        self.collisionNodePath.reparentTo(parentNP)

    def addCollider(self, collider):
        if False:
            for i in range(10):
                print('nop')
        self.collider = collider
        self.collisionNode.addSolid(self.collider)

    def collideWithBitMask(self, bitMask):
        if False:
            i = 10
            return i + 15
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        self.collisionNode.setFromCollideMask(bitMask)

    def collideWithGeom(self):
        if False:
            return 10
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        self.collisionNode.setFromCollideMask(GeomNode.getDefaultCollideMask())

    def collideWithWidget(self):
        if False:
            for i in range(10):
                print('nop')
        self.collisionNode.setIntoCollideMask(BitMask32().allOff())
        mask = BitMask32()
        mask.setWord(2147483648)
        self.collisionNode.setFromCollideMask(mask)

    def addUnpickable(self, item):
        if False:
            i = 10
            return i + 15
        if item not in self.unpickable:
            self.unpickable.append(item)

    def removeUnpickable(self, item):
        if False:
            print('Hello World!')
        if item in self.unpickable:
            self.unpickable.remove(item)

    def setCurrentIndex(self, index):
        if False:
            print('Hello World!')
        if index < 0 or index >= self.getNumEntries():
            self.index = -1
        else:
            self.index = index

    def setCurrentEntry(self, entry):
        if False:
            for i in range(10):
                print('nop')
        self.entry = entry

    def getCurrentEntry(self):
        if False:
            for i in range(10):
                print('nop')
        return self.entry

    def isEntryBackfacing(self, entry):
        if False:
            for i in range(10):
                print('nop')
        if not entry.hasSurfaceNormal():
            return 0
        if base.direct:
            cam = base.direct.cam
        else:
            cam = base.cam
        fromNodePath = entry.getFromNodePath()
        v = Vec3(entry.getSurfacePoint(fromNodePath))
        n = entry.getSurfaceNormal(fromNodePath)
        if self.collisionNodePath.getParent() != cam:
            p2cam = self.collisionNodePath.getParent().getMat(cam)
            v = Vec3(p2cam.xformPoint(v))
            n = p2cam.xformVec(n)
        v.normalize()
        return v.dot(n) >= 0

    def findNextCollisionEntry(self, skipFlags=DG.SKIP_NONE):
        if False:
            return 10
        return self.findCollisionEntry(skipFlags, self.index + 1)

    def findCollisionEntry(self, skipFlags=DG.SKIP_NONE, startIndex=0):
        if False:
            return 10
        self.setCurrentIndex(-1)
        self.setCurrentEntry(None)
        for i in range(startIndex, self.getNumEntries()):
            entry = self.getEntry(i)
            nodePath = entry.getIntoNodePath()
            if skipFlags & DG.SKIP_HIDDEN and nodePath.isHidden():
                pass
            elif skipFlags & DG.SKIP_BACKFACE and self.isEntryBackfacing(entry):
                pass
            elif skipFlags & DG.SKIP_CAMERA and base.camera in nodePath.getAncestors():
                pass
            elif skipFlags & DG.SKIP_UNPICKABLE and nodePath.getName() in self.unpickable:
                pass
            elif base.direct and (skipFlags & DG.SKIP_WIDGET and nodePath.getTag('WidgetName') != base.direct.widget.getName()):
                pass
            elif base.direct and (skipFlags & DG.SKIP_WIDGET and base.direct.fControl and (nodePath.getName()[2:] == 'ring')):
                pass
            else:
                self.setCurrentIndex(i)
                self.setCurrentEntry(entry)
                break
        return self.getCurrentEntry()

class SelectionRay(SelectionQueue):

    def __init__(self, parentNP=None):
        if False:
            return 10
        if parentNP is None:
            parentNP = render
        SelectionQueue.__init__(self, parentNP)
        self.addCollider(CollisionRay())

    def pick(self, targetNodePath, xy=None):
        if False:
            for i in range(10):
                print('nop')
        if xy:
            mx = xy[0]
            my = xy[1]
        elif base.direct:
            mx = base.direct.dr.mouseX
            my = base.direct.dr.mouseY
        else:
            if not base.mouseWatcherNode.hasMouse():
                self.clearEntries()
                return
            mx = base.mouseWatcherNode.getMouseX()
            my = base.mouseWatcherNode.getMouseY()
        if base.direct:
            self.collider.setFromLens(base.direct.camNode, mx, my)
        else:
            self.collider.setFromLens(base.camNode, mx, my)
        self.ct.traverse(targetNodePath)
        self.sortEntries()

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=None, skipFlags=DG.SKIP_ALL):
        if False:
            while True:
                i = 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithBitMask(bitMask)
        self.pick(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pickGeom(self, targetNodePath=None, skipFlags=DG.SKIP_ALL, xy=None):
        if False:
            return 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithGeom()
        self.pick(targetNodePath, xy=xy)
        return self.findCollisionEntry(skipFlags)

    def pickWidget(self, targetNodePath=None, skipFlags=DG.SKIP_NONE):
        if False:
            return 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithWidget()
        self.pick(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pick3D(self, targetNodePath, origin, dir):
        if False:
            while True:
                i = 10
        self.collider.setOrigin(origin)
        self.collider.setDirection(dir)
        self.ct.traverse(targetNodePath)
        self.sortEntries()

    def pickGeom3D(self, targetNodePath=None, origin=Point3(0), dir=Vec3(0, 0, -1), skipFlags=DG.SKIP_HIDDEN | DG.SKIP_CAMERA):
        if False:
            while True:
                i = 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithGeom()
        self.pick3D(targetNodePath, origin, dir)
        return self.findCollisionEntry(skipFlags)

    def pickBitMask3D(self, bitMask=BitMask32.allOff(), targetNodePath=None, origin=Point3(0), dir=Vec3(0, 0, -1), skipFlags=DG.SKIP_ALL):
        if False:
            for i in range(10):
                print('nop')
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithBitMask(bitMask)
        self.pick3D(targetNodePath, origin, dir)
        return self.findCollisionEntry(skipFlags)

class SelectionSegment(SelectionQueue):

    def __init__(self, parentNP=None, numSegments=1):
        if False:
            return 10
        if parentNP is None:
            parentNP = render
        SelectionQueue.__init__(self, parentNP)
        self.colliders = []
        self.numColliders = 0
        for i in range(numSegments):
            self.addCollider(CollisionSegment())

    def addCollider(self, collider):
        if False:
            for i in range(10):
                print('nop')
        self.colliders.append(collider)
        self.collisionNode.addSolid(collider)
        self.numColliders += 1

    def pickGeom(self, targetNodePath=None, endPointList=[], skipFlags=DG.SKIP_HIDDEN | DG.SKIP_CAMERA):
        if False:
            i = 10
            return i + 15
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithGeom()
        for i in range(min(len(endPointList), self.numColliders)):
            (pointA, pointB) = endPointList[i]
            collider = self.colliders[i]
            collider.setPointA(pointA)
            collider.setPointB(pointB)
        self.ct.traverse(targetNodePath)
        return self.findCollisionEntry(skipFlags)

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=None, endPointList=[], skipFlags=DG.SKIP_HIDDEN | DG.SKIP_CAMERA):
        if False:
            return 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithBitMask(bitMask)
        for i in range(min(len(endPointList), self.numColliders)):
            (pointA, pointB) = endPointList[i]
            collider = self.colliders[i]
            collider.setPointA(pointA)
            collider.setPointB(pointB)
        self.ct.traverse(targetNodePath)
        return self.findCollisionEntry(skipFlags)

class SelectionSphere(SelectionQueue):

    def __init__(self, parentNP=None, numSpheres=1):
        if False:
            while True:
                i = 10
        if parentNP is None:
            parentNP = render
        SelectionQueue.__init__(self, parentNP)
        self.colliders = []
        self.numColliders = 0
        for i in range(numSpheres):
            self.addCollider(CollisionSphere(Point3(0), 1))

    def addCollider(self, collider):
        if False:
            for i in range(10):
                print('nop')
        self.colliders.append(collider)
        self.collisionNode.addSolid(collider)
        self.numColliders += 1

    def setCenter(self, i, center):
        if False:
            return 10
        c = self.colliders[i]
        c.setCenter(center)

    def setRadius(self, i, radius):
        if False:
            return 10
        c = self.colliders[i]
        c.setRadius(radius)

    def setCenterRadius(self, i, center, radius):
        if False:
            i = 10
            return i + 15
        c = self.colliders[i]
        c.setCenter(center)
        c.setRadius(radius)

    def isEntryBackfacing(self, entry):
        if False:
            while True:
                i = 10
        fromNodePath = entry.getFromNodePath()
        v = Vec3(entry.getSurfacePoint(fromNodePath) - entry.getFrom().getCenter())
        n = entry.getSurfaceNormal(fromNodePath)
        if v.length() < 0.05:
            return 1
        v.normalize()
        return v.dot(n) >= 0

    def pick(self, targetNodePath, skipFlags):
        if False:
            i = 10
            return i + 15
        self.ct.traverse(targetNodePath)
        self.sortEntries()
        return self.findCollisionEntry(skipFlags)

    def pickGeom(self, targetNodePath=None, skipFlags=DG.SKIP_HIDDEN | DG.SKIP_CAMERA):
        if False:
            return 10
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithGeom()
        return self.pick(targetNodePath, skipFlags)

    def pickBitMask(self, bitMask=BitMask32.allOff(), targetNodePath=None, skipFlags=DG.SKIP_HIDDEN | DG.SKIP_CAMERA):
        if False:
            for i in range(10):
                print('nop')
        if targetNodePath is None:
            targetNodePath = render
        self.collideWithBitMask(bitMask)
        return self.pick(targetNodePath, skipFlags)