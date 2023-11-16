"""
ShadowPlacer.py places a shadow.

It traces a line from a light source to the opposing surface.
Or it may do that later, right now it puts a node on the surface under
the its parent node.
"""
__all__ = ['ShadowPlacer']
from direct.controls.ControlManager import CollisionHandlerRayStart
from direct.directnotify import DirectNotifyGlobal
from panda3d.core import BitMask32, CollisionHandlerFloor, CollisionNode, CollisionRay, CollisionTraverser, NodePath
from . import DirectObject

class ShadowPlacer(DirectObject.DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('ShadowPlacer')
    if __debug__:
        count = 0
        activeCount = 0

    def __init__(self, cTrav, shadowNodePath, wallCollideMask, floorCollideMask):
        if False:
            while True:
                i = 10
        self.isActive = 0
        assert self.notify.debugCall()
        DirectObject.DirectObject.__init__(self)
        self.setup(cTrav, shadowNodePath, wallCollideMask, floorCollideMask)
        if __debug__:
            self.count += 1
            self.debugDisplay()

    def setup(self, cTrav, shadowNodePath, wallCollideMask, floorCollideMask):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the collisions\n        '
        assert self.notify.debugCall()
        assert not shadowNodePath.isEmpty()
        assert not hasattr(self, 'cTrav')
        if not cTrav:
            base.initShadowTrav()
            cTrav = base.shadowTrav
        self.cTrav = cTrav
        self.shadowNodePath = shadowNodePath
        floorOffset = 0.025
        self.cRay = CollisionRay(0.0, 0.0, CollisionHandlerRayStart, 0.0, 0.0, -1.0)
        cRayNode = CollisionNode('shadowPlacer')
        cRayNode.addSolid(self.cRay)
        self.cRayNodePath = NodePath(cRayNode)
        self.cRayBitMask = floorCollideMask
        cRayNode.setFromCollideMask(self.cRayBitMask)
        cRayNode.setIntoCollideMask(BitMask32.allOff())
        self.lifter = CollisionHandlerFloor()
        self.lifter.setOffset(floorOffset)
        self.lifter.setReach(4.0)
        self.lifter.addCollider(self.cRayNodePath, shadowNodePath)

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugCall()
        self.off()
        if __debug__:
            assert not self.isActive
            self.count -= 1
            self.debugDisplay()
        del self.cTrav
        del self.shadowNodePath
        del self.cRay
        self.cRayNodePath.removeNode()
        del self.cRayNodePath
        del self.lifter

    def on(self):
        if False:
            return 10
        '\n        Turn on the shadow placement.  The shadow z position will\n        start being updated until a call to off() is made.\n        '
        assert self.notify.debugCall('activeCount=%s' % (self.activeCount,))
        if self.isActive:
            assert self.cTrav.hasCollider(self.cRayNodePath)
            return
        assert not self.cTrav.hasCollider(self.cRayNodePath)
        self.cRayNodePath.reparentTo(self.shadowNodePath.getParent())
        self.cTrav.addCollider(self.cRayNodePath, self.lifter)
        self.isActive = 1
        if __debug__:
            self.activeCount += 1
            self.debugDisplay()

    def off(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Turn off the shadow placement.  The shadow will still be\n        there, but the z position will not be updated until a call\n        to on() is made.\n        '
        assert self.notify.debugCall('activeCount=%s' % (self.activeCount,))
        if not self.isActive:
            assert not self.cTrav.hasCollider(self.cRayNodePath)
            return
        assert self.cTrav.hasCollider(self.cRayNodePath)
        didIt = self.cTrav.removeCollider(self.cRayNodePath)
        assert didIt
        self.oneTimeCollide()
        self.cRayNodePath.detachNode()
        self.isActive = 0
        if __debug__:
            self.activeCount -= 1
            self.debugDisplay()

    def oneTimeCollide(self):
        if False:
            i = 10
            return i + 15
        '\n        Makes one quick collision pass for the avatar, for instance as\n        a one-time straighten-things-up operation after collisions\n        have been disabled.\n        '
        assert self.notify.debugCall()
        tempCTrav = CollisionTraverser('oneTimeCollide')
        tempCTrav.addCollider(self.cRayNodePath, self.lifter)
        tempCTrav.traverse(render)

    def resetToOrigin(self):
        if False:
            return 10
        if self.shadowNodePath:
            self.shadowNodePath.setPos(0, 0, 0)
    if __debug__:

        def debugDisplay(self):
            if False:
                for i in range(10):
                    print('nop')
            'for debugging'
            if self.notify.getDebug():
                message = '%d active (%d total), %d colliders' % (self.activeCount, self.count, self.cTrav.getNumColliders())
                self.notify.debug(message)
                onScreenDebug.add('ShadowPlacers', message)
            return 1