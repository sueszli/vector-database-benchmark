"""
ObserverWalker.py is for avatars.

A walker control such as this one provides:

- creation of the collision nodes
- handling the keyboard and mouse input for avatar movement
- moving the avatar

it does not:

- play sounds
- play animations

although it does send messages that allow a listener to play sounds or
animations based on walker events.
"""
from panda3d.core import BitMask32, CollisionHandlerPusher, CollisionNode, CollisionSphere, CollisionTraverser
from direct.directnotify import DirectNotifyGlobal
from . import NonPhysicsWalker

class ObserverWalker(NonPhysicsWalker.NonPhysicsWalker):
    notify = DirectNotifyGlobal.directNotify.newCategory('ObserverWalker')
    slideName = 'jump'

    def initializeCollisions(self, collisionTraverser, avatarNodePath, avatarRadius=1.4, floorOffset=1.0, reach=1.0):
        if False:
            return 10
        '\n        Set up the avatar for collisions\n        '
        assert not avatarNodePath.isEmpty()
        self.cTrav = collisionTraverser
        self.avatarNodePath = avatarNodePath
        self.cSphere = CollisionSphere(0.0, 0.0, 0.0, avatarRadius)
        cSphereNode = CollisionNode('Observer.cSphereNode')
        cSphereNode.addSolid(self.cSphere)
        self.cSphereNodePath = avatarNodePath.attachNewNode(cSphereNode)
        cSphereNode.setFromCollideMask(self.cSphereBitMask)
        cSphereNode.setIntoCollideMask(BitMask32.allOff())
        self.pusher = CollisionHandlerPusher()
        self.pusher.setInPattern('enter%in')
        self.pusher.setOutPattern('exit%in')
        self.pusher.addCollider(self.cSphereNodePath, avatarNodePath)
        self.setCollisionsActive(1)

        class Foo:

            def hasContact(self):
                if False:
                    while True:
                        i = 10
                return 1
        self.lifter = Foo()

    def deleteCollisions(self):
        if False:
            print('Hello World!')
        del self.cTrav
        del self.cSphere
        self.cSphereNodePath.removeNode()
        del self.cSphereNodePath
        del self.pusher

    def setCollisionsActive(self, active=1):
        if False:
            while True:
                i = 10
        assert self.debugPrint('setCollisionsActive(active%s)' % (active,))
        if self.collisionsActive != active:
            self.collisionsActive = active
            if active:
                self.cTrav.addCollider(self.cSphereNodePath, self.pusher)
            else:
                self.cTrav.removeCollider(self.cSphereNodePath)
                self.oneTimeCollide()

    def oneTimeCollide(self):
        if False:
            print('Hello World!')
        '\n        Makes one quick collision pass for the avatar, for instance as\n        a one-time straighten-things-up operation after collisions\n        have been disabled.\n        '
        tempCTrav = CollisionTraverser('oneTimeCollide')
        tempCTrav.addCollider(self.cSphereNodePath, self.pusher)
        tempCTrav.traverse(render)

    def enableAvatarControls(self):
        if False:
            i = 10
            return i + 15
        '\n        Activate the arrow keys, etc.\n        '
        assert self.debugPrint('enableAvatarControls')

    def disableAvatarControls(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ignore the arrow keys, etc.\n        '
        assert self.debugPrint('disableAvatarControls')