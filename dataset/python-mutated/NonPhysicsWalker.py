"""
NonPhysicsWalker.py is for avatars.

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
from direct.directnotify import DirectNotifyGlobal
from direct.showbase import DirectObject
from direct.controls.ControlManager import CollisionHandlerRayStart
from direct.showbase.InputStateGlobal import inputState
from direct.showbase.MessengerGlobal import messenger
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import BitMask32, ClockObject, CollisionHandlerFloor, CollisionHandlerPusher, CollisionNode, CollisionRay, CollisionSphere, CollisionTraverser, ConfigVariableBool, Mat3, Point3, Vec3

class NonPhysicsWalker(DirectObject.DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('NonPhysicsWalker')
    wantDebugIndicator = ConfigVariableBool('want-avatar-physics-indicator', False)
    slideName = 'slide-is-disabled'

    def __init__(self):
        if False:
            while True:
                i = 10
        DirectObject.DirectObject.__init__(self)
        self.worldVelocity = Vec3.zero()
        self.collisionsActive = 0
        self.speed = 0.0
        self.rotationSpeed = 0.0
        self.slideSpeed = 0.0
        self.vel = Vec3(0.0, 0.0, 0.0)
        self.stopThisFrame = 0

    def setWalkSpeed(self, forward, jump, reverse, rotate):
        if False:
            i = 10
            return i + 15
        assert self.debugPrint('setWalkSpeed()')
        self.avatarControlForwardSpeed = forward
        self.avatarControlReverseSpeed = reverse
        self.avatarControlRotateSpeed = rotate

    def getSpeeds(self):
        if False:
            print('Hello World!')
        return (self.speed, self.rotationSpeed, self.slideSpeed)

    def setAvatar(self, avatar):
        if False:
            for i in range(10):
                print('nop')
        self.avatar = avatar
        if avatar is not None:
            pass

    def setAirborneHeightFunc(self, getAirborneHeight):
        if False:
            while True:
                i = 10
        self.getAirborneHeight = getAirborneHeight

    def setWallBitMask(self, bitMask):
        if False:
            i = 10
            return i + 15
        self.cSphereBitMask = bitMask

    def setFloorBitMask(self, bitMask):
        if False:
            while True:
                i = 10
        self.cRayBitMask = bitMask

    def swapFloorBitMask(self, oldMask, newMask):
        if False:
            for i in range(10):
                print('nop')
        self.cRayBitMask = self.cRayBitMask & ~oldMask
        self.cRayBitMask |= newMask
        if self.cRayNodePath and (not self.cRayNodePath.isEmpty()):
            self.cRayNodePath.node().setFromCollideMask(self.cRayBitMask)

    def initializeCollisions(self, collisionTraverser, avatarNodePath, avatarRadius=1.4, floorOffset=1.0, reach=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the avatar for collisions\n        '
        assert not avatarNodePath.isEmpty()
        self.cTrav = collisionTraverser
        self.avatarNodePath = avatarNodePath
        self.cSphere = CollisionSphere(0.0, 0.0, 0.0, avatarRadius)
        cSphereNode = CollisionNode('NPW.cSphereNode')
        cSphereNode.addSolid(self.cSphere)
        self.cSphereNodePath = avatarNodePath.attachNewNode(cSphereNode)
        cSphereNode.setFromCollideMask(self.cSphereBitMask)
        cSphereNode.setIntoCollideMask(BitMask32.allOff())
        self.cRay = CollisionRay(0.0, 0.0, CollisionHandlerRayStart, 0.0, 0.0, -1.0)
        cRayNode = CollisionNode('NPW.cRayNode')
        cRayNode.addSolid(self.cRay)
        self.cRayNodePath = avatarNodePath.attachNewNode(cRayNode)
        cRayNode.setFromCollideMask(self.cRayBitMask)
        cRayNode.setIntoCollideMask(BitMask32.allOff())
        self.pusher = CollisionHandlerPusher()
        self.pusher.setInPattern('enter%in')
        self.pusher.setOutPattern('exit%in')
        self.lifter = CollisionHandlerFloor()
        self.lifter.setInPattern('on-floor')
        self.lifter.setOutPattern('off-floor')
        self.lifter.setOffset(floorOffset)
        self.lifter.setReach(reach)
        self.lifter.setMaxVelocity(16.0)
        self.pusher.addCollider(self.cSphereNodePath, avatarNodePath)
        self.lifter.addCollider(self.cRayNodePath, avatarNodePath)
        self.setCollisionsActive(1)

    def deleteCollisions(self):
        if False:
            i = 10
            return i + 15
        del self.cTrav
        del self.cSphere
        self.cSphereNodePath.removeNode()
        del self.cSphereNodePath
        del self.cRay
        self.cRayNodePath.removeNode()
        del self.cRayNodePath
        del self.pusher
        del self.lifter

    def setTag(self, key, value):
        if False:
            return 10
        self.cSphereNodePath.setTag(key, value)

    def setCollisionsActive(self, active=1):
        if False:
            return 10
        assert self.debugPrint('setCollisionsActive(active%s)' % (active,))
        if self.collisionsActive != active:
            self.collisionsActive = active
            if active:
                self.cTrav.addCollider(self.cSphereNodePath, self.pusher)
                self.cTrav.addCollider(self.cRayNodePath, self.lifter)
            else:
                self.cTrav.removeCollider(self.cSphereNodePath)
                self.cTrav.removeCollider(self.cRayNodePath)
                self.oneTimeCollide()

    def placeOnFloor(self):
        if False:
            while True:
                i = 10
        '\n        Make a reasonable effor to place the avatar on the ground.\n        For example, this is useful when switching away from the\n        current walker.\n        '
        return

    def oneTimeCollide(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Makes one quick collision pass for the avatar, for instance as\n        a one-time straighten-things-up operation after collisions\n        have been disabled.\n        '
        tempCTrav = CollisionTraverser('oneTimeCollide')
        tempCTrav.addCollider(self.cSphereNodePath, self.pusher)
        tempCTrav.addCollider(self.cRayNodePath, self.lifter)
        tempCTrav.traverse(base.render)

    def addBlastForce(self, vector):
        if False:
            for i in range(10):
                print('nop')
        pass

    def displayDebugInfo(self):
        if False:
            while True:
                i = 10
        '\n        For debug use.\n        '
        onScreenDebug.add('controls', 'NonPhysicsWalker')

    def _calcSpeeds(self):
        if False:
            for i in range(10):
                print('nop')
        forward = inputState.isSet('forward')
        reverse = inputState.isSet('reverse')
        turnLeft = inputState.isSet('turnLeft')
        turnRight = inputState.isSet('turnRight')
        slide = inputState.isSet(self.slideName) or 0
        if base.localAvatar.getAutoRun():
            forward = 1
            reverse = 0
        self.speed = forward and self.avatarControlForwardSpeed or (reverse and -self.avatarControlReverseSpeed)
        self.slideSpeed = slide and (reverse and turnLeft and -self.avatarControlReverseSpeed * 0.75 or (reverse and turnRight and self.avatarControlReverseSpeed * 0.75) or (turnLeft and -self.avatarControlForwardSpeed * 0.75) or (turnRight and self.avatarControlForwardSpeed * 0.75))
        self.rotationSpeed = not slide and (turnLeft and self.avatarControlRotateSpeed or (turnRight and -self.avatarControlRotateSpeed))

    def handleAvatarControls(self, task):
        if False:
            return 10
        '\n        Check on the arrow keys and update the avatar.\n        '
        if not self.lifter.hasContact():
            messenger.send('walkerIsOutOfWorld', [self.avatarNodePath])
        self._calcSpeeds()
        if __debug__:
            debugRunning = inputState.isSet('debugRunning')
            if debugRunning:
                self.speed *= 4.0
                self.slideSpeed *= 4.0
                self.rotationSpeed *= 1.25
        if self.wantDebugIndicator:
            self.displayDebugInfo()
        dt = ClockObject.getGlobalClock().getDt()
        if self.speed or self.slideSpeed or self.rotationSpeed:
            if self.stopThisFrame:
                distance = 0.0
                slideDistance = 0.0
                rotation = 0.0
                self.stopThisFrame = 0
            else:
                distance = dt * self.speed
                slideDistance = dt * self.slideSpeed
                rotation = dt * self.rotationSpeed
            self.vel = Vec3(Vec3.forward() * distance + Vec3.right() * slideDistance)
            if self.vel != Vec3.zero():
                rotMat = Mat3.rotateMatNormaxis(self.avatarNodePath.getH(), Vec3.up())
                step = rotMat.xform(self.vel)
                self.avatarNodePath.setFluidPos(Point3(self.avatarNodePath.getPos() + step))
            self.avatarNodePath.setH(self.avatarNodePath.getH() + rotation)
            messenger.send('avatarMoving')
        else:
            self.vel.set(0.0, 0.0, 0.0)
        self.__oldPosDelta = self.avatarNodePath.getPosDelta(base.render)
        self.__oldDt = dt
        if self.__oldDt != 0:
            self.worldVelocity = self.__oldPosDelta * (1 / self.__oldDt)
        else:
            self.worldVelocity = 0
        return Task.cont

    def doDeltaPos(self):
        if False:
            return 10
        assert self.debugPrint('doDeltaPos()')

    def reset(self):
        if False:
            while True:
                i = 10
        assert self.debugPrint('reset()')

    def getVelocity(self):
        if False:
            print('Hello World!')
        return self.vel

    def enableAvatarControls(self):
        if False:
            i = 10
            return i + 15
        '\n        Activate the arrow keys, etc.\n        '
        assert self.debugPrint('enableAvatarControls')
        assert self.collisionsActive
        taskName = 'AvatarControls-%s' % (id(self),)
        taskMgr.remove(taskName)
        taskMgr.add(self.handleAvatarControls, taskName)

    def disableAvatarControls(self):
        if False:
            while True:
                i = 10
        '\n        Ignore the arrow keys, etc.\n        '
        assert self.debugPrint('disableAvatarControls')
        taskName = 'AvatarControls-%s' % (id(self),)
        taskMgr.remove(taskName)

    def flushEventHandlers(self):
        if False:
            return 10
        if hasattr(self, 'cTrav'):
            self.pusher.flush()
        self.lifter.flush()
    if __debug__:

        def debugPrint(self, message):
            if False:
                i = 10
                return i + 15
            'for debugging'
            return self.notify.debug(str(id(self)) + ' ' + message)