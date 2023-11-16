"""
DevWalker.py is for avatars.

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
from direct.showbase.InputStateGlobal import inputState
from direct.directnotify import DirectNotifyGlobal
from direct.showbase import DirectObject
from direct.showbase.MessengerGlobal import messenger
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ClockObject, ConfigVariableBool, ConfigVariableDouble, Mat3, Point3, Vec3

class DevWalker(DirectObject.DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('DevWalker')
    wantDebugIndicator = ConfigVariableBool('want-avatar-physics-indicator', False)
    runMultiplier = ConfigVariableDouble('dev-run-multiplier', 4.0)
    slideName = 'slide-is-disabled'

    def __init__(self):
        if False:
            return 10
        DirectObject.DirectObject.__init__(self)
        self.speed = 0.0
        self.rotationSpeed = 0.0
        self.slideSpeed = 0.0
        self.vel = Vec3(0.0, 0.0, 0.0)
        self.task = None

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
            for i in range(10):
                print('nop')
        return (self.speed, self.rotationSpeed, self.slideSpeed)

    def setAvatar(self, avatar):
        if False:
            i = 10
            return i + 15
        self.avatar = avatar
        if avatar is not None:
            pass

    def setWallBitMask(self, bitMask):
        if False:
            for i in range(10):
                print('nop')
        pass

    def setFloorBitMask(self, bitMask):
        if False:
            for i in range(10):
                print('nop')
        pass

    def initializeCollisions(self, collisionTraverser, avatarNodePath, wallCollideMask, floorCollideMask, avatarRadius=1.4, floorOffset=1.0, reach=1.0):
        if False:
            i = 10
            return i + 15
        assert not avatarNodePath.isEmpty()
        self.cTrav = collisionTraverser
        self.avatarNodePath = avatarNodePath

    def setAirborneHeightFunc(self, getAirborneHeight):
        if False:
            for i in range(10):
                print('nop')
        pass

    def deleteCollisions(self):
        if False:
            i = 10
            return i + 15
        pass

    def setTag(self, key, value):
        if False:
            print('Hello World!')
        pass

    def setCollisionsActive(self, active=1):
        if False:
            while True:
                i = 10
        pass

    def placeOnFloor(self):
        if False:
            return 10
        pass

    def oneTimeCollide(self):
        if False:
            print('Hello World!')
        pass

    def addBlastForce(self, vector):
        if False:
            return 10
        pass

    def displayDebugInfo(self):
        if False:
            while True:
                i = 10
        '\n        For debug use.\n        '
        onScreenDebug.add('w controls', 'DevWalker')

    def handleAvatarControls(self, task):
        if False:
            print('Hello World!')
        '\n        Check on the arrow keys and update the avatar.\n        '
        forward = inputState.isSet('forward')
        reverse = inputState.isSet('reverse')
        turnLeft = inputState.isSet('turnLeft')
        turnRight = inputState.isSet('turnRight')
        slideLeft = inputState.isSet('slideLeft')
        slideRight = inputState.isSet('slideRight')
        levitateUp = inputState.isSet('levitateUp')
        levitateDown = inputState.isSet('levitateDown')
        run = inputState.isSet('run') and self.runMultiplier.getValue() or 1.0
        if base.localAvatar.getAutoRun():
            forward = 1
            reverse = 0
        self.speed = forward and self.avatarControlForwardSpeed or (reverse and -self.avatarControlReverseSpeed)
        self.liftSpeed = levitateUp and self.avatarControlForwardSpeed or (levitateDown and -self.avatarControlReverseSpeed)
        self.slideSpeed = slideLeft and -self.avatarControlForwardSpeed or (slideRight and self.avatarControlForwardSpeed)
        self.rotationSpeed = turnLeft and self.avatarControlRotateSpeed or (turnRight and -self.avatarControlRotateSpeed)
        if self.wantDebugIndicator:
            self.displayDebugInfo()
        if self.speed or self.liftSpeed or self.slideSpeed or self.rotationSpeed:
            dt = ClockObject.getGlobalClock().getDt()
            distance = dt * self.speed * run
            lift = dt * self.liftSpeed * run
            slideDistance = dt * self.slideSpeed * run
            rotation = dt * self.rotationSpeed
            self.vel = Vec3(Vec3.forward() * distance + Vec3.up() * lift + Vec3.right() * slideDistance)
            if self.vel != Vec3.zero():
                rotMat = Mat3.rotateMatNormaxis(self.avatarNodePath.getH(), Vec3.up())
                step = rotMat.xform(self.vel)
                self.avatarNodePath.setFluidPos(Point3(self.avatarNodePath.getPos() + step))
            self.avatarNodePath.setH(self.avatarNodePath.getH() + rotation)
            messenger.send('avatarMoving')
        else:
            self.vel.set(0.0, 0.0, 0.0)
        return Task.cont

    def enableAvatarControls(self):
        if False:
            print('Hello World!')
        '\n        Activate the arrow keys, etc.\n        '
        assert self.debugPrint('enableAvatarControls')
        if self.task:
            self.task.remove(self.task)
        self.task = taskMgr.add(self.handleAvatarControls, 'AvatarControls-dev-%s' % (id(self),))

    def disableAvatarControls(self):
        if False:
            while True:
                i = 10
        '\n        Ignore the arrow keys, etc.\n        '
        assert self.debugPrint('disableAvatarControls')
        if self.task:
            self.task.remove()
            self.task = None

    def flushEventHandlers(self):
        if False:
            return 10
        pass
    if __debug__:

        def debugPrint(self, message):
            if False:
                print('Hello World!')
            'for debugging'
            return self.notify.debug(str(id(self)) + ' ' + message)