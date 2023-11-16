from direct.showbase.InputStateGlobal import inputState
from direct.showbase.MessengerGlobal import messenger
from direct.task.Task import Task
from panda3d.core import ClockObject, Mat3, Point3, Vec3
from . import GravityWalker
BattleStrafe = 0

def ToggleStrafe():
    if False:
        for i in range(10):
            print('nop')
    global BattleStrafe
    BattleStrafe = not BattleStrafe

def SetStrafe(status):
    if False:
        print('Hello World!')
    global BattleStrafe
    BattleStrafe = status

class BattleWalker(GravityWalker.GravityWalker):

    def __init__(self):
        if False:
            while True:
                i = 10
        GravityWalker.GravityWalker.__init__(self)
        self.slideSpeed = 0
        self.advanceSpeed = 0

    def getSpeeds(self):
        if False:
            return 10
        return (self.speed, self.rotationSpeed, self.slideSpeed, self.advanceSpeed)

    def handleAvatarControls(self, task):
        if False:
            return 10
        '\n        Check on the arrow keys and update the avatar.\n        '
        run = inputState.isSet('run')
        forward = inputState.isSet('forward')
        reverse = inputState.isSet('reverse')
        turnLeft = inputState.isSet('turnLeft')
        turnRight = inputState.isSet('turnRight')
        slideLeft = inputState.isSet('slideLeft')
        slideRight = inputState.isSet('slideRight')
        jump = inputState.isSet('jump')
        if base.localAvatar.getAutoRun():
            forward = 1
            reverse = 0
        self.speed = forward and self.avatarControlForwardSpeed or (reverse and -self.avatarControlReverseSpeed)
        self.slideSpeed = (slideLeft and -self.avatarControlForwardSpeed or (slideRight and self.avatarControlForwardSpeed)) * 0.5
        self.rotationSpeed = not (slideLeft or slideRight) and (turnLeft and self.avatarControlRotateSpeed or (turnRight and -self.avatarControlRotateSpeed))
        debugRunning = inputState.isSet('debugRunning')
        if debugRunning:
            self.speed *= base.debugRunningMultiplier
            self.slideSpeed *= base.debugRunningMultiplier
            self.rotationSpeed *= 1.25
        if self.needToDeltaPos:
            self.setPriorParentVector()
            self.needToDeltaPos = 0
        if self.wantDebugIndicator:
            self.displayDebugInfo()
        if self.lifter.isOnGround():
            if self.isAirborne:
                self.isAirborne = 0
                assert self.debugPrint('isAirborne 0 due to isOnGround() true')
                impact = self.lifter.getImpactVelocity()
                if impact < -30.0:
                    messenger.send('jumpHardLand')
                    self.startJumpDelay(0.3)
                else:
                    messenger.send('jumpLand')
                    if impact < -5.0:
                        self.startJumpDelay(0.2)
            assert self.isAirborne == 0
            self.priorParent = Vec3.zero()
            if jump and self.mayJump:
                self.lifter.addVelocity(self.avatarControlJumpForce)
                messenger.send('jumpStart')
                self.isAirborne = 1
                assert self.debugPrint('isAirborne 1 due to jump')
        else:
            if self.isAirborne == 0:
                assert self.debugPrint('isAirborne 1 due to isOnGround() false')
            self.isAirborne = 1
        self.__oldPosDelta = self.avatarNodePath.getPosDelta(render)
        self.__oldDt = ClockObject.getGlobalClock().getDt()
        dt = self.__oldDt
        self.moving = self.speed or self.slideSpeed or self.rotationSpeed or (self.priorParent != Vec3.zero())
        if self.moving:
            distance = dt * self.speed
            slideDistance = dt * self.slideSpeed
            rotation = dt * self.rotationSpeed
            if distance or slideDistance or self.priorParent != Vec3.zero():
                rotMat = Mat3.rotateMatNormaxis(self.avatarNodePath.getH(), Vec3.up())
                if self.isAirborne:
                    forward = Vec3.forward()
                else:
                    contact = self.lifter.getContactNormal()
                    forward = contact.cross(Vec3.right())
                    forward.normalize()
                self.vel = Vec3(forward * distance)
                if slideDistance:
                    if self.isAirborne:
                        right = Vec3.right()
                    else:
                        right = forward.cross(contact)
                        right.normalize()
                    self.vel = Vec3(self.vel + right * slideDistance)
                self.vel = Vec3(rotMat.xform(self.vel))
                step = self.vel + self.priorParent * dt
                self.avatarNodePath.setFluidPos(Point3(self.avatarNodePath.getPos() + step))
            self.avatarNodePath.setH(self.avatarNodePath.getH() + rotation)
        else:
            self.vel.set(0.0, 0.0, 0.0)
        if self.moving or jump:
            messenger.send('avatarMoving')
        return Task.cont