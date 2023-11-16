"""
PhysicsWalker.py is for avatars.

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
from direct.extensions_native import Mat3_extensions
from direct.extensions_native import VBase3_extensions
from direct.extensions_native import VBase4_extensions
from panda3d.core import BitMask32, ClockObject, CollisionHandlerFloor, CollisionHandlerQueue, CollisionNode, CollisionRay, CollisionSphere, CollisionTraverser, ConfigVariableBool, LRotationf, Mat3, NodePath, Point3, Vec3
from panda3d.physics import ActorNode, ForceNode, LinearEulerIntegrator, LinearFrictionForce, LinearVectorForce, PhysicsCollisionHandler, PhysicsManager
import math

class PhysicsWalker(DirectObject.DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('PhysicsWalker')
    wantDebugIndicator = ConfigVariableBool('want-avatar-physics-indicator', False)
    useLifter = 0
    useHeightRay = 0

    def __init__(self, gravity=-32.174, standableGround=0.707, hardLandingForce=16.0):
        if False:
            while True:
                i = 10
        assert self.debugPrint('PhysicsWalker(gravity=%s, standableGround=%s)' % (gravity, standableGround))
        DirectObject.DirectObject.__init__(self)
        self.__gravity = gravity
        self.__standableGround = standableGround
        self.__hardLandingForce = hardLandingForce
        self.needToDeltaPos = 0
        self.physVelocityIndicator = None
        self.avatarControlForwardSpeed = 0
        self.avatarControlJumpForce = 0
        self.avatarControlReverseSpeed = 0
        self.avatarControlRotateSpeed = 0
        self.__oldAirborneHeight = None
        self.getAirborneHeight = None
        self.__oldContact = None
        self.__oldPosDelta = Vec3(0)
        self.__oldDt = 0
        self.__speed = 0.0
        self.__rotationSpeed = 0.0
        self.__slideSpeed = 0.0
        self.__vel = Vec3(0.0)
        self.collisionsActive = 0
        self.isAirborne = 0
        self.highMark = 0

    def setWalkSpeed(self, forward, jump, reverse, rotate):
        if False:
            return 10
        assert self.debugPrint('setWalkSpeed()')
        self.avatarControlForwardSpeed = forward
        self.avatarControlJumpForce = jump
        self.avatarControlReverseSpeed = reverse
        self.avatarControlRotateSpeed = rotate

    def getSpeeds(self):
        if False:
            return 10
        return (self.__speed, self.__rotationSpeed)

    def setAvatar(self, avatar):
        if False:
            print('Hello World!')
        self.avatar = avatar
        if avatar is not None:
            self.setupPhysics(avatar)

    def setupRay(self, floorBitmask, floorOffset):
        if False:
            print('Hello World!')
        self.cRay = CollisionRay(0.0, 0.0, CollisionHandlerRayStart, 0.0, 0.0, -1.0)
        cRayNode = CollisionNode('PW.cRayNode')
        cRayNode.addSolid(self.cRay)
        self.cRayNodePath = self.avatarNodePath.attachNewNode(cRayNode)
        self.cRayBitMask = floorBitmask
        cRayNode.setFromCollideMask(self.cRayBitMask)
        cRayNode.setIntoCollideMask(BitMask32.allOff())
        if self.useLifter:
            self.lifter = CollisionHandlerFloor()
            self.lifter.setInPattern('enter%in')
            self.lifter.setOutPattern('exit%in')
            self.lifter.setOffset(floorOffset)
            self.lifter.addCollider(self.cRayNodePath, self.avatarNodePath)
        else:
            self.cRayQueue = CollisionHandlerQueue()
            self.cTrav.addCollider(self.cRayNodePath, self.cRayQueue)

    def determineHeight(self):
        if False:
            return 10
        '\n        returns the height of the avatar above the ground.\n        If there is no floor below the avatar, 0.0 is returned.\n        aka get airborne height.\n        '
        if self.useLifter:
            height = self.avatarNodePath.getPos(self.cRayNodePath)
            assert onScreenDebug.add('height', height.getZ())
            return height.getZ() - self.floorOffset
        else:
            height = 0.0
            if self.cRayQueue.getNumEntries() != 0:
                self.cRayQueue.sortEntries()
                floorPoint = self.cRayQueue.getEntry(0).getFromIntersectionPoint()
                height = -floorPoint.getZ()
            self.cRayQueue.clearEntries()
            if __debug__:
                onScreenDebug.add('height', height)
            return height

    def setupSphere(self, bitmask, avatarRadius):
        if False:
            i = 10
            return i + 15
        '\n        Set up the collision sphere\n        '
        self.avatarRadius = avatarRadius
        centerHeight = avatarRadius
        if self.useHeightRay:
            centerHeight *= 2.0
        self.cSphere = CollisionSphere(0.0, 0.0, centerHeight, avatarRadius)
        cSphereNode = CollisionNode('PW.cSphereNode')
        cSphereNode.addSolid(self.cSphere)
        self.cSphereNodePath = self.avatarNodePath.attachNewNode(cSphereNode)
        self.cSphereBitMask = bitmask
        cSphereNode.setFromCollideMask(self.cSphereBitMask)
        cSphereNode.setIntoCollideMask(BitMask32.allOff())
        self.pusher = PhysicsCollisionHandler()
        self.pusher.setInPattern('enter%in')
        self.pusher.setOutPattern('exit%in')
        self.pusher.addCollider(self.cSphereNodePath, self.avatarNodePath)

    def setupPhysics(self, avatarNodePath):
        if False:
            i = 10
            return i + 15
        assert self.debugPrint('setupPhysics()')
        self.actorNode = ActorNode('PW physicsActor')
        self.actorNode.getPhysicsObject().setOriented(1)
        self.actorNode.getPhysical(0).setViscosity(0.1)
        physicsActor = NodePath(self.actorNode)
        avatarNodePath.reparentTo(physicsActor)
        avatarNodePath.assign(physicsActor)
        self.phys = PhysicsManager()
        fn = ForceNode('gravity')
        fnp = NodePath(fn)
        fnp.reparentTo(render)
        gravity = LinearVectorForce(0.0, 0.0, self.__gravity)
        fn.addForce(gravity)
        self.phys.addLinearForce(gravity)
        self.gravity = gravity
        fn = ForceNode('priorParent')
        fnp = NodePath(fn)
        fnp.reparentTo(render)
        priorParent = LinearVectorForce(0.0, 0.0, 0.0)
        fn.addForce(priorParent)
        self.phys.addLinearForce(priorParent)
        self.priorParentNp = fnp
        self.priorParent = priorParent
        fn = ForceNode('viscosity')
        fnp = NodePath(fn)
        fnp.reparentTo(render)
        self.avatarViscosity = LinearFrictionForce(0.0, 1.0, 0)
        fn.addForce(self.avatarViscosity)
        self.phys.addLinearForce(self.avatarViscosity)
        self.phys.attachLinearIntegrator(LinearEulerIntegrator())
        self.phys.attachPhysicalNode(physicsActor.node())
        self.acForce = LinearVectorForce(0.0, 0.0, 0.0)
        fn = ForceNode('avatarControls')
        fnp = NodePath(fn)
        fnp.reparentTo(render)
        fn.addForce(self.acForce)
        self.phys.addLinearForce(self.acForce)
        return avatarNodePath

    def initializeCollisions(self, collisionTraverser, avatarNodePath, wallBitmask, floorBitmask, avatarRadius=1.4, floorOffset=1.0, reach=1.0):
        if False:
            print('Hello World!')
        '\n        Set up the avatar collisions\n        '
        assert self.debugPrint('initializeCollisions()')
        assert not avatarNodePath.isEmpty()
        self.cTrav = collisionTraverser
        self.floorOffset = floorOffset = 7.0
        self.avatarNodePath = self.setupPhysics(avatarNodePath)
        if self.useHeightRay:
            self.setupRay(floorBitmask, 0.0)
        self.setupSphere(wallBitmask | floorBitmask, avatarRadius)
        self.setCollisionsActive(1)

    def setAirborneHeightFunc(self, getAirborneHeight):
        if False:
            i = 10
            return i + 15
        self.getAirborneHeight = getAirborneHeight

    def setAvatarPhysicsIndicator(self, indicator):
        if False:
            print('Hello World!')
        '\n        indicator is a NodePath\n        '
        assert self.debugPrint('setAvatarPhysicsIndicator()')
        self.cSphereNodePath.show()
        if indicator:
            change = render.attachNewNode('change')
            change.setScale(0.1)
            indicator.reparentTo(change)
            indicatorNode = render.attachNewNode('physVelocityIndicator')
            indicatorNode.setPos(self.avatarNodePath, 0.0, 0.0, 6.0)
            indicatorNode.setColor(0.0, 0.0, 1.0, 1.0)
            change.reparentTo(indicatorNode)
            self.physVelocityIndicator = indicatorNode
            contactIndicatorNode = render.attachNewNode('physContactIndicator')
            contactIndicatorNode.setScale(0.25)
            contactIndicatorNode.setP(90.0)
            contactIndicatorNode.setPos(self.avatarNodePath, 0.0, 0.0, 5.0)
            contactIndicatorNode.setColor(1.0, 0.0, 0.0, 1.0)
            indicator.instanceTo(contactIndicatorNode)
            self.physContactIndicator = contactIndicatorNode
        else:
            print('failed load of physics indicator')

    def avatarPhysicsIndicator(self, task):
        if False:
            i = 10
            return i + 15
        self.physVelocityIndicator.setPos(self.avatarNodePath, 0.0, 0.0, 6.0)
        physObject = self.actorNode.getPhysicsObject()
        a = physObject.getVelocity()
        self.physVelocityIndicator.setScale(math.sqrt(a.length()))
        a += self.physVelocityIndicator.getPos()
        self.physVelocityIndicator.lookAt(Point3(a))
        contact = self.actorNode.getContactVector()
        if contact == Vec3.zero():
            self.physContactIndicator.hide()
        else:
            self.physContactIndicator.show()
            self.physContactIndicator.setPos(self.avatarNodePath, 0.0, 0.0, 5.0)
            point = Point3(contact + self.physContactIndicator.getPos())
            self.physContactIndicator.lookAt(point)
        return Task.cont

    def deleteCollisions(self):
        if False:
            print('Hello World!')
        assert self.debugPrint('deleteCollisions()')
        del self.cTrav
        if self.useHeightRay:
            del self.cRayQueue
            self.cRayNodePath.removeNode()
            del self.cRayNodePath
        del self.cSphere
        self.cSphereNodePath.removeNode()
        del self.cSphereNodePath
        del self.pusher
        del self.getAirborneHeight

    def setCollisionsActive(self, active=1):
        if False:
            print('Hello World!')
        assert self.debugPrint('collisionsActive(active=%s)' % (active,))
        if self.collisionsActive != active:
            self.collisionsActive = active
            if active:
                self.cTrav.addCollider(self.cSphereNodePath, self.pusher)
                if self.useHeightRay:
                    if self.useLifter:
                        self.cTrav.addCollider(self.cRayNodePath, self.lifter)
                    else:
                        self.cTrav.addCollider(self.cRayNodePath, self.cRayQueue)
            else:
                self.cTrav.removeCollider(self.cSphereNodePath)
                if self.useHeightRay:
                    self.cTrav.removeCollider(self.cRayNodePath)
                self.oneTimeCollide()

    def getCollisionsActive(self):
        if False:
            while True:
                i = 10
        assert self.debugPrint('getCollisionsActive() returning=%s' % (self.collisionsActive,))
        return self.collisionsActive

    def placeOnFloor(self):
        if False:
            while True:
                i = 10
        '\n        Make a reasonable effort to place the avatar on the ground.\n        For example, this is useful when switching away from the\n        current walker.\n        '
        self.oneTimeCollide()
        self.avatarNodePath.setZ(self.avatarNodePath.getZ() - self.getAirborneHeight())

    def oneTimeCollide(self):
        if False:
            return 10
        '\n        Makes one quick collision pass for the avatar, for instance as\n        a one-time straighten-things-up operation after collisions\n        have been disabled.\n        '
        assert self.debugPrint('oneTimeCollide()')
        tempCTrav = CollisionTraverser('oneTimeCollide')
        if self.useHeightRay:
            if self.useLifter:
                tempCTrav.addCollider(self.cRayNodePath, self.lifter)
            else:
                tempCTrav.addCollider(self.cRayNodePath, self.cRayQueue)
        tempCTrav.traverse(render)

    def addBlastForce(self, vector):
        if False:
            print('Hello World!')
        pass

    def displayDebugInfo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For debug use.\n        '
        onScreenDebug.add('w controls', 'PhysicsWalker')
        if self.useLifter:
            onScreenDebug.add('w airborneHeight', self.lifter.getAirborneHeight())
            onScreenDebug.add('w isOnGround', self.lifter.isOnGround())
            onScreenDebug.add('w contact normal', self.lifter.getContactNormal().pPrintValues())
            onScreenDebug.add('w impact', self.lifter.getImpactVelocity())
            onScreenDebug.add('w velocity', self.lifter.getVelocity())
            onScreenDebug.add('w hasContact', self.lifter.hasContact())
        onScreenDebug.add('w isAirborne', self.isAirborne)

    def handleAvatarControls(self, task):
        if False:
            i = 10
            return i + 15
        '\n        Check on the arrow keys and update the avatar.\n        '
        if __debug__:
            if self.wantDebugIndicator:
                onScreenDebug.append('localAvatar pos = %s\n' % (base.localAvatar.getPos().pPrintValues(),))
                onScreenDebug.append('localAvatar h = % 10.4f\n' % (base.localAvatar.getH(),))
                onScreenDebug.append('localAvatar anim = %s\n' % (base.localAvatar.animFSM.getCurrentState().getName(),))
        physObject = self.actorNode.getPhysicsObject()
        contact = self.actorNode.getContactVector()
        if contact == Vec3.zero() and self.avatarNodePath.getZ() < -50.0:
            self.reset()
            self.avatarNodePath.setZ(50.0)
            messenger.send('walkerIsOutOfWorld', [self.avatarNodePath])
        if self.wantDebugIndicator:
            self.displayDebugInfo()
        forward = inputState.isSet('forward')
        reverse = inputState.isSet('reverse')
        turnLeft = inputState.isSet('turnLeft')
        turnRight = inputState.isSet('turnRight')
        slide = 0
        slideLeft = 0
        slideRight = 0
        jump = inputState.isSet('jump')
        if base.localAvatar.getAutoRun():
            forward = 1
            reverse = 0
        self.__speed = forward and self.avatarControlForwardSpeed or (reverse and -self.avatarControlReverseSpeed)
        avatarSlideSpeed = self.avatarControlForwardSpeed * 0.5
        self.__slideSpeed = slideLeft and -avatarSlideSpeed or (slideRight and avatarSlideSpeed)
        self.__rotationSpeed = not slide and (turnLeft and self.avatarControlRotateSpeed or (turnRight and -self.avatarControlRotateSpeed))
        dt = ClockObject.getGlobalClock().getDt()
        if self.needToDeltaPos:
            self.setPriorParentVector()
            self.needToDeltaPos = 0
        self.__oldPosDelta = self.avatarNodePath.getPosDelta(render)
        self.__oldDt = dt
        if __debug__:
            if self.wantDebugIndicator:
                onScreenDebug.add('posDelta1', self.avatarNodePath.getPosDelta(render).pPrintValues())
                onScreenDebug.add('physObject vel', physObject.getVelocity().pPrintValues())
                onScreenDebug.add('physObject len', '% 10.4f' % physObject.getVelocity().length())
                onScreenDebug.add('priorParent', self.priorParent.getLocalVector().pPrintValues())
                onScreenDebug.add('contact', contact.pPrintValues())
        airborneHeight = self.getAirborneHeight()
        if airborneHeight > self.highMark:
            self.highMark = airborneHeight
            if __debug__:
                onScreenDebug.add('highMark', '% 10.4f' % (self.highMark,))
        if airborneHeight > self.avatarRadius * 0.5 or physObject.getVelocity().getZ() > 0.0:
            self.isAirborne = 1
        elif self.isAirborne and physObject.getVelocity().getZ() <= 0.0:
            contactLength = contact.length()
            if contactLength > self.__hardLandingForce:
                messenger.send('jumpHardLand')
            else:
                messenger.send('jumpLand')
            self.priorParent.setVector(Vec3.zero())
            self.isAirborne = 0
        elif jump:
            messenger.send('jumpStart')
            jumpVec = Vec3.up()
            jumpVec *= self.avatarControlJumpForce
            physObject.addImpulse(Vec3(jumpVec))
            self.isAirborne = 1
        else:
            self.isAirborne = 0
        if __debug__:
            onScreenDebug.add('isAirborne', '%d' % (self.isAirborne,))
        if contact != self.__oldContact:
            self.__oldContact = Vec3(contact)
        self.__oldAirborneHeight = airborneHeight
        moveToGround = Vec3.zero()
        if not self.useHeightRay or self.isAirborne:
            self.phys.doPhysics(dt)
            if __debug__:
                onScreenDebug.add('phys', 'on')
        else:
            physObject.setVelocity(Vec3.zero())
            moveToGround = Vec3(0.0, 0.0, -self.determineHeight())
            if __debug__:
                onScreenDebug.add('phys', 'off')
        if self.__speed or self.__slideSpeed or self.__rotationSpeed or (moveToGround != Vec3.zero()):
            distance = dt * self.__speed
            slideDistance = dt * self.__slideSpeed
            rotation = dt * self.__rotationSpeed
            assert self.avatarNodePath.getQuat().isSameDirection(physObject.getOrientation())
            assert self.avatarNodePath.getPos().almostEqual(physObject.getPosition(), 0.0001)
            self.__vel = Vec3(Vec3.forward() * distance + Vec3.right() * slideDistance)
            rotMat = Mat3.rotateMatNormaxis(self.avatarNodePath.getH(), Vec3.up())
            step = rotMat.xform(self.__vel)
            physObject.setPosition(Point3(physObject.getPosition() + step + moveToGround))
            o = physObject.getOrientation()
            r = LRotationf()
            r.setHpr(Vec3(rotation, 0.0, 0.0))
            physObject.setOrientation(o * r)
            self.actorNode.updateTransform()
            assert self.avatarNodePath.getQuat().isSameDirection(physObject.getOrientation())
            assert self.avatarNodePath.getPos().almostEqual(physObject.getPosition(), 0.0001)
            messenger.send('avatarMoving')
        else:
            self.__vel.set(0.0, 0.0, 0.0)
        self.actorNode.setContactVector(Vec3.zero())
        return Task.cont

    def doDeltaPos(self):
        if False:
            print('Hello World!')
        assert self.debugPrint('doDeltaPos()')
        self.needToDeltaPos = 1

    def setPriorParentVector(self):
        if False:
            i = 10
            return i + 15
        assert self.debugPrint('doDeltaPos()')
        print('self.__oldDt %s self.__oldPosDelta %s' % (self.__oldDt, self.__oldPosDelta))
        if __debug__:
            onScreenDebug.add('__oldDt', '% 10.4f' % self.__oldDt)
            onScreenDebug.add('self.__oldPosDelta', self.__oldPosDelta.pPrintValues())
        velocity = self.__oldPosDelta * (1 / self.__oldDt) * 4.0
        assert self.debugPrint('  __oldPosDelta=%s' % (self.__oldPosDelta,))
        assert self.debugPrint('  velocity=%s' % (velocity,))
        self.priorParent.setVector(Vec3(velocity))
        if __debug__:
            if self.wantDebugIndicator:
                onScreenDebug.add('velocity', velocity.pPrintValues())

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.debugPrint('reset()')
        self.actorNode.getPhysicsObject().resetPosition(self.avatarNodePath.getPos())
        self.priorParent.setVector(Vec3.zero())
        self.highMark = 0
        self.actorNode.setContactVector(Vec3.zero())
        if __debug__:
            contact = self.actorNode.getContactVector()
            onScreenDebug.add('priorParent po', self.priorParent.getVector(self.actorNode.getPhysicsObject()).pPrintValues())
            onScreenDebug.add('highMark', '% 10.4f' % (self.highMark,))
            onScreenDebug.add('contact', contact.pPrintValues())

    def getVelocity(self):
        if False:
            print('Hello World!')
        physObject = self.actorNode.getPhysicsObject()
        return physObject.getVelocity()

    def enableAvatarControls(self):
        if False:
            i = 10
            return i + 15
        '\n        Activate the arrow keys, etc.\n        '
        assert self.debugPrint('enableAvatarControls()')
        assert self.collisionsActive
        if __debug__:
            self.accept('f3', self.reset)
        taskName = 'AvatarControls-%s' % (id(self),)
        taskMgr.remove(taskName)
        taskMgr.add(self.handleAvatarControls, taskName, 25)
        if self.physVelocityIndicator:
            taskMgr.add(self.avatarPhysicsIndicator, 'AvatarControlsIndicator%s' % (id(self),), 35)

    def disableAvatarControls(self):
        if False:
            while True:
                i = 10
        '\n        Ignore the arrow keys, etc.\n        '
        assert self.debugPrint('disableAvatarControls()')
        taskName = 'AvatarControls-%s' % (id(self),)
        taskMgr.remove(taskName)
        taskName = 'AvatarControlsIndicator%s' % (id(self),)
        taskMgr.remove(taskName)
        if __debug__:
            self.ignore('control-f3')
            self.ignore('f3')

    def flushEventHandlers(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'cTrav'):
            if self.useLifter:
                self.lifter.flush()
            self.pusher.flush()
    if __debug__:

        def setupAvatarPhysicsIndicator(self):
            if False:
                print('Hello World!')
            if self.wantDebugIndicator:
                indicator = base.loader.loadModel('phase_5/models/props/dagger')

        def debugPrint(self, message):
            if False:
                for i in range(10):
                    print('nop')
            'for debugging'
            return self.notify.debug(str(id(self)) + ' ' + message)