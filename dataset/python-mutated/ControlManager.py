from direct.showbase.InputStateGlobal import inputState
from direct.showbase.MessengerGlobal import messenger
from direct.directnotify import DirectNotifyGlobal
from direct.task import Task
from panda3d.core import ConfigVariableBool
CollisionHandlerRayStart = 4000.0

class ControlManager:
    notify = DirectNotifyGlobal.directNotify.newCategory('ControlManager')
    wantWASD = ConfigVariableBool('want-WASD', False)

    def __init__(self, enable=True, passMessagesThrough=False):
        if False:
            i = 10
            return i + 15
        assert self.notify.debug('init control manager %s' % passMessagesThrough)
        assert self.notify.debugCall(id(self))
        self.passMessagesThrough = passMessagesThrough
        self.inputStateTokens = []
        self.WASDTurnTokens = []
        self.__WASDTurn = True
        self.controls = {}
        self.currentControls = None
        self.currentControlsName = None
        self.isEnabled = 0
        if enable:
            self.enable()
        self.forceAvJumpToken = None
        if self.passMessagesThrough:
            ist = self.inputStateTokens
            ist.append(inputState.watchWithModifiers('forward', 'arrow_up', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('reverse', 'arrow_down', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('turnLeft', 'arrow_left', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('turnRight', 'arrow_right', inputSource=inputState.ArrowKeys))

    def __str__(self):
        if False:
            while True:
                i = 10
        return "ControlManager: using '%s'" % self.currentControlsName

    def add(self, controls, name='basic'):
        if False:
            i = 10
            return i + 15
        'Add a control instance to the list of available control systems.\n\n        Args:\n            controls: an avatar control system.\n            name (str): any key that you want to use to refer to the controls\n                later (e.g. using the use(<name>) call).\n\n        See also: :meth:`use()`.\n        '
        assert self.notify.debugCall(id(self))
        assert controls is not None
        oldControls = self.controls.get(name)
        if oldControls is not None:
            assert self.notify.debug('Replacing controls: %s' % name)
            oldControls.disableAvatarControls()
            oldControls.setCollisionsActive(0)
            oldControls.delete()
        controls.disableAvatarControls()
        controls.setCollisionsActive(0)
        self.controls[name] = controls

    def get(self, name):
        if False:
            while True:
                i = 10
        return self.controls.get(name)

    def remove(self, name):
        if False:
            print('Hello World!')
        'Remove a control instance from the list of available control\n        systems.\n\n        Args:\n            name: any key that was used to refer to the controls when they were\n                added (e.g. using the add(<controls>, <name>) call).\n\n        See also: :meth:`add()`.\n        '
        assert self.notify.debugCall(id(self))
        oldControls = self.controls.pop(name, None)
        if oldControls is not None:
            assert self.notify.debug('Removing controls: %s' % name)
            oldControls.disableAvatarControls()
            oldControls.setCollisionsActive(0)
    if __debug__:

        def lockControls(self):
            if False:
                return 10
            self.ignoreUse = True

        def unlockControls(self):
            if False:
                i = 10
                return i + 15
            if hasattr(self, 'ignoreUse'):
                del self.ignoreUse

    def use(self, name, avatar):
        if False:
            while True:
                i = 10
        '\n        name is a key (string) that was previously passed to add().\n\n        Use a previously added control system.\n\n        See also: :meth:`add()`.\n        '
        assert self.notify.debugCall(id(self))
        if __debug__ and hasattr(self, 'ignoreUse'):
            return
        controls = self.controls.get(name)
        if controls is not None:
            if controls is not self.currentControls:
                if self.currentControls is not None:
                    self.currentControls.disableAvatarControls()
                    self.currentControls.setCollisionsActive(0)
                    self.currentControls.setAvatar(None)
                self.currentControls = controls
                self.currentControlsName = name
                self.currentControls.setAvatar(avatar)
                self.currentControls.setCollisionsActive(1)
                if self.isEnabled:
                    self.currentControls.enableAvatarControls()
                messenger.send('use-%s-controls' % (name,), [avatar])
        else:
            assert self.notify.debug('Unkown controls: %s' % name)

    def setSpeeds(self, forwardSpeed, jumpForce, reverseSpeed, rotateSpeed, strafeLeft=0, strafeRight=0):
        if False:
            print('Hello World!')
        assert self.notify.debugCall(id(self))
        for controls in self.controls.values():
            controls.setWalkSpeed(forwardSpeed, jumpForce, reverseSpeed, rotateSpeed)

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugCall(id(self))
        self.disable()
        for controls in list(self.controls.keys()):
            self.remove(controls)
        del self.controls
        del self.currentControls
        for token in self.inputStateTokens:
            token.release()
        for token in self.WASDTurnTokens:
            token.release()
        self.WASDTurnTokens = []

    def getSpeeds(self):
        if False:
            print('Hello World!')
        if self.currentControls:
            return self.currentControls.getSpeeds()
        return None

    def getIsAirborne(self):
        if False:
            return 10
        if self.currentControls:
            return self.currentControls.getIsAirborne()
        return False

    def setTag(self, key, value):
        if False:
            print('Hello World!')
        assert self.notify.debugCall(id(self))
        for controls in self.controls.values():
            controls.setTag(key, value)

    def deleteCollisions(self):
        if False:
            print('Hello World!')
        assert self.notify.debugCall(id(self))
        for controls in self.controls.values():
            controls.deleteCollisions()

    def collisionsOn(self):
        if False:
            while True:
                i = 10
        assert self.notify.debugCall(id(self))
        if self.currentControls:
            self.currentControls.setCollisionsActive(1)

    def collisionsOff(self):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugCall(id(self))
        if self.currentControls:
            self.currentControls.setCollisionsActive(0)

    def placeOnFloor(self):
        if False:
            while True:
                i = 10
        assert self.notify.debugCall(id(self))
        if self.currentControls:
            self.currentControls.placeOnFloor()

    def enable(self):
        if False:
            return 10
        assert self.notify.debugCall(id(self))
        if self.isEnabled:
            assert self.notify.debug('already isEnabled')
            return
        self.isEnabled = 1
        ist = self.inputStateTokens
        ist.append(inputState.watch('run', 'runningEvent', 'running-on', 'running-off'))
        ist.append(inputState.watchWithModifiers('forward', 'arrow_up', inputSource=inputState.ArrowKeys))
        ist.append(inputState.watch('forward', 'force-forward', 'force-forward-stop'))
        ist.append(inputState.watchWithModifiers('reverse', 'arrow_down', inputSource=inputState.ArrowKeys))
        ist.append(inputState.watchWithModifiers('reverse', 'mouse4', inputSource=inputState.Mouse))
        if self.wantWASD:
            ist.append(inputState.watchWithModifiers('turnLeft', 'arrow_left', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watch('turnLeft', 'mouse-look_left', 'mouse-look_left-done'))
            ist.append(inputState.watch('turnLeft', 'force-turnLeft', 'force-turnLeft-stop'))
            ist.append(inputState.watchWithModifiers('turnRight', 'arrow_right', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watch('turnRight', 'mouse-look_right', 'mouse-look_right-done'))
            ist.append(inputState.watch('turnRight', 'force-turnRight', 'force-turnRight-stop'))
            ist.append(inputState.watchWithModifiers('forward', 'w', inputSource=inputState.WASD))
            ist.append(inputState.watchWithModifiers('reverse', 's', inputSource=inputState.WASD))
            ist.append(inputState.watchWithModifiers('slideLeft', 'q', inputSource=inputState.QE))
            ist.append(inputState.watchWithModifiers('slideRight', 'e', inputSource=inputState.QE))
            self.setWASDTurn(self.__WASDTurn)
        else:
            ist.append(inputState.watchWithModifiers('turnLeft', 'arrow_left', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watch('turnLeft', 'mouse-look_left', 'mouse-look_left-done'))
            ist.append(inputState.watch('turnLeft', 'force-turnLeft', 'force-turnLeft-stop'))
            ist.append(inputState.watchWithModifiers('turnRight', 'arrow_right', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watch('turnRight', 'mouse-look_right', 'mouse-look_right-done'))
            ist.append(inputState.watch('turnRight', 'force-turnRight', 'force-turnRight-stop'))
        if self.wantWASD:
            ist.append(inputState.watchWithModifiers('jump', 'space'))
        else:
            ist.append(inputState.watch('jump', 'control', 'control-up'))
        if self.currentControls:
            self.currentControls.enableAvatarControls()

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugCall(id(self))
        self.isEnabled = 0
        for token in self.inputStateTokens:
            token.release()
        self.inputStateTokens = []
        for token in self.WASDTurnTokens:
            token.release()
        self.WASDTurnTokens = []
        if self.currentControls:
            self.currentControls.disableAvatarControls()
        if self.passMessagesThrough:
            ist = self.inputStateTokens
            ist.append(inputState.watchWithModifiers('forward', 'arrow_up', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('reverse', 'arrow_down', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('turnLeft', 'arrow_left', inputSource=inputState.ArrowKeys))
            ist.append(inputState.watchWithModifiers('turnRight', 'arrow_right', inputSource=inputState.ArrowKeys))

    def stop(self):
        if False:
            return 10
        self.disable()
        if self.currentControls:
            self.currentControls.setCollisionsActive(0)
            self.currentControls.setAvatar(None)
        self.currentControls = None

    def disableAvatarJump(self):
        if False:
            return 10
        '\n        prevent\n        '
        assert self.forceAvJumpToken is None
        self.forceAvJumpToken = inputState.force('jump', 0, 'ControlManager.disableAvatarJump')

    def enableAvatarJump(self):
        if False:
            i = 10
            return i + 15
        "\n        Stop forcing the ctrl key to return 0's\n        "
        assert self.forceAvJumpToken is not None
        self.forceAvJumpToken.release()
        self.forceAvJumpToken = None

    def monitor(self, _):
        if False:
            while True:
                i = 10
        return Task.cont

    def setWASDTurn(self, turn):
        if False:
            i = 10
            return i + 15
        self.__WASDTurn = turn
        if not self.isEnabled:
            return
        turnLeftWASDSet = inputState.isSet('turnLeft', inputSource=inputState.WASD)
        turnRightWASDSet = inputState.isSet('turnRight', inputSource=inputState.WASD)
        slideLeftWASDSet = inputState.isSet('slideLeft', inputSource=inputState.WASD)
        slideRightWASDSet = inputState.isSet('slideRight', inputSource=inputState.WASD)
        for token in self.WASDTurnTokens:
            token.release()
        if turn:
            self.WASDTurnTokens = (inputState.watchWithModifiers('turnLeft', 'a', inputSource=inputState.WASD), inputState.watchWithModifiers('turnRight', 'd', inputSource=inputState.WASD))
            inputState.set('turnLeft', slideLeftWASDSet, inputSource=inputState.WASD)
            inputState.set('turnRight', slideRightWASDSet, inputSource=inputState.WASD)
            inputState.set('slideLeft', False, inputSource=inputState.WASD)
            inputState.set('slideRight', False, inputSource=inputState.WASD)
        else:
            self.WASDTurnTokens = (inputState.watchWithModifiers('slideLeft', 'a', inputSource=inputState.WASD), inputState.watchWithModifiers('slideRight', 'd', inputSource=inputState.WASD))
            inputState.set('slideLeft', turnLeftWASDSet, inputSource=inputState.WASD)
            inputState.set('slideRight', turnRightWASDSet, inputSource=inputState.WASD)
            inputState.set('turnLeft', False, inputSource=inputState.WASD)
            inputState.set('turnRight', False, inputSource=inputState.WASD)