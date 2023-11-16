"""
Demonstrate usage of steering wheels

In this sample you can use a wheel type device to control the camera and
show some messages on screen.  You can acclerate forward using the
accleration pedal and slow down using the break pedal.
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, InputDevice, loadPrcFileData, Vec3
from direct.gui.OnscreenText import OnscreenText
loadPrcFileData('', '\n    default-fov 60\n    notify-level-device debug\n')

class App(ShowBase):

    def __init__(self):
        if False:
            print('Hello World!')
        ShowBase.__init__(self)
        self.lblWarning = OnscreenText(text='No devices found', fg=(1, 0, 0, 1), scale=0.25)
        self.lblAction = OnscreenText(text='Action', fg=(1, 1, 1, 1), scale=0.15)
        self.lblAction.hide()
        self.wheel = None
        devices = self.devices.getDevices(InputDevice.DeviceClass.steering_wheel)
        if devices:
            self.connect(devices[0])
        self.currentMoveSpeed = 0.0
        self.maxAccleration = 28.0
        self.deaccleration = 10.0
        self.deaclerationBreak = 37.0
        self.maxSpeed = 80.0
        self.accept('connect-device', self.connect)
        self.accept('disconnect-device', self.disconnect)
        self.accept('escape', exit)
        self.accept('steering_wheel0-face_a', self.action, extraArgs=['Action'])
        self.accept('steering_wheel0-face_a-up', self.actionUp)
        self.accept('steering_wheel0-hat_up', self.center_wheel)
        self.environment = loader.loadModel('environment')
        self.environment.reparentTo(render)
        self.wheelCenter = 0
        if self.wheel is not None:
            self.wheelCenter = self.wheel.findAxis(InputDevice.Axis.wheel).value
        self.disableMouse()
        self.reset()
        self.taskMgr.add(self.moveTask, 'movement update task')

    def connect(self, device):
        if False:
            return 10
        'Event handler that is called when a device is discovered.'
        if device.device_class == InputDevice.DeviceClass.steering_wheel and (not self.wheel):
            print('Found %s' % device)
            self.wheel = device
            self.attachInputDevice(device, prefix='steering_wheel0')
            self.lblWarning.hide()

    def disconnect(self, device):
        if False:
            while True:
                i = 10
        'Event handler that is called when a device is removed.'
        if self.wheel != device:
            return
        print('Disconnected %s' % device)
        self.detachInputDevice(device)
        self.wheel = None
        devices = self.devices.getDevices(InputDevice.DeviceClass.steering_wheel)
        if devices:
            self.connect(devices[0])
        else:
            self.lblWarning.show()

    def reset(self):
        if False:
            print('Hello World!')
        'Reset the camera to the initial position.'
        self.camera.setPosHpr(0, -200, 2, 0, 0, 0)

    def action(self, button):
        if False:
            while True:
                i = 10
        self.lblAction.text = 'Pressed %s' % button
        self.lblAction.show()

    def actionUp(self):
        if False:
            return 10
        self.lblAction.hide()

    def center_wheel(self):
        if False:
            while True:
                i = 10
        'Reset the wheels center rotation to the current rotation of the wheel'
        self.wheelCenter = self.wheel.findAxis(InputDevice.Axis.wheel).value

    def moveTask(self, task):
        if False:
            i = 10
            return i + 15
        dt = base.clock.dt
        movementVec = Vec3()
        if not self.wheel:
            return task.cont
        if self.currentMoveSpeed > 0:
            self.currentMoveSpeed -= dt * self.deaccleration
            if self.currentMoveSpeed < 0:
                self.currentMoveSpeed = 0
        accleratorPedal = self.wheel.findAxis(InputDevice.Axis.accelerator).value
        accleration = accleratorPedal * self.maxAccleration
        if self.currentMoveSpeed > accleratorPedal * self.maxSpeed:
            self.currentMoveSpeed -= dt * self.deaccleration
        self.currentMoveSpeed += dt * accleration
        breakPedal = self.wheel.findAxis(InputDevice.Axis.brake).value
        deacleration = breakPedal * self.deaclerationBreak
        self.currentMoveSpeed -= dt * deacleration
        if self.currentMoveSpeed < 0:
            self.currentMoveSpeed = 0
        rotation = self.wheelCenter - self.wheel.findAxis(InputDevice.Axis.wheel).value
        base.camera.setH(base.camera, 100 * dt * rotation)
        base.camera.setY(base.camera, dt * self.currentMoveSpeed)
        return task.cont
app = App()
app.run()