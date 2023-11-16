"""
Demonstrate usage of flight stick

In this sample you can use a flight stick to control the camera and show some
messages on screen.  You can accelerate using the throttle.
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, InputDevice, loadPrcFileData, Vec3
from direct.gui.OnscreenText import OnscreenText
loadPrcFileData('', '\n    default-fov 60\n    notify-level-device debug\n')
STICK_DEAD_ZONE = 0.02
THROTTLE_DEAD_ZONE = 0.02

class App(ShowBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ShowBase.__init__(self)
        self.lblWarning = OnscreenText(text='No devices found', fg=(1, 0, 0, 1), scale=0.25)
        self.lblAction = OnscreenText(text='Action', fg=(1, 1, 1, 1), scale=0.15)
        self.lblAction.hide()
        self.flightStick = None
        devices = self.devices.getDevices(InputDevice.DeviceClass.flight_stick)
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
        self.accept('flight_stick0-start', exit)
        self.accept('flight_stick0-trigger', self.action, extraArgs=['Trigger'])
        self.accept('flight_stick0-trigger-up', self.actionUp)
        self.environment = loader.loadModel('environment')
        self.environment.reparentTo(render)
        self.disableMouse()
        self.reset()
        self.taskMgr.add(self.moveTask, 'movement update task')

    def connect(self, device):
        if False:
            return 10
        'Event handler that is called when a device is discovered.'
        if device.device_class == InputDevice.DeviceClass.flight_stick and (not self.flightStick):
            print('Found %s' % device)
            self.flightStick = device
            self.attachInputDevice(device, prefix='flight_stick0')
            self.lblWarning.hide()

    def disconnect(self, device):
        if False:
            i = 10
            return i + 15
        'Event handler that is called when a device is removed.'
        if self.flightStick != device:
            return
        print('Disconnected %s' % device)
        self.detachInputDevice(device)
        self.flightStick = None
        devices = self.devices.getDevices(InputDevice.DeviceClass.flight_stick)
        if devices:
            self.connect(devices[0])
        else:
            self.lblWarning.show()

    def reset(self):
        if False:
            while True:
                i = 10
        'Reset the camera to the initial position.'
        self.camera.setPosHpr(0, -200, 10, 0, 0, 0)

    def action(self, button):
        if False:
            return 10
        self.lblAction.text = 'Pressed %s' % button
        self.lblAction.show()

    def actionUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.lblAction.hide()

    def moveTask(self, task):
        if False:
            while True:
                i = 10
        dt = base.clock.dt
        if not self.flightStick:
            return task.cont
        if self.currentMoveSpeed > 0:
            self.currentMoveSpeed -= dt * self.deaccleration
            if self.currentMoveSpeed < 0:
                self.currentMoveSpeed = 0
        throttle = self.flightStick.findAxis(InputDevice.Axis.throttle).value
        if abs(throttle) < THROTTLE_DEAD_ZONE:
            throttle = 0
        accleration = throttle * self.maxAccleration
        if self.currentMoveSpeed > throttle * self.maxSpeed:
            self.currentMoveSpeed -= dt * self.deaccleration
        self.currentMoveSpeed += dt * accleration
        stick_yaw = self.flightStick.findAxis(InputDevice.Axis.yaw)
        if abs(stick_yaw.value) > STICK_DEAD_ZONE:
            base.camera.setH(base.camera, 100 * dt * stick_yaw.value)
        stick_y = self.flightStick.findAxis(InputDevice.Axis.pitch)
        if abs(stick_y.value) > STICK_DEAD_ZONE:
            base.camera.setP(base.camera, 100 * dt * stick_y.value)
        stick_x = self.flightStick.findAxis(InputDevice.Axis.roll)
        if abs(stick_x.value) > STICK_DEAD_ZONE:
            base.camera.setR(base.camera, 100 * dt * stick_x.value)
        base.camera.setY(base.camera, dt * self.currentMoveSpeed)
        if base.camera.getZ() < 1:
            base.camera.setZ(1)
        return task.cont
app = App()
app.run()