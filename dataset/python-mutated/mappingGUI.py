"""
Demonstrate how a simple button mapping gui can be written
"""
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DGG, DirectFrame, DirectButton, DirectLabel, OkCancelDialog, DirectScrolledFrame
from panda3d.core import VBase4, TextNode, Vec2, InputDevice, loadPrcFileData, GamepadButton, KeyboardButton
loadPrcFileData('', 'textures-auto-power-2 #t')
DEAD_ZONE = 0.33

class InputMapping(object):
    """A container class for storing a mapping from a string action to either
    an axis or a button.  You could extend this with additional methods to load
    the default mappings from a configuration file. """
    actions = ('Move forward', 'Move backward', 'Move left', 'Move right', 'Jump', 'Buy', 'Use', 'Break', 'Fix', 'Trash', 'Change', 'Mail', 'Upgrade')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__map = dict.fromkeys(self.actions)

    def mapButton(self, action, button):
        if False:
            print('Hello World!')
        self.__map[action] = ('button', str(button))

    def mapAxis(self, action, axis):
        if False:
            while True:
                i = 10
        self.__map[action] = ('axis', axis.name)

    def unmap(self):
        if False:
            print('Hello World!')
        self.__map[action] = None

    def formatMapping(self, action):
        if False:
            print('Hello World!')
        'Returns a string label describing the mapping for a given action,\n        for displaying in a GUI. '
        mapping = self.__map.get(action)
        if not mapping:
            return 'Unmapped'
        label = mapping[1].replace('_', ' ').title()
        if mapping[0] == 'axis':
            return 'Axis: ' + label
        else:
            return 'Button: ' + label

class ChangeActionDialog(object):
    """Encapsulates the UI dialog that opens up when changing a mapping.  It
    holds the state of which action is being set and which button is pressed
    and invokes a callback function when the dialog is exited."""

    def __init__(self, action, button_geom, command):
        if False:
            return 10
        self.action = action
        self.newInputType = ''
        self.newInput = ''
        self.setKeyCalled = False
        self.__command = command
        self.attachedDevices = []
        self.dialog = OkCancelDialog(dialogName='dlg_device_input', pos=(0, 0, 0.25), text='Hit desired key:', text_fg=VBase4(0.898, 0.839, 0.73, 1.0), text_shadow=VBase4(0, 0, 0, 0.75), text_shadowOffset=Vec2(0.05, 0.05), text_scale=0.05, text_align=TextNode.ACenter, fadeScreen=0.65, frameColor=VBase4(0.3, 0.3, 0.3, 1), button_geom=button_geom, button_scale=0.15, button_text_scale=0.35, button_text_align=TextNode.ALeft, button_text_fg=VBase4(0.898, 0.839, 0.73, 1.0), button_text_pos=Vec2(-0.9, -0.125), button_relief=1, button_pad=Vec2(0.01, 0.01), button_frameColor=VBase4(0, 0, 0, 0), button_frameSize=VBase4(-1.0, 1.0, -0.25, 0.25), button_pressEffect=False, command=self.onClose)
        self.dialog.setTransparency(True)
        self.dialog.configureDialog()
        scale = self.dialog['image_scale']
        self.dialog['image_scale'] = (scale[0] / 2.0, scale[1], scale[2] / 2.0)
        self.dialog['text_pos'] = (self.dialog['text_pos'][0], self.dialog['text_pos'][1] + 0.06)

    def buttonPressed(self, button):
        if False:
            for i in range(10):
                print('nop')
        if any((button.guiItem.getState() == 1 for button in self.dialog.buttonList)):
            return
        text = str(button).replace('_', ' ').title()
        self.dialog['text'] = 'New event will be:\n\nButton: ' + text
        self.newInputType = 'button'
        self.newInput = button

    def axisMoved(self, axis):
        if False:
            i = 10
            return i + 15
        text = axis.name.replace('_', ' ').title()
        self.dialog['text'] = 'New event will be:\n\nAxis: ' + text
        self.newInputType = 'axis'
        self.newInput = axis

    def onClose(self, result):
        if False:
            print('Hello World!')
        'Called when the OK or Cancel button is pressed.'
        self.dialog.cleanup()
        if self.newInput and result == DGG.DIALOG_OK:
            self.__command(self.action, self.newInputType, self.newInput)
        else:
            self.__command(self.action, None, None)

class MappingGUIDemo(ShowBase):

    def __init__(self):
        if False:
            return 10
        ShowBase.__init__(self)
        self.setBackgroundColor(0, 0, 0)
        DGG.getDefaultFont().setPixelsPerUnit(100)
        self.mapping = InputMapping()
        self.mapping.mapAxis('Move forward', InputDevice.Axis.left_y)
        self.mapping.mapAxis('Move backward', InputDevice.Axis.left_y)
        self.mapping.mapAxis('Move left', InputDevice.Axis.left_x)
        self.mapping.mapAxis('Move right', InputDevice.Axis.left_x)
        self.mapping.mapButton('Jump', GamepadButton.face_a())
        self.mapping.mapButton('Use', GamepadButton.face_b())
        self.mapping.mapButton('Break', GamepadButton.face_x())
        self.mapping.mapButton('Fix', GamepadButton.face_y())
        maps = loader.loadModel('models/button_map')
        self.buttonGeom = (maps.find('**/ready'), maps.find('**/click'), maps.find('**/hover'), maps.find('**/disabled'))
        DGG.setDefaultDialogGeom('models/dialog.png')
        self.textscale = 0.1
        self.title = DirectLabel(scale=self.textscale, pos=(base.a2dLeft + 0.05, 0.0, base.a2dTop - (self.textscale + 0.05)), frameColor=VBase4(0, 0, 0, 0), text='Button Mapping', text_align=TextNode.ALeft, text_fg=VBase4(1, 1, 1, 1), text_shadow=VBase4(0, 0, 0, 0.75), text_shadowOffset=Vec2(0.05, 0.05))
        self.title.setTransparency(1)
        thumbMaps = loader.loadModel('models/thumb_map')
        thumbGeom = (thumbMaps.find('**/thumb_ready'), thumbMaps.find('**/thumb_click'), thumbMaps.find('**/thumb_hover'), thumbMaps.find('**/thumb_disabled'))
        incMaps = loader.loadModel('models/inc_map')
        incGeom = (incMaps.find('**/inc_ready'), incMaps.find('**/inc_click'), incMaps.find('**/inc_hover'), incMaps.find('**/inc_disabled'))
        decMaps = loader.loadModel('models/dec_map')
        decGeom = (decMaps.find('**/dec_ready'), decMaps.find('**/dec_click'), decMaps.find('**/dec_hover'), decMaps.find('**/dec_disabled'))
        self.lstActionMap = DirectScrolledFrame(frameSize=VBase4(base.a2dLeft, base.a2dRight, 0.0, 1.55), canvasSize=VBase4(base.a2dLeft, base.a2dRight, 0.0, 0.0), frameColor=VBase4(0, 0, 0.25, 0.75), pos=(0, 0, -0.8), verticalScroll_scrollSize=0.2, verticalScroll_frameColor=VBase4(0.02, 0.02, 0.02, 1), verticalScroll_thumb_relief=1, verticalScroll_thumb_geom=thumbGeom, verticalScroll_thumb_pressEffect=False, verticalScroll_thumb_frameColor=VBase4(0, 0, 0, 0), verticalScroll_incButton_relief=1, verticalScroll_incButton_geom=incGeom, verticalScroll_incButton_pressEffect=False, verticalScroll_incButton_frameColor=VBase4(0, 0, 0, 0), verticalScroll_decButton_relief=1, verticalScroll_decButton_geom=decGeom, verticalScroll_decButton_pressEffect=False, verticalScroll_decButton_frameColor=VBase4(0, 0, 0, 0))
        idx = 0
        self.listBGEven = base.loader.loadModel('models/list_item_even')
        self.listBGOdd = base.loader.loadModel('models/list_item_odd')
        self.actionLabels = {}
        for action in self.mapping.actions:
            mapped = self.mapping.formatMapping(action)
            item = self.__makeListItem(action, mapped, idx)
            item.reparentTo(self.lstActionMap.getCanvas())
            idx += 1
        self.lstActionMap['canvasSize'] = (base.a2dLeft + 0.05, base.a2dRight - 0.05, -(len(self.mapping.actions) * 0.1), 0.09)
        self.lstActionMap.setCanvasSize()

    def closeDialog(self, action, newInputType, newInput):
        if False:
            while True:
                i = 10
        'Called in callback when the dialog is closed.  newInputType will be\n        "button" or "axis", or None if the remapping was cancelled.'
        self.dlgInput = None
        if newInputType is not None:
            if newInputType == 'axis':
                self.mapping.mapAxis(action, newInput)
            else:
                self.mapping.mapButton(action, newInput)
            self.actionLabels[action]['text'] = self.mapping.formatMapping(action)
        for bt in base.buttonThrowers:
            bt.node().setSpecificFlag(True)
            bt.node().setButtonDownEvent('')
        for bt in base.deviceButtonThrowers:
            bt.node().setSpecificFlag(True)
            bt.node().setButtonDownEvent('')
        taskMgr.remove('checkControls')
        for device in self.attachedDevices:
            base.detachInputDevice(device)
        self.attachedDevices.clear()

    def changeMapping(self, action):
        if False:
            while True:
                i = 10
        self.dlgInput = ChangeActionDialog(action, button_geom=self.buttonGeom, command=self.closeDialog)
        devices = base.devices.getDevices()
        for device in devices:
            base.attachInputDevice(device)
        self.attachedDevices = devices
        for bt in base.buttonThrowers:
            bt.node().setSpecificFlag(False)
            bt.node().setButtonDownEvent('keyListenEvent')
        for bt in base.deviceButtonThrowers:
            bt.node().setSpecificFlag(False)
            bt.node().setButtonDownEvent('deviceListenEvent')
        self.accept('keyListenEvent', self.dlgInput.buttonPressed)
        self.accept('deviceListenEvent', self.dlgInput.buttonPressed)
        self.axisStates = {None: {}}
        for device in devices:
            for axis in device.axes:
                if device not in self.axisStates.keys():
                    self.axisStates.update({device: {axis.axis: axis.value}})
                else:
                    self.axisStates[device].update({axis.axis: axis.value})
        taskMgr.add(self.watchControls, 'checkControls')

    def watchControls(self, task):
        if False:
            while True:
                i = 10
        for device in self.attachedDevices:
            if device.device_class == InputDevice.DeviceClass.mouse:
                continue
            for axis in device.axes:
                if self.axisStates[device][axis.axis] + DEAD_ZONE < axis.value or self.axisStates[device][axis.axis] - DEAD_ZONE > axis.value:
                    self.axisStates[device][axis.axis] = axis.value
                    if axis.axis != InputDevice.Axis.none:
                        self.dlgInput.axisMoved(axis.axis)
        return task.cont

    def __makeListItem(self, action, event, index):
        if False:
            for i in range(10):
                print('nop')

        def dummy():
            if False:
                while True:
                    i = 10
            pass
        if index % 2 == 0:
            bg = self.listBGEven
        else:
            bg = self.listBGOdd
        item = DirectFrame(text=action, geom=bg, geom_scale=(base.a2dRight - 0.05, 1, 0.1), frameSize=VBase4(base.a2dLeft + 0.05, base.a2dRight - 0.05, -0.05, 0.05), frameColor=VBase4(1, 0, 0, 0), text_align=TextNode.ALeft, text_scale=0.05, text_fg=VBase4(1, 1, 1, 1), text_pos=(base.a2dLeft + 0.3, -0.015), text_shadow=VBase4(0, 0, 0, 0.35), text_shadowOffset=Vec2(-0.05, -0.05), pos=(0.05, 0, -(0.1 * index)))
        item.setTransparency(True)
        lbl = DirectLabel(text=event, text_fg=VBase4(1, 1, 1, 1), text_scale=0.05, text_pos=Vec2(0, -0.015), frameColor=VBase4(0, 0, 0, 0))
        lbl.reparentTo(item)
        lbl.setTransparency(True)
        self.actionLabels[action] = lbl
        buttonScale = 0.15
        btn = DirectButton(text='Change', geom=self.buttonGeom, scale=buttonScale, text_scale=0.25, text_align=TextNode.ALeft, text_fg=VBase4(0.898, 0.839, 0.73, 1.0), text_pos=Vec2(-0.9, -0.085), relief=1, pad=Vec2(0.01, 0.01), frameColor=VBase4(0, 0, 0, 0), frameSize=VBase4(-1.0, 1.0, -0.25, 0.25), pos=(base.a2dRight - (0.898 * buttonScale + 0.3), 0, 0), pressEffect=False, command=self.changeMapping, extraArgs=[action])
        btn.setTransparency(True)
        btn.reparentTo(item)
        return item
app = MappingGUIDemo()
app.run()