import sys
from direct.showbase.ShowBase import ShowBase
from direct.showbase.DirectObject import DirectObject
from panda3d.core import InputDeviceManager, InputDevice
from panda3d.core import VBase4, Vec2
from panda3d.core import TextNode
from direct.gui.DirectGui import DGG, DirectFrame, DirectButton, DirectLabel, DirectScrolledFrame, DirectSlider

class Main(ShowBase):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        base.disableMouse()
        self.accept('escape', sys.exit)
        self.device_connectivity_monitor = DeviceConnectivityMonitor()

class DeviceConnectivityMonitor(DirectObject):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.mgr = InputDeviceManager.get_global_ptr()
        self.create_device_menu()
        self.devices = {}
        for device in self.mgr.get_devices():
            self.connect_device(device)
        self.accept('connect-device', self.connect_device)
        self.accept('disconnect-device', self.disconnect_device)

    def create_device_menu(self):
        if False:
            print('Hello World!')
        self.current_panel = None
        self.buttons = {}
        self.devices_frame = DirectScrolledFrame(frameSize=VBase4(0, base.a2dLeft * -0.75, base.a2dBottom - base.a2dTop, 0), frameColor=VBase4(0, 0, 0.25, 1.0), canvasSize=VBase4(0, base.a2dLeft * -0.75, 0, 0), scrollBarWidth=0.08, manageScrollBars=True, autoHideScrollBars=True, pos=(base.a2dLeft, 0, base.a2dTop), parent=base.aspect2d)
        self.devices_frame.setCanvasSize()

    def create_menu_button(self, device):
        if False:
            for i in range(10):
                print('nop')
        button = DirectButton(command=self.switch_to_panel, extraArgs=[device], text=device.name, text_scale=0.05, text_align=TextNode.ALeft, text_fg=VBase4(0.0, 0.0, 0.0, 1.0), text_pos=Vec2(0.01, base.a2dBottom / 10.0), relief=1, pad=Vec2(0.01, 0.01), frameColor=VBase4(0.8, 0.8, 0.8, 1.0), frameSize=VBase4(0.0, base.a2dLeft * -0.75 - 0.081, base.a2dBottom / 5.0, 0.0), parent=self.devices_frame.getCanvas())
        self.buttons[device] = button

    def destroy_menu_button(self, device):
        if False:
            i = 10
            return i + 15
        self.buttons[device].detach_node()
        del self.buttons[device]

    def refresh_device_menu(self):
        if False:
            return 10
        self.devices_frame['canvasSize'] = VBase4(0, base.a2dLeft * -0.75, base.a2dBottom / 5.0 * len(self.buttons), 0)
        self.devices_frame.setCanvasSize()
        sorted_buttons = sorted(self.buttons.items(), key=lambda i: i[0].name)
        for (idx, (dev, button)) in enumerate(sorted_buttons):
            button.set_pos(0, 0, base.a2dBottom / 5.0 * idx)

    def switch_to_panel(self, device):
        if False:
            while True:
                i = 10
        if self.current_panel is not None:
            self.devices[self.current_panel].hide()
        self.current_panel = device
        self.devices[self.current_panel].show()

    def connect_device(self, device):
        if False:
            while True:
                i = 10
        if device in self.devices:
            return
        self.devices[device] = DeviceMonitor(device)
        self.switch_to_panel(device)
        self.create_menu_button(device)
        self.refresh_device_menu()

    def disconnect_device(self, device):
        if False:
            for i in range(10):
                print('nop')
        self.devices[device].deactivate()
        del self.devices[device]
        if self.current_panel == device:
            self.current_panel = None
            if len(self.devices) > 0:
                active_device = sorted(self.devices.keys(), key=lambda d: d.name)[0]
                self.switch_to_panel(active_device)
        self.destroy_menu_button(device)
        self.refresh_device_menu()

class DeviceMonitor(DirectObject):

    def __init__(self, device):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.device = device
        self.create_panel()
        self.activate()
        self.hide()

    def activate(self):
        if False:
            while True:
                i = 10
        print('Device connected')
        print('  Name        : {}'.format(self.device.name))
        print('  Type        : {}'.format(self.device.device_class.name))
        print('  Manufacturer: {}'.format(self.device.manufacturer))
        print('  ID          : {:04x}:{:04x}'.format(self.device.vendor_id, self.device.product_id))
        axis_names = [axis.axis.name for axis in self.device.axes]
        print('  Axes        : {} ({})'.format(len(self.device.axes), ', '.join(axis_names)))
        button_names = [button.handle.name for button in self.device.buttons]
        print('  Buttons     : {} ({})'.format(len(self.device.buttons), ', '.join(button_names)))
        base.attachInputDevice(self.device)
        self.task = base.taskMgr.add(self.update, 'Monitor for {}'.format(self.device.name), sort=10)

    def deactivate(self):
        if False:
            while True:
                i = 10
        print('"{}" disconnected'.format(self.device.name))
        base.taskMgr.remove(self.task)
        self.panel.detach_node()

    def create_panel(self):
        if False:
            for i in range(10):
                print('nop')
        panel_width = base.a2dLeft * -0.25 + base.a2dRight
        scroll_bar_width = 0.08
        canvas_width = panel_width - scroll_bar_width - 0.001
        canvas_height = base.a2dBottom - base.a2dTop
        self.panel = DirectScrolledFrame(frameSize=VBase4(0, panel_width, canvas_height, 0), frameColor=VBase4(0.8, 0.8, 0.8, 1), canvasSize=VBase4(0, canvas_width, canvas_height, 0), scrollBarWidth=scroll_bar_width, manageScrollBars=True, autoHideScrollBars=True, pos=(base.a2dLeft * 0.25, 0, base.a2dTop), parent=base.aspect2d)
        panel_canvas = self.panel.getCanvas()
        offset = -0.0
        half_width_entry = dict(frameSize=VBase4(0, canvas_width / 2, -0.1, 0), parent=panel_canvas, frameColor=VBase4(0.8, 0.8, 0.8, 1))
        left_aligned_small_text = dict(text_align=TextNode.ALeft, text_scale=0.05, text_fg=VBase4(0, 0, 0, 1), text_pos=(0.05, -0.06))
        half_width_text_frame = dict(**half_width_entry, **left_aligned_small_text)
        header = dict(frameSize=VBase4(0, canvas_width, -0.1, 0), parent=panel_canvas, frameColor=VBase4(0.6, 0.6, 0.6, 1), text_align=TextNode.ALeft, text_scale=0.1, text_fg=VBase4(0, 0, 0, 1), text_pos=(0.05, -0.075))
        self.device_header = DirectLabel(text='Device data', pos=(0, 0, offset), **header)
        offset -= 0.1

        def add_data_entry(offset, label, text):
            if False:
                print('Hello World!')
            self.name = DirectLabel(text=label, pos=(0, 0, offset), **half_width_text_frame)
            self.name = DirectLabel(text=text, pos=(canvas_width / 2, 0, offset), **half_width_text_frame)
        metadata = [('Name', self.device.name), ('Device class', self.device.device_class.name), ('Manufacturer', self.device.manufacturer), ('USB ID', '{:04x}:{:04x}'.format(self.device.vendor_id, self.device.product_id))]
        for (label, text) in metadata:
            add_data_entry(offset, label, text)
            offset -= 0.1
        self.axis_sliders = []
        if len(self.device.axes) > 0:
            offset -= 0.1
            self.axes_header = DirectLabel(text='Axes', pos=(0, 0, offset), **header)
            offset -= 0.1

            def add_axis(offset, axis_name):
                if False:
                    for i in range(10):
                        print('nop')
                slider_width = canvas_width / 2
                label = DirectLabel(text=axis_name, **left_aligned_small_text, pos=(0.05, 0, offset), parent=panel_canvas)
                slider = DirectSlider(value=0.0, range=(-1.0, 1.0), state=DGG.DISABLED, frameSize=VBase4(0, slider_width, -0.1, 0), thumb_frameSize=VBase4(0.0, 0.04, -0.04, 0.04), frameColor=VBase4(0.3, 0.3, 0.3, 1), pos=(canvas_width - slider_width, 0, offset), parent=panel_canvas)
                return slider
            for axis in self.device.axes:
                axis_slider = add_axis(offset, axis.axis.name)
                self.axis_sliders.append(axis_slider)
                offset -= 0.1
        self.button_buttons = []
        if len(self.device.buttons) > 0:
            offset -= 0.1
            self.buttons_header = DirectLabel(text='Buttons', pos=(0, 0, offset), **header)
            offset -= 0.1

            def add_button(offset, button_name):
                if False:
                    for i in range(10):
                        print('nop')
                button_width = canvas_width / 2
                label = DirectLabel(text=button_name, **left_aligned_small_text, pos=(0.05, 0, offset), parent=panel_canvas)
                button = DirectFrame(frameSize=VBase4(0, button_width, -0.1, 0), text='', text_align=TextNode.ACenter, text_scale=0.05, text_fg=VBase4(0, 0, 0, 1), text_pos=(button_width / 2, -0.06), frameColor=VBase4(0.3, 0.3, 0.3, 1), pos=(canvas_width - button_width, 0, offset), parent=panel_canvas)
                return button
            for i in range(len(self.device.buttons)):
                button_name = self.device.buttons[i].handle.name
                button_button = add_button(offset, button_name)
                self.button_buttons.append(button_button)
                offset -= 0.1
        self.vibration = []
        if self.device.has_feature(InputDevice.Feature.vibration):
            offset -= 0.1
            self.vibration_header = DirectLabel(text='Vibration', pos=(0, 0, offset), **header)
            offset -= 0.1

            def add_vibration(offset, axis_name, index):
                if False:
                    for i in range(10):
                        print('nop')
                slider_width = canvas_width / 2
                label = DirectLabel(text=axis_name, **left_aligned_small_text, pos=(0.05, 0, offset), parent=panel_canvas)
                slider = DirectSlider(value=0.0, range=(0.0, 1.0), command=self.update_vibration, frameSize=VBase4(0, slider_width, -0.1, 0), thumb_frameSize=VBase4(0.0, 0.04, -0.04, 0.04), frameColor=VBase4(0.3, 0.3, 0.3, 1), pos=(canvas_width - slider_width, 0, offset), parent=panel_canvas)
                return slider
            for (index, name) in enumerate(['low frequency', 'high frequency']):
                self.vibration.append(add_vibration(offset, name, index))
                offset -= 0.1
        if -offset > -canvas_height:
            self.panel['canvasSize'] = VBase4(0, canvas_width, offset, 0)
        self.panel.setCanvasSize()

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        self.panel.show()

    def hide(self):
        if False:
            for i in range(10):
                print('nop')
        self.panel.hide()

    def update_vibration(self):
        if False:
            while True:
                i = 10
        low = self.vibration[0]['value']
        high = self.vibration[1]['value']
        self.device.set_vibration(low, high)

    def update(self, task):
        if False:
            for i in range(10):
                print('nop')
        for (idx, slider) in enumerate(self.axis_sliders):
            slider['value'] = self.device.axes[idx].value
        for (idx, button) in enumerate(self.button_buttons):
            if self.device.buttons[idx].known:
                if self.device.buttons[idx].pressed:
                    button['frameColor'] = VBase4(0.0, 0.8, 0.0, 1)
                    button['text'] = 'down'
                else:
                    button['frameColor'] = VBase4(0.3, 0.3, 0.3, 1)
                    button['text'] = 'up'
            else:
                button['frameColor'] = VBase4(0.8, 0.8, 0.0, 1)
                button['text'] = 'unknown'
        return task.cont
if __name__ == '__main__':
    main = Main()
    main.run()