"""
Native support for HID input from the linux kernel
==================================================

Support starts from 2.6.32-ubuntu, or 2.6.34.

To configure HIDInput, add this to your configuration::

    [input]
    # devicename = hidinput,/dev/input/eventXX
    # example with Stantum MTP4.3" screen
    stantum = hidinput,/dev/input/event2

.. note::
    You must have read access to the input event.

You can use a custom range for the X, Y and pressure values.
For some drivers, the range reported is invalid.
To fix that, you can add these options to the argument line:

* invert_x : 1 to invert X axis
* invert_y : 1 to invert Y axis
* min_position_x : X relative minimum
* max_position_x : X relative maximum
* min_position_y : Y relative minimum
* max_position_y : Y relative maximum
* min_abs_x : X absolute minimum
* min_abs_y : Y absolute minimum
* max_abs_x : X absolute maximum
* max_abs_y : Y absolute maximum
* min_pressure : pressure minimum
* max_pressure : pressure maximum
* rotation : rotate the input coordinate (0, 90, 180, 270)

For example, on the Asus T101M, the touchscreen reports a range from 0-4095 for
the X and Y values, but the real values are in a range from 0-32768. To correct
this, you can add the following to the configuration::

    [input]
    t101m = hidinput,/dev/input/event7,max_position_x=32768,max_position_y=32768

.. versionadded:: 1.9.1

    `rotation` configuration token added.

"""
import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
__all__ = ('HIDInputMotionEventProvider', 'HIDMotionEvent')
Window = None
Keyboard = None

class HIDMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)

    def depack(self, args):
        if False:
            i = 10
            return i + 15
        self.sx = args['x']
        self.sy = args['y']
        self.profile = ['pos']
        if 'size_w' in args and 'size_h' in args:
            self.shape = ShapeRect()
            self.shape.width = args['size_w']
            self.shape.height = args['size_h']
            self.profile.append('shape')
        if 'pressure' in args:
            self.pressure = args['pressure']
            self.profile.append('pressure')
        if 'button' in args:
            self.button = args['button']
            self.profile.append('button')
        super().depack(args)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<HIDMotionEvent id=%d pos=(%f, %f) device=%s>' % (self.id, self.sx, self.sy, self.device)
if 'KIVY_DOC' in os.environ:
    HIDInputMotionEventProvider = None
else:
    import threading
    import collections
    import struct
    import fcntl
    from kivy.input.provider import MotionEventProvider
    from kivy.input.factory import MotionEventFactory
    from kivy.logger import Logger
    EV_SYN = 0
    EV_KEY = 1
    EV_REL = 2
    EV_ABS = 3
    EV_MSC = 4
    EV_SW = 5
    EV_LED = 17
    EV_SND = 18
    EV_REP = 20
    EV_FF = 21
    EV_PWR = 22
    EV_FF_STATUS = 23
    EV_MAX = 31
    EV_CNT = EV_MAX + 1
    KEY_MAX = 767
    SYN_REPORT = 0
    SYN_CONFIG = 1
    SYN_MT_REPORT = 2
    MSC_SERIAL = 0
    MSC_PULSELED = 1
    MSC_GESTURE = 2
    MSC_RAW = 3
    MSC_SCAN = 4
    MSC_MAX = 7
    MSC_CNT = MSC_MAX + 1
    ABS_X = 0
    ABS_Y = 1
    ABS_PRESSURE = 24
    ABS_MT_TOUCH_MAJOR = 48
    ABS_MT_TOUCH_MINOR = 49
    ABS_MT_WIDTH_MAJOR = 50
    ABS_MT_WIDTH_MINOR = 51
    ABS_MT_ORIENTATION = 52
    ABS_MT_POSITION_X = 53
    ABS_MT_POSITION_Y = 54
    ABS_MT_TOOL_TYPE = 55
    ABS_MT_BLOB_ID = 56
    ABS_MT_TRACKING_ID = 57
    ABS_MT_PRESSURE = 58
    EVIOCGNAME = 2147501318
    EVIOCGBIT = 2147501344
    EVIOCGABS = 2149074240
    keyboard_keys = {41: ('`', '~'), 2: ('1', '!'), 3: ('2', '@'), 4: ('3', '#'), 5: ('4', '$'), 6: ('5', '%'), 7: ('6', '^'), 8: ('7', '&'), 9: ('8', '*'), 10: ('9', '('), 11: ('0', ')'), 12: ('-', '_'), 13: ('=', '+'), 14: ('backspace',), 15: ('tab',), 16: ('q', 'Q'), 17: ('w', 'W'), 18: ('e', 'E'), 19: ('r', 'R'), 20: ('t', 'T'), 21: ('y', 'Y'), 22: ('u', 'U'), 23: ('i', 'I'), 24: ('o', 'O'), 25: ('p', 'P'), 26: ('[', '{'), 27: (']', '}'), 43: ('\\', '|'), 58: ('capslock',), 30: ('a', 'A'), 31: ('s', 'S'), 32: ('d', 'D'), 33: ('f', 'F'), 34: ('g', 'G'), 35: ('h', 'H'), 36: ('j', 'J'), 37: ('k', 'K'), 38: ('l', 'L'), 39: (';', ':'), 40: ("'", '"'), 255: ('non-US-1',), 28: ('enter',), 42: ('shift',), 44: ('z', 'Z'), 45: ('x', 'X'), 46: ('c', 'C'), 47: ('v', 'V'), 48: ('b', 'B'), 49: ('n', 'N'), 50: ('m', 'M'), 51: (',', '<'), 52: ('.', '>'), 53: ('/', '?'), 54: ('shift',), 86: ('pipe',), 29: ('lctrl',), 125: ('super',), 56: ('alt',), 57: ('spacebar',), 100: ('alt-gr',), 126: ('super',), 127: ('compose',), 97: ('rctrl',), 69: ('numlock',), 71: ('numpad7', 'home'), 75: ('numpad4', 'left'), 79: ('numpad1', 'end'), 72: ('numpad8', 'up'), 76: ('numpad5',), 80: ('numpad2', 'down'), 82: ('numpad0', 'insert'), 55: ('numpadmul',), 98: ('numpaddivide',), 73: ('numpad9', 'pageup'), 77: ('numpad6', 'right'), 81: ('numpad3', 'pagedown'), 83: ('numpaddecimal', 'delete'), 74: ('numpadsubstract',), 78: ('numpadadd',), 96: ('numpadenter',), 1: ('escape',), 59: ('f1',), 60: ('f2',), 61: ('f3',), 62: ('f4',), 63: ('f5',), 64: ('f6',), 65: ('f7',), 66: ('f8',), 67: ('f9',), 68: ('f10',), 87: ('f11',), 88: ('f12',), 84: ('Alt+SysRq',), 70: ('Screenlock',), 103: ('up',), 108: ('down',), 105: ('left',), 106: ('right',), 110: ('insert',), 111: ('delete',), 102: ('home',), 107: ('end',), 104: ('pageup',), 109: ('pagedown',), 99: ('print',), 119: ('pause',)}
    keys_str = {'spacebar': ' ', 'tab': '\t', 'shift': '', 'alt': '', 'ctrl': '', 'escape': '', 'numpad1': '1', 'numpad2': '2', 'numpad3': '3', 'numpad4': '4', 'numpad5': '5', 'numpad6': '6', 'numpad7': '7', 'numpad8': '8', 'numpad9': '9', 'numpad0': '0', 'numpadmul': '*', 'numpaddivide': '/', 'numpadadd': '+', 'numpaddecimal': '.', 'numpadsubstract': '-'}
    struct_input_event_sz = struct.calcsize('LLHHi')
    struct_input_absinfo_sz = struct.calcsize('iiiiii')
    sz_l = struct.calcsize('Q')

    class HIDInputMotionEventProvider(MotionEventProvider):
        options = ('min_position_x', 'max_position_x', 'min_position_y', 'max_position_y', 'min_pressure', 'max_pressure', 'min_abs_x', 'max_abs_x', 'min_abs_y', 'max_abs_y', 'invert_x', 'invert_y', 'rotation')

        def __init__(self, device, args):
            if False:
                while True:
                    i = 10
            super(HIDInputMotionEventProvider, self).__init__(device, args)
            global Window, Keyboard
            if Window is None:
                from kivy.core.window import Window
            if Keyboard is None:
                from kivy.core.window import Keyboard
            self.input_fn = None
            self.default_ranges = dict()
            args = args.split(',')
            if not args:
                Logger.error('HIDInput: Filename missing in configuration')
                Logger.error('HIDInput: Use /dev/input/event0 for example')
                return None
            self.input_fn = args[0]
            Logger.info('HIDInput: Read event from <%s>' % self.input_fn)
            for arg in args[1:]:
                if arg == '':
                    continue
                arg = arg.split('=')
                if len(arg) != 2:
                    Logger.error('HIDInput: invalid parameter %s, not in key=value format.' % arg)
                    continue
                (key, value) = arg
                if key not in HIDInputMotionEventProvider.options:
                    Logger.error('HIDInput: unknown %s option' % key)
                    continue
                try:
                    self.default_ranges[key] = int(value)
                except ValueError:
                    err = 'HIDInput: invalid value "%s" for "%s"' % (key, value)
                    Logger.error(err)
                    continue
                Logger.info('HIDInput: Set custom %s to %d' % (key, int(value)))
            if 'rotation' not in self.default_ranges:
                self.default_ranges['rotation'] = 0
            elif self.default_ranges['rotation'] not in (0, 90, 180, 270):
                Logger.error('HIDInput: invalid rotation value ({})'.format(self.default_ranges['rotation']))
                self.default_ranges['rotation'] = 0

        def start(self):
            if False:
                while True:
                    i = 10
            if self.input_fn is None:
                return
            self.uid = 0
            self.queue = collections.deque()
            self.dispatch_queue = []
            self.thread = threading.Thread(name=self.__class__.__name__, target=self._thread_run, kwargs=dict(queue=self.queue, input_fn=self.input_fn, device=self.device, default_ranges=self.default_ranges))
            self.thread.daemon = True
            self.thread.start()

        def _thread_run(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            input_fn = kwargs.get('input_fn')
            queue = self.queue
            dispatch_queue = self.dispatch_queue
            device = kwargs.get('device')
            drs = kwargs.get('default_ranges').get
            touches = {}
            touches_sent = []
            point = {}
            l_points = []
            range_min_position_x = 0
            range_max_position_x = 2048
            range_min_position_y = 0
            range_max_position_y = 2048
            range_min_pressure = 0
            range_max_pressure = 255
            range_min_abs_x = 0
            range_max_abs_x = 255
            range_min_abs_y = 0
            range_max_abs_y = 255
            range_min_abs_pressure = 0
            range_max_abs_pressure = 255
            invert_x = int(bool(drs('invert_x', 0)))
            invert_y = int(bool(drs('invert_y', 1)))
            rotation = drs('rotation', 0)

            def assign_coord(point, value, invert, coords):
                if False:
                    return 10
                (cx, cy) = coords
                if invert:
                    value = 1.0 - value
                if rotation == 0:
                    point[cx] = value
                elif rotation == 90:
                    point[cy] = value
                elif rotation == 180:
                    point[cx] = 1.0 - value
                elif rotation == 270:
                    point[cy] = 1.0 - value

            def assign_rel_coord(point, value, invert, coords):
                if False:
                    print('Hello World!')
                (cx, cy) = coords
                if invert:
                    value = -1 * value
                if rotation == 0:
                    point[cx] += value
                elif rotation == 90:
                    point[cy] += value
                elif rotation == 180:
                    point[cx] += -value
                elif rotation == 270:
                    point[cy] += -value
                point['x'] = min(1.0, max(0.0, point['x']))
                point['y'] = min(1.0, max(0.0, point['y']))

            def process_as_multitouch(tv_sec, tv_usec, ev_type, ev_code, ev_value):
                if False:
                    print('Hello World!')
                if ev_type == EV_SYN:
                    if ev_code == SYN_MT_REPORT:
                        if 'id' not in point:
                            return
                        l_points.append(point.copy())
                    elif ev_code == SYN_REPORT:
                        process(l_points)
                        del l_points[:]
                elif ev_type == EV_MSC and ev_code in (MSC_RAW, MSC_SCAN):
                    pass
                elif ev_code == ABS_MT_TRACKING_ID:
                    point.clear()
                    point['id'] = ev_value
                elif ev_code == ABS_MT_POSITION_X:
                    val = normalize(ev_value, range_min_position_x, range_max_position_x)
                    assign_coord(point, val, invert_x, 'xy')
                elif ev_code == ABS_MT_POSITION_Y:
                    val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
                    assign_coord(point, val, invert_y, 'yx')
                elif ev_code == ABS_MT_ORIENTATION:
                    point['orientation'] = ev_value
                elif ev_code == ABS_MT_BLOB_ID:
                    point['blobid'] = ev_value
                elif ev_code == ABS_MT_PRESSURE:
                    point['pressure'] = normalize(ev_value, range_min_pressure, range_max_pressure)
                elif ev_code == ABS_MT_TOUCH_MAJOR:
                    point['size_w'] = ev_value
                elif ev_code == ABS_MT_TOUCH_MINOR:
                    point['size_h'] = ev_value

            def process_as_mouse_or_keyboard(tv_sec, tv_usec, ev_type, ev_code, ev_value):
                if False:
                    for i in range(10):
                        print('nop')
                if ev_type == EV_SYN:
                    if ev_code == SYN_REPORT:
                        process([point])
                        if 'button' in point and point['button'].startswith('scroll'):
                            del point['button']
                            point['id'] += 1
                            point['_avoid'] = True
                            process([point])
                elif ev_type == EV_REL:
                    if ev_code == 0:
                        assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_x, 'xy')
                    elif ev_code == 1:
                        assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_y, 'yx')
                    elif ev_code == 8:
                        b = 'scrollup' if ev_value < 0 else 'scrolldown'
                        if 'button' not in point:
                            point['button'] = b
                            point['id'] += 1
                            if '_avoid' in point:
                                del point['_avoid']
                elif ev_type != EV_KEY:
                    if ev_code == ABS_X:
                        val = normalize(ev_value, range_min_abs_x, range_max_abs_x)
                        assign_coord(point, val, invert_x, 'xy')
                    elif ev_code == ABS_Y:
                        val = 1.0 - normalize(ev_value, range_min_abs_y, range_max_abs_y)
                        assign_coord(point, val, invert_y, 'yx')
                    elif ev_code == ABS_PRESSURE:
                        point['pressure'] = normalize(ev_value, range_min_abs_pressure, range_max_abs_pressure)
                else:
                    buttons = {272: 'left', 273: 'right', 274: 'middle', 275: 'side', 276: 'extra', 277: 'forward', 278: 'back', 279: 'task', 330: 'touch', 320: 'pen'}
                    if ev_code in buttons.keys():
                        if ev_value:
                            if 'button' not in point:
                                point['button'] = buttons[ev_code]
                                point['id'] += 1
                                if '_avoid' in point:
                                    del point['_avoid']
                        elif 'button' in point:
                            if point['button'] == buttons[ev_code]:
                                del point['button']
                                point['id'] += 1
                                point['_avoid'] = True
                    else:
                        if not 0 <= ev_value <= 1:
                            return
                        if ev_code not in keyboard_keys:
                            Logger.warn('HIDInput: unhandled HID code: {}'.format(ev_code))
                            return
                        z = keyboard_keys[ev_code][-1 if 'shift' in Window._modifiers else 0]
                        if z.lower() not in Keyboard.keycodes:
                            Logger.warn('HIDInput: unhandled character: {}'.format(z))
                            return
                        keycode = Keyboard.keycodes[z.lower()]
                        if ev_value == 1:
                            if z == 'shift' or z == 'alt':
                                Window._modifiers.append(z)
                            elif z.endswith('ctrl'):
                                Window._modifiers.append('ctrl')
                            dispatch_queue.append(('key_down', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
                        elif ev_value == 0:
                            dispatch_queue.append(('key_up', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
                            if (z == 'shift' or z == 'alt') and z in Window._modifiers:
                                Window._modifiers.remove(z)
                            elif z.endswith('ctrl') and 'ctrl' in Window._modifiers:
                                Window._modifiers.remove('ctrl')

            def process(points):
                if False:
                    while True:
                        i = 10
                if not is_multitouch:
                    dispatch_queue.append(('mouse_pos', (points[0]['x'] * Window.width, points[0]['y'] * Window.height)))
                actives = [args['id'] for args in points if 'id' in args and '_avoid' not in args]
                for args in points:
                    tid = args['id']
                    try:
                        touch = touches[tid]
                        if touch.sx == args['x'] and touch.sy == args['y']:
                            continue
                        touch.move(args)
                        if tid not in touches_sent:
                            queue.append(('begin', touch))
                            touches_sent.append(tid)
                        queue.append(('update', touch))
                    except KeyError:
                        if '_avoid' not in args:
                            touch = HIDMotionEvent(device, tid, args)
                            touches[touch.id] = touch
                            if tid not in touches_sent:
                                queue.append(('begin', touch))
                                touches_sent.append(tid)
                for tid in list(touches.keys())[:]:
                    if tid not in actives:
                        touch = touches[tid]
                        if tid in touches_sent:
                            touch.update_time_end()
                            queue.append(('end', touch))
                            touches_sent.remove(tid)
                        del touches[tid]

            def normalize(value, vmin, vmax):
                if False:
                    i = 10
                    return i + 15
                return (value - vmin) / float(vmax - vmin)
            fd = open(input_fn, 'rb')
            device_name = fcntl.ioctl(fd, EVIOCGNAME + (256 << 16), ' ' * 256).decode().strip()
            Logger.info('HIDMotionEvent: using <%s>' % device_name)
            bit = fcntl.ioctl(fd, EVIOCGBIT + (EV_MAX << 16), ' ' * sz_l)
            (bit,) = struct.unpack('Q', bit)
            is_multitouch = False
            for x in range(EV_MAX):
                if x != EV_ABS:
                    continue
                if bit & 1 << x == 0:
                    continue
                sbit = fcntl.ioctl(fd, EVIOCGBIT + x + (KEY_MAX << 16), ' ' * sz_l)
                (sbit,) = struct.unpack('Q', sbit)
                for y in range(KEY_MAX):
                    if sbit & 1 << y == 0:
                        continue
                    absinfo = fcntl.ioctl(fd, EVIOCGABS + y + (struct_input_absinfo_sz << 16), ' ' * struct_input_absinfo_sz)
                    (abs_value, abs_min, abs_max, abs_fuzz, abs_flat, abs_res) = struct.unpack('iiiiii', absinfo)
                    if y == ABS_MT_POSITION_X:
                        is_multitouch = True
                        range_min_position_x = drs('min_position_x', abs_min)
                        range_max_position_x = drs('max_position_x', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range position X is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_MT_POSITION_Y:
                        is_multitouch = True
                        range_min_position_y = drs('min_position_y', abs_min)
                        range_max_position_y = drs('max_position_y', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range position Y is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_MT_PRESSURE:
                        range_min_pressure = drs('min_pressure', abs_min)
                        range_max_pressure = drs('max_pressure', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range pressure is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_X:
                        range_min_abs_x = drs('min_abs_x', abs_min)
                        range_max_abs_x = drs('max_abs_x', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range ABS X position is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_Y:
                        range_min_abs_y = drs('min_abs_y', abs_min)
                        range_max_abs_y = drs('max_abs_y', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range ABS Y position is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_PRESSURE:
                        range_min_abs_pressure = drs('min_abs_pressure', abs_min)
                        range_max_abs_pressure = drs('max_abs_pressure', abs_max)
                        Logger.info('HIDMotionEvent: ' + '<%s> range ABS pressure is %d - %d' % (device_name, abs_min, abs_max))
            if not is_multitouch:
                point = {'x': 0.5, 'y': 0.5, 'id': 0, '_avoid': True}
            while fd:
                data = fd.read(struct_input_event_sz)
                if len(data) < struct_input_event_sz:
                    break
                for i in range(int(len(data) / struct_input_event_sz)):
                    ev = data[i * struct_input_event_sz:]
                    infos = struct.unpack('LLHHi', ev[:struct_input_event_sz])
                    if is_multitouch:
                        process_as_multitouch(*infos)
                    else:
                        process_as_mouse_or_keyboard(*infos)

        def update(self, dispatch_fn):
            if False:
                print('Hello World!')
            dispatch_queue = self.dispatch_queue
            n = len(dispatch_queue)
            for (name, args) in dispatch_queue[:n]:
                if name == 'mouse_pos':
                    Window.mouse_pos = args
                elif name == 'key_down':
                    if not Window.dispatch('on_key_down', *args):
                        Window.dispatch('on_keyboard', *args)
                elif name == 'key_up':
                    Window.dispatch('on_key_up', *args)
            del dispatch_queue[:n]
            try:
                while True:
                    (event_type, touch) = self.queue.popleft()
                    dispatch_fn(event_type, touch)
            except:
                pass
    MotionEventFactory.register('hidinput', HIDInputMotionEventProvider)