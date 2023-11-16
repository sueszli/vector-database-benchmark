"""
Native support of Wacom tablet from linuxwacom driver
=====================================================

To configure LinuxWacom, add this to your configuration::

    [input]
    pen = linuxwacom,/dev/input/event2,mode=pen
    finger = linuxwacom,/dev/input/event3,mode=touch

.. note::
    You must have read access to the input event.

You can use a custom range for the X, Y and pressure values.
On some drivers, the range reported is invalid.
To fix that, you can add these options to the argument line:

* invert_x : 1 to invert X axis
* invert_y : 1 to invert Y axis
* min_position_x : X minimum
* max_position_x : X maximum
* min_position_y : Y minimum
* max_position_y : Y maximum
* min_pressure : pressure minimum
* max_pressure : pressure maximum
"""
__all__ = ('LinuxWacomMotionEventProvider', 'LinuxWacomMotionEvent')
import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect

class LinuxWacomMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)

    def depack(self, args):
        if False:
            while True:
                i = 10
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
        super().depack(args)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<LinuxWacomMotionEvent id=%d pos=(%f, %f) device=%s>' % (self.id, self.sx, self.sy, self.device)
if 'KIVY_DOC' in os.environ:
    LinuxWacomMotionEventProvider = None
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
    ABS_MISC = 40
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
    struct_input_event_sz = struct.calcsize('LLHHi')
    struct_input_absinfo_sz = struct.calcsize('iiiiii')
    sz_l = struct.calcsize('Q')

    class LinuxWacomMotionEventProvider(MotionEventProvider):
        options = ('min_position_x', 'max_position_x', 'min_position_y', 'max_position_y', 'min_pressure', 'max_pressure', 'invert_x', 'invert_y')

        def __init__(self, device, args):
            if False:
                return 10
            super(LinuxWacomMotionEventProvider, self).__init__(device, args)
            self.input_fn = None
            self.default_ranges = dict()
            self.mode = 'touch'
            args = args.split(',')
            if not args:
                Logger.error('LinuxWacom: No filename given in config')
                Logger.error('LinuxWacom: Use /dev/input/event0 for example')
                return
            self.input_fn = args[0]
            Logger.info('LinuxWacom: Read event from <%s>' % self.input_fn)
            for arg in args[1:]:
                if arg == '':
                    continue
                arg = arg.split('=')
                if len(arg) != 2:
                    err = 'LinuxWacom: Bad parameter%s: Not in key=value format.' % arg
                    Logger.error(err)
                    continue
                (key, value) = arg
                if key == 'mode':
                    self.mode = value
                    continue
                if key not in LinuxWacomMotionEventProvider.options:
                    Logger.error('LinuxWacom: unknown %s option' % key)
                    continue
                try:
                    self.default_ranges[key] = int(value)
                except ValueError:
                    err = 'LinuxWacom: value %s invalid for %s' % (key, value)
                    Logger.error(err)
                    continue
                msg = 'LinuxWacom: Set custom %s to %d' % (key, int(value))
                Logger.info(msg)
            Logger.info('LinuxWacom: mode is <%s>' % self.mode)

        def start(self):
            if False:
                i = 10
                return i + 15
            if self.input_fn is None:
                return
            self.uid = 0
            self.queue = collections.deque()
            self.thread = threading.Thread(target=self._thread_run, kwargs=dict(queue=self.queue, input_fn=self.input_fn, device=self.device, default_ranges=self.default_ranges))
            self.thread.daemon = True
            self.thread.start()

        def _thread_run(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            input_fn = kwargs.get('input_fn')
            queue = kwargs.get('queue')
            device = kwargs.get('device')
            drs = kwargs.get('default_ranges').get
            touches = {}
            touches_sent = []
            l_points = {}
            range_min_position_x = 0
            range_max_position_x = 2048
            range_min_position_y = 0
            range_max_position_y = 2048
            range_min_pressure = 0
            range_max_pressure = 255
            invert_x = int(bool(drs('invert_x', 0)))
            invert_y = int(bool(drs('invert_y', 0)))
            reset_touch = False

            def process(points):
                if False:
                    for i in range(10):
                        print('nop')
                actives = list(points.keys())
                for args in points.values():
                    tid = args['id']
                    try:
                        touch = touches[tid]
                    except KeyError:
                        touch = LinuxWacomMotionEvent(device, tid, args)
                        touches[touch.id] = touch
                    if touch.sx == args['x'] and touch.sy == args['y'] and (tid in touches_sent):
                        continue
                    touch.move(args)
                    if tid not in touches_sent:
                        queue.append(('begin', touch))
                        touches_sent.append(tid)
                    queue.append(('update', touch))
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
                    while True:
                        i = 10
                return (value - vmin) / float(vmax - vmin)
            try:
                fd = open(input_fn, 'rb')
            except IOError:
                Logger.exception('Unable to open %s' % input_fn)
                return
            device_name = fcntl.ioctl(fd, EVIOCGNAME + (256 << 16), ' ' * 256).split('\x00')[0]
            Logger.info('LinuxWacom: using <%s>' % device_name)
            bit = fcntl.ioctl(fd, EVIOCGBIT + (EV_MAX << 16), ' ' * sz_l)
            (bit,) = struct.unpack('Q', bit)
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
                    if y == ABS_X:
                        range_min_position_x = drs('min_position_x', abs_min)
                        range_max_position_x = drs('max_position_x', abs_max)
                        Logger.info('LinuxWacom: ' + '<%s> range position X is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_Y:
                        range_min_position_y = drs('min_position_y', abs_min)
                        range_max_position_y = drs('max_position_y', abs_max)
                        Logger.info('LinuxWacom: ' + '<%s> range position Y is %d - %d' % (device_name, abs_min, abs_max))
                    elif y == ABS_PRESSURE:
                        range_min_pressure = drs('min_pressure', abs_min)
                        range_max_pressure = drs('max_pressure', abs_max)
                        Logger.info('LinuxWacom: ' + '<%s> range pressure is %d - %d' % (device_name, abs_min, abs_max))
            changed = False
            touch_id = 0
            touch_x = 0
            touch_y = 0
            touch_pressure = 0
            while fd:
                data = fd.read(struct_input_event_sz)
                if len(data) < struct_input_event_sz:
                    break
                for i in range(len(data) / struct_input_event_sz):
                    ev = data[i * struct_input_event_sz:]
                    (tv_sec, tv_usec, ev_type, ev_code, ev_value) = struct.unpack('LLHHi', ev[:struct_input_event_sz])
                    if ev_type == EV_SYN and ev_code == SYN_REPORT:
                        if touch_id in l_points:
                            p = l_points[touch_id]
                        else:
                            p = dict()
                            l_points[touch_id] = p
                        p['id'] = touch_id
                        if not reset_touch:
                            p['x'] = touch_x
                            p['y'] = touch_y
                            p['pressure'] = touch_pressure
                        if self.mode == 'pen' and touch_pressure == 0 and (not reset_touch):
                            del l_points[touch_id]
                        if changed:
                            if 'x' not in p:
                                reset_touch = False
                                continue
                            process(l_points)
                            changed = False
                        if reset_touch:
                            l_points.clear()
                            reset_touch = False
                            process(l_points)
                    elif ev_type == EV_MSC and ev_code == MSC_SERIAL:
                        touch_id = ev_value
                    elif ev_type == EV_ABS and ev_code == ABS_X:
                        val = normalize(ev_value, range_min_position_x, range_max_position_x)
                        if invert_x:
                            val = 1.0 - val
                        touch_x = val
                        changed = True
                    elif ev_type == EV_ABS and ev_code == ABS_Y:
                        val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
                        if invert_y:
                            val = 1.0 - val
                        touch_y = val
                        changed = True
                    elif ev_type == EV_ABS and ev_code == ABS_PRESSURE:
                        touch_pressure = normalize(ev_value, range_min_pressure, range_max_pressure)
                        changed = True
                    elif ev_type == EV_ABS and ev_code == ABS_MISC:
                        if ev_value == 0:
                            reset_touch = True

        def update(self, dispatch_fn):
            if False:
                i = 10
                return i + 15
            try:
                while True:
                    (event_type, touch) = self.queue.popleft()
                    dispatch_fn(event_type, touch)
            except:
                pass
    MotionEventFactory.register('linuxwacom', LinuxWacomMotionEventProvider)