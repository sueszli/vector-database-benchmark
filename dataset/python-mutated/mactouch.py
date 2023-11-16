"""
Native support of MultitouchSupport framework for MacBook (MaxOSX platform)
===========================================================================
"""
__all__ = ('MacMotionEventProvider',)
import ctypes
import threading
import collections
import os
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
if 'KIVY_DOC' not in os.environ:
    CFArrayRef = ctypes.c_void_p
    CFMutableArrayRef = ctypes.c_void_p
    CFIndex = ctypes.c_long
    dll = '/System/Library/PrivateFrameworks/' + 'MultitouchSupport.framework/MultitouchSupport'
    MultitouchSupport = ctypes.CDLL(dll)
    CFArrayGetCount = MultitouchSupport.CFArrayGetCount
    CFArrayGetCount.argtypes = [CFArrayRef]
    CFArrayGetCount.restype = CFIndex
    CFArrayGetValueAtIndex = MultitouchSupport.CFArrayGetValueAtIndex
    CFArrayGetValueAtIndex.argtypes = [CFArrayRef, CFIndex]
    CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    MTDeviceCreateList = MultitouchSupport.MTDeviceCreateList
    MTDeviceCreateList.argtypes = []
    MTDeviceCreateList.restype = CFMutableArrayRef

    class MTPoint(ctypes.Structure):
        _fields_ = [('x', ctypes.c_float), ('y', ctypes.c_float)]

    class MTVector(ctypes.Structure):
        _fields_ = [('position', MTPoint), ('velocity', MTPoint)]

    class MTData(ctypes.Structure):
        _fields_ = [('frame', ctypes.c_int), ('timestamp', ctypes.c_double), ('identifier', ctypes.c_int), ('state', ctypes.c_int), ('unknown1', ctypes.c_int), ('unknown2', ctypes.c_int), ('normalized', MTVector), ('size', ctypes.c_float), ('unknown3', ctypes.c_int), ('angle', ctypes.c_float), ('major_axis', ctypes.c_float), ('minor_axis', ctypes.c_float), ('unknown4', MTVector), ('unknown5_1', ctypes.c_int), ('unknown5_2', ctypes.c_int), ('unknown6', ctypes.c_float)]
    MTDataRef = ctypes.POINTER(MTData)
    MTContactCallbackFunction = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, MTDataRef, ctypes.c_int, ctypes.c_double, ctypes.c_int)
    MTDeviceRef = ctypes.c_void_p
    MTRegisterContactFrameCallback = MultitouchSupport.MTRegisterContactFrameCallback
    MTRegisterContactFrameCallback.argtypes = [MTDeviceRef, MTContactCallbackFunction]
    MTRegisterContactFrameCallback.restype = None
    MTDeviceStart = MultitouchSupport.MTDeviceStart
    MTDeviceStart.argtypes = [MTDeviceRef, ctypes.c_int]
    MTDeviceStart.restype = None
else:
    MTContactCallbackFunction = lambda x: None

class MacMotionEvent(MotionEvent):
    """MotionEvent representing a contact point on the touchpad. Supports pos
    and shape profiles.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ('pos', 'shape')

    def depack(self, args):
        if False:
            while True:
                i = 10
        self.shape = ShapeRect()
        (self.sx, self.sy) = (args[0], args[1])
        self.shape.width = args[2]
        self.shape.height = args[2]
        super().depack(args)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<MacMotionEvent id=%d pos=(%f, %f) device=%s>' % (self.id, self.sx, self.sy, self.device)
_instance = None

class MacMotionEventProvider(MotionEventProvider):

    def __init__(self, *largs, **kwargs):
        if False:
            print('Hello World!')
        global _instance
        if _instance is not None:
            raise Exception('Only one MacMotionEvent provider is allowed.')
        _instance = self
        super(MacMotionEventProvider, self).__init__(*largs, **kwargs)

    def start(self):
        if False:
            return 10
        self.uid = 0
        self.touches = {}
        self.lock = threading.Lock()
        self.queue = collections.deque()
        devices = MultitouchSupport.MTDeviceCreateList()
        num_devices = CFArrayGetCount(devices)
        for i in range(num_devices):
            device = CFArrayGetValueAtIndex(devices, i)
            data_id = str(device)
            self.touches[data_id] = {}
            MTRegisterContactFrameCallback(device, self._mts_callback)
            MTDeviceStart(device, 0)

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

    def stop(self):
        if False:
            i = 10
            return i + 15
        pass

    @MTContactCallbackFunction
    def _mts_callback(device, data_ptr, n_fingers, timestamp, frame):
        if False:
            print('Hello World!')
        global _instance
        devid = str(device)
        if devid not in _instance.touches:
            _instance.touches[devid] = {}
        touches = _instance.touches[devid]
        actives = []
        for i in range(n_fingers):
            data = data_ptr[i]
            actives.append(data.identifier)
            data_id = data.identifier
            norm_pos = data.normalized.position
            args = (norm_pos.x, norm_pos.y, data.size)
            if data_id not in touches:
                _instance.lock.acquire()
                _instance.uid += 1
                touch = MacMotionEvent(_instance.device, _instance.uid, args)
                _instance.lock.release()
                _instance.queue.append(('begin', touch))
                touches[data_id] = touch
            else:
                touch = touches[data_id]
                if data.normalized.position.x == touch.sx and data.normalized.position.y == touch.sy:
                    continue
                touch.move(args)
                _instance.queue.append(('update', touch))
        for tid in list(touches.keys())[:]:
            if tid not in actives:
                touch = touches[tid]
                touch.update_time_end()
                _instance.queue.append(('end', touch))
                del touches[tid]
        return 0
MotionEventFactory.register('mactouch', MacMotionEventProvider)