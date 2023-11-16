"""
Python mtdev
============

The mtdev module provides Python bindings to the `Kernel multi-touch
transformation library <https://launchpad.net/mtdev>`_, also known as mtdev
(MIT license).

The mtdev library transforms all variants of kernel MT events to the
slotted type B protocol. The events put into mtdev may be from any MT
device, specifically type A without contact tracking, type A with
contact tracking, or type B with contact tracking. See the kernel
documentation for further details.

.. warning::

    This is an external library and Kivy does not provide any support for it.
    It might change in the future and we advise you don't rely on it in your
    code.
"""
import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, c_void_p, pointer, POINTER, byref
if 'KIVY_DOC' not in os.environ:
    libmtdev = cdll.LoadLibrary('libmtdev.so.1')
MTDEV_CODE_SLOT = 47
MTDEV_CODE_TOUCH_MAJOR = 48
MTDEV_CODE_TOUCH_MINOR = 49
MTDEV_CODE_WIDTH_MAJOR = 50
MTDEV_CODE_WIDTH_MINOR = 51
MTDEV_CODE_ORIENTATION = 52
MTDEV_CODE_POSITION_X = 53
MTDEV_CODE_POSITION_Y = 54
MTDEV_CODE_TOOL_TYPE = 55
MTDEV_CODE_BLOB_ID = 56
MTDEV_CODE_TRACKING_ID = 57
MTDEV_CODE_PRESSURE = 58
MTDEV_CODE_ABS_X = 0
MTDEV_CODE_ABS_Y = 1
MTDEV_CODE_ABS_Z = 2
MTDEV_CODE_BTN_DIGI = 320
MTDEV_CODE_BTN_TOOL_PEN = 320
MTDEV_CODE_BTN_TOOL_RUBBER = 321
MTDEV_CODE_BTN_TOOL_BRUSH = 322
MTDEV_CODE_BTN_TOOL_PENCIL = 323
MTDEV_CODE_BTN_TOOL_AIRBRUSH = 324
MTDEV_CODE_BTN_TOOL_FINGER = 325
MTDEV_CODE_BTN_TOOL_MOUSE = 326
MTDEV_CODE_BTN_TOOL_LENS = 327
MTDEV_CODE_BTN_TOUCH = 330
MTDEV_CODE_BTN_STYLUS = 331
MTDEV_CODE_BTN_STYLUS2 = 332
MTDEV_CODE_BTN_TOOL_DOUBLETAP = 333
MTDEV_CODE_BTN_TOOL_TRIPLETAP = 334
MTDEV_CODE_BTN_TOOL_QUADTAP = 335
MTDEV_TYPE_EV_ABS = 3
MTDEV_TYPE_EV_SYN = 0
MTDEV_TYPE_EV_KEY = 1
MTDEV_TYPE_EV_REL = 2
MTDEV_TYPE_EV_ABS = 3
MTDEV_TYPE_EV_MSC = 4
MTDEV_TYPE_EV_SW = 5
MTDEV_TYPE_EV_LED = 17
MTDEV_TYPE_EV_SND = 18
MTDEV_TYPE_EV_REP = 20
MTDEV_TYPE_EV_FF = 21
MTDEV_TYPE_EV_PWR = 22
MTDEV_TYPE_EV_FF_STATUS = 23
MTDEV_ABS_TRACKING_ID = 9
MTDEV_ABS_POSITION_X = 5
MTDEV_ABS_POSITION_Y = 6
MTDEV_ABS_TOUCH_MAJOR = 0
MTDEV_ABS_TOUCH_MINOR = 1
MTDEV_ABS_WIDTH_MAJOR = 2
MTDEV_ABS_WIDTH_MINOR = 3
MTDEV_ABS_ORIENTATION = 4
MTDEV_ABS_SIZE = 11

class timeval(Structure):
    _fields_ = [('tv_sec', c_ulong), ('tv_usec', c_ulong)]

class input_event(Structure):
    _fields_ = [('time', timeval), ('type', c_ushort), ('code', c_ushort), ('value', c_int)]

class input_absinfo(Structure):
    _fields_ = [('value', c_int), ('minimum', c_int), ('maximum', c_int), ('fuzz', c_int), ('flat', c_int), ('resolution', c_int)]

class mtdev_caps(Structure):
    _fields_ = [('has_mtdata', c_int), ('has_slot', c_int), ('has_abs', c_int * MTDEV_ABS_SIZE), ('slot', input_absinfo), ('abs', input_absinfo * MTDEV_ABS_SIZE)]

class mtdev(Structure):
    _fields_ = [('caps', mtdev_caps), ('state', c_void_p)]
if 'KIVY_DOC' not in os.environ:
    mtdev_open = libmtdev.mtdev_open
    mtdev_open.argtypes = [POINTER(mtdev), c_int]
    mtdev_get = libmtdev.mtdev_get
    mtdev_get.argtypes = [POINTER(mtdev), c_int, POINTER(input_event), c_int]
    mtdev_idle = libmtdev.mtdev_idle
    mtdev_idle.argtypes = [POINTER(mtdev), c_int, c_int]
    mtdev_close = libmtdev.mtdev_close
    mtdev_close.argtypes = [POINTER(mtdev)]

class Device:

    def __init__(self, filename):
        if False:
            print('Hello World!')
        self._filename = filename
        self._fd = -1
        self._device = mtdev()
        permission_wait_until = time.time() + 3.0
        while self._fd == -1:
            try:
                self._fd = os.open(filename, os.O_NONBLOCK | os.O_RDONLY)
            except PermissionError:
                if time.time() > permission_wait_until:
                    raise
        ret = mtdev_open(pointer(self._device), self._fd)
        if ret != 0:
            os.close(self._fd)
            self._fd = -1
            raise Exception('Unable to open device')

    def close(self):
        if False:
            print('Hello World!')
        'Close the mtdev converter\n        '
        if self._fd == -1:
            return
        mtdev_close(pointer(self._device))
        os.close(self._fd)
        self._fd = -1

    def idle(self, ms):
        if False:
            i = 10
            return i + 15
        'Check state of kernel device\n\n        :Parameters:\n            `ms`: int\n                Number of milliseconds to wait for activity\n\n        :Return:\n            Return True if the device is idle, i.e, there are no fetched events\n            in the pipe and there is nothing to fetch from the device.\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        return bool(mtdev_idle(pointer(self._device), self._fd, ms))

    def get(self):
        if False:
            while True:
                i = 10
        if self._fd == -1:
            raise Exception('Device closed')
        ev = input_event()
        if mtdev_get(pointer(self._device), self._fd, byref(ev), 1) <= 0:
            return None
        return ev

    def has_mtdata(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the device has multitouch data.\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        return bool(self._device.caps.has_mtdata)

    def has_slot(self):
        if False:
            return 10
        'Return True if the device has slot information.\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        return bool(self._device.caps.has_slot)

    def has_abs(self, index):
        if False:
            i = 10
            return i + 15
        'Return True if the device has abs data.\n\n        :Parameters:\n            `index`: int\n                One of const starting with a name ABS_MT_\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        if index < 0 or index >= MTDEV_ABS_SIZE:
            raise IndexError('Invalid index')
        return bool(self._device.caps.has_abs[index])

    def get_max_abs(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the maximum number of abs information available.\n        '
        return MTDEV_ABS_SIZE

    def get_slot(self):
        if False:
            while True:
                i = 10
        'Return the slot data.\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        if self._device.caps.has_slot == 0:
            return
        return self._device.caps.slot

    def get_abs(self, index):
        if False:
            while True:
                i = 10
        'Return the abs data.\n\n        :Parameters:\n            `index`: int\n                One of const starting with a name ABS_MT_\n        '
        if self._fd == -1:
            raise Exception('Device closed')
        if index < 0 or index >= MTDEV_ABS_SIZE:
            raise IndexError('Invalid index')
        return self._device.caps.abs[index]