import struct
from subprocess import check_output
import re
from ._nixcommon import EV_KEY, EV_REL, EV_MSC, EV_SYN, EV_ABS, aggregate_devices
from ._mouse_event import ButtonEvent, WheelEvent, MoveEvent, LEFT, RIGHT, MIDDLE, X, X2, UP, DOWN
import ctypes
import ctypes.util
from ctypes import c_uint32, c_uint, c_int, byref
display = None
window = None
x11 = None

def build_display():
    if False:
        for i in range(10):
            print('nop')
    global display, window, x11
    if display and window and x11:
        return
    x11 = ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11'))
    x11.XInitThreads()
    display = x11.XOpenDisplay(None)
    window = x11.XDefaultRootWindow(display)

def get_position():
    if False:
        print('Hello World!')
    build_display()
    (root_id, child_id) = (c_uint32(), c_uint32())
    (root_x, root_y, win_x, win_y) = (c_int(), c_int(), c_int(), c_int())
    mask = c_uint()
    ret = x11.XQueryPointer(display, c_uint32(window), byref(root_id), byref(child_id), byref(root_x), byref(root_y), byref(win_x), byref(win_y), byref(mask))
    return (root_x.value, root_y.value)

def move_to(x, y):
    if False:
        for i in range(10):
            print('nop')
    build_display()
    x11.XWarpPointer(display, None, window, 0, 0, 0, 0, x, y)
    x11.XFlush(display)
REL_X = 0
REL_Y = 1
REL_Z = 2
REL_HWHEEL = 6
REL_WHEEL = 8
ABS_X = 0
ABS_Y = 1
BTN_MOUSE = 272
BTN_LEFT = 272
BTN_RIGHT = 273
BTN_MIDDLE = 274
BTN_SIDE = 275
BTN_EXTRA = 276
button_by_code = {BTN_LEFT: LEFT, BTN_RIGHT: RIGHT, BTN_MIDDLE: MIDDLE, BTN_SIDE: X, BTN_EXTRA: X2}
code_by_button = {button: code for (code, button) in button_by_code.items()}
device = None

def build_device():
    if False:
        while True:
            i = 10
    global device
    if device:
        return
    device = aggregate_devices('mouse')
init = build_device

def listen(queue):
    if False:
        i = 10
        return i + 15
    build_device()
    while True:
        (time, type, code, value, device_id) = device.read_event()
        if type == EV_SYN or type == EV_MSC:
            continue
        event = None
        arg = None
        if type == EV_KEY:
            event = ButtonEvent(DOWN if value else UP, button_by_code.get(code, '?'), time)
        elif type == EV_REL:
            (value,) = struct.unpack('i', struct.pack('I', value))
            if code == REL_WHEEL:
                event = WheelEvent(value, time)
            elif code in (REL_X, REL_Y):
                (x, y) = get_position()
                event = MoveEvent(x, y, time)
        if event is None:
            continue
        queue.put(event)

def press(button=LEFT):
    if False:
        print('Hello World!')
    build_device()
    device.write_event(EV_KEY, code_by_button[button], 1)

def release(button=LEFT):
    if False:
        i = 10
        return i + 15
    build_device()
    device.write_event(EV_KEY, code_by_button[button], 0)

def move_relative(x, y):
    if False:
        return 10
    build_device()
    if x < 0:
        x += 2 ** 32
    if y < 0:
        y += 2 ** 32
    device.write_event(EV_REL, REL_X, x)
    device.write_event(EV_REL, REL_Y, y)

def wheel(delta=1):
    if False:
        return 10
    build_device()
    if delta < 0:
        delta += 2 ** 32
    device.write_event(EV_REL, REL_WHEEL, delta)
if __name__ == '__main__':
    move_to(100, 200)