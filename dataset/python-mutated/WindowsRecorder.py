import pyWinhook
from pyWinhook import cpyHook, HookConstants
import win32api
from Event import ScreenWidth as SW, ScreenHeight as SH, flag_multiplemonitor
from loguru import logger
import Recorder.globals as globalv
import collections
from winreg import QueryValueEx, OpenKey, HKEY_CURRENT_USER, KEY_READ
swapmousebuttons = True if QueryValueEx(OpenKey(HKEY_CURRENT_USER, 'Control Panel\\Mouse', 0, KEY_READ), 'SwapMouseButtons')[0] == '1' else False
msgdic = {513: 'mouse left down', 514: 'mouse left up', 516: 'mouse right down', 517: 'mouse right up', 512: 'mouse move', 519: 'mouse middle down', 520: 'mouse middle up', 522: 'mouse wheel', 523: 'mouse x down', 524: 'mouse x up'}
datadic = {65536: 'x1', 131072: 'x2'}
MyMouseEvent = collections.namedtuple('MyMouseEvent', ['MessageName'])
record_signals = globalv.RecordSignal()

def on_mouse_event(event):
    if False:
        return 10
    message = event.MessageName
    if message == 'mouse wheel':
        message += ' up' if event.Wheel == 1 else ' down'
    elif message in globalv.swapmousemap and swapmousebuttons:
        message = globalv.swapmousemap[message]
    all_messages = ('mouse left down', 'mouse left up', 'mouse right down', 'mouse right up', 'mouse move', 'mouse middle down', 'mouse middle up', 'mouse wheel up', 'mouse wheel down', 'mouse x1 down', 'mouse x1 up', 'mouse x2 down', 'mouse x2 up')
    if message not in all_messages:
        return True
    pos = win32api.GetCursorPos()
    delay = globalv.current_ts() - globalv.latest_time
    mouse_move_interval_ms = globalv.mouse_interval_ms or 999999
    if message == 'mouse move' and delay < mouse_move_interval_ms:
        return True
    if globalv.latest_time < 0:
        delay = 0
    globalv.latest_time = globalv.current_ts()
    if not flag_multiplemonitor:
        (x, y) = pos
        pos = (x / SW, y / SH)
    sevent = globalv.ScriptEvent({'delay': delay, 'event_type': 'EM', 'message': message, 'action': pos, 'addon': None})
    record_signals.event_signal.emit(sevent)
    return True

def on_keyboard_event(event):
    if False:
        for i in range(10):
            print('nop')
    message = event.MessageName
    message = message.replace(' sys ', ' ')
    all_messages = ('key down', 'key up')
    if message not in all_messages:
        return True
    key_info = (event.KeyID, event.Key, event.Extended)
    delay = globalv.current_ts() - globalv.latest_time
    if globalv.latest_time < 0:
        delay = 0
    globalv.latest_time = globalv.current_ts()
    sevent = globalv.ScriptEvent({'delay': delay, 'event_type': 'EK', 'message': message, 'action': key_info, 'addon': None})
    record_signals.event_signal.emit(sevent)
    return True

def mouse_handler(msg, x, y, data, flags, time, hwnd, window_name):
    if False:
        while True:
            i = 10
    try:
        name = msgdic[msg]
        if name == 'mouse wheel':
            name = name + (' up' if data > 0 else ' down')
        elif name in ['mouse x down', 'mouse x up']:
            name = name.replace('x', datadic[data])
        on_mouse_event(MyMouseEvent(name))
    except KeyError as e:
        logger.debug('Unknown mouse event, keyid {0}'.format(e))
    finally:
        return True

def setuphook(commandline=False):
    if False:
        i = 10
        return i + 15
    hm = pyWinhook.HookManager()
    if not commandline:
        cpyHook.cSetHook(HookConstants.WH_MOUSE_LL, mouse_handler)
    hm.KeyAll = on_keyboard_event
    hm.HookKeyboard()