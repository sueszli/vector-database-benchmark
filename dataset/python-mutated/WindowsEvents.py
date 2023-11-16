import re
import pyperclip
import win32api
from Event.Event import Event
from loguru import logger
import ctypes
import win32con
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
numofmonitors = user32.GetSystemMetrics(win32con.SM_CMONITORS)
(SW, SH) = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

class WindowsEvent(Event):

    def changepos(self, pos: tuple):
        if False:
            i = 10
            return i + 15
        if self.event_type == 'EM':
            (x, y) = pos
            if isinstance(x, int):
                self.action[0] = int(x * 65535 / SW)
            else:
                self.action[0] = int(x * 65535)
            if isinstance(y, int):
                self.action[1] = int(y * 65535 / SH)
            else:
                self.action[1] = int(y * 65535)

    def execute(self, thd=None):
        if False:
            i = 10
            return i + 15
        self.sleep(thd)
        if self.event_type == 'EM':
            (x, y) = self.action
            if not isinstance(x, int) and (not isinstance(y, int)):
                x = float(re.match('([0-1].[0-9]+)%', x).group(1))
                y = float(re.match('([0-1].[0-9]+)%', y).group(1))
            if self.action == [-1, -1]:
                pass
            elif isinstance(x, int) and isinstance(y, int):
                if numofmonitors > 1:
                    win32api.SetCursorPos([x, y])
                else:
                    nx = int(x * 65535 / SW)
                    ny = int(y * 65535 / SH)
                    win32api.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE | win32con.MOUSEEVENTF_MOVE, nx, ny, 0, 0)
            else:
                nx = int(x * 65535)
                ny = int(y * 65535)
                win32api.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE | win32con.MOUSEEVENTF_MOVE, nx, ny, 0, 0)
            if self.message == 'mouse left down':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            elif self.message == 'mouse left up':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif self.message == 'mouse right down':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            elif self.message == 'mouse right up':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            elif self.message == 'mouse middle down':
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
            elif self.message == 'mouse middle up':
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            elif self.message == 'mouse wheel up':
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, win32con.WHEEL_DELTA, 0)
            elif self.message == 'mouse wheel down':
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -win32con.WHEEL_DELTA, 0)
            elif self.message == 'mouse move':
                pass
            else:
                logger.warning('Unknown mouse event:%s' % self.message)
        elif self.event_type == 'EK':
            (key_code, key_name, extended) = self.action
            base = 0
            if extended:
                base = win32con.KEYEVENTF_EXTENDEDKEY
            if self.message == 'key down':
                win32api.keybd_event(key_code, 0, base, 0)
            elif self.message == 'key up':
                win32api.keybd_event(key_code, 0, base | win32con.KEYEVENTF_KEYUP, 0)
            else:
                logger.warning('Unknown keyboard event:', self.message)
        elif self.event_type == 'EX':
            if self.message == 'input':
                text = self.action
                pyperclip.copy(text)
                win32api.keybd_event(162, 0, 0, 0)
                win32api.keybd_event(86, 0, 0, 0)
                win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(162, 0, win32con.KEYEVENTF_KEYUP, 0)
            else:
                logger.warning('Unknown extra event:%s' % self.message)