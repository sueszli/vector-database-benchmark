import datetime
import time
from .iomediator import IoMediator
from autokey.model.key import Key, MODIFIERS
from . import iomediator

class KeyGrabber:
    """
    Keygrabber used by the hotkey settings dialog to grab the key pressed
    """

    def __init__(self, parent):
        if False:
            print('Hello World!')
        self.target_parent = parent

    def start(self):
        if False:
            return 10
        time.sleep(0.1)
        IoMediator.listeners.append(self)
        iomediator.CURRENT_INTERFACE.grab_keyboard()

    def handle_keypress(self, raw_key, modifiers, key, *args):
        if False:
            while True:
                i = 10
        if raw_key not in MODIFIERS:
            IoMediator.listeners.remove(self)
            self.target_parent.set_key(raw_key, modifiers)
            iomediator.CURRENT_INTERFACE.ungrab_keyboard()

    def handle_mouseclick(self, root_x, root_y, rel_x, rel_y, button, window_info):
        if False:
            for i in range(10):
                print('nop')
        IoMediator.listeners.remove(self)
        iomediator.CURRENT_INTERFACE.ungrab_keyboard()
        self.target_parent.cancel_grab()

class Recorder(KeyGrabber):
    """
    Recorder used by the record macro functionality
    """

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        KeyGrabber.__init__(self, parent)
        self.insideKeys = False
        self.start_time = 0.0
        self.delay = 0.0
        self.delay_finished = False
        self.record_keyboard = self.record_mouse = False

    def start(self, delay: float):
        if False:
            print('Hello World!')
        time.sleep(0.1)
        IoMediator.listeners.append(self)
        self.target_parent.start_record()
        self.start_time = time.time()
        self.delay = delay
        self.delay_finished = False

    def start_withgrab(self):
        if False:
            print('Hello World!')
        time.sleep(0.1)
        IoMediator.listeners.append(self)
        self.target_parent.start_record()
        self.start_time = time.time()
        self.delay = 0
        self.delay_finished = True
        iomediator.CURRENT_INTERFACE.grab_keyboard()

    def stop(self):
        if False:
            while True:
                i = 10
        if self in IoMediator.listeners:
            IoMediator.listeners.remove(self)
            if self.insideKeys:
                self.target_parent.end_key_sequence()
            self.insideKeys = False

    def stop_withgrab(self):
        if False:
            for i in range(10):
                print('nop')
        iomediator.CURRENT_INTERFACE.ungrab_keyboard()
        if self in IoMediator.listeners:
            IoMediator.listeners.remove(self)
            if self.insideKeys:
                self.target_parent.end_key_sequence()
            self.insideKeys = False

    def set_record_keyboard(self, record: bool):
        if False:
            i = 10
            return i + 15
        self.record_keyboard = record

    def set_record_mouse(self, record: bool):
        if False:
            i = 10
            return i + 15
        self.record_mouse = record

    def _delay_passed(self) -> bool:
        if False:
            i = 10
            return i + 15
        if not self.delay_finished:
            now = time.time()
            delta = datetime.datetime.utcfromtimestamp(now - self.start_time)
            self.delay_finished = delta.second > self.delay
        return self.delay_finished

    def handle_keypress(self, raw_key, modifiers, key, *args):
        if False:
            while True:
                i = 10
        if self.record_keyboard and self._delay_passed():
            if not self.insideKeys:
                self.insideKeys = True
                self.target_parent.start_key_sequence()
            modifier_count = len(modifiers)
            if modifier_count > 1 or (modifier_count == 1 and Key.SHIFT not in modifiers) or (Key.SHIFT in modifiers and len(raw_key) > 1):
                self.target_parent.append_hotkey(raw_key, modifiers)
            elif key not in MODIFIERS:
                self.target_parent.append_key(key)

    def handle_mouseclick(self, root_x, root_y, rel_x, rel_y, button, window_info):
        if False:
            for i in range(10):
                print('nop')
        if self.record_mouse and self._delay_passed():
            if self.insideKeys:
                self.insideKeys = False
                self.target_parent.end_key_sequence()
            self.target_parent.append_mouseclick(rel_x, rel_y, button, window_info[0])