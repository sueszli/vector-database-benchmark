import typing
from autokey.model.helpers import TriggerMode
from autokey.model.abstract_window_filter import AbstractWindowFilter
from autokey.model.key import Key

class AbstractHotkey(AbstractWindowFilter):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.modifiers = []
        self.hotKey = None

    def get_serializable(self):
        if False:
            return 10
        d = {'modifiers': self.modifiers, 'hotKey': self.hotKey}
        return d

    def load_from_serialized(self, data):
        if False:
            while True:
                i = 10
        self.set_hotkey(data['modifiers'], data['hotKey'])

    def copy_hotkey(self, theHotkey):
        if False:
            for i in range(10):
                print('nop')
        [self.modifiers.append(modifier) for modifier in theHotkey.modifiers]
        self.hotKey = theHotkey.hotKey

    def set_hotkey(self, modifiers, key):
        if False:
            while True:
                i = 10
        modifiers.sort()
        self.modifiers = modifiers
        self.hotKey = key
        if key is not None and TriggerMode.HOTKEY not in self.modes:
            self.modes.append(TriggerMode.HOTKEY)

    def unset_hotkey(self):
        if False:
            i = 10
            return i + 15
        self.modifiers = []
        self.hotKey = None
        if TriggerMode.HOTKEY in self.modes:
            self.modes.remove(TriggerMode.HOTKEY)

    def check_hotkey(self, modifiers, key, windowTitle):
        if False:
            while True:
                i = 10
        if self.hotKey is not None and self._should_trigger_window_title(windowTitle):
            return self.modifiers == modifiers and self.hotKey == key
        else:
            return False

    def get_hotkey_string(self, key=None, modifiers=None):
        if False:
            for i in range(10):
                print('nop')
        if key is None and modifiers is None:
            if TriggerMode.HOTKEY not in self.modes:
                return ''
            key = self.hotKey
            modifiers = self.modifiers
        ret = ''
        for modifier in modifiers:
            ret += modifier
            ret += '+'
        if key == ' ':
            ret += '<space>'
        else:
            ret += key
        return ret