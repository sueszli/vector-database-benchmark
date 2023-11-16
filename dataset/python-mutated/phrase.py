import enum
import json
import os
import typing
from autokey.model.key import NAVIGATION_KEYS, Key, KEY_SPLIT_RE
from autokey.model.helpers import JSON_FILE_PATTERN, get_safe_path, TriggerMode
from autokey.model.abstract_abbreviation import AbstractAbbreviation
from autokey.model.abstract_window_filter import AbstractWindowFilter
from autokey.model.abstract_hotkey import AbstractHotkey
logger = __import__('autokey.logger').logger.get_logger(__name__)

class Phrase(AbstractAbbreviation, AbstractHotkey, AbstractWindowFilter):
    """
    Encapsulates all data and behaviour for a phrase.
    """

    def __init__(self, description, phrase, path=None):
        if False:
            for i in range(10):
                print('nop')
        AbstractAbbreviation.__init__(self)
        AbstractHotkey.__init__(self)
        AbstractWindowFilter.__init__(self)
        self.description = description
        self.phrase = phrase
        self.modes = []
        self.usageCount = 0
        self.prompt = False
        self.temporary = False
        self.omitTrigger = False
        self.matchCase = False
        self.parent = None
        self.show_in_tray_menu = False
        self.sendMode = SendMode.CB_CTRL_V
        self.path = path

    def build_path(self, base_name=None):
        if False:
            print('Hello World!')
        if base_name is None:
            base_name = self.description
        else:
            base_name = base_name[:-4]
        self.path = get_safe_path(self.parent.path, base_name, '.txt')

    def get_json_path(self):
        if False:
            i = 10
            return i + 15
        (directory, base_name) = os.path.split(self.path[:-4])
        return JSON_FILE_PATTERN.format(directory, base_name)

    def persist(self):
        if False:
            print('Hello World!')
        if self.path is None:
            self.build_path()
        with open(self.get_json_path(), 'w') as json_file:
            json.dump(self.get_serializable(), json_file, indent=4)
        with open(self.path, 'w') as out_file:
            out_file.write(self.phrase)

    def get_serializable(self):
        if False:
            return 10
        d = {'type': 'phrase', 'description': self.description, 'modes': [mode.value for mode in self.modes], 'usageCount': self.usageCount, 'prompt': self.prompt, 'omitTrigger': self.omitTrigger, 'matchCase': self.matchCase, 'showInTrayMenu': self.show_in_tray_menu, 'abbreviation': AbstractAbbreviation.get_serializable(self), 'hotkey': AbstractHotkey.get_serializable(self), 'filter': AbstractWindowFilter.get_serializable(self), 'sendMode': self.sendMode.value}
        return d

    def load(self, parent):
        if False:
            print('Hello World!')
        self.parent = parent
        with open(self.path, 'r') as inFile:
            self.phrase = inFile.read()
        if os.path.exists(self.get_json_path()):
            self.load_from_serialized()
        else:
            self.description = os.path.basename(self.path)[:-4]

    def load_from_serialized(self):
        if False:
            return 10
        try:
            with open(self.get_json_path(), 'r') as json_file:
                data = json.load(json_file)
                self.inject_json_data(data)
        except Exception:
            logger.exception('Error while loading json data for ' + self.description)
            logger.error('JSON data not loaded (or loaded incomplete)')

    def inject_json_data(self, data: dict):
        if False:
            while True:
                i = 10
        self.description = data['description']
        self.modes = [TriggerMode(item) for item in data['modes']]
        self.usageCount = data['usageCount']
        self.prompt = data['prompt']
        self.omitTrigger = data['omitTrigger']
        self.matchCase = data['matchCase']
        self.show_in_tray_menu = data['showInTrayMenu']
        self.sendMode = SendMode(data.get('sendMode', SendMode.KEYBOARD))
        AbstractAbbreviation.load_from_serialized(self, data['abbreviation'])
        AbstractHotkey.load_from_serialized(self, data['hotkey'])
        AbstractWindowFilter.load_from_serialized(self, data['filter'])

    def rebuild_path(self):
        if False:
            print('Hello World!')
        if self.path is not None:
            old_name = self.path
            old_json = self.get_json_path()
            self.build_path()
            os.rename(old_name, self.path)
            os.rename(old_json, self.get_json_path())
        else:
            self.build_path()

    def remove_data(self):
        if False:
            for i in range(10):
                print('nop')
        if self.path is not None:
            if os.path.exists(self.path):
                os.remove(self.path)
            if os.path.exists(self.get_json_path()):
                os.remove(self.get_json_path())

    def copy(self, source_phrase):
        if False:
            for i in range(10):
                print('nop')
        self.description = source_phrase.description
        self.phrase = source_phrase.phrase
        self.prompt = source_phrase.prompt
        self.omitTrigger = source_phrase.omitTrigger
        self.matchCase = source_phrase.matchCase
        self.parent = source_phrase.parent
        self.show_in_tray_menu = source_phrase.show_in_tray_menu
        self.copy_abbreviation(source_phrase)
        self.copy_hotkey(source_phrase)
        self.copy_window_filter(source_phrase)

    def get_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        return ('text-plain', self.description, self.get_abbreviations(), self.get_hotkey_string(), self)

    def set_modes(self, modes: typing.List[TriggerMode]):
        if False:
            return 10
        self.modes = modes

    def check_input(self, buffer, window_info):
        if False:
            while True:
                i = 10
        if TriggerMode.ABBREVIATION in self.modes:
            return self._should_trigger_abbreviation(buffer) and self._should_trigger_window_title(window_info)
        else:
            return False

    def build_phrase(self, buffer):
        if False:
            i = 10
            return i + 15
        self.usageCount += 1
        self.parent.increment_usage_count()
        expansion = Expansion(self.phrase)
        trigger_found = False
        if TriggerMode.ABBREVIATION in self.modes:
            if self._should_trigger_abbreviation(buffer):
                abbr = self._get_trigger_abbreviation(buffer)
                (stringBefore, typedAbbr, stringAfter) = self._partition_input(buffer, abbr)
                trigger_found = True
                if self.backspace:
                    expansion.backspaces = len(abbr) + len(stringAfter)
                else:
                    expansion.backspaces = len(stringAfter)
                if not self.omitTrigger:
                    expansion.string += stringAfter
                if self.matchCase:
                    if typedAbbr.istitle():
                        expansion.string = expansion.string.capitalize()
                    elif typedAbbr.isupper():
                        expansion.string = expansion.string.upper()
                    elif typedAbbr.islower():
                        expansion.string = expansion.string.lower()
        if not trigger_found:
            expansion.backspaces = self.parent.get_backspace_count(buffer)
        return expansion

    def calculate_input(self, buffer):
        if False:
            return 10
        '\n        Calculate how many keystrokes were used in triggering this phrase.\n        '
        if TriggerMode.ABBREVIATION in self.modes:
            if self._should_trigger_abbreviation(buffer):
                if self.immediate:
                    return len(self._get_trigger_abbreviation(buffer))
                else:
                    return len(self._get_trigger_abbreviation(buffer)) + 1
        if TriggerMode.HOTKEY in self.modes:
            if buffer == '':
                return len(self.modifiers) + 1
        return self.parent.calculate_input(buffer)

    def get_trigger_chars(self, buffer):
        if False:
            while True:
                i = 10
        abbr = self._get_trigger_abbreviation(buffer)
        (stringBefore, typedAbbr, stringAfter) = self._partition_input(buffer, abbr)
        return typedAbbr + stringAfter

    def should_prompt(self, buffer):
        if False:
            return 10
        '\n        Get a value indicating whether the user should be prompted to select the phrase.\n        Always returns true if the phrase has been triggered using predictive mode.\n        '
        return self.prompt

    def get_description(self, buffer):
        if False:
            print('Hello World!')
        return self.description
    'def _should_trigger_predictive(self, buffer):\n        if len(buffer) >= ConfigManager.SETTINGS[PREDICTIVE_LENGTH]:\n            typed = buffer[-ConfigManager.SETTINGS[PREDICTIVE_LENGTH]:]\n            return self.phrase.startswith(typed)\n        else:\n            return False'

    def parsePositionTokens(self, expansion):
        if False:
            return 10
        CURSOR_POSITION_TOKEN = '|'
        if CURSOR_POSITION_TOKEN in expansion.string:
            (firstpart, secondpart) = expansion.string.split(CURSOR_POSITION_TOKEN)
            foundNavigationKey = False
            for key in NAVIGATION_KEYS:
                if key in expansion.string:
                    expansion.lefts = 0
                    foundNavigationKey = True
                    break
            if not foundNavigationKey:
                for section in KEY_SPLIT_RE.split(secondpart):
                    if not Key.is_key(section) or section in [' ', '\n']:
                        expansion.lefts += len(section)
            expansion.string = firstpart + secondpart

    def __str__(self):
        if False:
            print('Hello World!')
        return "phrase '{}'".format(self.description)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "Phrase('" + self.description + "')"

class Expansion:

    def __init__(self, string):
        if False:
            i = 10
            return i + 15
        self.string = string
        self.lefts = 0
        self.backspaces = 0

class SendMode(enum.Enum):
    """
    Enumeration class for phrase send modes

    KEYBOARD: Send using key events
    CB_CTRL_V: Send via clipboard and paste with Ctrl+v
    CB_CTRL_SHIFT_V: Send via clipboard and paste with Ctrl+Shift+v
    SELECTION: Send via X selection and paste with middle mouse button
    """
    KEYBOARD = 'kb'
    CB_CTRL_V = Key.CONTROL + '+v'
    CB_CTRL_SHIFT_V = Key.CONTROL + '+' + Key.SHIFT + '+v'
    CB_SHIFT_INSERT = Key.SHIFT + '+' + Key.INSERT
    SELECTION = None
SEND_MODES = {'Keyboard': SendMode.KEYBOARD, 'Clipboard (Ctrl+V)': SendMode.CB_CTRL_V, 'Clipboard (Ctrl+Shift+V)': SendMode.CB_CTRL_SHIFT_V, 'Clipboard (Shift+Insert)': SendMode.CB_SHIFT_INSERT, 'Mouse Selection': SendMode.SELECTION}