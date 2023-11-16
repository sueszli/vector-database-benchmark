import typing
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialogButtonBox
import autokey.model.folder
import autokey.model.helpers
import autokey.model.phrase
import autokey.model.script
from autokey.qtui import common as qtui_common
from autokey import UI_common_functions as UI_common
from autokey import iomediator
import autokey.configmanager.configmanager as cm
from autokey.model.key import Key
logger = __import__('autokey.logger').logger.get_logger(__name__)
Item = typing.Union[autokey.model.folder.Folder, autokey.model.script.Script, autokey.model.phrase.Phrase]

class HotkeySettingsDialog(*qtui_common.inherits_from_ui_file_with_name('hotkeysettings')):
    KEY_MAP = {' ': '<space>'}
    REVERSE_KEY_MAP = {value: key for (key, value) in KEY_MAP.items()}
    DEFAULT_RECORDED_KEY_LABEL_CONTENT = '(None)'
    '\n    This signal is emitted whenever the key is assigned/deleted. This happens when the user records a key or cancels\n    a key recording.\n    '
    key_assigned = pyqtSignal(bool, name='key_assigned')
    recording_finished = pyqtSignal(bool, name='recording_finished')

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super(HotkeySettingsDialog, self).__init__(parent)
        self.setupUi(self)
        self.key_assigned.connect(self.buttonBox.button(QDialogButtonBox.Ok).setEnabled)
        self.recording_finished.connect(self.record_combination_button.setEnabled)
        self.key = ''
        self._update_key(None)
        self.target_item = None
        self.grabber = None
        self.MODIFIER_BUTTONS = {self.mod_control_button: Key.CONTROL, self.mod_alt_button: Key.ALT, self.mod_shift_button: Key.SHIFT, self.mod_super_button: Key.SUPER, self.mod_hyper_button: Key.HYPER, self.mod_meta_button: Key.META}

    def _update_key(self, key):
        if False:
            i = 10
            return i + 15
        self.key = key
        if key is None:
            self.recorded_key_label.setText('Key: {}'.format(self.DEFAULT_RECORDED_KEY_LABEL_CONTENT))
            self.key_assigned.emit(False)
        else:
            self.recorded_key_label.setText('Key: {}'.format(key))
            self.key_assigned.emit(True)

    def on_record_combination_button_pressed(self):
        if False:
            while True:
                i = 10
        '\n        Start recording a key combination when the user clicks on the record_combination_button.\n        The button itself is automatically disabled during the recording process.\n        '
        self.recorded_key_label.setText('Press a key or combination...')
        logger.debug('User starts to record a key combination.')
        self.grabber = iomediator.keygrabber.KeyGrabber(self)
        self.grabber.start()

    def load(self, item: Item):
        if False:
            return 10
        self.target_item = item
        UI_common.load_hotkey_settings_dialog(self, item)

    def populate_hotkey_details(self, item):
        if False:
            i = 10
            return i + 15
        self.activate_modifier_buttons(item.modifiers)
        key = item.hotKey
        key_text = UI_common.get_hotkey_text(self, key)
        self._update_key(key_text)
        logger.debug('Loaded item {}, key: {}, modifiers: {}'.format(item, key_text, item.modifiers))

    def activate_modifier_buttons(self, modifiers):
        if False:
            print('Hello World!')
        for (button, key) in self.MODIFIER_BUTTONS.items():
            button.setChecked(key in modifiers)

    def save(self, item):
        if False:
            while True:
                i = 10
        UI_common.save_hotkey_settings_dialog(self, item)

    def reset(self):
        if False:
            return 10
        for button in self.MODIFIER_BUTTONS:
            button.setChecked(False)
        self._update_key(None)

    def set_key(self, key, modifiers: typing.List[Key]=None):
        if False:
            i = 10
            return i + 15
        'This is called when the user successfully finishes recording a key combination.'
        if modifiers is None:
            modifiers = []
        if key in self.KEY_MAP:
            key = self.KEY_MAP[key]
        self._update_key(key)
        self.activate_modifier_buttons(modifiers)
        self.recording_finished.emit(True)

    def cancel_grab(self):
        if False:
            return 10
        '\n        This is called when the user cancels a recording.\n        Canceling is done by clicking with the left mouse button.\n        '
        logger.debug('User canceled hotkey recording.')
        self.recording_finished.emit(True)

    def build_modifiers(self):
        if False:
            print('Hello World!')
        modifiers = []
        for (button, key) in self.MODIFIER_BUTTONS.items():
            if button.isChecked():
                modifiers.append(key)
        modifiers.sort()
        return modifiers

    def reject(self):
        if False:
            for i in range(10):
                print('nop')
        self.load(self.target_item)
        super().reject()

class GlobalHotkeyDialog(HotkeySettingsDialog):

    def load(self, item: cm.GlobalHotkey):
        if False:
            for i in range(10):
                print('nop')
        self.target_item = item
        UI_common.load_global_hotkey_dialog(self, item)

    def save(self, item: cm.GlobalHotkey):
        if False:
            i = 10
            return i + 15
        UI_common.save_hotkey_settings_dialog(self, item)