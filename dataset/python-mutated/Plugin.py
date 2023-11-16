import logging
import os
from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSettings
from PyQt5.QtWidgets import QUndoCommand, QUndoStack
from urh.util.Logger import logger

class Plugin(QObject):
    enabled_changed = pyqtSignal()

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.__enabled = Qt.Unchecked
        self.name = name
        self.plugin_path = ''
        self.description = ''
        self.__settings_frame = None
        self.qsettings = QSettings(QSettings.IniFormat, QSettings.UserScope, 'urh', self.name + '-plugin')

    @property
    def settings_frame(self):
        if False:
            while True:
                i = 10
        if self.__settings_frame is None:
            logging.getLogger().setLevel(logging.WARNING)
            self.__settings_frame = uic.loadUi(os.path.join(self.plugin_path, 'settings.ui'))
            logging.getLogger().setLevel(logger.level)
            self.create_connects()
        return self.__settings_frame

    @property
    def enabled(self) -> bool:
        if False:
            return 10
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool):
        if False:
            i = 10
            return i + 15
        if value != self.__enabled:
            self.__enabled = Qt.Checked if value else Qt.Unchecked
            self.enabled_changed.emit()

    def load_description(self):
        if False:
            return 10
        descr_file = os.path.join(self.plugin_path, 'descr.txt')
        try:
            with open(descr_file, 'r') as f:
                self.description = f.read()
        except Exception as e:
            print(e)

    def destroy_settings_frame(self):
        if False:
            return 10
        self.__settings_frame = None

    def create_connects(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ProtocolPlugin(Plugin):

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        Plugin.__init__(self, name)

    def get_action(self, parent, undo_stack: QUndoStack, sel_range, groups, view: int) -> QUndoCommand:
        if False:
            while True:
                i = 10
        '\n        :type parent: QTableView\n        :type undo_stack: QUndoStack\n        :type groups: list of ProtocolGroups\n        '
        raise NotImplementedError('Abstract Method.')

class SDRPlugin(Plugin):

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        Plugin.__init__(self, name)

class SignalEditorPlugin(Plugin):

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        Plugin.__init__(self, name)