import os
from typing import Optional
from PyQt6.QtCore import QObject, QUrl, pyqtSlot, pyqtProperty, pyqtSignal
from UM.Logger import Logger
from UM.PluginObject import PluginObject
from UM.PluginRegistry import PluginRegistry

class MachineAction(QObject, PluginObject):
    """Machine actions are actions that are added to a specific machine type.

    Examples of such actions are updating the firmware, connecting with remote devices or doing bed leveling. A
    machine action can also have a qml, which should contain a :py:class:`cura.MachineAction.MachineAction` item.
    When activated, the item will be displayed in a dialog and this object will be added as "manager" (so all
    pyqtSlot() functions can be called by calling manager.func())
    """

    def __init__(self, key: str, label: str='') -> None:
        if False:
            while True:
                i = 10
        'Create a new Machine action.\n\n        :param key: unique key of the machine action\n        :param label: Human readable label used to identify the machine action.\n        '
        super().__init__()
        self._key = key
        self._label = label
        self._qml_url = ''
        self._view = None
        self._finished = False
        self._open_as_dialog = True
        self._visible = True
    labelChanged = pyqtSignal()
    visibilityChanged = pyqtSignal()
    onFinished = pyqtSignal()

    def getKey(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._key

    def needsUserInteraction(self) -> bool:
        if False:
            return 10
        "Whether this action needs to ask the user anything.\n\n         If not, we shouldn't present the user with certain screens which otherwise show up.\n\n        :return: Defaults to true to be in line with the old behaviour.\n        "
        return True

    @pyqtProperty(str, notify=labelChanged)
    def label(self) -> str:
        if False:
            while True:
                i = 10
        return self._label

    def setLabel(self, label: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._label != label:
            self._label = label
            self.labelChanged.emit()

    @pyqtSlot()
    def reset(self) -> None:
        if False:
            return 10
        "Reset the action to it's default state.\n\n        This should not be re-implemented by child classes, instead re-implement _reset.\n\n        :py:meth:`cura.MachineAction.MachineAction._reset`\n        "
        self._finished = False
        self._reset()

    def _reset(self) -> None:
        if False:
            i = 10
            return i + 15
        'Protected implementation of reset.\n\n        See also :py:meth:`cura.MachineAction.MachineAction.reset`\n        '
        pass

    @pyqtSlot()
    def execute(self) -> None:
        if False:
            while True:
                i = 10
        self._execute()

    def _execute(self) -> None:
        if False:
            print('Hello World!')
        'Protected implementation of execute.'
        pass

    @pyqtSlot()
    def setFinished(self) -> None:
        if False:
            return 10
        self._finished = True
        self._reset()
        self.onFinished.emit()

    @pyqtProperty(bool, notify=onFinished)
    def finished(self) -> bool:
        if False:
            print('Hello World!')
        return self._finished

    def _createViewFromQML(self) -> Optional['QObject']:
        if False:
            while True:
                i = 10
        'Protected helper to create a view object based on provided QML.'
        plugin_path = PluginRegistry.getInstance().getPluginPath(self.getPluginId())
        if plugin_path is None:
            Logger.error(f'Cannot create QML view: cannot find plugin path for plugin {self.getPluginId()}')
            return None
        path = os.path.join(plugin_path, self._qml_url)
        from cura.CuraApplication import CuraApplication
        view = CuraApplication.getInstance().createQmlComponent(path, {'manager': self})
        return view

    @pyqtProperty(QUrl, constant=True)
    def qmlPath(self) -> 'QUrl':
        if False:
            return 10
        plugin_path = PluginRegistry.getInstance().getPluginPath(self.getPluginId())
        if plugin_path is None:
            Logger.error(f'Cannot create QML view: cannot find plugin path for plugin {self.getPluginId()}')
            return QUrl('')
        path = os.path.join(plugin_path, self._qml_url)
        return QUrl.fromLocalFile(path)

    @pyqtSlot(result=QObject)
    def getDisplayItem(self) -> Optional['QObject']:
        if False:
            i = 10
            return i + 15
        return self._createViewFromQML()

    @pyqtProperty(bool, constant=True)
    def shouldOpenAsDialog(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether this action will show a dialog.\n\n         If not, the action will directly run the function inside execute().\n\n        :return: Defaults to true to be in line with the old behaviour.\n        '
        return self._open_as_dialog

    @pyqtSlot()
    def setVisible(self, visible: bool) -> None:
        if False:
            while True:
                i = 10
        if self._visible != visible:
            self._visible = visible
            self.visibilityChanged.emit()

    @pyqtProperty(bool, notify=visibilityChanged)
    def visible(self) -> bool:
        if False:
            return 10
        'Whether this action button will be visible.\n\n         Example: Show only when isLoggedIn\n\n        :return: Defaults to true to be in line with the old behaviour.\n        '
        return self._visible