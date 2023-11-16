import os
from typing import Optional
from PyQt6.QtCore import QObject
from UM.Qt.QtApplication import QtApplication
from UM.Signal import Signal
from .SubscribedPackagesModel import SubscribedPackagesModel

class DiscrepanciesPresenter(QObject):
    """Shows a list of packages to be added or removed. The user can select which packages to (un)install. The user's

    choices are emitted on the `packageMutations` Signal.
    """

    def __init__(self, app: QtApplication) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.packageMutations = Signal()
        self._app = app
        self._package_manager = app.getPackageManager()
        self._dialog: Optional[QObject] = None
        self._compatibility_dialog_path = 'resources/qml/CompatibilityDialog.qml'

    def present(self, plugin_path: str, model: SubscribedPackagesModel) -> None:
        if False:
            print('Hello World!')
        path = os.path.join(plugin_path, self._compatibility_dialog_path)
        self._dialog = self._app.createQmlComponent(path, {'subscribedPackagesModel': model, 'handler': self})
        assert self._dialog
        self._dialog.accepted.connect(lambda : self._onConfirmClicked(model))

    def _onConfirmClicked(self, model: SubscribedPackagesModel) -> None:
        if False:
            for i in range(10):
                print('nop')
        if model.getIncompatiblePackages():
            self._package_manager.dismissAllIncompatiblePackages(model.getIncompatiblePackages())
        if model.getCompatiblePackages():
            model.setItems(model.getCompatiblePackages())
            self.packageMutations.emit(model)