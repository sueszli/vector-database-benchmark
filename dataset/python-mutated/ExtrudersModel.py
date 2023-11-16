from PyQt6.QtCore import Qt, pyqtSignal, pyqtProperty, QTimer
from typing import Iterable, TYPE_CHECKING
from UM.i18n import i18nCatalog
from UM.Qt.ListModel import ListModel
from UM.Application import Application
import UM.FlameProfiler
if TYPE_CHECKING:
    from cura.Settings.ExtruderStack import ExtruderStack
catalog = i18nCatalog('cura')

class ExtrudersModel(ListModel):
    """Model that holds extruders.

    This model is designed for use by any list of extruders, but specifically intended for drop-down lists of the
    current machine's extruders in place of settings.
    """
    IdRole = Qt.ItemDataRole.UserRole + 1
    NameRole = Qt.ItemDataRole.UserRole + 2
    'Human-readable name of the extruder.'
    ColorRole = Qt.ItemDataRole.UserRole + 3
    'Colour of the material loaded in the extruder.'
    IndexRole = Qt.ItemDataRole.UserRole + 4
    'Index of the extruder, which is also the value of the setting itself.\n\n    An index of 0 indicates the first extruder, an index of 1 the second one, and so on. This is the value that will \n    be saved in instance containers. '
    DefinitionRole = Qt.ItemDataRole.UserRole + 5
    MaterialRole = Qt.ItemDataRole.UserRole + 6
    VariantRole = Qt.ItemDataRole.UserRole + 7
    StackRole = Qt.ItemDataRole.UserRole + 8
    MaterialBrandRole = Qt.ItemDataRole.UserRole + 9
    ColorNameRole = Qt.ItemDataRole.UserRole + 10
    EnabledRole = Qt.ItemDataRole.UserRole + 11
    'Is the extruder enabled?'
    MaterialTypeRole = Qt.ItemDataRole.UserRole + 12
    'The type of the material (e.g. PLA, ABS, PETG, etc.).'
    defaultColors = ['#ffc924', '#86ec21', '#22eeee', '#245bff', '#9124ff', '#ff24c8']
    'List of colours to display if there is no material or the material has no known colour. '
    MaterialNameRole = Qt.ItemDataRole.UserRole + 13

    def __init__(self, parent=None):
        if False:
            return 10
        'Initialises the extruders model, defining the roles and listening for changes in the data.\n\n        :param parent: Parent QtObject of this list.\n        '
        super().__init__(parent)
        self.addRoleName(self.IdRole, 'id')
        self.addRoleName(self.NameRole, 'name')
        self.addRoleName(self.EnabledRole, 'enabled')
        self.addRoleName(self.ColorRole, 'color')
        self.addRoleName(self.IndexRole, 'index')
        self.addRoleName(self.DefinitionRole, 'definition')
        self.addRoleName(self.MaterialRole, 'material')
        self.addRoleName(self.VariantRole, 'variant')
        self.addRoleName(self.StackRole, 'stack')
        self.addRoleName(self.MaterialBrandRole, 'material_brand')
        self.addRoleName(self.ColorNameRole, 'color_name')
        self.addRoleName(self.MaterialTypeRole, 'material_type')
        self.addRoleName(self.MaterialNameRole, 'material_name')
        self._update_extruder_timer = QTimer()
        self._update_extruder_timer.setInterval(100)
        self._update_extruder_timer.setSingleShot(True)
        self._update_extruder_timer.timeout.connect(self.__updateExtruders)
        self._active_machine_extruders = []
        self._add_optional_extruder = False
        Application.getInstance().globalContainerStackChanged.connect(self._extrudersChanged)
        Application.getInstance().getExtruderManager().extrudersChanged.connect(self._extrudersChanged)
        Application.getInstance().getContainerRegistry().containerMetaDataChanged.connect(self._onExtruderStackContainersChanged)
        self._extrudersChanged()
    addOptionalExtruderChanged = pyqtSignal()

    def setAddOptionalExtruder(self, add_optional_extruder):
        if False:
            for i in range(10):
                print('nop')
        if add_optional_extruder != self._add_optional_extruder:
            self._add_optional_extruder = add_optional_extruder
            self.addOptionalExtruderChanged.emit()
            self._updateExtruders()

    @pyqtProperty(bool, fset=setAddOptionalExtruder, notify=addOptionalExtruderChanged)
    def addOptionalExtruder(self):
        if False:
            for i in range(10):
                print('nop')
        return self._add_optional_extruder

    def _extrudersChanged(self, machine_id=None):
        if False:
            while True:
                i = 10
        "Links to the stack-changed signal of the new extruders when an extruder is swapped out or added in the\n         current machine.\n\n        :param machine_id: The machine for which the extruders changed. This is filled by the\n        ExtruderManager.extrudersChanged signal when coming from that signal. Application.globalContainerStackChanged\n        doesn't fill this signal; it's assumed to be the current printer in that case.\n        "
        machine_manager = Application.getInstance().getMachineManager()
        if machine_id is not None:
            if machine_manager.activeMachine is None:
                return
            if machine_id != machine_manager.activeMachine.getId():
                return
        for extruder in self._active_machine_extruders:
            extruder.containersChanged.disconnect(self._onExtruderStackContainersChanged)
            extruder.enabledChanged.disconnect(self._updateExtruders)
        self._active_machine_extruders = []
        extruder_manager = Application.getInstance().getExtruderManager()
        for extruder in extruder_manager.getActiveExtruderStacks():
            if extruder is None:
                continue
            extruder.containersChanged.connect(self._onExtruderStackContainersChanged)
            extruder.enabledChanged.connect(self._updateExtruders)
            self._active_machine_extruders.append(extruder)
        self._updateExtruders()

    def _onExtruderStackContainersChanged(self, container):
        if False:
            for i in range(10):
                print('nop')
        if container.getMetaDataEntry('type') in ['material', 'variant', None]:
            self._updateExtruders()
    modelChanged = pyqtSignal()

    def _updateExtruders(self):
        if False:
            i = 10
            return i + 15
        self._update_extruder_timer.start()

    @UM.FlameProfiler.profile
    def __updateExtruders(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the list of extruders.\n\n        This should be called whenever the list of extruders changes.\n        '
        extruders_changed = False
        if self.count != 0:
            extruders_changed = True
        items = []
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack:
            machine_extruder_count = global_container_stack.getProperty('machine_extruder_count', 'value')
            for extruder in Application.getInstance().getExtruderManager().getActiveExtruderStacks():
                position = extruder.getMetaDataEntry('position', default='0')
                try:
                    position = int(position)
                except ValueError:
                    position = -1
                if position >= machine_extruder_count:
                    continue
                default_color = self.defaultColors[position] if 0 <= position < len(self.defaultColors) else self.defaultColors[0]
                color = extruder.material.getMetaDataEntry('color_code', default=default_color) if extruder.material else default_color
                material_brand = extruder.material.getMetaDataEntry('brand', default='generic')
                color_name = extruder.material.getMetaDataEntry('color_name')
                item = {'id': extruder.getId(), 'name': extruder.getName(), 'enabled': extruder.isEnabled, 'color': color, 'index': position, 'definition': extruder.getBottom().getId(), 'material': extruder.material.getName() if extruder.material else '', 'variant': extruder.variant.getName() if extruder.variant else '', 'stack': extruder, 'material_brand': material_brand, 'color_name': color_name, 'material_type': extruder.material.getMetaDataEntry('material') if extruder.material else '', 'material_name': extruder.material.getMetaDataEntry('name') if extruder.material else ''}
                items.append(item)
                extruders_changed = True
        if extruders_changed:
            items.sort(key=lambda i: i['index'])
            if self._add_optional_extruder:
                item = {'id': '', 'name': catalog.i18nc('@menuitem', 'Not overridden'), 'enabled': True, 'color': 'transparent', 'index': -1, 'definition': '', 'material': '', 'variant': '', 'stack': None, 'material_brand': '', 'color_name': '', 'material_type': '', 'material_label': ''}
                items.append(item)
            if self._items != items:
                self.setItems(items)
                self.modelChanged.emit()