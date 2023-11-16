from typing import Dict, Set
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtProperty
from UM.Qt.ListModel import ListModel
from UM.Logger import Logger
import cura.CuraApplication
from cura.Machines.ContainerTree import ContainerTree
from cura.Machines.MaterialNode import MaterialNode
from cura.Settings.CuraContainerRegistry import CuraContainerRegistry

class BaseMaterialsModel(ListModel):
    """This is the base model class for GenericMaterialsModel and MaterialBrandsModel.

    Those 2 models are used by the material drop down menu to show generic materials and branded materials
    separately. The extruder position defined here is being used to bound a menu to the correct extruder. This is
    used in the top bar menu "Settings" -> "Extruder nr" -> "Material" -> this menu
    """
    extruderPositionChanged = pyqtSignal()
    enabledChanged = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        from cura.CuraApplication import CuraApplication
        self._application = CuraApplication.getInstance()
        self._available_materials = {}
        self._favorite_ids = set()
        self._container_registry = self._application.getInstance().getContainerRegistry()
        self._machine_manager = self._application.getMachineManager()
        self._extruder_position = 0
        self._extruder_stack = None
        self._enabled = True
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(100)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._update)
        self._machine_manager.globalContainerChanged.connect(self._updateExtruderStack)
        self._updateExtruderStack()
        self._machine_manager.activeStackChanged.connect(self._onChanged)
        ContainerTree.getInstance().materialsChanged.connect(self._materialsListChanged)
        self._application.getMaterialManagementModel().favoritesChanged.connect(self._onChanged)
        self.addRoleName(Qt.ItemDataRole.UserRole + 1, 'root_material_id')
        self.addRoleName(Qt.ItemDataRole.UserRole + 2, 'id')
        self.addRoleName(Qt.ItemDataRole.UserRole + 3, 'GUID')
        self.addRoleName(Qt.ItemDataRole.UserRole + 4, 'name')
        self.addRoleName(Qt.ItemDataRole.UserRole + 5, 'brand')
        self.addRoleName(Qt.ItemDataRole.UserRole + 6, 'description')
        self.addRoleName(Qt.ItemDataRole.UserRole + 7, 'material')
        self.addRoleName(Qt.ItemDataRole.UserRole + 8, 'color_name')
        self.addRoleName(Qt.ItemDataRole.UserRole + 9, 'color_code')
        self.addRoleName(Qt.ItemDataRole.UserRole + 10, 'density')
        self.addRoleName(Qt.ItemDataRole.UserRole + 11, 'diameter')
        self.addRoleName(Qt.ItemDataRole.UserRole + 12, 'approximate_diameter')
        self.addRoleName(Qt.ItemDataRole.UserRole + 13, 'adhesion_info')
        self.addRoleName(Qt.ItemDataRole.UserRole + 14, 'is_read_only')
        self.addRoleName(Qt.ItemDataRole.UserRole + 15, 'container_node')
        self.addRoleName(Qt.ItemDataRole.UserRole + 16, 'is_favorite')

    def _onChanged(self) -> None:
        if False:
            return 10
        self._update_timer.start()

    def _updateExtruderStack(self):
        if False:
            return 10
        global_stack = self._machine_manager.activeMachine
        if global_stack is None:
            return
        if self._extruder_stack is not None:
            self._extruder_stack.pyqtContainersChanged.disconnect(self._onChanged)
            self._extruder_stack.approximateMaterialDiameterChanged.disconnect(self._onChanged)
        try:
            self._extruder_stack = global_stack.extruderList[self._extruder_position]
        except IndexError:
            self._extruder_stack = None
        if self._extruder_stack is not None:
            self._extruder_stack.pyqtContainersChanged.connect(self._onChanged)
            self._extruder_stack.approximateMaterialDiameterChanged.connect(self._onChanged)
        self._onChanged()

    def setExtruderPosition(self, position: int):
        if False:
            for i in range(10):
                print('nop')
        if self._extruder_stack is None or self._extruder_position != position:
            self._extruder_position = position
            self._updateExtruderStack()
            self.extruderPositionChanged.emit()

    @pyqtProperty(int, fset=setExtruderPosition, notify=extruderPositionChanged)
    def extruderPosition(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._extruder_position

    def setEnabled(self, enabled):
        if False:
            i = 10
            return i + 15
        if self._enabled != enabled:
            self._enabled = enabled
            if self._enabled:
                self._onChanged()
            self.enabledChanged.emit()

    @pyqtProperty(bool, fset=setEnabled, notify=enabledChanged)
    def enabled(self):
        if False:
            while True:
                i = 10
        return self._enabled

    def _materialsListChanged(self, material: MaterialNode) -> None:
        if False:
            while True:
                i = 10
        'Triggered when a list of materials changed somewhere in the container\n\n        tree. This change may trigger an _update() call when the materials changed for the configuration that this\n        model is looking for.\n        '
        if self._extruder_stack is None:
            return
        if material.variant.container_id != self._extruder_stack.variant.getId():
            return
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack:
            return
        if material.variant.machine.container_id != global_stack.definition.getId():
            return
        self._onChanged()

    def _favoritesChanged(self, material_base_file: str) -> None:
        if False:
            print('Hello World!')
        'Triggered when the list of favorite materials is changed.'
        if material_base_file in self._available_materials:
            self._onChanged()

    def _update(self):
        if False:
            for i in range(10):
                print('nop')
        'This is an abstract method that needs to be implemented by the specific models themselves. '
        self._favorite_ids = set(cura.CuraApplication.CuraApplication.getInstance().getPreferences().getValue('cura/favorite_materials').split(';'))
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack or not global_stack.hasMaterials:
            return
        extruder_list = global_stack.extruderList
        if self._extruder_position > len(extruder_list):
            return
        extruder_stack = extruder_list[self._extruder_position]
        nozzle_name = extruder_stack.variant.getName()
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        if nozzle_name not in machine_node.variants:
            Logger.log('w', 'Unable to find variant %s in container tree', nozzle_name)
            self._available_materials = {}
            return
        materials = machine_node.variants[nozzle_name].materials
        approximate_material_diameter = extruder_stack.getApproximateMaterialDiameter()
        self._available_materials = {key: material for (key, material) in materials.items() if float(material.getMetaDataEntry('approximate_diameter', -1)) == approximate_material_diameter}

    def _canUpdate(self):
        if False:
            print('Hello World!')
        "This method is used by all material models in the beginning of the _update() method in order to prevent\n        errors. It's the same in all models so it's placed here for easy access. "
        global_stack = self._machine_manager.activeMachine
        if global_stack is None or not self._enabled:
            return False
        if self._extruder_position >= len(global_stack.extruderList):
            return False
        return True

    def _createMaterialItem(self, root_material_id, container_node):
        if False:
            return 10
        "This is another convenience function which is shared by all material models so it's put here to avoid having\n         so much duplicated code. "
        metadata_list = CuraContainerRegistry.getInstance().findContainersMetadata(id=container_node.container_id)
        if not metadata_list:
            return None
        metadata = metadata_list[0]
        item = {'root_material_id': root_material_id, 'id': metadata['id'], 'container_id': metadata['id'], 'GUID': metadata['GUID'], 'name': metadata['name'], 'brand': metadata['brand'], 'description': metadata['description'], 'material': metadata['material'], 'color_name': metadata['color_name'], 'color_code': metadata.get('color_code', ''), 'density': metadata.get('properties', {}).get('density', ''), 'diameter': metadata.get('properties', {}).get('diameter', ''), 'approximate_diameter': metadata['approximate_diameter'], 'adhesion_info': metadata['adhesion_info'], 'is_read_only': self._container_registry.isReadOnly(metadata['id']), 'container_node': container_node, 'is_favorite': root_material_id in self._favorite_ids}
        return item