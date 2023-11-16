from typing import Optional, Dict, Any, Set, List
from PyQt6.QtCore import Qt, QObject, pyqtProperty, pyqtSignal, QTimer
import cura.CuraApplication
from UM.Qt.ListModel import ListModel
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Logger import Logger
from cura.Machines.ContainerTree import ContainerTree
from cura.Machines.MaterialNode import MaterialNode
from cura.Machines.Models.MachineModelUtils import fetchLayerHeight
from cura.Machines.QualityGroup import QualityGroup

class IntentModel(ListModel):
    NameRole = Qt.ItemDataRole.UserRole + 1
    QualityTypeRole = Qt.ItemDataRole.UserRole + 2
    LayerHeightRole = Qt.ItemDataRole.UserRole + 3
    AvailableRole = Qt.ItemDataRole.UserRole + 4
    IntentRole = Qt.ItemDataRole.UserRole + 5

    def __init__(self, parent: Optional[QObject]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.addRoleName(self.NameRole, 'name')
        self.addRoleName(self.QualityTypeRole, 'quality_type')
        self.addRoleName(self.LayerHeightRole, 'layer_height')
        self.addRoleName(self.AvailableRole, 'available')
        self.addRoleName(self.IntentRole, 'intent_category')
        self._intent_category = 'engineering'
        self._update_timer = QTimer()
        self._update_timer.setInterval(100)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._update)
        machine_manager = cura.CuraApplication.CuraApplication.getInstance().getMachineManager()
        machine_manager.globalContainerChanged.connect(self._updateDelayed)
        machine_manager.extruderChanged.connect(self._updateDelayed)
        ContainerRegistry.getInstance().containerAdded.connect(self._onChanged)
        ContainerRegistry.getInstance().containerRemoved.connect(self._onChanged)
        self._layer_height_unit = ''
        self._update()
    intentCategoryChanged = pyqtSignal()

    def setIntentCategory(self, new_category: str) -> None:
        if False:
            print('Hello World!')
        if self._intent_category != new_category:
            self._intent_category = new_category
            self.intentCategoryChanged.emit()
            self._update()

    @pyqtProperty(str, fset=setIntentCategory, notify=intentCategoryChanged)
    def intentCategory(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._intent_category

    def _updateDelayed(self):
        if False:
            return 10
        self._update_timer.start()

    def _onChanged(self, container):
        if False:
            return 10
        if container.getMetaDataEntry('type') == 'intent':
            self._updateDelayed()

    def _update(self) -> None:
        if False:
            i = 10
            return i + 15
        new_items = []
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack:
            self.setItems(new_items)
            return
        quality_groups = ContainerTree.getInstance().getCurrentQualityGroups()
        material_nodes = self._getActiveMaterials()
        added_quality_type_set = set()
        for material_node in material_nodes:
            intents = self._getIntentsForMaterial(material_node, quality_groups)
            for intent in intents:
                if intent['quality_type'] not in added_quality_type_set:
                    new_items.append(intent)
                    added_quality_type_set.add(intent['quality_type'])
        for (quality_type, quality_group) in quality_groups.items():
            if quality_type not in added_quality_type_set:
                layer_height = fetchLayerHeight(quality_group)
                new_items.append({'name': 'Unavailable', 'quality_type': quality_type, 'layer_height': layer_height, 'intent_category': self._intent_category, 'available': False})
                added_quality_type_set.add(quality_type)
        new_items = sorted(new_items, key=lambda x: x['layer_height'])
        self.setItems(new_items)

    def _getActiveMaterials(self) -> Set['MaterialNode']:
        if False:
            while True:
                i = 10
        'Get the active materials for all extruders. No duplicates will be returned'
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack is None:
            return set()
        container_tree = ContainerTree.getInstance()
        machine_node = container_tree.machines[global_stack.definition.getId()]
        nodes = set()
        for extruder in global_stack.extruderList:
            active_variant_name = extruder.variant.getMetaDataEntry('name')
            if active_variant_name not in machine_node.variants:
                Logger.log('w', 'Could not find the variant %s', active_variant_name)
                continue
            active_variant_node = machine_node.variants[active_variant_name]
            active_material_node = active_variant_node.materials.get(extruder.material.getMetaDataEntry('base_file'))
            if active_material_node is None:
                Logger.log('w', 'Could not find the material %s', extruder.material.getMetaDataEntry('base_file'))
                continue
            nodes.add(active_material_node)
        return nodes

    def _getIntentsForMaterial(self, active_material_node: 'MaterialNode', quality_groups: Dict[str, 'QualityGroup']) -> List[Dict[str, Any]]:
        if False:
            return 10
        extruder_intents = []
        for (quality_id, quality_node) in active_material_node.qualities.items():
            if quality_node.quality_type not in quality_groups:
                continue
            quality_group = quality_groups[quality_node.quality_type]
            layer_height = fetchLayerHeight(quality_group)
            for (intent_id, intent_node) in quality_node.intents.items():
                if intent_node.intent_category != self._intent_category:
                    continue
                extruder_intents.append({'name': quality_group.name, 'quality_type': quality_group.quality_type, 'layer_height': layer_height, 'available': quality_group.is_available, 'intent_category': self._intent_category})
        return extruder_intents

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.items)