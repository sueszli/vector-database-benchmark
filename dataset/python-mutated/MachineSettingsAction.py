from typing import Optional, TYPE_CHECKING
from PyQt6.QtCore import pyqtProperty
import UM.i18n
from UM.FlameProfiler import pyqtSlot
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Util import parseBool
import cura.CuraApplication
from cura.MachineAction import MachineAction
from cura.Machines.ContainerTree import ContainerTree
from cura.Settings.CuraStackBuilder import CuraStackBuilder
from cura.Settings.cura_empty_instance_containers import isEmptyContainer
if TYPE_CHECKING:
    from PyQt6.QtCore import QObject
catalog = UM.i18n.i18nCatalog('cura')

class MachineSettingsAction(MachineAction):
    """This action allows for certain settings that are "machine only") to be modified.

    It automatically detects machine definitions that it knows how to change and attaches itself to those.
    """

    def __init__(self, parent: Optional['QObject']=None) -> None:
        if False:
            return 10
        super().__init__('MachineSettingsAction', catalog.i18nc('@action', 'Machine Settings'))
        self._qml_url = 'MachineSettingsAction.qml'
        from cura.CuraApplication import CuraApplication
        self._application = CuraApplication.getInstance()
        from cura.Settings.CuraContainerStack import _ContainerIndexes
        self._store_container_index = _ContainerIndexes.DefinitionChanges
        self._container_registry = ContainerRegistry.getInstance()
        self._container_registry.containerAdded.connect(self._onContainerAdded)
        self._backend = self._application.getBackend()
        self.onFinished.connect(self._onFinished)
        self._application.globalContainerStackChanged.connect(self._updateHasMaterialsInContainerTree)

    @pyqtProperty(int, constant=True)
    def storeContainerIndex(self) -> int:
        if False:
            return 10
        return self._store_container_index

    def _onContainerAdded(self, container):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(container, DefinitionContainer) and container.getMetaDataEntry('type') == 'machine':
            self._application.getMachineActionManager().addSupportedAction(container.getId(), self.getKey())

    def _updateHasMaterialsInContainerTree(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Triggered when the global container stack changes or when the g-code\n\n        flavour setting is changed.\n        '
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack is None:
            return
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        if machine_node.has_materials != parseBool(global_stack.getMetaDataEntry('has_materials')):
            machine_node.has_materials = parseBool(global_stack.getMetaDataEntry('has_materials'))
            machine_node._loadAll()

    def _reset(self):
        if False:
            i = 10
            return i + 15
        global_stack = self._application.getMachineManager().activeMachine
        if not global_stack:
            return
        definition_changes_id = global_stack.definitionChanges.getId()
        if isEmptyContainer(definition_changes_id):
            CuraStackBuilder.createDefinitionChangesContainer(global_stack, global_stack.getName() + '_settings')
        if self._backend:
            self._backend.disableTimer()

    def _onFinished(self):
        if False:
            for i in range(10):
                print('nop')
        if self._backend and self._backend.determineAutoSlicing():
            self._backend.enableTimer()
            self._backend.tickle()

    @pyqtSlot(int)
    def setMachineExtruderCount(self, extruder_count: int) -> None:
        if False:
            while True:
                i = 10
        self._application.getMachineManager().setActiveMachineExtruderCount(extruder_count)

    @pyqtSlot()
    def forceUpdate(self) -> None:
        if False:
            return 10
        self._application.getMachineManager().globalContainerChanged.emit()
        self._application.getMachineManager().forceUpdateAllSettings()

    @pyqtSlot()
    def updateHasMaterialsMetadata(self) -> None:
        if False:
            return 10
        global_stack = self._application.getMachineManager().activeMachine
        if not global_stack:
            return
        definition = global_stack.getDefinition()
        if definition.getProperty('machine_gcode_flavor', 'value') != 'UltiGCode' or parseBool(definition.getMetaDataEntry('has_materials', False)):
            return
        machine_manager = self._application.getMachineManager()
        has_materials = global_stack.getProperty('machine_gcode_flavor', 'value') != 'UltiGCode'
        if has_materials:
            global_stack.setMetaDataEntry('has_materials', True)
        elif 'has_materials' in global_stack.getMetaData():
            global_stack.removeMetaDataEntry('has_materials')
        self._updateHasMaterialsInContainerTree()
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        for (position, extruder) in enumerate(global_stack.extruderList):
            approximate_diameter = round(extruder.getProperty('material_diameter', 'value'))
            material_node = machine_node.variants[extruder.variant.getName()].preferredMaterial(approximate_diameter)
            machine_manager.setMaterial(str(position), material_node)
        self._application.globalContainerStackChanged.emit()

    @pyqtSlot(int)
    def updateMaterialForDiameter(self, extruder_position: int) -> None:
        if False:
            while True:
                i = 10
        self._application.getMachineManager().updateMaterialWithVariant(str(extruder_position))