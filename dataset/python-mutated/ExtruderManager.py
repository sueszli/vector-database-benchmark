from PyQt6.QtCore import pyqtSignal, pyqtProperty, QObject, QVariant
from UM.Application import Application
from UM.FlameProfiler import pyqtSlot
import cura.CuraApplication
from UM.Util import parseBool
from cura.Settings.GlobalStack import GlobalStack
from UM.Logger import Logger
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Scene.Selection import Selection
from UM.Scene.Iterator.BreadthFirstIterator import BreadthFirstIterator
from UM.Settings.ContainerRegistry import ContainerRegistry
from cura.Machines.ContainerTree import ContainerTree
from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from cura.Settings.ExtruderStack import ExtruderStack

class ExtruderManager(QObject):
    """Manages all existing extruder stacks.

    This keeps a list of extruder stacks for each machine.
    """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        'Registers listeners and such to listen to changes to the extruders.'
        if ExtruderManager.__instance is not None:
            raise RuntimeError("Try to create singleton '%s' more than once" % self.__class__.__name__)
        super().__init__(parent)
        ExtruderManager.__instance = self
        self._application = cura.CuraApplication.CuraApplication.getInstance()
        self._extruder_trains = {}
        self._active_extruder_index = -1
        self._selected_object_extruders = []
        Selection.selectionChanged.connect(self.resetSelectedObjectExtruders)
        Application.getInstance().globalContainerStackChanged.connect(self.emitGlobalStackExtrudersChanged)
    extrudersChanged = pyqtSignal(QVariant)
    'Signal to notify other components when the list of extruders for a machine definition changes.'
    activeExtruderChanged = pyqtSignal()
    'Notify when the user switches the currently active extruder.'

    def emitGlobalStackExtrudersChanged(self):
        if False:
            i = 10
            return i + 15
        self.extrudersChanged.emit(self._application.getGlobalContainerStack().getId())

    @pyqtProperty(int, notify=extrudersChanged)
    def enabledExtruderCount(self) -> int:
        if False:
            return 10
        global_container_stack = self._application.getGlobalContainerStack()
        if global_container_stack:
            return len([extruder for extruder in global_container_stack.extruderList if parseBool(extruder.getMetaDataEntry('enabled', 'True'))])
        return 0

    @pyqtProperty(str, notify=activeExtruderChanged)
    def activeExtruderStackId(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Gets the unique identifier of the currently active extruder stack.\n\n        The currently active extruder stack is the stack that is currently being\n        edited.\n\n        :return: The unique ID of the currently active extruder stack.\n        '
        if not self._application.getGlobalContainerStack():
            return None
        try:
            return self._extruder_trains[self._application.getGlobalContainerStack().getId()][str(self.activeExtruderIndex)].getId()
        except KeyError:
            return None

    @pyqtProperty('QVariantMap', notify=extrudersChanged)
    def extruderIds(self) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        'Gets a dict with the extruder stack ids with the extruder number as the key.'
        extruder_stack_ids = {}
        global_container_stack = self._application.getGlobalContainerStack()
        if global_container_stack:
            extruder_stack_ids = {extruder.getMetaDataEntry('position', ''): extruder.id for extruder in global_container_stack.extruderList}
        return extruder_stack_ids

    @pyqtSlot(int)
    def setActiveExtruderIndex(self, index: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Changes the active extruder by index.\n\n        :param index: The index of the new active extruder.\n        '
        if self._active_extruder_index != index:
            self._active_extruder_index = index
            self.activeExtruderChanged.emit()

    @pyqtProperty(int, notify=activeExtruderChanged)
    def activeExtruderIndex(self) -> int:
        if False:
            print('Hello World!')
        return self._active_extruder_index
    selectedObjectExtrudersChanged = pyqtSignal()
    'Emitted whenever the selectedObjectExtruders property changes.'

    @pyqtProperty('QVariantList', notify=selectedObjectExtrudersChanged)
    def selectedObjectExtruders(self) -> List[Union[str, 'ExtruderStack']]:
        if False:
            print('Hello World!')
        'Provides a list of extruder IDs used by the current selected objects.'
        if not self._selected_object_extruders:
            object_extruders = set()
            selected_nodes = []
            for node in Selection.getAllSelectedObjects():
                if node.callDecoration('isGroup'):
                    for grouped_node in BreadthFirstIterator(node):
                        if grouped_node.callDecoration('isGroup'):
                            continue
                        selected_nodes.append(grouped_node)
                else:
                    selected_nodes.append(node)
            current_extruder_trains = self.getActiveExtruderStacks()
            for node in selected_nodes:
                extruder = node.callDecoration('getActiveExtruder')
                if extruder:
                    object_extruders.add(extruder)
                elif current_extruder_trains:
                    object_extruders.add(current_extruder_trains[0].getId())
            self._selected_object_extruders = list(object_extruders)
        return self._selected_object_extruders

    def resetSelectedObjectExtruders(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset the internal list used for the selectedObjectExtruders property\n\n        This will trigger a recalculation of the extruders used for the\n        selection.\n        '
        self._selected_object_extruders = []
        self.selectedObjectExtrudersChanged.emit()

    @pyqtSlot(result=QObject)
    def getActiveExtruderStack(self) -> Optional['ExtruderStack']:
        if False:
            while True:
                i = 10
        return self.getExtruderStack(self.activeExtruderIndex)

    def getExtruderStack(self, index) -> Optional['ExtruderStack']:
        if False:
            for i in range(10):
                print('nop')
        'Get an extruder stack by index'
        global_container_stack = self._application.getGlobalContainerStack()
        if global_container_stack:
            if global_container_stack.getId() in self._extruder_trains:
                if str(index) in self._extruder_trains[global_container_stack.getId()]:
                    return self._extruder_trains[global_container_stack.getId()][str(index)]
        return None

    def getAllExtruderSettings(self, setting_key: str, prop: str) -> List[Any]:
        if False:
            while True:
                i = 10
        'Gets a property of a setting for all extruders.\n\n        :param setting_key:  :type{str} The setting to get the property of.\n        :param prop:  :type{str} The property to get.\n        :return: :type{List} the list of results\n        '
        result = []
        for extruder_stack in self.getActiveExtruderStacks():
            result.append(extruder_stack.getProperty(setting_key, prop))
        return result

    def extruderValueWithDefault(self, value: str) -> str:
        if False:
            print('Hello World!')
        machine_manager = self._application.getMachineManager()
        if value == '-1':
            return machine_manager.defaultExtruderPosition
        else:
            return value

    def getUsedExtruderStacks(self) -> List['ExtruderStack']:
        if False:
            return 10
        'Gets the extruder stacks that are actually being used at the moment.\n\n        An extruder stack is being used if it is the extruder to print any mesh\n        with, or if it is the support infill extruder, the support interface\n        extruder, or the bed adhesion extruder.\n\n        If there are no extruders, this returns the global stack as a singleton\n        list.\n\n        :return: A list of extruder stacks.\n        '
        global_stack = self._application.getGlobalContainerStack()
        container_registry = ContainerRegistry.getInstance()
        used_extruder_stack_ids = set()
        support_enabled = False
        support_bottom_enabled = False
        support_roof_enabled = False
        scene_root = self._application.getController().getScene().getRoot()
        if len(self.extruderIds) == 0:
            return []
        number_active_extruders = len([extruder for extruder in self.getActiveExtruderStacks() if extruder.isEnabled])
        nodes = [node for node in DepthFirstIterator(scene_root) if node.isSelectable() and (not node.callDecoration('isAntiOverhangMesh')) and (not node.callDecoration('isSupportMesh'))]
        for node in nodes:
            extruder_stack_id = node.callDecoration('getActiveExtruder')
            if not extruder_stack_id:
                extruder_stack_id = self.extruderIds['0']
            used_extruder_stack_ids.add(extruder_stack_id)
            if len(used_extruder_stack_ids) == number_active_extruders:
                break
            stack_to_use = node.callDecoration('getStack')
            if not stack_to_use:
                stack_to_use = container_registry.findContainerStacks(id=extruder_stack_id)[0]
            if not support_enabled:
                support_enabled |= stack_to_use.getProperty('support_enable', 'value')
            if not support_bottom_enabled:
                support_bottom_enabled |= stack_to_use.getProperty('support_bottom_enable', 'value')
            if not support_roof_enabled:
                support_roof_enabled |= stack_to_use.getProperty('support_roof_enable', 'value')
        limit_to_extruder_feature_list = ['wall_0_extruder_nr', 'wall_x_extruder_nr', 'roofing_extruder_nr', 'top_bottom_extruder_nr', 'infill_extruder_nr']
        for extruder_nr_feature_name in limit_to_extruder_feature_list:
            extruder_nr = int(global_stack.getProperty(extruder_nr_feature_name, 'value'))
            if extruder_nr == -1:
                continue
            if str(extruder_nr) not in self.extruderIds:
                extruder_nr = int(self._application.getMachineManager().defaultExtruderPosition)
            used_extruder_stack_ids.add(self.extruderIds[str(extruder_nr)])
        if support_enabled:
            used_extruder_stack_ids.add(self.extruderIds[self.extruderValueWithDefault(str(global_stack.getProperty('support_infill_extruder_nr', 'value')))])
            used_extruder_stack_ids.add(self.extruderIds[self.extruderValueWithDefault(str(global_stack.getProperty('support_extruder_nr_layer_0', 'value')))])
            if support_bottom_enabled:
                used_extruder_stack_ids.add(self.extruderIds[self.extruderValueWithDefault(str(global_stack.getProperty('support_bottom_extruder_nr', 'value')))])
            if support_roof_enabled:
                used_extruder_stack_ids.add(self.extruderIds[self.extruderValueWithDefault(str(global_stack.getProperty('support_roof_extruder_nr', 'value')))])
        used_adhesion_extruders = set()
        adhesion_type = global_stack.getProperty('adhesion_type', 'value')
        if adhesion_type == 'skirt' and (global_stack.getProperty('skirt_line_count', 'value') > 0 or global_stack.getProperty('skirt_brim_minimal_length', 'value') > 0):
            used_adhesion_extruders.add('skirt_brim_extruder_nr')
        if (adhesion_type == 'brim' or global_stack.getProperty('prime_tower_brim_enable', 'value')) and (global_stack.getProperty('brim_line_count', 'value') > 0 or global_stack.getProperty('skirt_brim_minimal_length', 'value') > 0):
            used_adhesion_extruders.add('skirt_brim_extruder_nr')
        if adhesion_type == 'raft':
            used_adhesion_extruders.add('raft_base_extruder_nr')
            if global_stack.getProperty('raft_interface_layers', 'value') > 0:
                used_adhesion_extruders.add('raft_interface_extruder_nr')
            if global_stack.getProperty('raft_surface_layers', 'value') > 0:
                used_adhesion_extruders.add('raft_surface_extruder_nr')
        for extruder_setting in used_adhesion_extruders:
            extruder_str_nr = str(global_stack.getProperty(extruder_setting, 'value'))
            if extruder_str_nr == '-1':
                continue
            if extruder_str_nr in self.extruderIds:
                used_extruder_stack_ids.add(self.extruderIds[extruder_str_nr])
        try:
            return [container_registry.findContainerStacks(id=stack_id)[0] for stack_id in used_extruder_stack_ids]
        except IndexError:
            Logger.log('e', 'Unable to find one or more of the extruders in %s', used_extruder_stack_ids)
            return []

    def getInitialExtruderNr(self) -> int:
        if False:
            print('Hello World!')
        'Get the extruder that the print will start with.\n\n        This should mirror the implementation in CuraEngine of\n        ``FffGcodeWriter::getStartExtruder()``.\n        '
        application = cura.CuraApplication.CuraApplication.getInstance()
        global_stack = application.getGlobalContainerStack()
        adhesion_type = global_stack.getProperty('adhesion_type', 'value')
        if adhesion_type in {'skirt', 'brim'}:
            return max(0, int(global_stack.getProperty('skirt_brim_extruder_nr', 'value')))
        if adhesion_type == 'raft':
            return global_stack.getProperty('raft_base_extruder_nr', 'value')
        if (global_stack.getProperty('support_enable', 'value') or global_stack.getProperty('support_structure', 'value') == 'tree') and global_stack.getProperty('support_brim_enable', 'value'):
            return global_stack.getProperty('support_infill_extruder_nr', 'value')
        return self.getUsedExtruderStacks()[0].getProperty('extruder_nr', 'value')

    def removeMachineExtruders(self, machine_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Removes the container stack and user profile for the extruders for a specific machine.\n\n        :param machine_id: The machine to remove the extruders for.\n        '
        for extruder in self.getMachineExtruders(machine_id):
            ContainerRegistry.getInstance().removeContainer(extruder.userChanges.getId())
            ContainerRegistry.getInstance().removeContainer(extruder.definitionChanges.getId())
            ContainerRegistry.getInstance().removeContainer(extruder.getId())
        if machine_id in self._extruder_trains:
            del self._extruder_trains[machine_id]

    def getMachineExtruders(self, machine_id: str) -> List['ExtruderStack']:
        if False:
            return 10
        'Returns extruders for a specific machine.\n\n        :param machine_id: The machine to get the extruders of.\n        '
        if machine_id not in self._extruder_trains:
            return []
        return [self._extruder_trains[machine_id][name] for name in self._extruder_trains[machine_id]]

    def getActiveExtruderStacks(self) -> List['ExtruderStack']:
        if False:
            while True:
                i = 10
        'Returns the list of active extruder stacks, taking into account the machine extruder count.\n\n        :return: :type{List[ContainerStack]} a list of\n        '
        global_stack = self._application.getGlobalContainerStack()
        if not global_stack:
            return []
        return global_stack.extruderList

    def _globalContainerStackChanged(self) -> None:
        if False:
            return 10
        self._addCurrentMachineExtruders()
        self.resetSelectedObjectExtruders()

    def addMachineExtruders(self, global_stack: GlobalStack) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds the extruders to the selected machine.'
        extruders_changed = False
        container_registry = ContainerRegistry.getInstance()
        global_stack_id = global_stack.getId()
        extruder_trains = container_registry.findContainerStacks(type='extruder_train', machine=global_stack_id)
        if global_stack_id not in self._extruder_trains:
            self._extruder_trains[global_stack_id] = {}
            extruders_changed = True
        for extruder_train in extruder_trains:
            extruder_position = extruder_train.getMetaDataEntry('position')
            self._extruder_trains[global_stack_id][extruder_position] = extruder_train
            extruder_train.setParent(global_stack)
            extruder_train.setNextStack(global_stack)
            extruders_changed = True
        self.fixSingleExtrusionMachineExtruderDefinition(global_stack)
        if extruders_changed:
            self.extrudersChanged.emit(global_stack_id)

    def fixSingleExtrusionMachineExtruderDefinition(self, global_stack: 'GlobalStack') -> None:
        if False:
            return 10
        container_registry = ContainerRegistry.getInstance()
        expected_extruder_stack = global_stack.getMetaDataEntry('machine_extruder_trains')
        if expected_extruder_stack is None:
            return
        expected_extruder_definition_0_id = expected_extruder_stack['0']
        try:
            extruder_stack_0 = global_stack.extruderList[0]
        except IndexError:
            extruder_stack_0 = None
        if not global_stack.extruderList:
            extruder_trains = container_registry.findContainerStacks(type='extruder_train', machine=global_stack.getId())
            if extruder_trains:
                for extruder in extruder_trains:
                    if extruder.getMetaDataEntry('position') == '0':
                        extruder_stack_0 = extruder
                        break
        if extruder_stack_0 is None:
            Logger.log('i', 'No extruder stack for global stack [%s], create one', global_stack.getId())
            from cura.Settings.CuraStackBuilder import CuraStackBuilder
            CuraStackBuilder.createExtruderStackWithDefaultSetup(global_stack, 0)
        elif extruder_stack_0.definition.getId() != expected_extruder_definition_0_id:
            Logger.log('e', "Single extruder printer [{printer}] expected extruder [{expected}], but got [{got}]. I'm making it [{expected}].".format(printer=global_stack.getId(), expected=expected_extruder_definition_0_id, got=extruder_stack_0.definition.getId()))
            try:
                extruder_definition = container_registry.findDefinitionContainers(id=expected_extruder_definition_0_id)[0]
            except IndexError:
                msg = 'Unable to find extruder definition with the id [%s]' % expected_extruder_definition_0_id
                Logger.logException('e', msg)
                raise IndexError(msg)
            extruder_stack_0.definition = extruder_definition

    @pyqtSlot('QVariant', result=bool)
    def getExtruderHasQualityForMaterial(self, extruder_stack: 'ExtruderStack') -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks if quality nodes exist for the variant/material combination.'
        application = cura.CuraApplication.CuraApplication.getInstance()
        global_stack = application.getGlobalContainerStack()
        if not global_stack or not extruder_stack:
            return False
        if not global_stack.getMetaDataEntry('has_materials'):
            return True
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        active_variant_name = extruder_stack.variant.getMetaDataEntry('name')
        if active_variant_name not in machine_node.variants:
            Logger.log('w', 'Could not find the variant %s', active_variant_name)
            return True
        active_variant_node = machine_node.variants[active_variant_name]
        try:
            active_material_node = active_variant_node.materials[extruder_stack.material.getMetaDataEntry('base_file')]
        except KeyError:
            return False
        active_material_node_qualities = active_material_node.qualities
        if not active_material_node_qualities:
            return False
        return list(active_material_node_qualities.keys())[0] != 'empty_quality'

    @pyqtSlot(str, result='QVariant')
    def getInstanceExtruderValues(self, key: str) -> List:
        if False:
            return 10
        'Get all extruder values for a certain setting.\n\n        This is exposed to qml for display purposes\n\n        :param key: The key of the setting to retrieve values for.\n\n        :return: String representing the extruder values\n        '
        return self._application.getCuraFormulaFunctions().getValuesInAllExtruders(key)

    @staticmethod
    def getResolveOrValue(key: str) -> Any:
        if False:
            return 10
        'Get the resolve value or value for a given key\n\n        This is the effective value for a given key, it is used for values in the global stack.\n        This is exposed to SettingFunction to use in value functions.\n        :param key: The key of the setting to get the value of.\n\n        :return: The effective value\n        '
        global_stack = cast(GlobalStack, cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack())
        resolved_value = global_stack.getProperty(key, 'value')
        return resolved_value
    __instance = None

    @classmethod
    def getInstance(cls, *args, **kwargs) -> 'ExtruderManager':
        if False:
            while True:
                i = 10
        return cls.__instance