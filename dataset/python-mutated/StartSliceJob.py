import os
import numpy
from string import Formatter
from enum import IntEnum
import time
from typing import Any, cast, Dict, List, Optional, Set
import re
import pyArcus as Arcus
from PyQt6.QtCore import QCoreApplication
from UM.Job import Job
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.Interfaces import ContainerInterface
from UM.Settings.SettingDefinition import SettingDefinition
from UM.Settings.SettingRelation import SettingRelation
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Scene.Scene import Scene
from UM.Settings.Validator import ValidatorState
from UM.Settings.SettingRelation import RelationType
from UM.Settings.SettingFunction import SettingFunction
from cura.CuraApplication import CuraApplication
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.OneAtATimeIterator import OneAtATimeIterator
from cura.Settings.ExtruderManager import ExtruderManager
NON_PRINTING_MESH_SETTINGS = ['anti_overhang_mesh', 'infill_mesh', 'cutting_mesh']

class StartJobResult(IntEnum):
    Finished = 1
    Error = 2
    SettingError = 3
    NothingToSlice = 4
    MaterialIncompatible = 5
    BuildPlateError = 6
    ObjectSettingError = 7
    ObjectsWithDisabledExtruder = 8

class GcodeStartEndFormatter(Formatter):
    _extruder_regex = re.compile('^\\s*(?P<expression>.*)\\s*,\\s*(?P<extruder_nr>\\d+)\\s*$')

    def __init__(self, default_extruder_nr: int=-1, *, additional_per_extruder_settings: Optional[Dict[str, Dict[str, any]]]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._default_extruder_nr: int = default_extruder_nr
        self._additional_per_extruder_settings: Optional[Dict[str, Dict[str, any]]] = additional_per_extruder_settings

    def get_value(self, expression: str, args: [str], kwargs: dict) -> str:
        if False:
            print('Hello World!')
        extruder_nr = self._default_extruder_nr
        match = self._extruder_regex.match(expression)
        if match:
            expression = match.group('expression')
            extruder_nr = int(match.group('extruder_nr'))
        if self._additional_per_extruder_settings is not None and str(extruder_nr) in self._additional_per_extruder_settings:
            additional_variables = self._additional_per_extruder_settings[str(extruder_nr)]
        else:
            additional_variables = dict()
        for (key, value) in enumerate(args):
            additional_variables[key] = value
        for (key, value) in kwargs.items():
            additional_variables[key] = value
        if extruder_nr == -1:
            container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        else:
            container_stack = ExtruderManager.getInstance().getExtruderStack(extruder_nr)
        setting_function = SettingFunction(expression)
        value = setting_function(container_stack, additional_variables=additional_variables)
        return value

class StartSliceJob(Job):
    """Job class that builds up the message of scene data to send to CuraEngine."""

    def __init__(self, slice_message: Arcus.PythonMessage) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._scene = CuraApplication.getInstance().getController().getScene()
        self._slice_message: Arcus.PythonMessage = slice_message
        self._is_cancelled = False
        self._build_plate_number = None
        self._all_extruders_settings = None

    def getSliceMessage(self) -> Arcus.PythonMessage:
        if False:
            print('Hello World!')
        return self._slice_message

    def setBuildPlate(self, build_plate_number: int) -> None:
        if False:
            while True:
                i = 10
        self._build_plate_number = build_plate_number

    def _checkStackForErrors(self, stack: ContainerStack) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if a stack has any errors.'
        'returns true if it has errors, false otherwise.'
        top_of_stack = cast(InstanceContainer, stack.getTop())
        changed_setting_keys = top_of_stack.getAllKeys()
        for key in top_of_stack.getAllKeys():
            instance = top_of_stack.getInstance(key)
            if instance is None:
                continue
            self._addRelations(changed_setting_keys, instance.definition.relations)
            Job.yieldThread()
        for changed_setting_key in changed_setting_keys:
            if not stack.getProperty(changed_setting_key, 'enabled'):
                continue
            validation_state = stack.getProperty(changed_setting_key, 'validationState')
            if validation_state is None:
                definition = cast(SettingDefinition, stack.getSettingDefinition(changed_setting_key))
                validator_type = SettingDefinition.getValidatorForType(definition.type)
                if validator_type:
                    validator = validator_type(changed_setting_key)
                    validation_state = validator(stack)
            if validation_state in (ValidatorState.Exception, ValidatorState.MaximumError, ValidatorState.MinimumError, ValidatorState.Invalid):
                Logger.log('w', 'Setting %s is not valid, but %s. Aborting slicing.', changed_setting_key, validation_state)
                return True
            Job.yieldThread()
        return False

    def run(self) -> None:
        if False:
            print('Hello World!')
        'Runs the job that initiates the slicing.'
        if self._build_plate_number is None:
            self.setResult(StartJobResult.Error)
            return
        stack = CuraApplication.getInstance().getGlobalContainerStack()
        if not stack:
            self.setResult(StartJobResult.Error)
            return
        if CuraApplication.getInstance().getMachineManager().stacksHaveErrors:
            self.setResult(StartJobResult.SettingError)
            return
        if CuraApplication.getInstance().getBuildVolume().hasErrors():
            self.setResult(StartJobResult.BuildPlateError)
            return
        while CuraApplication.getInstance().getMachineErrorChecker().needToWaitForResult:
            time.sleep(0.1)
        if CuraApplication.getInstance().getMachineErrorChecker().hasError:
            self.setResult(StartJobResult.SettingError)
            return
        if not CuraApplication.getInstance().getMachineManager().variantBuildplateCompatible and (not CuraApplication.getInstance().getMachineManager().variantBuildplateUsable):
            self.setResult(StartJobResult.MaterialIncompatible)
            return
        for extruder_stack in stack.extruderList:
            material = extruder_stack.findContainer({'type': 'material'})
            if not extruder_stack.isEnabled:
                continue
            if material:
                if material.getMetaDataEntry('compatible') == False:
                    self.setResult(StartJobResult.MaterialIncompatible)
                    return
        for node in DepthFirstIterator(self._scene.getRoot()):
            if not isinstance(node, CuraSceneNode) or not node.isSelectable():
                continue
            if self._checkStackForErrors(node.callDecoration('getStack')):
                self.setResult(StartJobResult.ObjectSettingError)
                return
        for node in DepthFirstIterator(self._scene.getRoot()):
            if node.callDecoration('getLayerData') and node.callDecoration('getBuildPlateNumber') == self._build_plate_number:
                cast(SceneNode, node.getParent()).removeChild(node)
                break
        object_groups = []
        if stack.getProperty('print_sequence', 'value') == 'one_at_a_time':
            modifier_mesh_nodes = []
            for node in DepthFirstIterator(self._scene.getRoot()):
                build_plate_number = node.callDecoration('getBuildPlateNumber')
                if node.callDecoration('isNonPrintingMesh') and build_plate_number == self._build_plate_number:
                    modifier_mesh_nodes.append(node)
            for node in OneAtATimeIterator(self._scene.getRoot()):
                temp_list = []
                build_plate_number = node.callDecoration('getBuildPlateNumber')
                if build_plate_number is not None and build_plate_number != self._build_plate_number:
                    continue
                children = node.getAllChildren()
                children.append(node)
                for child_node in children:
                    mesh_data = child_node.getMeshData()
                    if mesh_data and mesh_data.getVertices() is not None:
                        temp_list.append(child_node)
                if temp_list:
                    object_groups.append(temp_list + modifier_mesh_nodes)
                Job.yieldThread()
            if len(object_groups) == 0:
                Logger.log('w', 'No objects suitable for one at a time found, or no correct order found')
        else:
            temp_list = []
            has_printing_mesh = False
            for node in DepthFirstIterator(self._scene.getRoot()):
                mesh_data = node.getMeshData()
                if node.callDecoration('isSliceable') and mesh_data and (mesh_data.getVertices() is not None):
                    is_non_printing_mesh = bool(node.callDecoration('isNonPrintingMesh'))
                    if node.callDecoration('getBuildPlateNumber') != self._build_plate_number:
                        continue
                    if getattr(node, '_outside_buildarea', False) and (not is_non_printing_mesh):
                        continue
                    temp_list.append(node)
                    if not is_non_printing_mesh:
                        has_printing_mesh = True
                Job.yieldThread()
            if not has_printing_mesh:
                temp_list.clear()
            if temp_list:
                object_groups.append(temp_list)
        global_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack:
            return
        extruders_enabled = [stack.isEnabled for stack in global_stack.extruderList]
        filtered_object_groups = []
        has_model_with_disabled_extruders = False
        associated_disabled_extruders = set()
        for group in object_groups:
            stack = global_stack
            skip_group = False
            for node in group:
                is_non_printing_mesh = node.callDecoration('evaluateIsNonPrintingMesh')
                extruder_position = int(node.callDecoration('getActiveExtruderPosition'))
                if not is_non_printing_mesh and (not extruders_enabled[extruder_position]):
                    skip_group = True
                    has_model_with_disabled_extruders = True
                    associated_disabled_extruders.add(extruder_position)
            if not skip_group:
                filtered_object_groups.append(group)
        if has_model_with_disabled_extruders:
            self.setResult(StartJobResult.ObjectsWithDisabledExtruder)
            associated_disabled_extruders = {p + 1 for p in associated_disabled_extruders}
            self.setMessage(', '.join(map(str, sorted(associated_disabled_extruders))))
            return
        if not filtered_object_groups:
            self.setResult(StartJobResult.NothingToSlice)
            return
        self._buildGlobalSettingsMessage(stack)
        self._buildGlobalInheritsStackMessage(stack)
        for extruder_stack in global_stack.extruderList:
            self._buildExtruderMessage(extruder_stack)
        for plugin in CuraApplication.getInstance().getBackendPlugins():
            if not plugin.usePlugin():
                continue
            for slot in plugin.getSupportedSlots():
                plugin_message = self._slice_message.addRepeatedMessage('engine_plugins')
                plugin_message.id = slot
                plugin_message.address = plugin.getAddress()
                plugin_message.port = plugin.getPort()
                plugin_message.plugin_name = plugin.getPluginId()
                plugin_message.plugin_version = plugin.getVersion()
        for group in filtered_object_groups:
            group_message = self._slice_message.addRepeatedMessage('object_lists')
            parent = group[0].getParent()
            if parent is not None and parent.callDecoration('isGroup'):
                self._handlePerObjectSettings(cast(CuraSceneNode, parent), group_message)
            for object in group:
                mesh_data = object.getMeshData()
                if mesh_data is None:
                    continue
                rot_scale = object.getWorldTransformation().getTransposed().getData()[0:3, 0:3]
                translate = object.getWorldTransformation().getData()[:3, 3]
                verts = mesh_data.getVertices()
                verts = verts.dot(rot_scale)
                verts += translate
                verts[:, [1, 2]] = verts[:, [2, 1]]
                verts[:, 1] *= -1
                obj = group_message.addRepeatedMessage('objects')
                obj.id = id(object)
                obj.name = object.getName()
                indices = mesh_data.getIndices()
                if indices is not None:
                    flat_verts = numpy.take(verts, indices.flatten(), axis=0)
                else:
                    flat_verts = numpy.array(verts)
                obj.vertices = flat_verts
                self._handlePerObjectSettings(cast(CuraSceneNode, object), obj)
                Job.yieldThread()
        self.setResult(StartJobResult.Finished)

    def cancel(self) -> None:
        if False:
            print('Hello World!')
        super().cancel()
        self._is_cancelled = True

    def isCancelled(self) -> bool:
        if False:
            return 10
        return self._is_cancelled

    def setIsCancelled(self, value: bool):
        if False:
            while True:
                i = 10
        self._is_cancelled = value

    def _buildReplacementTokens(self, stack: ContainerStack) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Creates a dictionary of tokens to replace in g-code pieces.\n\n        This indicates what should be replaced in the start and end g-codes.\n        :param stack: The stack to get the settings from to replace the tokens with.\n        :return: A dictionary of replacement tokens to the values they should be replaced with.\n        '
        result = {}
        for key in stack.getAllKeys():
            result[key] = stack.getProperty(key, 'value')
            Job.yieldThread()
        result['material_id'] = stack.material.getMetaDataEntry('base_file', '')
        result['material_type'] = stack.material.getMetaDataEntry('material', '')
        result['material_name'] = stack.material.getMetaDataEntry('name', '')
        result['material_brand'] = stack.material.getMetaDataEntry('brand', '')
        result['quality_name'] = stack.quality.getMetaDataEntry('name', '')
        result['quality_changes_name'] = stack.qualityChanges.getMetaDataEntry('name')
        result['print_bed_temperature'] = result['material_bed_temperature']
        result['print_temperature'] = result['material_print_temperature']
        result['travel_speed'] = result['speed_travel']
        result['time'] = time.strftime('%H:%M:%S')
        result['date'] = time.strftime('%d-%m-%Y')
        result['day'] = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][int(time.strftime('%w'))]
        result['initial_extruder_nr'] = CuraApplication.getInstance().getExtruderManager().getInitialExtruderNr()
        return result

    def _cacheAllExtruderSettings(self):
        if False:
            return 10
        global_stack = cast(ContainerStack, CuraApplication.getInstance().getGlobalContainerStack())
        self._all_extruders_settings = {'-1': self._buildReplacementTokens(global_stack)}
        QCoreApplication.processEvents()
        for extruder_stack in ExtruderManager.getInstance().getActiveExtruderStacks():
            extruder_nr = extruder_stack.getProperty('extruder_nr', 'value')
            self._all_extruders_settings[str(extruder_nr)] = self._buildReplacementTokens(extruder_stack)
            QCoreApplication.processEvents()

    def _expandGcodeTokens(self, value: str, default_extruder_nr: int=-1) -> str:
        if False:
            while True:
                i = 10
        'Replace setting tokens in a piece of g-code.\n\n        :param value: A piece of g-code to replace tokens in.\n        :param default_extruder_nr: Stack nr to use when no stack nr is specified, defaults to the global stack\n        '
        if not self._all_extruders_settings:
            self._cacheAllExtruderSettings()
        try:
            additional_per_extruder_settings = self._all_extruders_settings.copy()
            additional_per_extruder_settings['default_extruder_nr'] = default_extruder_nr
            fmt = GcodeStartEndFormatter(default_extruder_nr=default_extruder_nr, additional_per_extruder_settings=additional_per_extruder_settings)
            return str(fmt.format(value))
        except:
            Logger.logException('w', 'Unable to do token replacement on start/end g-code')
            return str(value)

    def _buildExtruderMessage(self, stack: ContainerStack) -> None:
        if False:
            return 10
        'Create extruder message from stack'
        message = self._slice_message.addRepeatedMessage('extruders')
        message.id = int(stack.getMetaDataEntry('position'))
        if not self._all_extruders_settings:
            self._cacheAllExtruderSettings()
        if self._all_extruders_settings is None:
            return
        extruder_nr = stack.getProperty('extruder_nr', 'value')
        settings = self._all_extruders_settings[str(extruder_nr)].copy()
        settings['material_guid'] = stack.material.getMetaDataEntry('GUID', '')
        extruder_nr = stack.getProperty('extruder_nr', 'value')
        settings['machine_extruder_start_code'] = self._expandGcodeTokens(settings['machine_extruder_start_code'], extruder_nr)
        settings['machine_extruder_end_code'] = self._expandGcodeTokens(settings['machine_extruder_end_code'], extruder_nr)
        global_definition = cast(ContainerInterface, cast(ContainerStack, stack.getNextStack()).getBottom())
        own_definition = cast(ContainerInterface, stack.getBottom())
        for (key, value) in settings.items():
            if not global_definition.getProperty(key, 'settable_per_extruder') and (not own_definition.getProperty(key, 'settable_per_extruder')):
                continue
            setting = message.getMessage('settings').addRepeatedMessage('settings')
            setting.name = key
            setting.value = str(value).encode('utf-8')
            Job.yieldThread()

    def _buildGlobalSettingsMessage(self, stack: ContainerStack) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sends all global settings to the engine.\n\n        The settings are taken from the global stack. This does not include any\n        per-extruder settings or per-object settings.\n        '
        if not self._all_extruders_settings:
            self._cacheAllExtruderSettings()
        if self._all_extruders_settings is None:
            return
        settings = self._all_extruders_settings['-1'].copy()
        start_gcode = settings['machine_start_gcode']
        start_gcode = re.sub(';.+?(\\n|$)', '\n', start_gcode)
        bed_temperature_settings = ['material_bed_temperature', 'material_bed_temperature_layer_0']
        pattern = '\\{(%s)(,\\s?\\w+)?\\}' % '|'.join(bed_temperature_settings)
        settings['material_bed_temp_prepend'] = re.search(pattern, start_gcode) == None
        print_temperature_settings = ['material_print_temperature', 'material_print_temperature_layer_0', 'default_material_print_temperature', 'material_initial_print_temperature', 'material_final_print_temperature', 'material_standby_temperature', 'print_temperature']
        pattern = '\\{(%s)(,\\s?\\w+)?\\}' % '|'.join(print_temperature_settings)
        settings['material_print_temp_prepend'] = re.search(pattern, start_gcode) is None
        initial_extruder_nr = CuraApplication.getInstance().getExtruderManager().getInitialExtruderNr()
        settings['machine_start_gcode'] = self._expandGcodeTokens(settings['machine_start_gcode'], initial_extruder_nr)
        settings['machine_end_gcode'] = self._expandGcodeTokens(settings['machine_end_gcode'], initial_extruder_nr)
        settings['nozzle_offsetting_for_disallowed_areas'] = CuraApplication.getInstance().getGlobalContainerStack().getMetaDataEntry('nozzle_offsetting_for_disallowed_areas', True)
        for (key, value) in settings.items():
            setting_message = self._slice_message.getMessage('global_settings').addRepeatedMessage('settings')
            setting_message.name = key
            setting_message.value = str(value).encode('utf-8')
            Job.yieldThread()

    def _buildGlobalInheritsStackMessage(self, stack: ContainerStack) -> None:
        if False:
            return 10
        'Sends for some settings which extruder they should fallback to if not set.\n\n        This is only set for settings that have the limit_to_extruder\n        property.\n\n        :param stack: The global stack with all settings, from which to read the\n            limit_to_extruder property.\n        '
        for key in stack.getAllKeys():
            extruder_position = int(round(float(stack.getProperty(key, 'limit_to_extruder'))))
            if extruder_position >= 0:
                setting_extruder = self._slice_message.addRepeatedMessage('limit_to_extruder')
                setting_extruder.name = key
                setting_extruder.extruder = extruder_position
            Job.yieldThread()

    def _handlePerObjectSettings(self, node: CuraSceneNode, message: Arcus.PythonMessage):
        if False:
            i = 10
            return i + 15
        'Check if a node has per object settings and ensure that they are set correctly in the message\n\n        :param node: Node to check.\n        :param message: object_lists message to put the per object settings in\n        '
        stack = node.callDecoration('getStack')
        if not stack:
            return
        top_of_stack = stack.getTop()
        changed_setting_keys = top_of_stack.getAllKeys()
        for key in top_of_stack.getAllKeys():
            instance = top_of_stack.getInstance(key)
            self._addRelations(changed_setting_keys, instance.definition.relations)
            Job.yieldThread()
        changed_setting_keys.add('extruder_nr')
        for key in changed_setting_keys:
            setting = message.addRepeatedMessage('settings')
            setting.name = key
            extruder = int(round(float(stack.getProperty(key, 'limit_to_extruder'))))
            if extruder >= 0 and key not in changed_setting_keys:
                limited_stack = ExtruderManager.getInstance().getActiveExtruderStacks()[extruder]
            else:
                limited_stack = stack
            setting.value = str(limited_stack.getProperty(key, 'value')).encode('utf-8')
            Job.yieldThread()

    def _addRelations(self, relations_set: Set[str], relations: List[SettingRelation]):
        if False:
            i = 10
            return i + 15
        'Recursive function to put all settings that require each other for value changes in a list\n\n        :param relations_set: Set of keys of settings that are influenced\n        :param relations: list of relation objects that need to be checked.\n        '
        for relation in filter(lambda r: r.role == 'value' or r.role == 'limit_to_extruder', relations):
            if relation.type == RelationType.RequiresTarget:
                continue
            relations_set.add(relation.target.key)
            self._addRelations(relations_set, relation.target.relations)