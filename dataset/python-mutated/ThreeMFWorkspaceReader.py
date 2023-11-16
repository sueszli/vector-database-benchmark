from configparser import ConfigParser
import zipfile
import os
import json
from typing import cast, Dict, List, Optional, Tuple, Any, Set
import xml.etree.ElementTree as ET
from UM.Util import parseBool
from UM.Workspace.WorkspaceReader import WorkspaceReader
from UM.Application import Application
from UM.Logger import Logger
from UM.Message import Message
from UM.i18n import i18nCatalog
from UM.Settings.ContainerFormatError import ContainerFormatError
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
from UM.Job import Job
from UM.Preferences import Preferences
from cura.CuraPackageManager import CuraPackageManager
from cura.Machines.ContainerTree import ContainerTree
from cura.Settings.CuraStackBuilder import CuraStackBuilder
from cura.Settings.ExtruderManager import ExtruderManager
from cura.Settings.ExtruderStack import ExtruderStack
from cura.Settings.GlobalStack import GlobalStack
from cura.Settings.IntentManager import IntentManager
from cura.Settings.CuraContainerStack import _ContainerIndexes
from cura.CuraApplication import CuraApplication
from cura.Utils.Threading import call_on_qt_thread
from PyQt6.QtCore import QCoreApplication
from .WorkspaceDialog import WorkspaceDialog
i18n_catalog = i18nCatalog('cura')
_ignored_machine_network_metadata: Set[str] = {'um_cloud_cluster_id', 'um_network_key', 'um_linked_to_account', 'host_guid', 'removal_warning', 'group_name', 'group_size', 'connection_type', 'capabilities', 'octoprint_api_key', 'is_abstract_machine'}

class ContainerInfo:

    def __init__(self, file_name: Optional[str], serialized: Optional[str], parser: Optional[ConfigParser]) -> None:
        if False:
            i = 10
            return i + 15
        self.file_name = file_name
        self.serialized = serialized
        self.parser = parser
        self.container = None
        self.definition_id = None

class QualityChangesInfo:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.name: Optional[str] = None
        self.global_info = None
        self.extruder_info_dict: Dict[str, ContainerInfo] = {}

class MachineInfo:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.container_id: Optional[str] = None
        self.name: Optional[str] = None
        self.definition_id: Optional[str] = None
        self.metadata_dict: Dict[str, str] = {}
        self.quality_type: Optional[str] = None
        self.intent_category: Optional[str] = None
        self.custom_quality_name: Optional[str] = None
        self.quality_changes_info: Optional[QualityChangesInfo] = None
        self.variant_info: Optional[ContainerInfo] = None
        self.definition_changes_info: Optional[ContainerInfo] = None
        self.user_changes_info: Optional[ContainerInfo] = None
        self.extruder_info_dict: Dict[str, str] = {}

class ExtruderInfo:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.position = None
        self.enabled = True
        self.variant_info: Optional[ContainerInfo] = None
        self.root_material_id: Optional[str] = None
        self.definition_changes_info: Optional[ContainerInfo] = None
        self.user_changes_info: Optional[ContainerInfo] = None
        self.intent_info: Optional[ContainerInfo] = None

class ThreeMFWorkspaceReader(WorkspaceReader):
    """Base implementation for reading 3MF workspace files."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._supported_extensions = ['.3mf']
        self._dialog = WorkspaceDialog()
        self._3mf_mesh_reader = None
        self._container_registry = ContainerRegistry.getInstance()
        self._definition_container_suffix = '.' + cast(MimeType, ContainerRegistry.getMimeTypeForContainer(DefinitionContainer)).preferredSuffix
        self._material_container_suffix = None
        self._instance_container_suffix = '.' + cast(MimeType, ContainerRegistry.getMimeTypeForContainer(InstanceContainer)).preferredSuffix
        self._container_stack_suffix = '.' + cast(MimeType, ContainerRegistry.getMimeTypeForContainer(ContainerStack)).preferredSuffix
        self._extruder_stack_suffix = '.' + cast(MimeType, ContainerRegistry.getMimeTypeForContainer(ExtruderStack)).preferredSuffix
        self._global_stack_suffix = '.' + cast(MimeType, ContainerRegistry.getMimeTypeForContainer(GlobalStack)).preferredSuffix
        self._ignored_instance_container_types = {'quality', 'variant'}
        self._resolve_strategies: Dict[str, str] = {}
        self._id_mapping: Dict[str, str] = {}
        self._old_empty_profile_id_dict = {'empty_%s' % k: 'empty' for k in ['material', 'variant']}
        self._old_new_materials: Dict[str, str] = {}
        self._machine_info = None

    def _clearState(self):
        if False:
            print('Hello World!')
        self._id_mapping = {}
        self._old_new_materials = {}
        self._machine_info = None

    def getNewId(self, old_id: str):
        if False:
            for i in range(10):
                print('nop')
        'Get a unique name based on the old_id. This is different from directly calling the registry in that it caches results.\n\n        This has nothing to do with speed, but with getting consistent new naming for instances & objects.\n        '
        if old_id not in self._id_mapping:
            self._id_mapping[old_id] = self._container_registry.uniqueName(old_id)
        return self._id_mapping[old_id]

    def _determineGlobalAndExtruderStackFiles(self, project_file_name: str, file_list: List[str]) -> Tuple[str, List[str]]:
        if False:
            while True:
                i = 10
        'Separates the given file list into a list of GlobalStack files and a list of ExtruderStack files.\n\n        In old versions, extruder stack files have the same suffix as container stack files ".stack.cfg".\n        '
        archive = zipfile.ZipFile(project_file_name, 'r')
        global_stack_file_list = [name for name in file_list if name.endswith(self._global_stack_suffix)]
        extruder_stack_file_list = [name for name in file_list if name.endswith(self._extruder_stack_suffix)]
        files_to_determine = [name for name in file_list if name.endswith(self._container_stack_suffix)]
        for file_name in files_to_determine:
            serialized = archive.open(file_name).read().decode('utf-8')
            stack_config = ConfigParser(interpolation=None)
            stack_config.read_string(serialized)
            if not stack_config.has_option('metadata', 'type'):
                Logger.log('e', "%s in %s doesn't seem to be valid stack file", file_name, project_file_name)
                continue
            stack_type = stack_config.get('metadata', 'type')
            if stack_type == 'extruder_train':
                extruder_stack_file_list.append(file_name)
            elif stack_type == 'machine':
                global_stack_file_list.append(file_name)
            else:
                Logger.log('w', "Unknown container stack type '%s' from %s in %s", stack_type, file_name, project_file_name)
        if len(global_stack_file_list) > 1:
            Logger.log('e', 'More than one global stack file found: [{file_list}]'.format(file_list=global_stack_file_list))
        if len(global_stack_file_list) == 0:
            Logger.log('e', 'No global stack file found!')
            raise FileNotFoundError('No global stack file found!')
        return (global_stack_file_list[0], extruder_stack_file_list)

    def preRead(self, file_name, show_dialog=True, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Read some info so we can make decisions\n\n        :param file_name:\n        :param show_dialog: In case we use preRead() to check if a file is a valid project file,\n                            we don't want to show a dialog.\n        "
        self._clearState()
        self._3mf_mesh_reader = Application.getInstance().getMeshFileHandler().getReaderForFile(file_name)
        if self._3mf_mesh_reader and self._3mf_mesh_reader.preRead(file_name) == WorkspaceReader.PreReadResult.accepted:
            pass
        else:
            Logger.log('w', 'Could not find reader that was able to read the scene data for 3MF workspace')
            return WorkspaceReader.PreReadResult.failed
        self._machine_info = MachineInfo()
        machine_type = ''
        variant_type_name = i18n_catalog.i18nc('@label', 'Nozzle')
        archive = zipfile.ZipFile(file_name, 'r')
        cura_file_names = [name for name in archive.namelist() if name.startswith('Cura/')]
        resolve_strategy_keys = ['machine', 'material', 'quality_changes']
        self._resolve_strategies = {k: None for k in resolve_strategy_keys}
        containers_found_dict = {k: False for k in resolve_strategy_keys}
        machine_definition_id = None
        updatable_machines = []
        machine_definition_container_count = 0
        extruder_definition_container_count = 0
        definition_container_files = [name for name in cura_file_names if name.endswith(self._definition_container_suffix)]
        for definition_container_file in definition_container_files:
            container_id = self._stripFileToId(definition_container_file)
            definitions = self._container_registry.findDefinitionContainersMetadata(id=container_id)
            serialized = archive.open(definition_container_file).read().decode('utf-8')
            if not definitions:
                definition_container = DefinitionContainer.deserializeMetadata(serialized, container_id)[0]
            else:
                definition_container = definitions[0]
            definition_container_type = definition_container.get('type')
            if definition_container_type == 'machine':
                machine_definition_id = container_id
                machine_definition_containers = self._container_registry.findDefinitionContainers(id=machine_definition_id)
                if machine_definition_containers:
                    updatable_machines = [machine for machine in self._container_registry.findContainerStacks(type='machine') if machine.definition == machine_definition_containers[0]]
                machine_type = definition_container['name']
                variant_type_name = definition_container.get('variants_name', variant_type_name)
                machine_definition_container_count += 1
            elif definition_container_type == 'extruder':
                extruder_definition_container_count += 1
            else:
                Logger.log('w', 'Unknown definition container type %s for %s', definition_container_type, definition_container_file)
            QCoreApplication.processEvents()
            Job.yieldThread()
        if machine_definition_container_count != 1:
            return WorkspaceReader.PreReadResult.failed
        material_ids_to_names_map = {}
        material_conflict = False
        xml_material_profile = self._getXmlProfileClass()
        reverse_material_id_dict = {}
        if self._material_container_suffix is None:
            self._material_container_suffix = ContainerRegistry.getMimeTypeForContainer(xml_material_profile).preferredSuffix
        if xml_material_profile:
            material_container_files = [name for name in cura_file_names if name.endswith(self._material_container_suffix)]
            for material_container_file in material_container_files:
                container_id = self._stripFileToId(material_container_file)
                serialized = archive.open(material_container_file).read().decode('utf-8')
                metadata_list = xml_material_profile.deserializeMetadata(serialized, container_id)
                reverse_map = {metadata['id']: container_id for metadata in metadata_list}
                reverse_material_id_dict.update(reverse_map)
                material_ids_to_names_map[container_id] = self._getMaterialLabelFromSerialized(serialized)
                if self._container_registry.findContainersMetadata(id=container_id):
                    containers_found_dict['material'] = True
                    if not self._container_registry.isReadOnly(container_id):
                        material_conflict = True
                QCoreApplication.processEvents()
                Job.yieldThread()
        instance_container_files = [name for name in cura_file_names if name.endswith(self._instance_container_suffix)]
        quality_name = ''
        custom_quality_name = ''
        intent_name = ''
        intent_category = ''
        num_settings_overridden_by_quality_changes = 0
        num_user_settings = 0
        quality_changes_conflict = False
        self._machine_info.quality_changes_info = QualityChangesInfo()
        quality_changes_info_list = []
        instance_container_info_dict = {}
        for instance_container_file_name in instance_container_files:
            container_id = self._stripFileToId(instance_container_file_name)
            serialized = archive.open(instance_container_file_name).read().decode('utf-8')
            parser = ConfigParser(interpolation=None, comment_prefixes=())
            parser.read_string(serialized)
            container_type = parser['metadata']['type']
            if container_type not in ('quality', 'variant'):
                serialized = InstanceContainer._updateSerialized(serialized, instance_container_file_name)
            parser = ConfigParser(interpolation=None, comment_prefixes=())
            parser.read_string(serialized)
            container_info = ContainerInfo(instance_container_file_name, serialized, parser)
            instance_container_info_dict[container_id] = container_info
            container_type = parser['metadata']['type']
            if container_type == 'quality_changes':
                quality_changes_info_list.append(container_info)
                if not parser.has_option('metadata', 'position'):
                    self._machine_info.quality_changes_info.name = parser['general']['name']
                    self._machine_info.quality_changes_info.global_info = container_info
                else:
                    position = parser['metadata']['position']
                    self._machine_info.quality_changes_info.extruder_info_dict[position] = container_info
                custom_quality_name = parser['general']['name']
                values = parser['values'] if parser.has_section('values') else dict()
                num_settings_overridden_by_quality_changes += len(values)
                quality_changes = self._container_registry.findInstanceContainers(name=custom_quality_name, type='quality_changes')
                if quality_changes:
                    containers_found_dict['quality_changes'] = True
                    instance_container = InstanceContainer(container_id)
                    try:
                        instance_container.deserialize(serialized, file_name=instance_container_file_name)
                    except ContainerFormatError:
                        Logger.logException('e', 'Failed to deserialize InstanceContainer %s from project file %s', instance_container_file_name, file_name)
                        return ThreeMFWorkspaceReader.PreReadResult.failed
                    if quality_changes[0] != instance_container:
                        quality_changes_conflict = True
            elif container_type == 'quality':
                if not quality_name:
                    quality_name = parser['general']['name']
            elif container_type == 'intent':
                if not intent_name:
                    intent_name = parser['general']['name']
                    intent_category = parser['metadata']['intent_category']
            elif container_type == 'user':
                num_user_settings += len(parser['values'])
            elif container_type in self._ignored_instance_container_types:
                Logger.log('w', 'Ignoring instance container [%s] with type [%s]', container_id, container_type)
                continue
            QCoreApplication.processEvents()
            Job.yieldThread()
        if self._machine_info.quality_changes_info.global_info is None:
            self._machine_info.quality_changes_info = None
        try:
            (global_stack_file, extruder_stack_files) = self._determineGlobalAndExtruderStackFiles(file_name, cura_file_names)
        except FileNotFoundError:
            return WorkspaceReader.PreReadResult.failed
        machine_conflict = False
        global_stack_id = self._stripFileToId(global_stack_file)
        serialized = archive.open(global_stack_file).read().decode('utf-8')
        serialized = GlobalStack._updateSerialized(serialized, global_stack_file)
        machine_name = self._getMachineNameFromSerializedStack(serialized)
        self._machine_info.metadata_dict = self._getMetaDataDictFromSerializedStack(serialized)
        id_list = self._getContainerIdListFromSerialized(serialized)
        if id_list[7] != machine_definition_id:
            machine_definition_id = id_list[7]
        stacks = self._container_registry.findContainerStacks(name=machine_name, type='machine')
        existing_global_stack = None
        global_stack = None
        if stacks:
            global_stack = stacks[0]
            existing_global_stack = global_stack
            containers_found_dict['machine'] = True
            for (index, container_id) in enumerate(id_list):
                container_id = self._old_empty_profile_id_dict.get(container_id, container_id)
                if global_stack.getContainer(index).getId() != container_id:
                    machine_conflict = True
                    break
        if updatable_machines and (not containers_found_dict['machine']):
            containers_found_dict['machine'] = True
        parser = ConfigParser(interpolation=None)
        parser.read_string(serialized)
        quality_container_id = parser['containers'][str(_ContainerIndexes.Quality)]
        quality_type = 'empty_quality'
        if quality_container_id not in ('empty', 'empty_quality'):
            if quality_container_id in instance_container_info_dict:
                quality_type = instance_container_info_dict[quality_container_id].parser['metadata']['quality_type']
            else:
                quality_matches = ContainerRegistry.getInstance().findContainersMetadata(id=quality_container_id)
                if quality_matches:
                    quality_type = quality_matches[0]['quality_type']
        serialized = archive.open(global_stack_file).read().decode('utf-8')
        serialized = GlobalStack._updateSerialized(serialized, global_stack_file)
        parser = ConfigParser(interpolation=None)
        parser.read_string(serialized)
        definition_changes_id = parser['containers'][str(_ContainerIndexes.DefinitionChanges)]
        if definition_changes_id not in ('empty', 'empty_definition_changes'):
            self._machine_info.definition_changes_info = instance_container_info_dict[definition_changes_id]
        user_changes_id = parser['containers'][str(_ContainerIndexes.UserChanges)]
        if user_changes_id not in ('empty', 'empty_user_changes'):
            self._machine_info.user_changes_info = instance_container_info_dict[user_changes_id]
        if not extruder_stack_files:
            position = '0'
            extruder_info = ExtruderInfo()
            extruder_info.position = position
            variant_id = parser['containers'][str(_ContainerIndexes.Variant)]
            material_id = parser['containers'][str(_ContainerIndexes.Material)]
            if variant_id not in ('empty', 'empty_variant'):
                extruder_info.variant_info = instance_container_info_dict[variant_id]
            if material_id not in ('empty', 'empty_material'):
                root_material_id = reverse_material_id_dict[material_id]
                extruder_info.root_material_id = root_material_id
            self._machine_info.extruder_info_dict[position] = extruder_info
        else:
            variant_id = parser['containers'][str(_ContainerIndexes.Variant)]
            if variant_id not in ('empty', 'empty_variant'):
                self._machine_info.variant_info = instance_container_info_dict[variant_id]
        QCoreApplication.processEvents()
        Job.yieldThread()
        materials_in_extruders_dict = {}
        for extruder_stack_file in extruder_stack_files:
            serialized = archive.open(extruder_stack_file).read().decode('utf-8')
            not_upgraded_parser = ConfigParser(interpolation=None)
            not_upgraded_parser.read_string(serialized)
            serialized = ExtruderStack._updateSerialized(serialized, extruder_stack_file)
            parser = ConfigParser(interpolation=None)
            parser.read_string(serialized)
            position = parser['metadata']['position']
            variant_id = parser['containers'][str(_ContainerIndexes.Variant)]
            material_id = parser['containers'][str(_ContainerIndexes.Material)]
            extruder_info = ExtruderInfo()
            extruder_info.position = position
            if parser.has_option('metadata', 'enabled'):
                extruder_info.enabled = parser['metadata']['enabled']
            if variant_id not in ('empty', 'empty_variant'):
                if variant_id in instance_container_info_dict:
                    extruder_info.variant_info = instance_container_info_dict[variant_id]
            if material_id not in ('empty', 'empty_material'):
                root_material_id = reverse_material_id_dict[material_id]
                extruder_info.root_material_id = root_material_id
                materials_in_extruders_dict[position] = material_ids_to_names_map[reverse_material_id_dict[material_id]]
            definition_changes_id = parser['containers'][str(_ContainerIndexes.DefinitionChanges)]
            if definition_changes_id not in ('empty', 'empty_definition_changes'):
                extruder_info.definition_changes_info = instance_container_info_dict[definition_changes_id]
            user_changes_id = parser['containers'][str(_ContainerIndexes.UserChanges)]
            if user_changes_id not in ('empty', 'empty_user_changes'):
                extruder_info.user_changes_info = instance_container_info_dict[user_changes_id]
            self._machine_info.extruder_info_dict[position] = extruder_info
            intent_container_id = parser['containers'][str(_ContainerIndexes.Intent)]
            intent_id = parser['containers'][str(_ContainerIndexes.Intent)]
            if intent_id not in ('empty', 'empty_intent'):
                if intent_container_id in instance_container_info_dict:
                    extruder_info.intent_info = instance_container_info_dict[intent_id]
                else:
                    extruder_info.intent_info = instance_container_info_dict[not_upgraded_parser['containers'][str(_ContainerIndexes.Intent)]]
            if not machine_conflict and containers_found_dict['machine'] and global_stack:
                if int(position) >= len(global_stack.extruderList):
                    continue
                existing_extruder_stack = global_stack.extruderList[int(position)]
                id_list = self._getContainerIdListFromSerialized(serialized)
                for (index, container_id) in enumerate(id_list):
                    container_id = self._old_empty_profile_id_dict.get(container_id, container_id)
                    if existing_extruder_stack.getContainer(index).getId() != container_id:
                        machine_conflict = True
                        break
        material_labels = [material_name for (pos, material_name) in sorted(materials_in_extruders_dict.items())]
        machine_extruder_count = self._getMachineExtruderCount()
        if machine_extruder_count:
            material_labels = material_labels[:machine_extruder_count]
        num_visible_settings = 0
        try:
            temp_preferences = Preferences()
            serialized = archive.open('Cura/preferences.cfg').read().decode('utf-8')
            temp_preferences.deserialize(serialized)
            visible_settings_string = temp_preferences.getValue('general/visible_settings')
            has_visible_settings_string = visible_settings_string is not None
            if visible_settings_string is not None:
                num_visible_settings = len(visible_settings_string.split(';'))
            active_mode = temp_preferences.getValue('cura/active_mode')
            if not active_mode:
                active_mode = Application.getInstance().getPreferences().getValue('cura/active_mode')
        except KeyError:
            Logger.log('w', 'File %s is not a valid workspace.', file_name)
            return WorkspaceReader.PreReadResult.failed
        def_results = self._container_registry.findDefinitionContainersMetadata(id=machine_definition_id)
        if not def_results:
            message = Message(i18n_catalog.i18nc("@info:status Don't translate the XML tags <filename> or <message>!", 'Project file <filename>{0}</filename> contains an unknown machine type <message>{1}</message>. Cannot import the machine. Models will be imported instead.', file_name, machine_definition_id), title=i18n_catalog.i18nc('@info:title', 'Open Project File'), message_type=Message.MessageType.WARNING)
            message.show()
            Logger.log('i', 'Could unknown machine definition %s in project file %s, cannot import it.', self._machine_info.definition_id, file_name)
            return WorkspaceReader.PreReadResult.failed
        if not show_dialog:
            return WorkspaceReader.PreReadResult.accepted
        num_extruders = extruder_definition_container_count
        if num_extruders == 0:
            num_extruders = 1
        extruders = num_extruders * ['']
        quality_name = custom_quality_name if custom_quality_name else quality_name
        self._machine_info.container_id = global_stack_id
        self._machine_info.name = machine_name
        self._machine_info.definition_id = machine_definition_id
        self._machine_info.quality_type = quality_type
        self._machine_info.custom_quality_name = quality_name
        self._machine_info.intent_category = intent_category
        is_printer_group = False
        if machine_conflict:
            group_name = existing_global_stack.getMetaDataEntry('group_name')
            if group_name is not None:
                is_printer_group = True
                machine_name = group_name
        package_metadata = self._parse_packages_metadata(archive)
        missing_package_metadata = self._filter_missing_package_metadata(package_metadata)
        self._dialog.setMachineConflict(machine_conflict)
        self._dialog.setIsPrinterGroup(is_printer_group)
        self._dialog.setQualityChangesConflict(quality_changes_conflict)
        self._dialog.setMaterialConflict(material_conflict)
        self._dialog.setHasVisibleSettingsField(has_visible_settings_string)
        self._dialog.setNumVisibleSettings(num_visible_settings)
        self._dialog.setQualityName(quality_name)
        self._dialog.setQualityType(quality_type)
        self._dialog.setIntentName(intent_category)
        self._dialog.setNumSettingsOverriddenByQualityChanges(num_settings_overridden_by_quality_changes)
        self._dialog.setNumUserSettings(num_user_settings)
        self._dialog.setActiveMode(active_mode)
        self._dialog.setUpdatableMachines(updatable_machines)
        self._dialog.setMaterialLabels(material_labels)
        self._dialog.setMachineType(machine_type)
        self._dialog.setExtruders(extruders)
        self._dialog.setVariantType(variant_type_name)
        self._dialog.setHasObjectsOnPlate(Application.getInstance().platformActivity)
        self._dialog.setMissingPackagesMetadata(missing_package_metadata)
        self._dialog.show()
        is_networked_machine = False
        is_abstract_machine = False
        if global_stack and isinstance(global_stack, GlobalStack):
            is_networked_machine = global_stack.hasNetworkedConnection()
            is_abstract_machine = parseBool(existing_global_stack.getMetaDataEntry('is_abstract_machine', False))
            self._dialog.setMachineToOverride(global_stack.getId())
            self._dialog.setResolveStrategy('machine', 'override')
        elif self._dialog.updatableMachinesModel.count > 0:
            machine = self._dialog.updatableMachinesModel.getItem(0)
            machine_name = machine['name']
            is_networked_machine = machine['isNetworked']
            is_abstract_machine = machine['isAbstractMachine']
            self._dialog.setMachineToOverride(machine['id'])
            self._dialog.setResolveStrategy('machine', 'override')
        else:
            machine_name = i18n_catalog.i18nc('@button', 'Create new')
            is_networked_machine = False
            is_abstract_machine = False
            self._dialog.setMachineToOverride(None)
            self._dialog.setResolveStrategy('machine', 'new')
        self._dialog.setIsNetworkedMachine(is_networked_machine)
        self._dialog.setIsAbstractMachine(is_abstract_machine)
        self._dialog.setMachineName(machine_name)
        self._dialog.waitForClose()
        if self._dialog.getResult() == {}:
            return WorkspaceReader.PreReadResult.cancelled
        self._resolve_strategies = self._dialog.getResult()
        for (key, strategy) in self._resolve_strategies.items():
            if key not in containers_found_dict or strategy is not None:
                continue
            self._resolve_strategies[key] = 'override' if containers_found_dict[key] else 'new'
        return WorkspaceReader.PreReadResult.accepted

    @call_on_qt_thread
    def read(self, file_name):
        if False:
            while True:
                i = 10
        'Read the project file\n\n        Add all the definitions / materials / quality changes that do not exist yet. Then it loads\n        all the stacks into the container registry. In some cases it will reuse the container for the global stack.\n        It handles old style project files containing .stack.cfg as well as new style project files\n        containing global.cfg / extruder.cfg\n\n        :param file_name:\n        '
        application = CuraApplication.getInstance()
        try:
            archive = zipfile.ZipFile(file_name, 'r')
        except EnvironmentError as e:
            message = Message(i18n_catalog.i18nc("@info:error Don't translate the XML tags <filename> or <message>!", 'Project file <filename>{0}</filename> is suddenly inaccessible: <message>{1}</message>.', file_name, str(e)), title=i18n_catalog.i18nc('@info:title', "Can't Open Project File"), message_type=Message.MessageType.ERROR)
            message.show()
            self.setWorkspaceName('')
            return ([], {})
        except zipfile.BadZipFile as e:
            message = Message(i18n_catalog.i18nc("@info:error Don't translate the XML tags <filename> or <message>!", 'Project file <filename>{0}</filename> is corrupt: <message>{1}</message>.', file_name, str(e)), title=i18n_catalog.i18nc('@info:title', "Can't Open Project File"), message_type=Message.MessageType.ERROR)
            message.show()
            self.setWorkspaceName('')
            return ([], {})
        cura_file_names = [name for name in archive.namelist() if name.startswith('Cura/')]
        temp_preferences = Preferences()
        try:
            serialized = archive.open('Cura/preferences.cfg').read().decode('utf-8')
        except KeyError as e:
            Logger.log('w', 'File %s is not a valid workspace.', file_name)
            message = Message(i18n_catalog.i18nc("@info:error Don't translate the XML tags <filename> or <message>!", 'Project file <filename>{0}</filename> is corrupt: <message>{1}</message>.', file_name, str(e)), title=i18n_catalog.i18nc('@info:title', "Can't Open Project File"), message_type=Message.MessageType.ERROR)
            message.show()
            self.setWorkspaceName('')
            return ([], {})
        temp_preferences.deserialize(serialized)
        global_preferences = application.getInstance().getPreferences()
        visible_settings = temp_preferences.getValue('general/visible_settings')
        if visible_settings is None:
            Logger.log('w', 'Workspace did not contain visible settings. Leaving visibility unchanged')
        else:
            global_preferences.setValue('general/visible_settings', visible_settings)
            global_preferences.setValue('cura/active_setting_visibility_preset', 'custom')
        categories_expanded = temp_preferences.getValue('cura/categories_expanded')
        if categories_expanded is None:
            Logger.log('w', 'Workspace did not contain expanded categories. Leaving them unchanged')
        else:
            global_preferences.setValue('cura/categories_expanded', categories_expanded)
        application.expandedCategoriesChanged.emit()
        if self._resolve_strategies['machine'] != 'override' or self._dialog.updatableMachinesModel.count == 0:
            machine_name = self._container_registry.uniqueName(self._machine_info.name)
            machine_extruder_count: Optional[int] = self._getMachineExtruderCount()
            global_stack = CuraStackBuilder.createMachine(machine_name, self._machine_info.definition_id, machine_extruder_count)
            if global_stack:
                extruder_stack_dict = {str(position): extruder for (position, extruder) in enumerate(global_stack.extruderList)}
                self._container_registry.addContainer(global_stack)
        else:
            global_stacks = self._container_registry.findContainerStacks(id=self._dialog.getMachineToOverride(), type='machine')
            if not global_stacks:
                message = Message(i18n_catalog.i18nc("@info:error Don't translate the XML tag <filename>!", 'Project file <filename>{0}</filename> is made using profiles that are unknown to this version of UltiMaker Cura.', file_name), message_type=Message.MessageType.ERROR)
                message.show()
                self.setWorkspaceName('')
                return ([], {})
            global_stack = global_stacks[0]
            extruder_stacks = self._container_registry.findContainerStacks(machine=global_stack.getId(), type='extruder_train')
            extruder_stack_dict = {stack.getMetaDataEntry('position'): stack for stack in extruder_stacks}
            for stack in extruder_stacks:
                stack.setNextStack(global_stack, connect_signals=False)
        Logger.log('d', 'Workspace loading is checking definitions...')
        definition_container_files = [name for name in cura_file_names if name.endswith(self._definition_container_suffix)]
        for definition_container_file in definition_container_files:
            container_id = self._stripFileToId(definition_container_file)
            definitions = self._container_registry.findDefinitionContainersMetadata(id=container_id)
            if not definitions:
                definition_container = DefinitionContainer(container_id)
                try:
                    definition_container.deserialize(archive.open(definition_container_file).read().decode('utf-8'), file_name=definition_container_file)
                except ContainerFormatError:
                    Logger.logException('e', 'Failed to deserialize definition file %s in project file %s', definition_container_file, file_name)
                    definition_container = self._container_registry.findDefinitionContainers(id='fdmprinter')[0]
                self._container_registry.addContainer(definition_container)
            Job.yieldThread()
            QCoreApplication.processEvents()
        Logger.log('d', 'Workspace loading is checking materials...')
        xml_material_profile = self._getXmlProfileClass()
        if self._material_container_suffix is None:
            self._material_container_suffix = ContainerRegistry.getMimeTypeForContainer(xml_material_profile).suffixes[0]
        if xml_material_profile:
            material_container_files = [name for name in cura_file_names if name.endswith(self._material_container_suffix)]
            for material_container_file in material_container_files:
                to_deserialize_material = False
                container_id = self._stripFileToId(material_container_file)
                need_new_name = False
                materials = self._container_registry.findInstanceContainers(id=container_id)
                if not materials:
                    to_deserialize_material = True
                else:
                    material_container = materials[0]
                    old_material_root_id = material_container.getMetaDataEntry('base_file')
                    if old_material_root_id is not None and (not self._container_registry.isReadOnly(old_material_root_id)):
                        to_deserialize_material = True
                        if self._resolve_strategies['material'] == 'override':
                            root_material_id = material_container.getMetaDataEntry('base_file')
                            application.getContainerRegistry().removeContainer(root_material_id)
                        elif self._resolve_strategies['material'] == 'new':
                            container_id = self.getNewId(container_id)
                            self._old_new_materials[old_material_root_id] = container_id
                            need_new_name = True
                if to_deserialize_material:
                    material_container = xml_material_profile(container_id)
                    try:
                        material_container.deserialize(archive.open(material_container_file).read().decode('utf-8'), file_name=container_id + '.' + self._material_container_suffix)
                    except ContainerFormatError:
                        Logger.logException('e', 'Failed to deserialize material file %s in project file %s', material_container_file, file_name)
                        continue
                    if need_new_name:
                        new_name = ContainerRegistry.getInstance().uniqueName(material_container.getName())
                        material_container.setName(new_name)
                    material_container.setDirty(True)
                    self._container_registry.addContainer(material_container)
                Job.yieldThread()
                QCoreApplication.processEvents()
        if global_stack:
            self._processQualityChanges(global_stack)
            self._applyChangesToMachine(global_stack, extruder_stack_dict)
            Logger.log('d', 'Workspace loading is notifying rest of the code of changes...')
            self._updateActiveMachine(global_stack)
        nodes = self._3mf_mesh_reader.read(file_name)
        if nodes is None:
            nodes = []
        base_file_name = os.path.basename(file_name)
        self.setWorkspaceName(base_file_name)
        return (nodes, self._loadMetadata(file_name))

    @staticmethod
    def _loadMetadata(file_name: str) -> Dict[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        result: Dict[str, Dict[str, Any]] = dict()
        try:
            archive = zipfile.ZipFile(file_name, 'r')
        except zipfile.BadZipFile:
            Logger.logException('w', 'Unable to retrieve metadata from {fname}: 3MF archive is corrupt.'.format(fname=file_name))
            return result
        except EnvironmentError as e:
            Logger.logException('w', 'Unable to retrieve metadata from {fname}: File is inaccessible. Error: {err}'.format(fname=file_name, err=str(e)))
            return result
        metadata_files = [name for name in archive.namelist() if name.endswith('plugin_metadata.json')]
        for metadata_file in metadata_files:
            try:
                plugin_id = metadata_file.split('/')[0]
                result[plugin_id] = json.loads(archive.open('%s/plugin_metadata.json' % plugin_id).read().decode('utf-8'))
            except Exception:
                Logger.logException('w', 'Unable to retrieve metadata for %s', metadata_file)
        return result

    def _processQualityChanges(self, global_stack):
        if False:
            while True:
                i = 10
        if self._machine_info.quality_changes_info is None:
            return
        quality_changes_name = self._machine_info.quality_changes_info.name
        if self._machine_info.quality_changes_info is not None:
            Logger.log('i', 'Loading custom profile [%s] from project file', self._machine_info.quality_changes_info.name)
            machine_definition_id_for_quality = ContainerTree.getInstance().machines[global_stack.definition.getId()].quality_definition
            machine_definition_for_quality = self._container_registry.findDefinitionContainers(id=machine_definition_id_for_quality)[0]
            quality_changes_info = self._machine_info.quality_changes_info
            quality_changes_quality_type = quality_changes_info.global_info.parser['metadata']['quality_type']
            quality_changes_intent_category_per_extruder = {position: 'default' for position in self._machine_info.extruder_info_dict}
            for (position, info) in quality_changes_info.extruder_info_dict.items():
                quality_changes_intent_category_per_extruder[position] = info.parser['metadata'].get('intent_category', 'default')
            quality_changes_name = quality_changes_info.name
            create_new = self._resolve_strategies.get('quality_changes') != 'override'
            if create_new:
                container_info_dict = {None: self._machine_info.quality_changes_info.global_info}
                container_info_dict.update(quality_changes_info.extruder_info_dict)
                quality_changes_name = self._container_registry.uniqueName(quality_changes_name)
                for (position, container_info) in container_info_dict.items():
                    extruder_stack = None
                    intent_category: Optional[str] = None
                    if position is not None:
                        try:
                            extruder_stack = global_stack.extruderList[int(position)]
                        except IndexError:
                            continue
                        intent_category = quality_changes_intent_category_per_extruder[position]
                    container = self._createNewQualityChanges(quality_changes_quality_type, intent_category, quality_changes_name, global_stack, extruder_stack)
                    container_info.container = container
                    self._container_registry.addContainer(container)
                    Logger.log('d', 'Created new quality changes container [%s]', container.getId())
            else:
                quality_changes_containers = self._container_registry.findInstanceContainers(name=quality_changes_name, type='quality_changes')
                for container in quality_changes_containers:
                    extruder_position = container.getMetaDataEntry('position')
                    if extruder_position is None:
                        quality_changes_info.global_info.container = container
                    else:
                        if extruder_position not in quality_changes_info.extruder_info_dict:
                            quality_changes_info.extruder_info_dict[extruder_position] = ContainerInfo(None, None, None)
                        container_info = quality_changes_info.extruder_info_dict[extruder_position]
                        container_info.container = container
            if not quality_changes_info.extruder_info_dict:
                container_info = ContainerInfo(None, None, None)
                quality_changes_info.extruder_info_dict['0'] = container_info
                if len(global_stack.extruderList) == 0:
                    ExtruderManager.getInstance().fixSingleExtrusionMachineExtruderDefinition(global_stack)
                try:
                    extruder_stack = global_stack.extruderList[0]
                except IndexError:
                    extruder_stack = None
                intent_category = quality_changes_intent_category_per_extruder['0']
                container = self._createNewQualityChanges(quality_changes_quality_type, intent_category, quality_changes_name, global_stack, extruder_stack)
                container_info.container = container
                self._container_registry.addContainer(container)
                Logger.log('d', 'Created new quality changes container [%s]', container.getId())
            quality_changes_info.global_info.container.clear()
            for container_info in quality_changes_info.extruder_info_dict.values():
                if container_info.container:
                    container_info.container.clear()
            global_info = quality_changes_info.global_info
            global_info.container.clear()
            for (key, value) in global_info.parser['values'].items():
                if not machine_definition_for_quality.getProperty(key, 'settable_per_extruder'):
                    global_info.container.setProperty(key, 'value', value)
                else:
                    quality_changes_info.extruder_info_dict['0'].container.setProperty(key, 'value', value)
            for (position, container_info) in quality_changes_info.extruder_info_dict.items():
                if container_info.parser is None:
                    continue
                if container_info.container is None:
                    try:
                        extruder_stack = global_stack.extruderList[int(position)]
                    except IndexError:
                        continue
                    intent_category = quality_changes_intent_category_per_extruder[position]
                    container = self._createNewQualityChanges(quality_changes_quality_type, intent_category, quality_changes_name, global_stack, extruder_stack)
                    container_info.container = container
                    self._container_registry.addContainer(container)
                for (key, value) in container_info.parser['values'].items():
                    container_info.container.setProperty(key, 'value', value)
        self._machine_info.quality_changes_info.name = quality_changes_name

    def _getMachineExtruderCount(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        "\n        Extracts the machine extruder count from the definition_changes file of the printer. If it is not specified in\n        the file, None is returned instead.\n\n        :return: The count of the machine's extruders\n        "
        machine_extruder_count = None
        if self._machine_info and self._machine_info.definition_changes_info and ('values' in self._machine_info.definition_changes_info.parser) and ('machine_extruder_count' in self._machine_info.definition_changes_info.parser['values']):
            try:
                machine_extruder_count = int(self._machine_info.definition_changes_info.parser['values']['machine_extruder_count'])
            except ValueError:
                Logger.log('w', "'machine_extruder_count' in file '{file_name}' is not a number.".format(file_name=self._machine_info.definition_changes_info.file_name))
        return machine_extruder_count

    def _createNewQualityChanges(self, quality_type: str, intent_category: Optional[str], name: str, global_stack: GlobalStack, extruder_stack: Optional[ExtruderStack]) -> InstanceContainer:
        if False:
            while True:
                i = 10
        "Helper class to create a new quality changes profile.\n\n        This will then later be filled with the appropriate data.\n\n        :param quality_type: The quality type of the new profile.\n        :param intent_category: The intent category of the new profile.\n        :param name: The name for the profile. This will later be made unique so\n            it doesn't need to be unique yet.\n        :param global_stack: The global stack showing the configuration that the\n            profile should be created for.\n        :param extruder_stack: The extruder stack showing the configuration that\n            the profile should be created for. If this is None, it will be created\n            for the global stack.\n        "
        container_registry = CuraApplication.getInstance().getContainerRegistry()
        base_id = global_stack.definition.getId() if extruder_stack is None else extruder_stack.getId()
        new_id = base_id + '_' + name
        new_id = new_id.lower().replace(' ', '_')
        new_id = container_registry.uniqueName(new_id)
        quality_changes = InstanceContainer(new_id)
        quality_changes.setName(name)
        quality_changes.setMetaDataEntry('type', 'quality_changes')
        quality_changes.setMetaDataEntry('quality_type', quality_type)
        if intent_category is not None:
            quality_changes.setMetaDataEntry('intent_category', intent_category)
        if extruder_stack is not None:
            quality_changes.setMetaDataEntry('position', extruder_stack.getMetaDataEntry('position'))
        machine_definition_id = ContainerTree.getInstance().machines[global_stack.definition.getId()].quality_definition
        quality_changes.setDefinition(machine_definition_id)
        quality_changes.setMetaDataEntry('setting_version', CuraApplication.getInstance().SettingVersion)
        quality_changes.setDirty(True)
        return quality_changes

    @staticmethod
    def _clearStack(stack):
        if False:
            return 10
        application = CuraApplication.getInstance()
        stack.definitionChanges.clear()
        stack.variant = application.empty_variant_container
        stack.material = application.empty_material_container
        stack.quality = application.empty_quality_container
        stack.qualityChanges = application.empty_quality_changes_container
        stack.userChanges.clear()

    def _applyDefinitionChanges(self, global_stack, extruder_stack_dict):
        if False:
            return 10
        values_to_set_for_extruders = {}
        if self._machine_info.definition_changes_info is not None:
            parser = self._machine_info.definition_changes_info.parser
            for (key, value) in parser['values'].items():
                if global_stack.getProperty(key, 'settable_per_extruder'):
                    values_to_set_for_extruders[key] = value
                elif not self._settingIsFromMissingPackage(key, value):
                    global_stack.definitionChanges.setProperty(key, 'value', value)
        for (position, extruder_stack) in extruder_stack_dict.items():
            if position not in self._machine_info.extruder_info_dict:
                continue
            extruder_info = self._machine_info.extruder_info_dict[position]
            if extruder_info.definition_changes_info is None:
                continue
            parser = extruder_info.definition_changes_info.parser
            for (key, value) in values_to_set_for_extruders.items():
                extruder_stack.definitionChanges.setProperty(key, 'value', value)
            if parser is not None:
                for (key, value) in parser['values'].items():
                    if not self._settingIsFromMissingPackage(key, value):
                        extruder_stack.definitionChanges.setProperty(key, 'value', value)

    def _applyUserChanges(self, global_stack, extruder_stack_dict):
        if False:
            i = 10
            return i + 15
        values_to_set_for_extruder_0 = {}
        if self._machine_info.user_changes_info is not None:
            parser = self._machine_info.user_changes_info.parser
            for (key, value) in parser['values'].items():
                if global_stack.getProperty(key, 'settable_per_extruder'):
                    values_to_set_for_extruder_0[key] = value
                elif not self._settingIsFromMissingPackage(key, value):
                    global_stack.userChanges.setProperty(key, 'value', value)
        for (position, extruder_stack) in extruder_stack_dict.items():
            if position not in self._machine_info.extruder_info_dict:
                continue
            extruder_info = self._machine_info.extruder_info_dict[position]
            if extruder_info.user_changes_info is not None:
                parser = self._machine_info.extruder_info_dict[position].user_changes_info.parser
                if position == '0':
                    for (key, value) in values_to_set_for_extruder_0.items():
                        extruder_stack.userChanges.setProperty(key, 'value', value)
                if parser is not None:
                    for (key, value) in parser['values'].items():
                        if not self._settingIsFromMissingPackage(key, value):
                            extruder_stack.userChanges.setProperty(key, 'value', value)

    def _applyVariants(self, global_stack, extruder_stack_dict):
        if False:
            while True:
                i = 10
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        if self._machine_info.variant_info is not None:
            variant_name = self._machine_info.variant_info.parser['general']['name']
            if variant_name in machine_node.variants:
                global_stack.variant = machine_node.variants[variant_name].container
            else:
                Logger.log('w', "Could not find global variant '{0}'.".format(variant_name))
        for (position, extruder_stack) in extruder_stack_dict.items():
            if position not in self._machine_info.extruder_info_dict:
                continue
            extruder_info = self._machine_info.extruder_info_dict[position]
            if extruder_info.variant_info is None:
                node = machine_node.variants.get(machine_node.preferred_variant_name, next(iter(machine_node.variants.values())))
            else:
                variant_name = extruder_info.variant_info.parser['general']['name']
                node = ContainerTree.getInstance().machines[global_stack.definition.getId()].variants[variant_name]
            extruder_stack.variant = node.container

    def _applyMaterials(self, global_stack, extruder_stack_dict):
        if False:
            for i in range(10):
                print('nop')
        machine_node = ContainerTree.getInstance().machines[global_stack.definition.getId()]
        for (position, extruder_stack) in extruder_stack_dict.items():
            if position not in self._machine_info.extruder_info_dict:
                continue
            extruder_info = self._machine_info.extruder_info_dict[position]
            if extruder_info.root_material_id is None:
                continue
            root_material_id = extruder_info.root_material_id
            root_material_id = self._old_new_materials.get(root_material_id, root_material_id)
            material_node = machine_node.variants[extruder_stack.variant.getName()].materials[root_material_id]
            extruder_stack.material = material_node.container

    def _applyChangesToMachine(self, global_stack, extruder_stack_dict):
        if False:
            return 10
        self._clearStack(global_stack)
        for extruder_stack in extruder_stack_dict.values():
            self._clearStack(extruder_stack)
        self._applyDefinitionChanges(global_stack, extruder_stack_dict)
        self._applyUserChanges(global_stack, extruder_stack_dict)
        self._applyVariants(global_stack, extruder_stack_dict)
        self._applyMaterials(global_stack, extruder_stack_dict)
        self._quality_changes_to_apply = None
        self._quality_type_to_apply = None
        self._intent_category_to_apply = None
        if self._machine_info.quality_changes_info is not None:
            self._quality_changes_to_apply = self._machine_info.quality_changes_info.name
        else:
            self._quality_type_to_apply = self._machine_info.quality_type
            self._intent_category_to_apply = self._machine_info.intent_category
        for (position, extruder_stack) in extruder_stack_dict.items():
            extruder_info = self._machine_info.extruder_info_dict.get(position)
            if not extruder_info:
                continue
            if 'enabled' not in extruder_stack.getMetaData():
                extruder_stack.setMetaDataEntry('enabled', 'True')
            extruder_stack.setMetaDataEntry('enabled', str(extruder_info.enabled))
        for (key, value) in self._machine_info.metadata_dict.items():
            if key not in _ignored_machine_network_metadata:
                global_stack.setMetaDataEntry(key, value)

    def _settingIsFromMissingPackage(self, key, value):
        if False:
            while True:
                i = 10
        for package in self._dialog.missingPackages:
            if value.startswith('PLUGIN::'):
                if package['id'] + '@' + package['package_version'] in value:
                    Logger.log('w', f'Ignoring {key} value {value} from missing package')
                    return True
        return False

    def _updateActiveMachine(self, global_stack):
        if False:
            for i in range(10):
                print('nop')
        machine_manager = Application.getInstance().getMachineManager()
        container_tree = ContainerTree.getInstance()
        machine_manager.setActiveMachine(global_stack.getId())
        for (key, value) in self._machine_info.metadata_dict.items():
            if key not in global_stack.getMetaData() and key not in _ignored_machine_network_metadata:
                global_stack.setMetaDataEntry(key, value)
        if self._quality_changes_to_apply:
            quality_changes_group_list = container_tree.getCurrentQualityChangesGroups()
            quality_changes_group = next((qcg for qcg in quality_changes_group_list if qcg.name == self._quality_changes_to_apply), None)
            if not quality_changes_group:
                Logger.log('e', 'Could not find quality_changes [%s]', self._quality_changes_to_apply)
                return
            machine_manager.setQualityChangesGroup(quality_changes_group, no_dialog=True)
        else:
            self._quality_type_to_apply = self._quality_type_to_apply.lower() if self._quality_type_to_apply else None
            quality_group_dict = container_tree.getCurrentQualityGroups()
            if self._quality_type_to_apply in quality_group_dict:
                quality_group = quality_group_dict[self._quality_type_to_apply]
            else:
                Logger.log('i', 'Could not find quality type [%s], switch to default', self._quality_type_to_apply)
                preferred_quality_type = global_stack.getMetaDataEntry('preferred_quality_type')
                quality_group = quality_group_dict.get(preferred_quality_type)
                if quality_group is None:
                    Logger.log('e', 'Could not get preferred quality type [%s]', preferred_quality_type)
            if quality_group is not None:
                machine_manager.setQualityGroup(quality_group, no_dialog=True)
                available_intent_category_list = IntentManager.getInstance().currentAvailableIntentCategories()
                if self._intent_category_to_apply is not None and self._intent_category_to_apply in available_intent_category_list:
                    machine_manager.setIntentByCategory(self._intent_category_to_apply)
                else:
                    machine_manager.resetIntents()
        global_stack.containersChanged.emit(global_stack.getTop())

    @staticmethod
    def _stripFileToId(file):
        if False:
            return 10
        mime_type = MimeTypeDatabase.getMimeTypeForFile(file)
        file = mime_type.stripExtension(file)
        return file.replace('Cura/', '')

    def _getXmlProfileClass(self):
        if False:
            print('Hello World!')
        return self._container_registry.getContainerForMimeType(MimeTypeDatabase.getMimeType('application/x-ultimaker-material-profile'))

    @staticmethod
    def _getContainerIdListFromSerialized(serialized):
        if False:
            while True:
                i = 10
        "Get the list of ID's of all containers in a container stack by partially parsing it's serialized data."
        parser = ConfigParser(interpolation=None, empty_lines_in_values=False)
        parser.read_string(serialized)
        container_ids = []
        if 'containers' in parser:
            for (index, container_id) in parser.items('containers'):
                container_ids.append(container_id)
        elif parser.has_option('general', 'containers'):
            container_string = parser['general'].get('containers', '')
            container_list = container_string.split(',')
            container_ids = [container_id for container_id in container_list if container_id != '']
        if len(container_ids) == 6:
            container_ids.insert(5, 'empty')
        return container_ids

    @staticmethod
    def _getMachineNameFromSerializedStack(serialized):
        if False:
            for i in range(10):
                print('nop')
        parser = ConfigParser(interpolation=None, empty_lines_in_values=False)
        parser.read_string(serialized)
        return parser['general'].get('name', '')

    @staticmethod
    def _getMetaDataDictFromSerializedStack(serialized: str) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        parser = ConfigParser(interpolation=None, empty_lines_in_values=False)
        parser.read_string(serialized)
        return dict(parser['metadata'])

    @staticmethod
    def _getMaterialLabelFromSerialized(serialized):
        if False:
            return 10
        data = ET.fromstring(serialized)
        metadata = data.iterfind('./um:metadata/um:name/um:label', {'um': 'http://www.ultimaker.com/material'})
        for entry in metadata:
            return entry.text

    @staticmethod
    def _parse_packages_metadata(archive: zipfile.ZipFile) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        try:
            package_metadata = json.loads(archive.open('Cura/packages.json').read().decode('utf-8'))
            return package_metadata['packages']
        except KeyError:
            Logger.warning('No package metadata was found in .3mf file.')
        except Exception:
            Logger.error('Failed to load packages metadata from .3mf file.')
        return []

    @staticmethod
    def _filter_missing_package_metadata(package_metadata: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if False:
            i = 10
            return i + 15
        'Filters out installed packages from package_metadata'
        missing_packages = []
        package_manager = cast(CuraPackageManager, CuraApplication.getInstance().getPackageManager())
        for package in package_metadata:
            package_id = package['id']
            if not package_manager.isPackageInstalled(package_id):
                missing_packages.append(package)
        return missing_packages