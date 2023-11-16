import os
import re
import configparser
from typing import Any, cast, Dict, Optional, List, Union, Tuple
from PyQt6.QtWidgets import QMessageBox
from UM.Decorators import override
from UM.Settings.ContainerFormatError import ContainerFormatError
from UM.Settings.Interfaces import ContainerInterface
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.SettingInstance import SettingInstance
from UM.Logger import Logger
from UM.Message import Message
from UM.Platform import Platform
from UM.PluginRegistry import PluginRegistry
from UM.Resources import Resources
from UM.Util import parseBool
from cura.ReaderWriters.ProfileWriter import ProfileWriter
from . import ExtruderStack
from . import GlobalStack
import cura.CuraApplication
from cura.Settings.cura_empty_instance_containers import empty_quality_container
from cura.Machines.ContainerTree import ContainerTree
from cura.ReaderWriters.ProfileReader import NoProfileException, ProfileReader
from UM.i18n import i18nCatalog
from .DatabaseHandlers.IntentDatabaseHandler import IntentDatabaseHandler
from .DatabaseHandlers.QualityDatabaseHandler import QualityDatabaseHandler
from .DatabaseHandlers.VariantDatabaseHandler import VariantDatabaseHandler
catalog = i18nCatalog('cura')

class CuraContainerRegistry(ContainerRegistry):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.containerAdded.connect(self._onContainerAdded)
        self._database_handlers['variant'] = VariantDatabaseHandler()
        self._database_handlers['quality'] = QualityDatabaseHandler()
        self._database_handlers['intent'] = IntentDatabaseHandler()

    @override(ContainerRegistry)
    def addContainer(self, container: ContainerInterface) -> bool:
        if False:
            return 10
        'Overridden from ContainerRegistry\n\n        Adds a container to the registry.\n\n        This will also try to convert a ContainerStack to either Extruder or\n        Global stack based on metadata information.\n        '
        if type(container) == ContainerStack:
            container = self._convertContainerStack(cast(ContainerStack, container))
        if isinstance(container, InstanceContainer) and type(container) != type(self.getEmptyInstanceContainer()):
            required_setting_version = cura.CuraApplication.CuraApplication.SettingVersion
            actual_setting_version = int(container.getMetaDataEntry('setting_version', default=0))
            if required_setting_version != actual_setting_version:
                Logger.log('w', 'Instance container {container_id} is outdated. Its setting version is {actual_setting_version} but it should be {required_setting_version}.'.format(container_id=container.getId(), actual_setting_version=actual_setting_version, required_setting_version=required_setting_version))
                return False
        return super().addContainer(container)

    def createUniqueName(self, container_type: str, current_name: str, new_name: str, fallback_name: str) -> str:
        if False:
            i = 10
            return i + 15
        'Create a name that is not empty and unique\n\n        :param container_type: :type{string} Type of the container (machine, quality, ...)\n        :param current_name: :type{} Current name of the container, which may be an acceptable option\n        :param new_name: :type{string} Base name, which may not be unique\n        :param fallback_name: :type{string} Name to use when (stripped) new_name is empty\n        :return: :type{string} Name that is unique for the specified type and name/id\n        '
        new_name = new_name.strip()
        num_check = re.compile('(.*?)\\s*#\\d+$').match(new_name)
        if num_check:
            new_name = num_check.group(1)
        if new_name == '':
            new_name = fallback_name
        unique_name = new_name
        i = 1
        while self._containerExists(container_type, unique_name) and unique_name != current_name:
            i += 1
            unique_name = '%s #%d' % (new_name, i)
        return unique_name

    def _containerExists(self, container_type: str, container_name: str):
        if False:
            return 10
        'Check if a container with of a certain type and a certain name or id exists\n\n        Both the id and the name are checked, because they may not be the same and it is better if they are both unique\n        :param container_type: :type{string} Type of the container (machine, quality, ...)\n        :param container_name: :type{string} Name to check\n        '
        container_class = ContainerStack if 'machine' in container_type else InstanceContainer
        return self.findContainersMetadata(container_type=container_class, id=container_name, type=container_type, ignore_case=True) or self.findContainersMetadata(container_type=container_class, name=container_name, type=container_type)

    def exportQualityProfile(self, container_list: List[InstanceContainer], file_name: str, file_type: str) -> bool:
        if False:
            while True:
                i = 10
        'Exports an profile to a file\n\n        :param container_list: :type{list} the containers to export. This is not\n        necessarily in any order!\n        :param file_name: :type{str} the full path and filename to export to.\n        :param file_type: :type{str} the file type with the format "<description> (*.<extension>)"\n        :return: True if the export succeeded, false otherwise.\n        '
        split = file_type.rfind(' (*.')
        if split < 0:
            Logger.log('e', 'Invalid file format identifier %s', file_type)
            return False
        description = file_type[:split]
        extension = file_type[split + 4:-1]
        if not file_name.endswith('.' + extension):
            file_name += '.' + extension
        if not Platform.isWindows():
            if os.path.exists(file_name):
                result = QMessageBox.question(None, catalog.i18nc('@title:window', 'File Already Exists'), catalog.i18nc("@label Don't translate the XML tag <filename>!", 'The file <filename>{0}</filename> already exists. Are you sure you want to overwrite it?').format(file_name))
                if result == QMessageBox.StandardButton.No:
                    return False
        profile_writer = self._findProfileWriter(extension, description)
        try:
            if profile_writer is None:
                raise Exception('Unable to find a profile writer')
            success = profile_writer.write(file_name, container_list)
        except Exception as e:
            Logger.log('e', 'Failed to export profile to %s: %s', file_name, str(e))
            m = Message(catalog.i18nc("@info:status Don't translate the XML tags <filename> or <message>!", 'Failed to export profile to <filename>{0}</filename>: <message>{1}</message>', file_name, str(e)), lifetime=0, title=catalog.i18nc('@info:title', 'Error'), message_type=Message.MessageType.ERROR)
            m.show()
            return False
        if not success:
            Logger.log('w', 'Failed to export profile to %s: Writer plugin reported failure.', file_name)
            m = Message(catalog.i18nc("@info:status Don't translate the XML tag <filename>!", 'Failed to export profile to <filename>{0}</filename>: Writer plugin reported failure.', file_name), lifetime=0, title=catalog.i18nc('@info:title', 'Error'), message_type=Message.MessageType.ERROR)
            m.show()
            return False
        m = Message(catalog.i18nc("@info:status Don't translate the XML tag <filename>!", 'Exported profile to <filename>{0}</filename>', file_name), title=catalog.i18nc('@info:title', 'Export succeeded'), message_type=Message.MessageType.POSITIVE)
        m.show()
        return True

    def _findProfileWriter(self, extension: str, description: str) -> Optional[ProfileWriter]:
        if False:
            print('Hello World!')
        'Gets the plugin object matching the criteria\n\n        :param extension:\n        :param description:\n        :return: The plugin object matching the given extension and description.\n        '
        plugin_registry = PluginRegistry.getInstance()
        for (plugin_id, meta_data) in self._getIOPlugins('profile_writer'):
            for supported_type in meta_data['profile_writer']:
                supported_extension = supported_type.get('extension', None)
                if supported_extension == extension:
                    supported_description = supported_type.get('description', None)
                    if supported_description == description:
                        return cast(ProfileWriter, plugin_registry.getPluginObject(plugin_id))
        return None

    def importProfile(self, file_name: str) -> Dict[str, str]:
        if False:
            return 10
        "Imports a profile from a file\n\n        :param file_name: The full path and filename of the profile to import.\n        :return: Dict with a 'status' key containing the string 'ok', 'warning' or 'error',\n            and a 'message' key containing a message for the user.\n        "
        Logger.log('d', 'Attempting to import profile %s', file_name)
        if not file_name:
            return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", 'Failed to import profile from <filename>{0}</filename>: {1}', file_name, 'Invalid path')}
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack:
            return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", "Can't import profile from <filename>{0}</filename> before a printer is added.", file_name)}
        container_tree = ContainerTree.getInstance()
        machine_extruders = global_stack.extruderList
        plugin_registry = PluginRegistry.getInstance()
        extension = file_name.split('.')[-1]
        for (plugin_id, meta_data) in self._getIOPlugins('profile_reader'):
            if meta_data['profile_reader'][0]['extension'] != extension:
                continue
            profile_reader = cast(ProfileReader, plugin_registry.getPluginObject(plugin_id))
            try:
                profile_or_list = profile_reader.read(file_name)
            except NoProfileException:
                return {'status': 'ok', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", 'No custom profile to import in file <filename>{0}</filename>', file_name)}
            except Exception as e:
                Logger.log('e', 'Failed to import profile from %s: %s while using profile reader. Got exception %s', file_name, profile_reader.getPluginId(), str(e))
                return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", 'Failed to import profile from <filename>{0}</filename>:', file_name) + '\n<message>' + str(e) + '</message>'}
            if profile_or_list:
                if not isinstance(profile_or_list, list):
                    profile_or_list = [profile_or_list]
                global_profile = None
                extruder_profiles = []
                if len(profile_or_list) == 1:
                    global_profile = profile_or_list[0]
                else:
                    for profile in profile_or_list:
                        if not profile.getMetaDataEntry('position'):
                            global_profile = profile
                        else:
                            extruder_profiles.append(profile)
                extruder_profiles = sorted(extruder_profiles, key=lambda x: int(x.getMetaDataEntry('position', default='0')))
                profile_or_list = [global_profile] + extruder_profiles
                if not global_profile:
                    Logger.log('e', 'Incorrect profile [%s]. Could not find global profile', file_name)
                    return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", 'This profile <filename>{0}</filename> contains incorrect data, could not import it.', file_name)}
                profile_definition = global_profile.getMetaDataEntry('definition')
                if profile_definition is None:
                    break
                machine_definitions = self.findContainers(id=profile_definition)
                if not machine_definitions:
                    Logger.log('e', 'Incorrect profile [%s]. Unknown machine type [%s]', file_name, profile_definition)
                    return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tags <filename>!", 'This profile <filename>{0}</filename> contains incorrect data, could not import it.', file_name)}
                machine_definition = machine_definitions[0]
                has_machine_quality = parseBool(machine_definition.getMetaDataEntry('has_machine_quality', 'false'))
                profile_definition = machine_definition.getMetaDataEntry('quality_definition', machine_definition.getId()) if has_machine_quality else 'fdmprinter'
                expected_machine_definition = container_tree.machines[global_stack.definition.getId()].quality_definition
                if profile_definition != expected_machine_definition:
                    Logger.log('d', "Profile {file_name} is for machine {profile_definition}, but the current active machine is {expected_machine_definition}. Changing profile's definition.".format(file_name=file_name, profile_definition=profile_definition, expected_machine_definition=expected_machine_definition))
                    global_profile.setMetaDataEntry('definition', expected_machine_definition)
                    for extruder_profile in extruder_profiles:
                        extruder_profile.setMetaDataEntry('definition', expected_machine_definition)
                quality_name = global_profile.getName()
                quality_type = global_profile.getMetaDataEntry('quality_type')
                name_seed = os.path.splitext(os.path.basename(file_name))[0]
                new_name = self.uniqueName(name_seed)
                if type(profile_or_list) is not list:
                    profile_or_list = [profile_or_list]
                if len(profile_or_list) == 1:
                    global_profile = profile_or_list[0]
                    extruder_profiles = []
                    for (idx, extruder) in enumerate(global_stack.extruderList):
                        profile_id = ContainerRegistry.getInstance().uniqueName(global_stack.getId() + '_extruder_' + str(idx + 1))
                        profile = InstanceContainer(profile_id)
                        profile.setName(quality_name)
                        profile.setMetaDataEntry('setting_version', cura.CuraApplication.CuraApplication.SettingVersion)
                        profile.setMetaDataEntry('type', 'quality_changes')
                        profile.setMetaDataEntry('definition', expected_machine_definition)
                        profile.setMetaDataEntry('quality_type', quality_type)
                        profile.setDirty(True)
                        if idx == 0:
                            for qc_setting_key in global_profile.getAllKeys():
                                settable_per_extruder = global_stack.getProperty(qc_setting_key, 'settable_per_extruder')
                                if settable_per_extruder:
                                    setting_value = global_profile.getProperty(qc_setting_key, 'value')
                                    setting_definition = global_stack.getSettingDefinition(qc_setting_key)
                                    if setting_definition is not None:
                                        new_instance = SettingInstance(setting_definition, profile)
                                        new_instance.setProperty('value', setting_value)
                                        new_instance.resetState()
                                        profile.addInstance(new_instance)
                                        profile.setDirty(True)
                                    global_profile.removeInstance(qc_setting_key, postpone_emit=True)
                        extruder_profiles.append(profile)
                    for profile in extruder_profiles:
                        profile_or_list.append(profile)
                profile_ids_added = []
                additional_message = None
                for (profile_index, profile) in enumerate(profile_or_list):
                    if profile_index == 0:
                        profile_id = (cast(ContainerInterface, global_stack.getBottom()).getId() + '_' + name_seed).lower().replace(' ', '_')
                    elif profile_index < len(machine_extruders) + 1:
                        extruder_id = machine_extruders[profile_index - 1].definition.getId()
                        extruder_position = str(profile_index - 1)
                        if not profile.getMetaDataEntry('position'):
                            profile.setMetaDataEntry('position', extruder_position)
                        else:
                            profile.setMetaDataEntry('position', extruder_position)
                        profile_id = (extruder_id + '_' + name_seed).lower().replace(' ', '_')
                    else:
                        continue
                    (configuration_successful, message) = self._configureProfile(profile, profile_id, new_name, expected_machine_definition)
                    if configuration_successful:
                        additional_message = message
                    else:
                        for profile_id in profile_ids_added + [profile.getId()]:
                            self.removeContainer(profile_id)
                        if not message:
                            message = ''
                        return {'status': 'error', 'message': catalog.i18nc("@info:status Don't translate the XML tag <filename>!", 'Failed to import profile from <filename>{0}</filename>:', file_name) + ' ' + message}
                    profile_ids_added.append(profile.getId())
                result_status = 'ok'
                success_message = catalog.i18nc('@info:status', 'Successfully imported profile {0}.', profile_or_list[0].getName())
                if additional_message:
                    result_status = 'warning'
                    success_message += additional_message
                return {'status': result_status, 'message': success_message}
            return {'status': 'error', 'message': catalog.i18nc('@info:status', 'File {0} does not contain any valid profile.', file_name)}
        return {'status': 'error', 'message': catalog.i18nc('@info:status', 'Profile {0} has an unknown file type or is corrupted.', file_name)}

    @override(ContainerRegistry)
    def load(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().load()
        self._registerSingleExtrusionMachinesExtruderStacks()
        self._connectUpgradedExtruderStacksToMachines()

    @override(ContainerRegistry)
    def loadAllMetadata(self) -> None:
        if False:
            print('Hello World!')
        super().loadAllMetadata()
        self._cleanUpInvalidQualityChanges()

    def _cleanUpInvalidQualityChanges(self) -> None:
        if False:
            return 10
        quality_changes = ContainerRegistry.getInstance().findContainersMetadata(type='quality_changes')
        profile_count_by_name = {}
        for quality_change in quality_changes:
            name = str(quality_change.get('name', ''))
            if name == 'empty':
                continue
            if name not in profile_count_by_name:
                profile_count_by_name[name] = 0
            profile_count_by_name[name] += 1
        for (profile_name, profile_count) in profile_count_by_name.items():
            if profile_count > 1:
                continue
            invalid_quality_changes = ContainerRegistry.getInstance().findContainersMetadata(name=profile_name)
            if invalid_quality_changes:
                Logger.log('d', 'Found an invalid quality_changes profile with the name %s. Going to remove that now', profile_name)
                self.removeContainer(invalid_quality_changes[0]['id'])

    @override(ContainerRegistry)
    def _isMetadataValid(self, metadata: Optional[Dict[str, Any]]) -> bool:
        if False:
            return 10
        'Check if the metadata for a container is okay before adding it.\n\n        This overrides the one from UM.Settings.ContainerRegistry because we\n        also require that the setting_version is correct.\n        '
        if metadata is None:
            return False
        if 'setting_version' not in metadata:
            return False
        try:
            if int(metadata['setting_version']) != cura.CuraApplication.CuraApplication.SettingVersion:
                return False
        except ValueError:
            return False
        except TypeError:
            return False
        return True

    def _configureProfile(self, profile: InstanceContainer, id_seed: str, new_name: str, machine_definition_id: str) -> Tuple[bool, Optional[str]]:
        if False:
            return 10
        'Update an imported profile to match the current machine configuration.\n\n        :param profile: The profile to configure.\n        :param id_seed: The base ID for the profile. May be changed so it does not conflict with existing containers.\n        :param new_name: The new name for the profile.\n\n        :returns: tuple (configuration_successful, message)\n                WHERE\n                bool configuration_successful: Whether the process of configuring the profile was successful\n                optional str message: A message indicating the outcome of configuring the profile. If the configuration\n                                      is successful, this message can be None or contain a warning\n        '
        profile.setDirty(True)
        new_id = self.createUniqueName('quality_changes', '', id_seed, catalog.i18nc('@label', 'Custom profile'))
        profile.setMetaDataEntry('id', new_id)
        profile.setName(new_name)
        profile.setMetaDataEntry('id', new_id)
        profile.setMetaDataEntry('definition', machine_definition_id)
        if 'type' in profile.getMetaData():
            profile.setMetaDataEntry('type', 'quality_changes')
        else:
            profile.setMetaDataEntry('type', 'quality_changes')
        quality_type = profile.getMetaDataEntry('quality_type')
        if not quality_type:
            return (False, catalog.i18nc('@info:status', 'Profile is missing a quality type.'))
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if not global_stack:
            return (False, catalog.i18nc('@info:status', 'There is no active printer yet.'))
        definition_id = ContainerTree.getInstance().machines[global_stack.definition.getId()].quality_definition
        profile.setDefinition(definition_id)
        if not self.addContainer(profile):
            return (False, catalog.i18nc('@info:status', 'Unable to add the profile.'))
        if quality_type == empty_quality_container.getMetaDataEntry('quality_type'):
            return (True, None)
        available_quality_groups_dict = {name: quality_group for (name, quality_group) in ContainerTree.getInstance().getCurrentQualityGroups().items() if quality_group.is_available}
        all_quality_groups_dict = ContainerTree.getInstance().getCurrentQualityGroups()
        if quality_type not in all_quality_groups_dict:
            return (False, catalog.i18nc('@info:status', "Quality type '{0}' is not compatible with the current active machine definition '{1}'.", quality_type, definition_id))
        if quality_type not in available_quality_groups_dict:
            return (True, '\n\n' + catalog.i18nc('@info:status', "Warning: The profile is not visible because its quality type '{0}' is not available for the current configuration. Switch to a material/nozzle combination that can use this quality type.", quality_type))
        return (True, None)

    @override(ContainerRegistry)
    def saveDirtyContainers(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.lockFile():
            for instance in self.findDirtyContainers(container_type=InstanceContainer):
                if instance.getMetaDataEntry('removed'):
                    continue
                if instance.getId() == instance.getMetaData().get('base_file'):
                    self.saveContainer(instance)
            for instance in self.findDirtyContainers(container_type=InstanceContainer):
                if instance.getMetaDataEntry('removed'):
                    continue
                self.saveContainer(instance)
            for stack in self.findContainerStacks():
                self.saveContainer(stack)

    def _getIOPlugins(self, io_type):
        if False:
            i = 10
            return i + 15
        'Gets a list of profile writer plugins\n\n        :return: List of tuples of (plugin_id, meta_data).\n        '
        plugin_registry = PluginRegistry.getInstance()
        active_plugin_ids = plugin_registry.getActivePlugins()
        result = []
        for plugin_id in active_plugin_ids:
            meta_data = plugin_registry.getMetaData(plugin_id)
            if io_type in meta_data:
                result.append((plugin_id, meta_data))
        return result

    def _convertContainerStack(self, container: ContainerStack) -> Union[ExtruderStack.ExtruderStack, GlobalStack.GlobalStack]:
        if False:
            while True:
                i = 10
        'Convert an "old-style" pure ContainerStack to either an Extruder or Global stack.'
        assert type(container) == ContainerStack
        container_type = container.getMetaDataEntry('type')
        if container_type not in ('extruder_train', 'machine'):
            return container
        Logger.log('d', 'Converting ContainerStack {stack} to {type}', stack=container.getId(), type=container_type)
        if container_type == 'extruder_train':
            new_stack = ExtruderStack.ExtruderStack(container.getId())
        else:
            new_stack = GlobalStack.GlobalStack(container.getId())
        container_contents = container.serialize()
        new_stack.deserialize(container_contents)
        if os.path.isfile(container.getPath()):
            os.remove(container.getPath())
        return new_stack

    def _registerSingleExtrusionMachinesExtruderStacks(self) -> None:
        if False:
            i = 10
            return i + 15
        machines = self.findContainerStacks(type='machine', machine_extruder_trains={'0': 'fdmextruder'})
        for machine in machines:
            extruder_stacks = self.findContainerStacks(type='extruder_train', machine=machine.getId())
            if not extruder_stacks:
                self.addExtruderStackForSingleExtrusionMachine(machine, 'fdmextruder')

    def _onContainerAdded(self, container: ContainerInterface) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(container, ContainerStack) or container.getMetaDataEntry('type') != 'machine':
            return
        machine_extruder_trains = container.getMetaDataEntry('machine_extruder_trains')
        if machine_extruder_trains is not None and machine_extruder_trains != {'0': 'fdmextruder'}:
            return
        extruder_stacks = self.findContainerStacks(type='extruder_train', machine=container.getId())
        if not extruder_stacks:
            self.addExtruderStackForSingleExtrusionMachine(container, 'fdmextruder')

    def addExtruderStackForSingleExtrusionMachine(self, machine, extruder_id, new_global_quality_changes=None, create_new_ids=True):
        if False:
            while True:
                i = 10
        new_extruder_id = extruder_id
        application = cura.CuraApplication.CuraApplication.getInstance()
        extruder_definitions = self.findDefinitionContainers(id=new_extruder_id)
        if not extruder_definitions:
            Logger.log('w', 'Could not find definition containers for extruder %s', new_extruder_id)
            return
        extruder_definition = extruder_definitions[0]
        unique_name = self.uniqueName(machine.getName() + ' ' + new_extruder_id) if create_new_ids else machine.getName() + ' ' + new_extruder_id
        extruder_stack = ExtruderStack.ExtruderStack(unique_name)
        extruder_stack.setName(extruder_definition.getName())
        extruder_stack.setDefinition(extruder_definition)
        extruder_stack.setMetaDataEntry('position', extruder_definition.getMetaDataEntry('position'))
        definition_changes_id = self.uniqueName(extruder_stack.getId() + '_settings') if create_new_ids else extruder_stack.getId() + '_settings'
        definition_changes_name = definition_changes_id
        definition_changes = InstanceContainer(definition_changes_id, parent=application)
        definition_changes.setName(definition_changes_name)
        definition_changes.setMetaDataEntry('setting_version', application.SettingVersion)
        definition_changes.setMetaDataEntry('type', 'definition_changes')
        definition_changes.setMetaDataEntry('definition', extruder_definition.getId())
        for setting_key in definition_changes.getAllKeys():
            if machine.definition.getProperty(setting_key, 'settable_per_extruder'):
                setting_value = machine.definitionChanges.getProperty(setting_key, 'value')
                if setting_value is not None:
                    setting_definition = machine.getSettingDefinition(setting_key)
                    new_instance = SettingInstance(setting_definition, definition_changes)
                    new_instance.setProperty('value', setting_value)
                    new_instance.resetState()
                    definition_changes.addInstance(new_instance)
                    definition_changes.setDirty(True)
                    machine.definitionChanges.removeInstance(setting_key, postpone_emit=True)
        self.addContainer(definition_changes)
        extruder_stack.setDefinitionChanges(definition_changes)
        user_container_id = self.uniqueName(extruder_stack.getId() + '_user') if create_new_ids else extruder_stack.getId() + '_user'
        user_container_name = user_container_id
        user_container = InstanceContainer(user_container_id, parent=application)
        user_container.setName(user_container_name)
        user_container.setMetaDataEntry('type', 'user')
        user_container.setMetaDataEntry('machine', machine.getId())
        user_container.setMetaDataEntry('setting_version', application.SettingVersion)
        user_container.setDefinition(machine.definition.getId())
        user_container.setMetaDataEntry('position', extruder_stack.getMetaDataEntry('position'))
        if machine.userChanges:
            for user_setting_key in machine.userChanges.getAllKeys():
                settable_per_extruder = machine.getProperty(user_setting_key, 'settable_per_extruder')
                if settable_per_extruder:
                    setting_value = machine.getProperty(user_setting_key, 'value')
                    setting_definition = machine.getSettingDefinition(user_setting_key)
                    new_instance = SettingInstance(setting_definition, definition_changes)
                    new_instance.setProperty('value', setting_value)
                    new_instance.resetState()
                    user_container.addInstance(new_instance)
                    user_container.setDirty(True)
                    machine.userChanges.removeInstance(user_setting_key, postpone_emit=True)
        self.addContainer(user_container)
        extruder_stack.setUserChanges(user_container)
        empty_variant = application.empty_variant_container
        empty_material = application.empty_material_container
        empty_quality = application.empty_quality_container
        if machine.variant.getId() not in ('empty', 'empty_variant'):
            variant = machine.variant
        else:
            variant = empty_variant
        extruder_stack.variant = variant
        if machine.material.getId() not in ('empty', 'empty_material'):
            material = machine.material
        else:
            material = empty_material
        extruder_stack.material = material
        if machine.quality.getId() not in ('empty', 'empty_quality'):
            quality = machine.quality
        else:
            quality = empty_quality
        extruder_stack.quality = quality
        machine_quality_changes = machine.qualityChanges
        if new_global_quality_changes is not None:
            machine_quality_changes = new_global_quality_changes
        if machine_quality_changes.getId() not in ('empty', 'empty_quality_changes'):
            extruder_quality_changes_container = self.findInstanceContainers(name=machine_quality_changes.getName(), extruder=extruder_id)
            if extruder_quality_changes_container:
                extruder_quality_changes_container = extruder_quality_changes_container[0]
                quality_changes_id = extruder_quality_changes_container.getId()
                extruder_stack.qualityChanges = self.findInstanceContainers(id=quality_changes_id)[0]
            else:
                extruder_quality_changes_container = self._findQualityChangesContainerInCuraFolder(machine_quality_changes.getName())
                if extruder_quality_changes_container:
                    quality_changes_id = extruder_quality_changes_container.getId()
                    extruder_quality_changes_container.setMetaDataEntry('position', extruder_definition.getMetaDataEntry('position'))
                    extruder_stack.qualityChanges = self.findInstanceContainers(id=quality_changes_id)[0]
                else:
                    container_name = machine_quality_changes.getName()
                    container_id = self.uniqueName(extruder_stack.getId() + '_qc_' + container_name)
                    extruder_quality_changes_container = InstanceContainer(container_id, parent=application)
                    extruder_quality_changes_container.setName(container_name)
                    extruder_quality_changes_container.setMetaDataEntry('type', 'quality_changes')
                    extruder_quality_changes_container.setMetaDataEntry('setting_version', application.SettingVersion)
                    extruder_quality_changes_container.setMetaDataEntry('position', extruder_definition.getMetaDataEntry('position'))
                    extruder_quality_changes_container.setMetaDataEntry('quality_type', machine_quality_changes.getMetaDataEntry('quality_type'))
                    extruder_quality_changes_container.setMetaDataEntry('intent_category', 'default')
                    extruder_quality_changes_container.setDefinition(machine_quality_changes.getDefinition().getId())
                    self.addContainer(extruder_quality_changes_container)
                    extruder_stack.qualityChanges = extruder_quality_changes_container
            if not extruder_quality_changes_container:
                Logger.log('w', 'Could not find quality_changes named [%s] for extruder [%s]', machine_quality_changes.getName(), extruder_stack.getId())
            else:
                for qc_setting_key in machine_quality_changes.getAllKeys():
                    settable_per_extruder = machine.getProperty(qc_setting_key, 'settable_per_extruder')
                    if settable_per_extruder:
                        setting_value = machine_quality_changes.getProperty(qc_setting_key, 'value')
                        setting_definition = machine.getSettingDefinition(qc_setting_key)
                        new_instance = SettingInstance(setting_definition, definition_changes)
                        new_instance.setProperty('value', setting_value)
                        new_instance.resetState()
                        extruder_quality_changes_container.addInstance(new_instance)
                        extruder_quality_changes_container.setDirty(True)
                        machine_quality_changes.removeInstance(qc_setting_key, postpone_emit=True)
        else:
            extruder_stack.qualityChanges = self.findInstanceContainers(id='empty_quality_changes')[0]
        self.addContainer(extruder_stack)
        if machine_quality_changes.getId() not in ('empty', 'empty_quality_changes'):
            quality_changes_machine_definition_id = machine_quality_changes.getDefinition().getId()
        else:
            whole_machine_definition = machine.definition
            machine_entry = machine.definition.getMetaDataEntry('machine')
            if machine_entry is not None:
                container_registry = ContainerRegistry.getInstance()
                whole_machine_definition = container_registry.findDefinitionContainers(id=machine_entry)[0]
            quality_changes_machine_definition_id = 'fdmprinter'
            if whole_machine_definition.getMetaDataEntry('has_machine_quality'):
                quality_changes_machine_definition_id = machine.definition.getMetaDataEntry('quality_definition', whole_machine_definition.getId())
        qcs = self.findInstanceContainers(type='quality_changes', definition=quality_changes_machine_definition_id)
        qc_groups = {}
        for qc in qcs:
            qc_name = qc.getName()
            if qc_name not in qc_groups:
                qc_groups[qc_name] = []
            qc_groups[qc_name].append(qc)
            quality_changes_container = self._findQualityChangesContainerInCuraFolder(machine_quality_changes.getName())
            if quality_changes_container:
                qc_groups[qc_name].append(quality_changes_container)
        for (qc_name, qc_list) in qc_groups.items():
            qc_dict = {'global': None, 'extruders': []}
            for qc in qc_list:
                extruder_position = qc.getMetaDataEntry('position')
                if extruder_position is not None:
                    qc_dict['extruders'].append(qc)
                else:
                    qc_dict['global'] = qc
            if qc_dict['global'] is not None and len(qc_dict['extruders']) == 1:
                for qc_setting_key in qc_dict['global'].getAllKeys():
                    settable_per_extruder = machine.getProperty(qc_setting_key, 'settable_per_extruder')
                    if settable_per_extruder:
                        setting_value = qc_dict['global'].getProperty(qc_setting_key, 'value')
                        setting_definition = machine.getSettingDefinition(qc_setting_key)
                        new_instance = SettingInstance(setting_definition, definition_changes)
                        new_instance.setProperty('value', setting_value)
                        new_instance.resetState()
                        qc_dict['extruders'][0].addInstance(new_instance)
                        qc_dict['extruders'][0].setDirty(True)
                        qc_dict['global'].removeInstance(qc_setting_key, postpone_emit=True)
        extruder_stack.setNextStack(machine)
        return extruder_stack

    def _findQualityChangesContainerInCuraFolder(self, name: str) -> Optional[InstanceContainer]:
        if False:
            print('Hello World!')
        quality_changes_dir = Resources.getPath(cura.CuraApplication.CuraApplication.ResourceTypes.QualityChangesInstanceContainer)
        instance_container = None
        for item in os.listdir(quality_changes_dir):
            file_path = os.path.join(quality_changes_dir, item)
            if not os.path.isfile(file_path):
                continue
            parser = configparser.ConfigParser(interpolation=None)
            try:
                parser.read([file_path])
            except Exception:
                continue
            if not parser.has_option('general', 'name'):
                continue
            if parser['general']['name'] == name:
                container_id = os.path.basename(file_path).replace('.inst.cfg', '')
                if self.findInstanceContainers(id=container_id):
                    continue
                instance_container = InstanceContainer(container_id)
                with open(file_path, 'r', encoding='utf-8') as f:
                    serialized = f.read()
                try:
                    instance_container.deserialize(serialized, file_path)
                except ContainerFormatError:
                    Logger.logException('e', 'Unable to deserialize InstanceContainer %s', file_path)
                    continue
                self.addContainer(instance_container)
                break
        return instance_container

    def _connectUpgradedExtruderStacksToMachines(self) -> None:
        if False:
            while True:
                i = 10
        extruder_stacks = self.findContainers(container_type=ExtruderStack.ExtruderStack)
        for extruder_stack in extruder_stacks:
            if extruder_stack.getNextStack():
                continue
            machines = ContainerRegistry.getInstance().findContainerStacks(id=extruder_stack.getMetaDataEntry('machine', ''))
            if machines:
                extruder_stack.setNextStack(machines[0])
            else:
                Logger.log('w', 'Could not find machine {machine} for extruder {extruder}', machine=extruder_stack.getMetaDataEntry('machine'), extruder=extruder_stack.getId())

    @classmethod
    @override(ContainerRegistry)
    def getInstance(cls, *args, **kwargs) -> 'CuraContainerRegistry':
        if False:
            print('Hello World!')
        return cast(CuraContainerRegistry, super().getInstance(*args, **kwargs))