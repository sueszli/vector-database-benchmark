import configparser
from io import StringIO
import zipfile
from UM.Application import Application
from UM.Logger import Logger
from UM.Preferences import Preferences
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Workspace.WorkspaceWriter import WorkspaceWriter
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')
from cura.Utils.Threading import call_on_qt_thread

class ThreeMFWorkspaceWriter(WorkspaceWriter):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    @call_on_qt_thread
    def write(self, stream, nodes, mode=WorkspaceWriter.OutputMode.BinaryMode):
        if False:
            for i in range(10):
                print('nop')
        application = Application.getInstance()
        machine_manager = application.getMachineManager()
        mesh_writer = application.getMeshFileHandler().getWriter('3MFWriter')
        if not mesh_writer:
            self.setInformation(catalog.i18nc('@error:zip', '3MF Writer plug-in is corrupt.'))
            Logger.error("3MF Writer class is unavailable. Can't write workspace.")
            return False
        global_stack = machine_manager.activeMachine
        if global_stack is None:
            self.setInformation(catalog.i18nc('@error', 'There is no workspace yet to write. Please add a printer first.'))
            Logger.error('Tried to write a 3MF workspace before there was a global stack.')
            return False
        mesh_writer.setStoreArchive(True)
        if not mesh_writer.write(stream, nodes, mode):
            self.setInformation(mesh_writer.getInformation())
            return False
        archive = mesh_writer.getArchive()
        if archive is None:
            archive = zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED)
        try:
            self._writeContainerToArchive(global_stack, archive)
            for container in global_stack.getContainers():
                self._writeContainerToArchive(container, archive)
            for extruder_stack in global_stack.extruderList:
                self._writeContainerToArchive(extruder_stack, archive)
                for container in extruder_stack.getContainers():
                    self._writeContainerToArchive(container, archive)
        except PermissionError:
            self.setInformation(catalog.i18nc('@error:zip', 'No permission to write the workspace here.'))
            Logger.error('No permission to write workspace to this stream.')
            return False
        original_preferences = Application.getInstance().getPreferences()
        temp_preferences = Preferences()
        for preference in {'general/visible_settings', 'cura/active_mode', 'cura/categories_expanded', 'metadata/setting_version'}:
            temp_preferences.addPreference(preference, None)
            temp_preferences.setValue(preference, original_preferences.getValue(preference))
        preferences_string = StringIO()
        temp_preferences.writeToFile(preferences_string)
        preferences_file = zipfile.ZipInfo('Cura/preferences.cfg')
        try:
            archive.writestr(preferences_file, preferences_string.getvalue())
            version_file = zipfile.ZipInfo('Cura/version.ini')
            version_config_parser = configparser.ConfigParser(interpolation=None)
            version_config_parser.add_section('versions')
            version_config_parser.set('versions', 'cura_version', application.getVersion())
            version_config_parser.set('versions', 'build_type', application.getBuildType())
            version_config_parser.set('versions', 'is_debug_mode', str(application.getIsDebugMode()))
            version_file_string = StringIO()
            version_config_parser.write(version_file_string)
            archive.writestr(version_file, version_file_string.getvalue())
            self._writePluginMetadataToArchive(archive)
            archive.close()
        except PermissionError:
            self.setInformation(catalog.i18nc('@error:zip', 'No permission to write the workspace here.'))
            Logger.error('No permission to write workspace to this stream.')
            return False
        except EnvironmentError as e:
            self.setInformation(catalog.i18nc('@error:zip', str(e)))
            Logger.error('EnvironmentError when writing workspace to this stream: {err}'.format(err=str(e)))
            return False
        mesh_writer.setStoreArchive(False)
        return True

    @staticmethod
    def _writePluginMetadataToArchive(archive: zipfile.ZipFile) -> None:
        if False:
            print('Hello World!')
        file_name_template = '%s/plugin_metadata.json'
        for (plugin_id, metadata) in Application.getInstance().getWorkspaceMetadataStorage().getAllData().items():
            file_name = file_name_template % plugin_id
            file_in_archive = zipfile.ZipInfo(file_name)
            file_in_archive.compress_type = zipfile.ZIP_DEFLATED
            import json
            archive.writestr(file_in_archive, json.dumps(metadata, separators=(', ', ': '), indent=4, skipkeys=True))

    @staticmethod
    def _writeContainerToArchive(container, archive):
        if False:
            i = 10
            return i + 15
        'Helper function that writes ContainerStacks, InstanceContainers and DefinitionContainers to the archive.\n\n        :param container: That follows the :type{ContainerInterface} to archive.\n        :param archive: The archive to write to.\n        '
        if isinstance(container, type(ContainerRegistry.getInstance().getEmptyInstanceContainer())):
            return
        file_suffix = ContainerRegistry.getMimeTypeForContainer(type(container)).preferredSuffix
        if 'base_file' in container.getMetaData():
            base_file = container.getMetaDataEntry('base_file')
            if base_file != container.getId():
                container = ContainerRegistry.getInstance().findContainers(id=base_file)[0]
        file_name = 'Cura/%s.%s' % (container.getId(), file_suffix)
        try:
            if file_name in archive.namelist():
                return
            file_in_archive = zipfile.ZipInfo(file_name)
            file_in_archive.compress_type = zipfile.ZIP_DEFLATED
            ignore_keys = {'um_cloud_cluster_id', 'um_network_key', 'um_linked_to_account', 'removal_warning', 'host_guid', 'group_name', 'group_size', 'connection_type', 'capabilities', 'octoprint_api_key', 'is_online'}
            serialized_data = container.serialize(ignored_metadata_keys=ignore_keys)
            archive.writestr(file_in_archive, serialized_data)
        except (FileNotFoundError, EnvironmentError):
            Logger.error('File became inaccessible while writing to it: {archive_filename}'.format(archive_filename=archive.fp.name))
            return