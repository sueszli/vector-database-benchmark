import io
import os
import re
import shutil
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED, BadZipfile
from typing import Dict, Optional, TYPE_CHECKING, List
from UM import i18nCatalog
from UM.Logger import Logger
from UM.Message import Message
from UM.Platform import Platform
from UM.Resources import Resources
from UM.Version import Version
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class Backup:
    """The back-up class holds all data about a back-up.

    It is also responsible for reading and writing the zip file to the user data folder.
    """
    IGNORED_FILES = ['cura\\.log', 'plugins\\.json', 'cache', '__pycache__', '\\.qmlc', '\\.pyc']
    'These files should be ignored when making a backup.'
    IGNORED_FOLDERS = []
    SECRETS_SETTINGS = ['general/ultimaker_auth_data']
    'Secret preferences that need to obfuscated when making a backup of Cura'
    catalog = i18nCatalog('cura')
    'Re-use translation catalog'

    def __init__(self, application: 'CuraApplication', zip_file: bytes=None, meta_data: Dict[str, str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._application = application
        self.zip_file = zip_file
        self.meta_data = meta_data

    def makeFromCurrent(self) -> None:
        if False:
            return 10
        'Create a back-up from the current user config folder.'
        cura_release = self._application.getVersion()
        version_data_dir = Resources.getDataStoragePath()
        Logger.log('d', 'Creating backup for Cura %s, using folder %s', cura_release, version_data_dir)
        secrets = self._obfuscate()
        self._application.saveSettings()
        if Platform.isLinux():
            preferences_file_name = self._application.getApplicationName()
            preferences_file = Resources.getPath(Resources.Preferences, '{}.cfg'.format(preferences_file_name))
            backup_preferences_file = os.path.join(version_data_dir, '{}.cfg'.format(preferences_file_name))
            if os.path.exists(preferences_file) and (not os.path.exists(backup_preferences_file) or not os.path.samefile(preferences_file, backup_preferences_file)):
                Logger.log('d', 'Copying preferences file from %s to %s', preferences_file, backup_preferences_file)
                shutil.copyfile(preferences_file, backup_preferences_file)
        buffer = io.BytesIO()
        archive = self._makeArchive(buffer, version_data_dir)
        if archive is None:
            return
        files = archive.namelist()
        machine_count = max(len([s for s in files if 'machine_instances/' in s]) - 1, 0)
        material_count = max(len([s for s in files if 'materials/' in s]) - 1, 0)
        profile_count = max(len([s for s in files if 'quality_changes/' in s]) - 1, 0)
        plugin_count = 0
        self.zip_file = buffer.getvalue()
        self.meta_data = {'cura_release': cura_release, 'machine_count': str(machine_count), 'material_count': str(material_count), 'profile_count': str(profile_count), 'plugin_count': str(plugin_count)}
        self._illuminate(**secrets)

    def _makeArchive(self, buffer: 'io.BytesIO', root_path: str) -> Optional[ZipFile]:
        if False:
            while True:
                i = 10
        'Make a full archive from the given root path with the given name.\n\n        :param root_path: The root directory to archive recursively.\n        :return: The archive as bytes.\n        '
        ignore_string = re.compile('|'.join(self.IGNORED_FILES + self.IGNORED_FOLDERS))
        try:
            archive = ZipFile(buffer, 'w', ZIP_DEFLATED)
            for (root, folders, files) in os.walk(root_path):
                for item_name in folders + files:
                    absolute_path = os.path.join(root, item_name)
                    if ignore_string.search(absolute_path):
                        continue
                    archive.write(absolute_path, absolute_path[len(root_path) + len(os.sep):])
            archive.close()
            return archive
        except (IOError, OSError, BadZipfile) as error:
            Logger.log('e', 'Could not create archive from user data directory: %s', error)
            self._showMessage(self.catalog.i18nc('@info:backup_failed', 'Could not create archive from user data directory: {}'.format(error)), message_type=Message.MessageType.ERROR)
            return None

    def _showMessage(self, message: str, message_type: Message.MessageType=Message.MessageType.NEUTRAL) -> None:
        if False:
            i = 10
            return i + 15
        'Show a UI message.'
        Message(message, title=self.catalog.i18nc('@info:title', 'Backup'), message_type=message_type).show()

    def restore(self) -> bool:
        if False:
            print('Hello World!')
        'Restore this back-up.\n\n        :return: Whether we had success or not.\n        '
        if not self.zip_file or not self.meta_data or (not self.meta_data.get('cura_release', None)):
            Logger.log('w', 'Tried to restore a Cura backup without having proper data or meta data.')
            self._showMessage(self.catalog.i18nc('@info:backup_failed', 'Tried to restore a Cura backup without having proper data or meta data.'), message_type=Message.MessageType.ERROR)
            return False
        current_version = Version(self._application.getVersion())
        version_to_restore = Version(self.meta_data.get('cura_release', 'dev'))
        if current_version < version_to_restore:
            Logger.log('d', 'Tried to restore a Cura backup of version {version_to_restore} with cura version {current_version}'.format(version_to_restore=version_to_restore, current_version=current_version))
            self._showMessage(self.catalog.i18nc('@info:backup_failed', 'Tried to restore a Cura backup that is higher than the current version.'), message_type=Message.MessageType.ERROR)
            return False
        secrets = self._obfuscate()
        version_data_dir = Resources.getDataStoragePath()
        try:
            archive = ZipFile(io.BytesIO(self.zip_file), 'r')
        except LookupError as e:
            Logger.log('d', f'The following error occurred while trying to restore a Cura backup: {str(e)}')
            Message(self.catalog.i18nc('@info:backup_failed', 'The following error occurred while trying to restore a Cura backup:') + str(e), title=self.catalog.i18nc('@info:title', 'Backup'), message_type=Message.MessageType.ERROR).show()
            return False
        extracted = self._extractArchive(archive, version_data_dir)
        if Platform.isLinux():
            preferences_file_name = self._application.getApplicationName()
            preferences_file = Resources.getPath(Resources.Preferences, '{}.cfg'.format(preferences_file_name))
            backup_preferences_file = os.path.join(version_data_dir, '{}.cfg'.format(preferences_file_name))
            Logger.log('d', 'Moving preferences file from %s to %s', backup_preferences_file, preferences_file)
            try:
                shutil.move(backup_preferences_file, preferences_file)
            except EnvironmentError as e:
                Logger.error(f'Unable to back-up preferences file: {type(e)} - {str(e)}')
        self._application.readPreferencesFromConfiguration()
        self._illuminate(**secrets)
        return extracted

    def _extractArchive(self, archive: 'ZipFile', target_path: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Extract the whole archive to the given target path.\n\n        :param archive: The archive as ZipFile.\n        :param target_path: The target path.\n        :return: Whether we had success or not.\n        '
        from cura.CuraApplication import CuraApplication
        config_filename = CuraApplication.getInstance().getApplicationName() + '.cfg'
        if config_filename not in [file.filename for file in archive.filelist]:
            Logger.logException('e', 'Unable to extract the backup due to corruption of compressed file(s).')
            return False
        Logger.log('d', 'Removing current data in location: %s', target_path)
        Resources.factoryReset()
        Logger.log('d', 'Extracting backup to location: %s', target_path)
        name_list = archive.namelist()
        ignore_string = re.compile('|'.join(self.IGNORED_FILES + self.IGNORED_FOLDERS))
        for archive_filename in name_list:
            if ignore_string.search(archive_filename):
                Logger.warning(f"File ({archive_filename}) in archive that doesn't fit current backup policy; ignored.")
                continue
            try:
                archive.extract(archive_filename, target_path)
            except (PermissionError, EnvironmentError):
                Logger.logException('e', f'Unable to extract the file {archive_filename} from the backup due to permission or file system errors.')
            except UnicodeEncodeError:
                Logger.error(f'Unable to extract the file {archive_filename} because of an encoding error.')
            CuraApplication.getInstance().processEvents()
        return True

    def _obfuscate(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        "\n        Obfuscate and remove the secret preferences that are specified in SECRETS_SETTINGS\n\n        :return: a dictionary of the removed secrets. Note: the '/' is replaced by '__'\n        "
        preferences = self._application.getPreferences()
        secrets = {}
        for secret in self.SECRETS_SETTINGS:
            secrets[secret.replace('/', '__')] = deepcopy(preferences.getValue(secret))
            preferences.setValue(secret, None)
        self._application.savePreferences()
        return secrets

    def _illuminate(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Restore the obfuscated settings\n\n        :param kwargs: a dict of obscured preferences. Note: the '__' of the keys will be replaced by '/'\n        "
        preferences = self._application.getPreferences()
        for (key, value) in kwargs.items():
            preferences.setValue(key.replace('__', '/'), value)
        self._application.savePreferences()