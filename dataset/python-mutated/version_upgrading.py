"""
Sometimes, changes in code require updating the format of existing user data.
This module handles upgrading user data, if required.
The stored user data contains a version field with the autokey version that created it. This is used to determine
if any patches must be applied.

Each converter function altering the configuration data MUST NOT change the "version" item in the configuration data to
a different version (so NEVER set to common.VERSION), because this might skip additional conversion steps.

Example: Let the current version be 0.97.0. If 0.96.1 introduces a conversion step, and old 0.70.x data is found,
setting config_data["version"] to common.VERSION ("0.97.0") inside the conversion function convert_v0_70_to_v0_80 will
skip the required conversion task for 0.96.1.

Additionally, it will skip further conversion tasks that require the model to be present, which are executed in the
ConfigManager later. The ConfigManager is responsible for updating the data version.

Do not require the user to install all versions one after another in order to get all data patches. Such skips might
happen with LTS distribution releases that skip several autokey versions during their lifetime.
"""
import os
from pathlib import Path
import glob
import re
from packaging.version import parse as vparse
import autokey.model.folder
import autokey.model.phrase
import autokey.model.script
from autokey import common
import autokey.configmanager.configmanager_constants as cm_constants
from autokey.iomediator.constants import X_RECORD_INTERFACE
logger = __import__('autokey.logger').logger.get_logger(__name__)

def upgrade_configuration_format(configuration_manager, config_data: dict):
    if False:
        for i in range(10):
            print('nop')
    '\n    Updates the global configuration data to the latest version.\n    Run before configuration loaded.\n    '
    version = vparse(config_data['version'])
    logger.info('Checking if upgrade is needed from version %s', version)
    if not version < vparse(common.VERSION):
        return
    if version < vparse('0.80.0'):
        convert_v0_70_to_v0_80(config_data)
    if version < vparse('0.95.3'):
        convert_autostart_entries_for_v0_95_3()
    if version < vparse('0.96'):
        convertDotFiles_v96(configuration_manager, config_data)
        convert_folder_attributes_0_96(configuration_manager, config_data)

def upgrade_configuration_after_load(configuration_manager, config_data: dict):
    if False:
        return 10
    '\n    Updates the global configuration data to the latest version.\n    Run after configuration loaded.\n    '
    version = vparse(config_data['version'])
    logger.info('Checking if upgrade is needed from version %s', version)
    if not version < vparse(common.VERSION):
        return
    configuration_manager.SETTINGS[cm_constants.INTERFACE_TYPE] = X_RECORD_INTERFACE
    logger.info('Resetting interface type, new type: %s', configuration_manager.SETTINGS[cm_constants.INTERFACE_TYPE])
    if version < vparse('0.82.3'):
        convert_to_v0_82_3(configuration_manager)
    if version < vparse('0.70.0'):
        convert_to_v0_70(configuration_manager)
    configuration_manager.VERSION = common.VERSION
    configuration_manager.config_altered(True)

def convert_to_v0_70(config_manager):
    if False:
        return 10
    logger.info('Doing upgrade to 0.70.0')
    for item in config_manager.allItems:
        if isinstance(item, autokey.model.phrase.Phrase):
            item.sendMode = autokey.model.phrase.SendMode.KEYBOARD

def convert_to_v0_82_3(configuration_manager):
    if False:
        i = 10
        return i + 15
    logger.info('Doing upgrade to 0.82.3')
    configuration_manager.SETTINGS[cm_constants.WORKAROUND_APP_REGEX] += '|krdc.Krdc'
    configuration_manager.workAroundApps = re.compile(configuration_manager.SETTINGS[cm_constants.WORKAROUND_APP_REGEX])
    configuration_manager.SETTINGS[cm_constants.SCRIPT_GLOBALS] = {}

def convert_v0_70_to_v0_80(config_data):
    if False:
        for i in range(10):
            print('nop')
    old_version = config_data['version']
    try:
        _convert_v0_70_to_v0_80(config_data, old_version)
    except Exception:
        logger.exception('Problem occurred during conversion of configuration data format from v0.70 to v0.80Existing config file has been saved as {}{}'.format(cm_constants.CONFIG_FILE, old_version))
        raise

def _convert_v0_70_to_v0_80(config_data, old_version: str):
    if False:
        i = 10
        return i + 15
    os.rename(cm_constants.CONFIG_FILE, cm_constants.CONFIG_FILE + old_version)
    logger.info('Converting v{} configuration data to v0.80.0'.format(old_version))
    for folder_data in config_data['folders']:
        _convert_v0_70_to_v0_80_folder(folder_data, None)
    config_data['folders'] = []
    config_data['settings'][cm_constants.NOTIFICATION_ICON] = common.ICON_FILE_NOTIFICATION
    if os.path.exists(cm_constants.CONFIG_FILE_BACKUP):
        os.remove(cm_constants.CONFIG_FILE_BACKUP)
    logger.info('Conversion succeeded')

def _convert_v0_70_to_v0_80_folder(folder_data, parent):
    if False:
        print('Hello World!')
    f = autokey.model.folder.Folder('')
    f.inject_json_data(folder_data)
    f.parent = parent
    f.persist()
    for subfolder in folder_data['folders']:
        _convert_v0_70_to_v0_80_folder(subfolder, f)
    for itemData in folder_data['items']:
        i = None
        if itemData['type'] == 'script':
            i = autokey.model.script.Script('', '')
            i.code = itemData['code']
        elif itemData['type'] == 'phrase':
            i = autokey.model.phrase.Phrase('', '')
            i.phrase = itemData['phrase']
        if i is not None:
            i.inject_json_data(itemData)
            i.parent = f
            i.persist()

def convert_autostart_entries_for_v0_95_3():
    if False:
        print('Hello World!')
    '\n    In versions <= 0.95.2, the autostart option in autokey-gtk copied the default autokey-gtk.desktop file into\n    $XDG_CONFIG_DIR/autostart (with minor, unrelated modifications).\n    For versions >= 0.95.3, the autostart file is renamed to autokey.desktop. In 0.95.3, the autostart functionality\n    is implemented for autokey-qt. Thus, it becomes possible to have an autostart file for both GUIs in the autostart\n    directory simultaneously. Because of the singleton nature of autokey, this becomes an issue and race-conditions\n    determine which GUI starts first. To prevent this, both GUIs will share a single autokey.desktop autostart entry,\n    allowing only one GUI to be started during login. This allows for much simpler code.\n    '
    logger.info('Version update: Converting autostart entry for 0.95.3')
    old_autostart_file = Path(common.AUTOSTART_DIR) / 'autokey-gtk.desktop'
    if old_autostart_file.exists():
        new_file_name = Path(common.AUTOSTART_DIR) / 'autokey.desktop'
        logger.info("Found old autostart entry: '{}'. Rename to: '{}'".format(old_autostart_file, new_file_name))
        old_autostart_file.rename(new_file_name)

def convertDotFiles_v95_11_folder(p: Path):
    if False:
        print('Hello World!')
    for name in p.glob('.*.json'):
        new_json = p / name.name[1:]
        name.rename(new_json)
        logger.debug('Converted to {}'.format(new_json))
    for name in p.iterdir():
        if name.is_dir():
            convertDotFiles_v95_11_folder(name)

def convertDotFiles_v96(cm, configData):
    if False:
        return 10
    logger.info('Version update: Unhiding sidecar dotfiles for versions > 0.95.10')
    for name in cm.get_all_config_folder_paths(configData):
        convertDotFiles_v95_11_folder(Path(name))

def convert_folder_attributes_0_96(cm, config_data):
    if False:
        while True:
            i = 10
    for folder in cm.get_all_folders():
        logger.debug(folder)
        try:
            _ = folder.temporary
        except AttributeError:
            folder.temporary = False