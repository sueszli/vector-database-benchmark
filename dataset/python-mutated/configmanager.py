import os
import os.path
import shutil
import glob
import threading
import typing
import re
import json
import itertools
import autokey.model.abstract_hotkey
import autokey.model.folder
import autokey.model.helpers
import autokey.model.phrase
import autokey.model.script
from autokey.model import key
from autokey import common
from autokey.configmanager.configmanager_constants import CONFIG_FILE, CONFIG_DEFAULT_FOLDER, CONFIG_FILE_BACKUP, RECENT_ENTRIES_FOLDER, IS_FIRST_RUN, SERVICE_RUNNING, MENU_TAKES_FOCUS, SHOW_TRAY_ICON, SORT_BY_USAGE_COUNT, PROMPT_TO_SAVE, ENABLE_QT4_WORKAROUND, UNDO_USING_BACKSPACE, WINDOW_DEFAULT_SIZE, HPANE_POSITION, COLUMN_WIDTHS, SHOW_TOOLBAR, NOTIFICATION_ICON, WORKAROUND_APP_REGEX, TRIGGER_BY_INITIAL, SCRIPT_GLOBALS, INTERFACE_TYPE, DISABLED_MODIFIERS, GTK_THEME, GTK_TREE_VIEW_EXPANDED_ROWS, PATH_LAST_OPEN
import autokey.configmanager.version_upgrading as version_upgrade
import autokey.configmanager.predefined_user_files
from autokey.iomediator.constants import X_RECORD_INTERFACE
from autokey.model.key import MODIFIERS
logger = __import__('autokey.logger').logger.get_logger(__name__)

def create_config_manager_instance(auto_key_app, had_error=False):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(CONFIG_DEFAULT_FOLDER):
        os.mkdir(CONFIG_DEFAULT_FOLDER)
    try:
        config_manager = ConfigManager(auto_key_app)
    except Exception as e:
        if had_error or not os.path.exists(CONFIG_FILE_BACKUP) or (not os.path.exists(CONFIG_FILE)):
            logger.exception('Error while loading configuration. Cannot recover.')
            raise
        logger.exception('Error while loading configuration. Backup has been restored.')
        os.remove(CONFIG_FILE)
        shutil.copy2(CONFIG_FILE_BACKUP, CONFIG_FILE)
        return create_config_manager_instance(auto_key_app, True)
    logger.debug('Global settings: %r', ConfigManager.SETTINGS)
    return config_manager

def save_config(config_manager):
    if False:
        for i in range(10):
            print('nop')
    logger.info('Persisting configuration')
    config_manager.app.monitor.suspend()
    if os.path.exists(CONFIG_FILE):
        logger.info('Backing up existing config file')
        shutil.copy2(CONFIG_FILE, CONFIG_FILE_BACKUP)
    try:
        _persist_settings(config_manager)
        logger.info('Finished persisting configuration - no errors')
    except Exception as e:
        if os.path.exists(CONFIG_FILE_BACKUP):
            shutil.copy2(CONFIG_FILE_BACKUP, CONFIG_FILE)
        logger.exception('Error while saving configuration. Backup has been restored (if found).')
        raise Exception('Error while saving configuration. Backup has been restored (if found).')
    finally:
        config_manager.app.monitor.unsuspend()

def _persist_settings(config_manager):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write the settings, including the persistent global script Store.\n    The Store instance might contain arbitrary user data, like function objects, OpenCL contexts, or whatever other\n    non-serializable objects, both as keys or values.\n    Try to serialize the data, and if it fails, fall back to checking the store and removing all non-serializable\n    data.\n    '
    serializable_data = config_manager.get_serializable()
    try:
        _try_persist_settings(serializable_data)
    except (TypeError, ValueError):
        _remove_non_serializable_store_entries(serializable_data['settings'][SCRIPT_GLOBALS])
        _try_persist_settings(serializable_data)

def _try_persist_settings(serializable_data: dict):
    if False:
        i = 10
        return i + 15
    '\n    Write the settings as JSON to the configuration file\n    :raises TypeError: If the user tries to store non-serializable types\n    :raises ValueError: If the user tries to store circular referenced (recursive) structures.\n    '
    with open(CONFIG_FILE, 'w') as json_file:
        json.dump(serializable_data, json_file, indent=4)

def _remove_non_serializable_store_entries(store: dict):
    if False:
        i = 10
        return i + 15
    '\n    This function is called if there are non-serializable items in the global script storage.\n    This function removes all such items.\n    '
    removed_key_list = []
    for (key, value) in store.items():
        if not (_is_serializable(key) and _is_serializable(value)):
            logger.info("Remove non-serializable item from the global script store. Key: '{}', Value: '{}'. This item cannot be saved and therefore will be lost.".format(key, value))
            removed_key_list.append(key)
    for key in removed_key_list:
        del store[key]

def _is_serializable(data):
    if False:
        for i in range(10):
            print('nop')
    'Check, if data is json serializable.'
    try:
        json.dumps(data)
    except (TypeError, ValueError):
        return False
    else:
        return True

def apply_settings(settings):
    if False:
        for i in range(10):
            print('nop')
    '\n    Allows new settings to be added without users having to lose all their configuration\n    '
    for (key, value) in settings.items():
        ConfigManager.SETTINGS[key] = value

class ConfigManager:
    """
    Contains all application configuration, and provides methods for updating and
    maintaining consistency of the configuration.
    """
    '\n    Static member for global application settings.\n    '
    CLASS_VERSION = common.VERSION
    SETTINGS = {IS_FIRST_RUN: True, SERVICE_RUNNING: True, MENU_TAKES_FOCUS: False, SHOW_TRAY_ICON: True, SORT_BY_USAGE_COUNT: True, PROMPT_TO_SAVE: False, ENABLE_QT4_WORKAROUND: False, INTERFACE_TYPE: X_RECORD_INTERFACE, UNDO_USING_BACKSPACE: True, WINDOW_DEFAULT_SIZE: (600, 400), HPANE_POSITION: 150, COLUMN_WIDTHS: [150, 50, 100], SHOW_TOOLBAR: True, NOTIFICATION_ICON: common.ICON_FILE_NOTIFICATION, WORKAROUND_APP_REGEX: '.*VirtualBox.*|krdc.Krdc', TRIGGER_BY_INITIAL: False, DISABLED_MODIFIERS: [], SCRIPT_GLOBALS: {}, GTK_THEME: 'classic', GTK_TREE_VIEW_EXPANDED_ROWS: [], PATH_LAST_OPEN: '0'}

    def __init__(self, app):
        if False:
            return 10
        '\n        Create initial default configuration\n        '
        self.VERSION = self.__class__.CLASS_VERSION
        self.lock = threading.Lock()
        self.app = app
        self.folders = []
        self.userCodeDir = None
        self.configHotkey = GlobalHotkey()
        self.configHotkey.set_hotkey(['<super>'], 'k')
        self.configHotkey.enabled = True
        self.toggleServiceHotkey = GlobalHotkey()
        self.toggleServiceHotkey.set_hotkey(['<super>', '<shift>'], 'k')
        self.toggleServiceHotkey.enabled = True
        self.workAroundApps = re.compile(self.SETTINGS[WORKAROUND_APP_REGEX])
        app.init_global_hotkeys(self)
        self.load_global_config()
        self.app.monitor.add_watch(CONFIG_DEFAULT_FOLDER)
        self.app.monitor.add_watch(common.CONFIG_DIR)
        if self.folders:
            return
        logger.info('No configuration found - creating new one')
        self.folders.append(autokey.configmanager.predefined_user_files.create_my_phrases_folder())
        self.folders.append(autokey.configmanager.predefined_user_files.create_sample_scripts_folder())
        logger.debug('Initial folders generated and populated with example data.')
        self.recentEntries = []
        self.config_altered(True)

    def get_serializable(self):
        if False:
            for i in range(10):
                print('nop')
        extraFolders = []
        for folder in self.folders:
            if not folder.path.startswith(CONFIG_DEFAULT_FOLDER):
                extraFolders.append(folder.path)
        d = {'version': self.VERSION, 'userCodeDir': self.userCodeDir, 'settings': ConfigManager.SETTINGS, 'folders': extraFolders, 'toggleServiceHotkey': self.toggleServiceHotkey.get_serializable(), 'configHotkey': self.configHotkey.get_serializable()}
        return d

    def load_global_config(self):
        if False:
            while True:
                i = 10
        if os.path.exists(CONFIG_FILE):
            logger.info('Loading config from existing file: ' + CONFIG_FILE)
            with open(CONFIG_FILE, 'r') as pFile:
                data = json.load(pFile)
            version_upgrade.upgrade_configuration_format(self, data)
            self.VERSION = data['version']
            self.userCodeDir = data['userCodeDir']
            apply_settings(data['settings'])
            self.load_disabled_modifiers()
            self.workAroundApps = re.compile(self.SETTINGS[WORKAROUND_APP_REGEX])
            self.__load_folders(data)
            self.toggleServiceHotkey.load_from_serialized(data['toggleServiceHotkey'])
            self.configHotkey.load_from_serialized(data['configHotkey'])
            if self.VERSION < self.CLASS_VERSION:
                version_upgrade.upgrade_configuration_after_load(self, data)
            self.config_altered(False)
            logger.info('Successfully loaded configuration')

    def __load_folders(self, data):
        if False:
            i = 10
            return i + 15
        for path in self.get_all_config_folder_paths(data):
            f = autokey.model.folder.Folder('', path=path)
            f.load()
            logger.debug("Loading folder at '%s'", path)
            self.folders.append(f)

    def get_all_config_folder_paths(self, data):
        if False:
            i = 10
            return i + 15
        for path in glob.glob(CONFIG_DEFAULT_FOLDER + '/*'):
            if os.path.isdir(path):
                yield path
        for path in data['folders']:
            yield path

    def get_all_folders(self):
        if False:
            return 10
        out = []
        for folder in self.folders:
            out.append(folder)
            out.extend(folder.get_child_folders())
        return out

    def __checkExisting(self, path):
        if False:
            i = 10
            return i + 15
        for item in self.allItems:
            if item.path == path:
                return item
        return None

    def __checkExistingFolder(self, path):
        if False:
            while True:
                i = 10
        for folder in self.allFolders:
            if folder.path == path:
                return folder
        return None

    def path_created_or_modified(self, path):
        if False:
            return 10
        (directory, baseName) = os.path.split(path)
        loaded = False
        if path == CONFIG_FILE:
            self.reload_global_config()
        elif directory != common.CONFIG_DIR:
            if os.path.isdir(path):
                f = autokey.model.folder.Folder('', path=path)
                if directory == CONFIG_DEFAULT_FOLDER:
                    self.folders.append(f)
                    f.load()
                    loaded = True
                else:
                    folder = self.__checkExistingFolder(directory)
                    if folder is not None:
                        f.load(folder)
                        folder.add_folder(f)
                        loaded = True
            elif os.path.isfile(path):
                i = self.__checkExisting(path)
                isNew = False
                if i is None:
                    isNew = True
                    if baseName.endswith('.txt'):
                        i = autokey.model.phrase.Phrase('', '', path=path)
                    elif baseName.endswith('.py'):
                        i = autokey.model.script.Script('', '', path=path)
                if i is not None:
                    folder = self.__checkExistingFolder(directory)
                    if folder is not None:
                        i.load(folder)
                        if isNew:
                            folder.add_item(i)
                        loaded = True
                if baseName == 'folder.json':
                    folder = self.__checkExistingFolder(directory)
                    if folder is not None:
                        folder.load_from_serialized()
                        loaded = True
                if baseName.endswith('.json'):
                    for item in self.allItems:
                        if item.get_json_path() == path:
                            item.load_from_serialized()
                            loaded = True
            if not loaded:
                logger.warning('No action taken for create/update event at %s', path)
            else:
                self.config_altered(False)
            return loaded

    def path_removed(self, path):
        if False:
            return 10
        (directory, baseName) = os.path.split(path)
        deleted = False
        if directory == common.CONFIG_DIR:
            return
        folder = self.__checkExistingFolder(path)
        item = self.__checkExisting(path)
        if folder is not None:
            if folder.parent is None:
                self.folders.remove(folder)
            else:
                folder.parent.remove_folder(folder)
            deleted = True
        elif item is not None:
            item.parent.remove_item(item)
            deleted = True
        if not deleted:
            logger.warning('No action taken for delete event at %s', path)
        else:
            self.config_altered(False)
        return deleted

    def load_disabled_modifiers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load all disabled modifier keys from the configuration file. Called during startup, after the configuration\n        is read into the SETTINGS dictionary.\n        :return:\n        '
        try:
            self.SETTINGS[DISABLED_MODIFIERS] = [key.Key(value) for value in self.SETTINGS[DISABLED_MODIFIERS]]
        except ValueError:
            logger.error('Unknown value in the disabled modifier list found. Unexpected: {}'.format(self.SETTINGS[DISABLED_MODIFIERS]))
            self.SETTINGS[DISABLED_MODIFIERS] = []
        for possible_modifier in self.SETTINGS[DISABLED_MODIFIERS]:
            self._check_if_modifier(possible_modifier)
            logger.info('Disabling modifier key {} based on the stored configuration file.'.format(possible_modifier))
            MODIFIERS.remove(possible_modifier)

    @staticmethod
    def is_modifier_disabled(modifier: key.Key) -> bool:
        if False:
            return 10
        'Checks, if the given modifier key is disabled. '
        ConfigManager._check_if_modifier(modifier)
        return modifier in ConfigManager.SETTINGS[DISABLED_MODIFIERS]

    @staticmethod
    def disable_modifier(modifier: typing.Union[key.Key, str]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Permanently disable a modifier key. This can be used to disable unwanted modifier keys, like CAPSLOCK,\n        if the user remapped the physical key to something else.\n        :param modifier: Modifier key to disable.\n        :return:\n        '
        if isinstance(modifier, str):
            modifier = key.Key(modifier)
        ConfigManager._check_if_modifier(modifier)
        try:
            logger.info('Disabling modifier key {} on user request.'.format(modifier))
            MODIFIERS.remove(modifier)
        except ValueError:
            logger.warning('Disabling already disabled modifier key. Affected key: {}'.format(modifier))
        else:
            ConfigManager.SETTINGS[DISABLED_MODIFIERS].append(modifier)

    @staticmethod
    def enable_modifier(modifier: typing.Union[key.Key, str]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enable a previously disabled modifier key.\n        :param modifier: Modifier key to re-enable\n        :return:\n        '
        if isinstance(modifier, str):
            modifier = key.Key(modifier)
        ConfigManager._check_if_modifier(modifier)
        if modifier not in MODIFIERS:
            logger.info('Re-eabling modifier key {} on user request.'.format(modifier))
            MODIFIERS.append(modifier)
            ConfigManager.SETTINGS[DISABLED_MODIFIERS].remove(modifier)
        else:
            logger.warning('Enabling already enabled modifier key. Affected key: {}'.format(modifier))

    @staticmethod
    def _check_if_modifier(modifier: key.Key):
        if False:
            print('Hello World!')
        if not isinstance(modifier, key.Key):
            raise TypeError('The given value must be an AutoKey Key instance, got {}'.format(type(modifier)))
        if not modifier in key._ALL_MODIFIERS_:
            raise ValueError("The given key '{}' is not a modifier. Expected one of {}.".format(modifier, key._ALL_MODIFIERS_))

    def reload_global_config(self):
        if False:
            while True:
                i = 10
        logger.info('Reloading global configuration')
        with open(CONFIG_FILE, 'r') as pFile:
            data = json.load(pFile)
        self.userCodeDir = data['userCodeDir']
        apply_settings(data['settings'])
        self.workAroundApps = re.compile(self.SETTINGS[WORKAROUND_APP_REGEX])
        existingPaths = []
        for folder in self.folders:
            if folder.parent is None and (not folder.path.startswith(CONFIG_DEFAULT_FOLDER)):
                existingPaths.append(folder.path)
        for folderPath in data['folders']:
            if folderPath not in existingPaths:
                f = autokey.model.folder.Folder('', path=folderPath)
                f.load()
                self.folders.append(f)
        self.toggleServiceHotkey.load_from_serialized(data['toggleServiceHotkey'])
        self.configHotkey.load_from_serialized(data['configHotkey'])
        self.config_altered(False)
        logger.info('Successfully reloaded global configuration')

    def config_altered(self, persistGlobal):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when some element of configuration has been altered, to update\n        the lists of phrases/folders.\n\n        @param persistGlobal: save the global configuration at the end of the process\n        '
        logger.info('Configuration changed - rebuilding in-memory structures')
        self.lock.acquire()
        self.hotKeyFolders = []
        self.hotKeys = []
        self.abbreviations = []
        self.allFolders = []
        self.allItems = []
        for folder in self.folders:
            if autokey.model.helpers.TriggerMode.HOTKEY in folder.modes:
                self.hotKeyFolders.append(folder)
            self.allFolders.append(folder)
            if not self.app.monitor.has_watch(folder.path):
                self.app.monitor.add_watch(folder.path)
            self.__processFolder(folder)
        self.globalHotkeys = []
        self.globalHotkeys.append(self.configHotkey)
        self.globalHotkeys.append(self.toggleServiceHotkey)
        if persistGlobal:
            save_config(self)
        self.lock.release()

    def __processFolder(self, parentFolder):
        if False:
            while True:
                i = 10
        if not self.app.monitor.has_watch(parentFolder.path):
            self.app.monitor.add_watch(parentFolder.path)
        for folder in parentFolder.folders:
            if autokey.model.helpers.TriggerMode.HOTKEY in folder.modes:
                self.hotKeyFolders.append(folder)
            self.allFolders.append(folder)
            if not self.app.monitor.has_watch(folder.path):
                self.app.monitor.add_watch(folder.path)
            self.__processFolder(folder)
        for item in parentFolder.items:
            if autokey.model.helpers.TriggerMode.HOTKEY in item.modes:
                self.hotKeys.append(item)
            if autokey.model.helpers.TriggerMode.ABBREVIATION in item.modes:
                self.abbreviations.append(item)
            self.allItems.append(item)

    def add_recent_entry(self, entry):
        if False:
            for i in range(10):
                print('nop')
        if RECENT_ENTRIES_FOLDER not in self.folders:
            folder = autokey.model.folder.Folder(RECENT_ENTRIES_FOLDER)
            folder.set_hotkey(['<super>'], '<f7>')
            folder.set_modes([autokey.model.helpers.TriggerMode.HOTKEY])
            self.folders[RECENT_ENTRIES_FOLDER] = folder
            self.recentEntries = []
        folder = self.folders[RECENT_ENTRIES_FOLDER]
        if entry not in self.recentEntries:
            self.recentEntries.append(entry)
            while len(self.recentEntries) > self.SETTINGS[RECENT_ENTRY_COUNT]:
                self.recentEntries.pop(0)
            folder.items = []
            for theEntry in self.recentEntries:
                if len(theEntry) > 17:
                    description = theEntry[:17] + '...'
                else:
                    description = theEntry
                p = autokey.model.phrase.Phrase(description, theEntry)
                if self.SETTINGS[RECENT_ENTRY_SUGGEST]:
                    p.set_modes([autokey.model.helpers.TriggerMode.PREDICTIVE])
                folder.add_item(p)
            self.config_altered(False)

    def check_abbreviation_unique(self, abbreviation, filterPattern, targetItem):
        if False:
            return 10
        '\n        Checks that the given abbreviation is not already in use.\n\n        @param abbreviation: the abbreviation to check\n        @param filterPattern: The filter pattern associated with the abbreviation\n        @param targetItem: the phrase for which the abbreviation to be used\n        '
        for item in itertools.chain(self.allFolders, self.allItems):
            if ConfigManager.item_has_abbreviation(item, abbreviation) and item.filter_matches(filterPattern):
                return (item is targetItem, item)
        return (True, None)

    @staticmethod
    def item_has_abbreviation(item, abbreviation):
        if False:
            print('Hello World!')
        return autokey.model.helpers.TriggerMode.ABBREVIATION in item.modes and abbreviation in item.abbreviations
    'def check_abbreviation_substring(self, abbreviation, targetItem):\n        for item in self.allFolders:\n            if model.TriggerMode.ABBREVIATION in item.modes:\n                if abbreviation in item.abbreviation or item.abbreviation in abbreviation:\n                    return item is targetItem, item.title\n\n        for item in self.allItems:\n            if model.TriggerMode.ABBREVIATION in item.modes:\n                if abbreviation in item.abbreviation or item.abbreviation in abbreviation:\n                    return item is targetItem, item.description\n\n        return True, ""\n\n    def __checkSubstringAbbr(self, item1, item2, abbr):\n        # Check if the given abbreviation is a substring match for the given item\n        # If it is, check a few other rules to see if it matters\n        print ("substring check {} against {}".format(item.abbreviation, abbr))\n        try:\n            index = item.abbreviation.index(abbr)\n            print (index)\n            if index == 0 and len(abbr) < len(item.abbreviation):\n                return item.immediate\n            elif (index + len(abbr)) == len(item.abbreviation):\n                return item.triggerInside\n            elif len(abbr) != len(item.abbreviation):\n                return item.triggerInside and item.immediate\n            else:\n                return False\n        except ValueError:\n            return False'

    def check_hotkey_unique(self, modifiers, hotKey, newFilterPattern, targetItem):
        if False:
            while True:
                i = 10
        '\n        Checks that the given hotkey is not already in use. Also checks the\n        special hotkeys configured from the advanced settings dialog.\n\n        @param modifiers: modifiers for the hotkey\n        @param hotKey: the hotkey to check\n        @param newFilterPattern:\n        @param targetItem: the phrase for which the hotKey to be used\n        '
        item = self.get_item_with_hotkey(modifiers, hotKey, newFilterPattern)
        if item:
            return (item is targetItem, item)
        else:
            return (True, None)

    def get_item_with_hotkey(self, modifiers, hotKey, newFilterPattern=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets first item with the specified hotkey. Also checks the\n        special hotkeys configured from the advanced settings dialog.\n        Checks folders first, then phrases, then special hotkeys.\n\n        @param modifiers: modifiers for the hotkey\n        @param hotKey: the hotkey to check\n        @param newFilterPattern:\n        '
        for item in itertools.chain(self.allFolders, self.allItems):
            if autokey.model.helpers.TriggerMode.HOTKEY in item.modes and ConfigManager.item_has_same_hotkey(item, modifiers, hotKey, newFilterPattern):
                return item
        for item in self.globalHotkeys:
            if item.enabled and ConfigManager.item_has_same_hotkey(item, modifiers, hotKey, newFilterPattern):
                return item
        return None

    @staticmethod
    def item_has_same_hotkey(item, modifiers, hotKey, newFilterPattern):
        if False:
            print('Hello World!')
        return item.modifiers == modifiers and item.hotKey == hotKey and item.filter_matches(newFilterPattern)

    def remove_all_temporary(self, folder=None, in_temp_parent=False):
        if False:
            print('Hello World!')
        '\n        Removes all temporary folders and phrases, as well as any within temporary folders.\n        Useful for rc-style scripts that want to change a set of keys.\n        '
        if folder is None:
            searchFolders = self.allFolders
            searchItems = self.allItems
        else:
            searchFolders = folder.folders
            searchItems = folder.items
        for item in searchItems:
            try:
                if item.temporary or in_temp_parent:
                    self.__deleteHotkeys(item)
                    searchItems.remove(item)
            except AttributeError:
                pass
        for subfolder in searchFolders:
            self.__deleteHotkeys(subfolder)
            try:
                if subfolder.temporary or in_temp_parent:
                    in_temp_parent = True
                    if folder is not None:
                        folder.remove_folder(subfolder)
                    else:
                        searchFolders.remove(subfolder)
            except AttributeError:
                pass
            self.remove_all_temporary(subfolder, in_temp_parent)

    def delete_hotkeys(self, removed_item):
        if False:
            i = 10
            return i + 15
        return self.__deleteHotkeys(removed_item)

    def __deleteHotkeys(self, removed_item):
        if False:
            i = 10
            return i + 15
        removed_item.unset_hotkey()
        app = self.app
        if autokey.model.helpers.TriggerMode.HOTKEY in removed_item.modes:
            app.hotkey_removed(removed_item)
        if isinstance(removed_item, autokey.model.folder.Folder):
            for subFolder in removed_item.folders:
                self.delete_hotkeys(subFolder)
            for item in removed_item.items:
                if autokey.model.helpers.TriggerMode.HOTKEY in item.modes:
                    app.hotkey_removed(item)

class GlobalHotkey(autokey.model.abstract_hotkey.AbstractHotkey):
    """
    A global application hotkey, configured from the advanced settings dialog.
    Allows a method call to be attached to the hotkey.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        autokey.model.abstract_hotkey.AbstractHotkey.__init__(self)
        self.enabled = False
        self.windowInfoRegex = None
        self.isRecursive = False
        self.parent = None
        self.modes = []

    def get_serializable(self):
        if False:
            i = 10
            return i + 15
        d = {'enabled': self.enabled}
        d.update(autokey.model.abstract_hotkey.AbstractHotkey.get_serializable(self))
        return d

    def load_from_serialized(self, data):
        if False:
            while True:
                i = 10
        autokey.model.abstract_hotkey.AbstractHotkey.load_from_serialized(self, data)
        self.enabled = data['enabled']

    def set_closure(self, closure):
        if False:
            i = 10
            return i + 15
        '\n        Set the callable to be executed when the hotkey is triggered.\n        '
        self.closure = closure

    def check_hotkey(self, modifiers, key, windowTitle):
        if False:
            while True:
                i = 10
        if autokey.model.abstract_hotkey.AbstractHotkey.check_hotkey(self, modifiers, key, windowTitle) and self.enabled:
            logger.debug('Triggered global hotkey using modifiers: %r key: %r', modifiers, key)
            self.closure()
        return False

    def get_hotkey_string(self, key=None, modifiers=None):
        if False:
            while True:
                i = 10
        if key is None and modifiers is None:
            if not self.enabled:
                return ''
            key = self.hotKey
            modifiers = self.modifiers
        ret = ''
        for modifier in modifiers:
            ret += modifier
            ret += '+'
        if key == ' ':
            ret += '<space>'
        else:
            ret += key
        return ret

    def __str__(self):
        if False:
            return 10
        return 'AutoKey global hotkeys'