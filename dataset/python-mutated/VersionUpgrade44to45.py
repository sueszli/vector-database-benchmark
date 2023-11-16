import configparser
from typing import Tuple, List
import fnmatch
import io
import os
import urllib.parse
import re
from UM.Logger import Logger
from UM.Resources import Resources
from UM.Version import Version
from UM.VersionUpgrade import VersionUpgrade
_merged_settings = {'machine_head_with_fans_polygon': 'machine_head_polygon', 'support_wall_count': 'support_tree_wall_count'}
_removed_settings = {'support_tree_wall_thickness'}

class VersionUpgrade44to45(VersionUpgrade):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Creates the version upgrade plug-in from 4.4 to 4.5.\n\n        In this case the plug-in will also check for stacks that need to be\n        deleted.\n        '
        super().__init__()
        data_storage_root = os.path.dirname(Resources.getDataStoragePath())
        if os.path.exists(data_storage_root):
            folders = set(os.listdir(data_storage_root))
            folders = set(filter(lambda p: re.fullmatch('\\d+\\.\\d+', p), folders))
            folders.difference_update({os.path.basename(Resources.getDataStoragePath())})
            if folders:
                latest_version = max(folders, key=Version)
                if latest_version == '4.4':
                    self.removeHiddenStacks()

    def removeHiddenStacks(self) -> None:
        if False:
            while True:
                i = 10
        "\n        If starting the upgrade from 4.4, this will remove any hidden printer\n        stacks from the configuration folder as well as all of the user profiles\n        and definition changes profiles.\n\n        This will ONLY run when upgrading from 4.4, not when e.g. upgrading from\n        4.3 to 4.6 (through 4.4). This is because it's to fix a bug\n        (https://github.com/Ultimaker/Cura/issues/6731) that occurred in 4.4\n        only, so only there will it have hidden stacks that need to be deleted.\n        If people upgrade from 4.3 they don't need to be deleted. If people\n        upgrade from 4.5 they have already been deleted previously or never got\n        the broken hidden stacks.\n        "
        Logger.log('d', 'Removing all hidden container stacks.')
        hidden_global_stacks = set()
        hidden_extruder_stacks = set()
        hidden_instance_containers = set()
        exclude_directories = {'plugins'}
        data_storage = Resources.getDataStoragePath()
        for (root, dirs, files) in os.walk(data_storage):
            dirs[:] = [dir for dir in dirs if dir not in exclude_directories]
            for filename in fnmatch.filter(files, '*.global.cfg'):
                parser = configparser.ConfigParser(interpolation=None)
                try:
                    parser.read(os.path.join(root, filename))
                except OSError:
                    continue
                except configparser.Error:
                    continue
                if 'metadata' in parser and 'hidden' in parser['metadata'] and (parser['metadata']['hidden'] == 'True'):
                    stack_id = urllib.parse.unquote_plus(os.path.basename(filename).split('.')[0])
                    hidden_global_stacks.add(stack_id)
                    if 'containers' in parser:
                        if '0' in parser['containers']:
                            hidden_instance_containers.add(parser['containers']['0'])
                        if '6' in parser['containers']:
                            hidden_instance_containers.add(parser['containers']['6'])
                    os.remove(os.path.join(root, filename))
        for (root, dirs, files) in os.walk(data_storage):
            dirs[:] = [dir for dir in dirs if dir not in exclude_directories]
            for filename in fnmatch.filter(files, '*.extruder.cfg'):
                parser = configparser.ConfigParser(interpolation=None)
                try:
                    parser.read(os.path.join(root, filename))
                except OSError:
                    continue
                except configparser.Error:
                    continue
                if 'metadata' in parser and 'machine' in parser['metadata'] and (parser['metadata']['machine'] in hidden_global_stacks):
                    stack_id = urllib.parse.unquote_plus(os.path.basename(filename).split('.')[0])
                    hidden_extruder_stacks.add(stack_id)
                    if 'containers' in parser:
                        if '0' in parser['containers']:
                            hidden_instance_containers.add(parser['containers']['0'])
                        if '6' in parser['containers']:
                            hidden_instance_containers.add(parser['containers']['6'])
                    os.remove(os.path.join(root, filename))
        for (root, dirs, files) in os.walk(data_storage):
            dirs[:] = [dir for dir in dirs if dir not in exclude_directories]
            for filename in fnmatch.filter(files, '*.inst.cfg'):
                container_id = urllib.parse.unquote_plus(os.path.basename(filename).split('.')[0])
                if container_id in hidden_instance_containers:
                    try:
                        os.remove(os.path.join(root, filename))
                    except OSError:
                        continue

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        'Upgrades Preferences to have the new version number.\n\n        This renames the renamed settings in the list of visible settings.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '11'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Upgrades instance containers to have the new version number.\n\n        This renames the renamed settings in the containers.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '11'
        if 'values' in parser:
            for (preferred, removed) in _merged_settings.items():
                if removed in parser['values']:
                    if preferred not in parser['values']:
                        parser['values'][preferred] = parser['values'][removed]
                    del parser['values'][removed]
            for removed in _removed_settings:
                if removed in parser['values']:
                    del parser['values'][removed]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        'Upgrades stacks to have the new version number.'
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '11'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])