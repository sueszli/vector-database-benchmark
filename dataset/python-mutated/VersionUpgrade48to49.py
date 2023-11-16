import configparser
from typing import Tuple, List
import io
import json
from UM.VersionUpgrade import VersionUpgrade
from cura.API import Account

class VersionUpgrade48to49(VersionUpgrade):
    _moved_visibility_settings = ['top_bottom_extruder_nr', 'top_bottom_thickness', 'top_thickness', 'top_layers', 'bottom_thickness', 'bottom_layers', 'ironing_enabled']

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        '\n        Upgrades preferences to have the new version number.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '7'
        if 'visible_settings' in parser['general']:
            parser['general']['visible_settings'] += ';top_bottom'
        if 'cura' in parser and 'categories_expanded' in parser['cura'] and any([setting in parser['cura']['categories_expanded'] for setting in self._moved_visibility_settings]):
            parser['cura']['categories_expanded'] += ';top_bottom'
        if 'ultimaker_auth_data' in parser['general']:
            ultimaker_auth_data = json.loads(parser['general']['ultimaker_auth_data'])
            if set(Account.CLIENT_SCOPES.split(' ')) - set(ultimaker_auth_data['scope'].split(' ')):
                parser['general']['ultimaker_auth_data'] = '{}'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades stacks to have the new version number.\n\n        This updates the post-processing scripts with new parameters.\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'general' not in parser:
            parser['general'] = {}
        parser['general']['version'] = '5'
        if 'post_processing_scripts' in parser['metadata']:
            new_scripts_entries = []
            for script_str in parser['metadata']['post_processing_scripts'].split('\n'):
                if not script_str:
                    continue
                script_str = script_str.replace('\\\\\\n', '\n').replace('\\\\\\\\', '\\\\')
                script_parser = configparser.ConfigParser(interpolation=None)
                script_parser.optionxform = str
                script_parser.read_string(script_str)
                script_id = script_parser.sections()[0]
                if script_id == 'DisplayProgressOnLCD':
                    script_parser[script_id]['time_remaining_method'] = 'm117' if script_parser[script_id]['time_remaining'] == 'True' else 'none'
                script_io = io.StringIO()
                script_parser.write(script_io)
                script_str = script_io.getvalue()
                script_str = script_str.replace('\\\\', '\\\\\\\\').replace('\n', '\\\\\\n')
                new_scripts_entries.append(script_str)
            parser['metadata']['post_processing_scripts'] = '\n'.join(new_scripts_entries)
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeSettingVisibility(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Upgrades setting visibility to have a version number and move moved settings to a different category\n\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, allow_no_value=True)
        parser.read_string(serialized)
        parser['general']['version'] = '2'
        if 'top_bottom' not in parser:
            parser['top_bottom'] = {}
        if 'shell' in parser:
            for setting in parser['shell']:
                if setting in self._moved_visibility_settings:
                    parser['top_bottom'][setting] = None
                    del parser['shell'][setting]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])