import configparser
from typing import Tuple, List, Dict, Set
import io
from UM.Util import parseBool
from UM.VersionUpgrade import VersionUpgrade
_RENAMED_DEFINITION_DICT = {'dagoma_discoeasy200': 'dagoma_discoeasy200_bicolor'}
_removed_settings = {'spaghetti_infill_enabled', 'spaghetti_infill_stepped', 'spaghetti_max_infill_angle', 'spaghetti_max_height', 'spaghetti_inset', 'spaghetti_flow', 'spaghetti_infill_extra_volume', 'support_tree_enable'}

class VersionUpgrade462to47(VersionUpgrade):

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades preferences to have the new version number.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '15'
        if 'general' in parser and 'visible_settings' in parser['general']:
            settings = set(parser['general']['visible_settings'].split(';'))
            if 'support_tree_enable' in parser['general']['visible_settings']:
                settings.add('support_structure')
            settings.difference_update(_removed_settings)
            parser['general']['visible_settings'] = ';'.join(settings)
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        '\n        Upgrades instance containers to have the new version number.\n\n        This changes the maximum deviation setting if that setting was present\n        in the profile.\n        :param serialized: The original contents of the instance container.\n        :param filename: The original file name of the instance container.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '15'
        if 'values' in parser:
            if 'meshfix_maximum_deviation' in parser['values']:
                maximum_deviation = parser['values']['meshfix_maximum_deviation']
                if maximum_deviation.startswith('='):
                    maximum_deviation = maximum_deviation[1:]
                maximum_deviation = '=(' + maximum_deviation + ') / 2'
                parser['values']['meshfix_maximum_deviation'] = maximum_deviation
            if 'ironing_inset' in parser['values']:
                ironing_inset = parser['values']['ironing_inset']
                if ironing_inset.startswith('='):
                    ironing_inset = ironing_inset[1:]
                if 'ironing_pattern' in parser['values'] and parser['values']['ironing_pattern'] == 'concentric':
                    correction = ' + ironing_line_spacing - skin_line_width * (1.0 + ironing_flow / 100) / 2'
                else:
                    correction = ' + skin_line_width * (1.0 - ironing_flow / 100) / 2'
                ironing_inset = '=(' + ironing_inset + ')' + correction
                parser['values']['ironing_inset'] = ironing_inset
            if 'support_tree_enable' in parser['values']:
                if parseBool(parser['values']['support_tree_enable']):
                    parser['values']['support_structure'] = 'tree'
                    parser['values']['support_enable'] = 'True'
            for removed in set(parser['values'].keys()).intersection(_removed_settings):
                del parser['values'][removed]
        if 'definition' in parser['general'] and parser['general']['definition'] in _RENAMED_DEFINITION_DICT:
            parser['general']['definition'] = _RENAMED_DEFINITION_DICT[parser['general']['definition']]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Upgrades stacks to have the new version number.\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '15'
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
                if script_id in ['BQ_PauseAtHeight', 'PauseAtHeightRepRapFirmwareDuet', 'PauseAtHeightforRepetier']:
                    script_settings = script_parser.items(script_id)
                    script_settings.append(('pause_method', {'BQ_PauseAtHeight': 'bq', 'PauseAtHeightforRepetier': 'repetier', 'PauseAtHeightRepRapFirmwareDuet': 'reprap'}[script_id]))
                    script_parser.remove_section(script_id)
                    script_id = 'PauseAtHeight'
                    script_parser.add_section(script_id)
                    for setting_tuple in script_settings:
                        script_parser.set(script_id, setting_tuple[0], setting_tuple[1])
                if 'PauseAtHeight' in script_parser:
                    if 'redo_layers' in script_parser['PauseAtHeight']:
                        script_parser['PauseAtHeight']['redo_layer'] = str(int(script_parser['PauseAtHeight']['redo_layers']) > 0)
                        del script_parser['PauseAtHeight']['redo_layers']
                if script_id == 'DisplayRemainingTimeOnLCD':
                    was_enabled = parseBool(script_parser[script_id]['TurnOn']) if 'TurnOn' in script_parser[script_id] else False
                    script_parser.remove_section(script_id)
                    script_id = 'DisplayProgressOnLCD'
                    script_parser.add_section(script_id)
                    if was_enabled:
                        script_parser.set(script_id, 'time_remaining', 'True')
                script_io = io.StringIO()
                script_parser.write(script_io)
                script_str = script_io.getvalue()
                script_str = script_str.replace('\\\\', '\\\\\\\\').replace('\n', '\\\\\\n')
                new_scripts_entries.append(script_str)
            parser['metadata']['post_processing_scripts'] = '\n'.join(new_scripts_entries)
        if parser.has_option('containers', '7') and parser['containers']['7'] in _RENAMED_DEFINITION_DICT:
            parser['containers']['7'] = _RENAMED_DEFINITION_DICT[parser['containers']['7']]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])