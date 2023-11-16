import configparser
from typing import Tuple, List
import io
from UM.VersionUpgrade import VersionUpgrade
_removed_settings = {'travel_compensate_overlapping_walls_enabled', 'travel_compensate_overlapping_walls_0_enabled', 'travel_compensate_overlapping_walls_x_enabled', 'fill_perimeter_gaps', 'filter_out_tiny_gaps', 'wall_min_flow', 'wall_min_flow_retract', 'speed_equalize_flow_max'}
_transformed_settings = {'outer_inset_first': 'inset_direction', 'speed_equalize_flow_enabled': 'speed_equalize_flow_width_factor'}

class VersionUpgrade413to50(VersionUpgrade):

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades preferences to remove from the visibility list the settings that were removed in this version.\n        It also changes the preferences to have the new version number.\n\n        This removes any settings that were removed in the new Cura version.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '20'
        if 'general' in parser and 'visible_settings' in parser['general']:
            visible_settings = set(parser['general']['visible_settings'].split(';'))
            for removed in _removed_settings:
                if removed in visible_settings:
                    visible_settings.remove(removed)
            for (old, new) in _transformed_settings.items():
                if old in visible_settings:
                    visible_settings.remove(old)
                    visible_settings.add(new)
            parser['general']['visible_settings'] = ';'.join(visible_settings)
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        '\n        Upgrades instance containers to remove the settings that were removed in this version.\n        It also changes the instance containers to have the new version number.\n\n        This removes any settings that were removed in the new Cura version and updates settings that need to be updated\n        with a new value.\n\n        :param serialized: The original contents of the instance container.\n        :param filename: The original file name of the instance container.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '20'
        if 'values' in parser:
            for removed in _removed_settings:
                if removed in parser['values']:
                    del parser['values'][removed]
            if 'outer_inset_first' in parser['values']:
                old_value = parser['values']['outer_inset_first']
                if old_value.startswith('='):
                    old_value = old_value[1:]
                parser['values']['inset_direction'] = f"='outside_in' if ({old_value}) else 'inside_out'"
            if 'speed_equalize_flow_enabled' in parser['values']:
                old_value = parser['values']['speed_equalize_flow_enabled']
                if old_value.startswith('='):
                    old_value = old_value[1:]
                parser['values']['speed_equalize_flow_width_factor'] = f'=100 if ({old_value}) else 0'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Upgrades stacks to have the new version number.\n\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '20'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])