import configparser
import io
import uuid
from typing import Dict, List, Tuple
from UM.VersionUpgrade import VersionUpgrade
_renamed_quality_profiles = {'gmax15plus_pla_dual_normal': 'gmax15plus_global_dual_normal', 'gmax15plus_pla_dual_thick': 'gmax15plus_global_dual_thick', 'gmax15plus_pla_dual_thin': 'gmax15plus_global_dual_thin', 'gmax15plus_pla_dual_very_thick': 'gmax15plus_global_dual_very_thick', 'gmax15plus_pla_normal': 'gmax15plus_global_normal', 'gmax15plus_pla_thick': 'gmax15plus_global_thick', 'gmax15plus_pla_thin': 'gmax15plus_global_thin', 'gmax15plus_pla_very_thick': 'gmax15plus_global_very_thick'}

class VersionUpgrade40to41(VersionUpgrade):
    """Upgrades configurations from the state they were in at version 4.0 to the
    state they should be in at version 4.1.
    """

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        'Upgrades instance containers to have the new version number.'
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '4'
        parser['metadata']['setting_version'] = '7'
        if 'values' in parser and 'meshfix_maximum_resolution' in parser['values']:
            resolution = parser['values']['meshfix_maximum_resolution']
            if resolution.startswith('='):
                resolution = resolution[1:]
            deviation = '=(' + resolution + ') / 2'
            parser['values']['meshfix_maximum_deviation'] = deviation
            del parser['values']['meshfix_maximum_resolution']
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        'Upgrades Preferences to have the new version number.'
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '6'
        if 'metadata' not in parser:
            parser['metadata'] = {}
        if 'latest_version_changelog_shown' in parser['general']:
            del parser['general']['latest_version_changelog_shown']
        parser['metadata']['setting_version'] = '7'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Upgrades stacks to have the new version number.'
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '4'
        parser['metadata']['setting_version'] = '7'
        if parser['containers']['4'] in _renamed_quality_profiles:
            parser['containers']['4'] = _renamed_quality_profiles[parser['containers']['4']]
        if 'um_network_key' in parser['metadata']:
            parser['metadata']['group_id'] = parser['metadata']['um_network_key']
        elif 'group_id' not in parser['metadata']:
            parser['metadata']['group_id'] = str(uuid.uuid4())
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])