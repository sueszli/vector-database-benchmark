import configparser
import io
import json
import os.path
from typing import List, Tuple
from UM.VersionUpgrade import VersionUpgrade

class VersionUpgrade411to412(VersionUpgrade):
    """
    Upgrades configurations from the state they were in at version 4.11 to the
    state they should be in at version 4.12.
    """
    _flsun_profile_mapping = {'extra_coarse': 'flsun_sr_normal', 'coarse': 'flsun_sr_normal', 'extra_fast': 'flsun_sr_normal', 'draft': 'flsun_sr_normal', 'fast': 'flsun_sr_normal', 'normal': 'flsun_sr_normal', 'high': 'flsun_sr_fine'}
    _flsun_quality_type_mapping = {'extra coarse': 'normal', 'coarse': 'normal', 'verydraft': 'normal', 'draft': 'normal', 'fast': 'normal', 'normal': 'normal', 'high': 'fine'}

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        '\n        Upgrades preferences to have the new version number.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '19'
        new_scopes = {'account.user.read', 'drive.backup.read', 'drive.backup.write', 'packages.download', 'packages.rating.read', 'packages.rating.write', 'connect.cluster.read', 'connect.cluster.write', 'library.project.read', 'library.project.write', 'cura.printjob.read', 'cura.printjob.write', 'cura.mesh.read', 'cura.mesh.write', 'cura.material.write'}
        if 'ultimaker_auth_data' in parser['general']:
            ultimaker_auth_data = json.loads(parser['general']['ultimaker_auth_data'])
            if new_scopes - set(ultimaker_auth_data['scope'].split(' ')):
                parser['general']['ultimaker_auth_data'] = '{}'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades instance containers to have the new version number.\n        :param serialized: The original contents of the instance container.\n        :param filename: The file name of the instance container.\n        :return: A list of file names, and a list of the new contents for those\n        files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '19'
        file_base_name = os.path.basename(filename)
        if file_base_name.startswith('flsun_sr_') and parser['metadata'].get('type') == 'quality_changes':
            if 'general' in parser and parser['general'].get('definition') == 'fdmprinter':
                old_quality_type = parser['metadata'].get('quality_type', 'normal')
                parser['general']['definition'] = 'flsun_sr'
                parser['metadata']['quality_type'] = self._flsun_quality_type_mapping.get(old_quality_type, 'normal')
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades container stacks to have the new version number.\n        Upgrades container stacks for FLSun Racer to change their profiles.\n        :param serialized: The original contents of the container stack.\n        :param filename: The file name of the container stack.\n        :return: A list of file names, and a list of the new contents for those\n        files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '19'
        if 'containers' in parser:
            definition_id = parser['containers'].get('7')
            if definition_id == 'flsun_sr':
                if parser['metadata'].get('type', 'machine') == 'machine':
                    old_quality = parser['containers'].get('3')
                    new_quality = self._flsun_profile_mapping.get(old_quality, 'flsun_sr_normal')
                    parser['containers']['3'] = new_quality
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])