import configparser
from typing import Tuple, List
import io
from UM.VersionUpgrade import VersionUpgrade

class VersionUpgrade47to48(VersionUpgrade):

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Upgrades preferences to have the new version number.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '16'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Upgrades instance containers to have the new version number.\n\n        This this also changes the unit of the Scaling Factor Shrinkage\n        Compensation setting.\n        :param serialized: The original contents of the instance container.\n        :param filename: The original file name of the instance container.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '16'
        if 'values' in parser:
            if 'material_shrinkage_percentage' in parser['values']:
                shrinkage_percentage = parser['values']['meshfix_maximum_deviation']
                if shrinkage_percentage.startswith('='):
                    shrinkage_percentage = shrinkage_percentage[1:]
                shrinkage_percentage = '=(' + shrinkage_percentage + ') + 100'
                parser['values']['material_shrinkage_percentage'] = shrinkage_percentage
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        '\n        Upgrades stacks to have the new version number.\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '16'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])