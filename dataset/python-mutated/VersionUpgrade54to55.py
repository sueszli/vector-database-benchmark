import configparser
from typing import Tuple, List
import io
from UM.VersionUpgrade import VersionUpgrade
import re

class VersionUpgrade54to55(VersionUpgrade):
    profile_regex = re.compile('um\\_(?P<machine>s(3|5|7))_(?P<core_type>aa|cc|bb)(?P<nozzle_size>0\\.(6|4|8))_(?P<material>pla|petg|abs|tough_pla)_(?P<layer_height>0\\.\\d{1,2}mm)')

    @staticmethod
    def _isUpgradedUltimakerDefinitionId(definition_id: str) -> bool:
        if False:
            return 10
        if definition_id.startswith('ultimaker_s5'):
            return True
        if definition_id.startswith('ultimaker_s3'):
            return True
        if definition_id.startswith('ultimaker_s7'):
            return True
        return False

    @staticmethod
    def _isBrandedMaterialID(material_id: str) -> bool:
        if False:
            print('Hello World!')
        return material_id.startswith('ultimaker_')

    @staticmethod
    def upgradeStack(serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Upgrades stacks to have the new version number.\n\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if 'general' not in parser:
            parser['general'] = {}
        extruder_definition_id = parser['containers']['7']
        if parser['metadata']['type'] == 'extruder_train' and VersionUpgrade54to55._isUpgradedUltimakerDefinitionId(extruder_definition_id):
            material_id = parser['containers']['4']
            quality_id = parser['containers']['3']
            intent_id = parser['containers']['2']
            if VersionUpgrade54to55._isBrandedMaterialID(material_id):
                quality_id = VersionUpgrade54to55.profile_regex.sub('um_\\g<machine>_\\g<core_type>\\g<nozzle_size>_um-\\g<material>_\\g<layer_height>', quality_id)
                intent_id = VersionUpgrade54to55.profile_regex.sub('um_\\g<machine>_\\g<core_type>\\g<nozzle_size>_um-\\g<material>_\\g<layer_height>', intent_id)
            parser['containers']['3'] = quality_id
            parser['containers']['2'] = intent_id
        parser['general']['version'] = '6'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])