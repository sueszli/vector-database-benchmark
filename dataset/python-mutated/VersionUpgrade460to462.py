import configparser
import copy
from typing import Tuple, List
import io
from UM.VersionUpgrade import VersionUpgrade
renamed_nozzles = {'deltacomb_025_e3d': 'deltacomb_dc20_fbe025', 'deltacomb_040_e3d': 'deltacomb_dc20_fbe040', 'deltacomb_080_e3d': 'deltacomb_dc20_vfbe080'}
default_qualities_per_nozzle_and_material = {'deltacomb_dc20_fbe025': {'generic_pla_175': 'deltacomb_FBE0.25_PLA_C', 'generic_abs_175': 'deltacomb_FBE0.25_ABS_C'}, 'deltacomb_dc20_fbe040': {'generic_pla_175': 'deltacomb_FBE0.40_PLA_C', 'generic_abs_175': 'deltacomb_FBE0.40_ABS_C', 'generic_petg_175': 'deltacomb_FBE0.40_PETG_C', 'generic_tpu_175': 'deltacomb_FBE0.40_TPU_C'}, 'deltacomb_dc20_vfbe080': {'generic_pla_175': 'deltacomb_VFBE0.80_PLA_D', 'generic_abs_175': 'deltacomb_VFBE0.80_ABS_D'}}

class VersionUpgrade460to462(VersionUpgrade):

    def getCfgVersion(self, serialised: str) -> int:
        if False:
            while True:
                i = 10
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        format_version = int(parser.get('general', 'version'))
        setting_version = int(parser.get('metadata', 'setting_version', fallback='0'))
        return format_version * 1000000 + setting_version

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Upgrades preferences to have the new version number.\n        :param serialized: The original contents of the preferences file.\n        :param filename: The file name of the preferences file.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '14'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeExtruderInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Upgrades per-extruder instance containers to the new version number.\n\n        This applies all of the changes that are applied in other instance\n        containers as well.\n\n        In the case of Deltacomb printers, it splits the 2 extruders into 4 and\n        changes the definition.\n        :param serialized: The original contents of the instance container.\n        :param filename: The original file name of the instance container.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        results = [(parser, filename)]
        if 'general' in parser and 'definition' in parser['general']:
            if parser['general']['definition'] == 'deltacomb_extruder_0':
                parser['general']['definition'] = 'deltacomb_base_extruder_0'
            elif parser['general']['definition'] == 'deltacomb_extruder_1':
                parser_e2 = configparser.ConfigParser(interpolation=None)
                parser_e3 = configparser.ConfigParser(interpolation=None)
                parser_e2.read_dict(parser)
                parser_e3.read_dict(parser)
                parser['general']['definition'] = 'deltacomb_base_extruder_1'
                parser_e2['general']['definition'] = 'deltacomb_base_extruder_2'
                parser_e3['general']['definition'] = 'deltacomb_base_extruder_3'
                results.append((parser_e2, filename + '_e2_upgrade'))
                results.append((parser_e3, filename + '_e3_upgrade'))
            elif parser['general']['definition'] == 'deltacomb':
                parser['general']['definition'] = 'deltacomb_dc20'
                if 'metadata' in parser and ('extruder' in parser['metadata'] or 'position' in parser['metadata']):
                    parser_e2 = configparser.ConfigParser(interpolation=None)
                    parser_e3 = configparser.ConfigParser(interpolation=None)
                    parser_e2.read_dict(parser)
                    parser_e3.read_dict(parser)
                    if 'extruder' in parser['metadata']:
                        parser_e2['metadata']['extruder'] += '_e2_upgrade'
                        parser_e3['metadata']['extruder'] += '_e3_upgrade'
                    results.append((parser_e2, filename + '_e2_upgrade'))
                    results.append((parser_e3, filename + '_e3_upgrade'))
        final_serialized = []
        final_filenames = []
        for (result_parser, result_filename) in results:
            result_ss = io.StringIO()
            result_parser.write(result_ss)
            result_serialized = result_ss.getvalue()
            (this_filenames_upgraded, this_serialized_upgraded) = self.upgradeInstanceContainer(result_serialized, result_filename)
            final_serialized += this_serialized_upgraded
            final_filenames += this_filenames_upgraded
        return (final_filenames, final_serialized)

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades instance containers to have the new version number.\n\n        This changes the maximum deviation setting if that setting was present\n        in the profile.\n        :param serialized: The original contents of the instance container.\n        :param filename: The original file name of the instance container.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '14'
        if 'values' in parser:
            if 'meshfix_maximum_deviation' in parser['values']:
                maximum_deviation = parser['values']['meshfix_maximum_deviation']
                if maximum_deviation.startswith('='):
                    maximum_deviation = maximum_deviation[1:]
                maximum_deviation = '=(' + maximum_deviation + ') * 2'
                parser['values']['meshfix_maximum_deviation'] = maximum_deviation
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upgrades stacks to have the new version number.\n\n        This upgrades Deltacomb printers to their new profile structure, and\n        gives them 4 extruders.\n        :param serialized: The original contents of the stack.\n        :param filename: The original file name of the stack.\n        :return: A list of new file names, and a list of the new contents for\n        those files.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        results = [(parser, filename)]
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '14'
        if 'containers' in parser and '7' in parser['containers']:
            if parser['containers']['7'] == 'deltacomb_extruder_0' or parser['containers']['7'] == 'deltacomb_extruder_1':
                if '5' in parser['containers']:
                    parser['containers']['5'] = renamed_nozzles.get(parser['containers']['5'], parser['containers']['5'])
                    if '3' in parser['containers'] and '4' in parser['containers'] and (parser['containers']['3'] == 'empty_quality'):
                        parser['containers']['3'] = default_qualities_per_nozzle_and_material[parser['containers']['5']].get(parser['containers']['4'], 'empty_quality')
                if parser['containers']['7'] == 'deltacomb_extruder_0':
                    parser['containers']['7'] = 'deltacomb_base_extruder_0'
                else:
                    parser['containers']['7'] = 'deltacomb_base_extruder_1'
                    extruder3 = configparser.ConfigParser(interpolation=None)
                    extruder4 = configparser.ConfigParser(interpolation=None)
                    extruder3.read_dict(parser)
                    extruder4.read_dict(parser)
                    extruder3['general']['id'] += '_e2_upgrade'
                    extruder3['metadata']['position'] = '2'
                    extruder3['containers']['0'] += '_e2_upgrade'
                    if extruder3['containers']['1'] != 'empty_quality_changes':
                        extruder3['containers']['1'] += '_e2_upgrade'
                    extruder3['containers']['6'] += '_e2_upgrade'
                    extruder3['containers']['7'] = 'deltacomb_base_extruder_2'
                    results.append((extruder3, filename + '_e2_upgrade'))
                    extruder4['general']['id'] += '_e3_upgrade'
                    extruder4['metadata']['position'] = '3'
                    extruder4['containers']['0'] += '_e3_upgrade'
                    if extruder4['containers']['1'] != 'empty_quality_changes':
                        extruder4['containers']['1'] += '_e3_upgrade'
                    extruder4['containers']['6'] += '_e3_upgrade'
                    extruder4['containers']['7'] = 'deltacomb_base_extruder_3'
                    results.append((extruder4, filename + '_e3_upgrade'))
            elif parser['containers']['7'] == 'deltacomb':
                parser['containers']['7'] = 'deltacomb_dc20'
                parser['containers']['3'] = 'deltacomb_global_C'
        result_serialized = []
        result_filenames = []
        for (result_parser, result_filename) in results:
            result_ss = io.StringIO()
            result_parser.write(result_ss)
            result_serialized.append(result_ss.getvalue())
            result_filenames.append(result_filename)
        return (result_filenames, result_serialized)