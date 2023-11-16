import configparser
import io
import os.path
from typing import Dict, List, Tuple
from UM.VersionUpgrade import VersionUpgrade
_renamed_settings = {'support_minimal_diameter': 'support_tower_maximum_supported_diameter'}
_removed_settings = ['prime_tower_circular', 'max_feedrate_z_override']
_renamed_profiles = {'creawsome_base': 'creality_base', 'creawsome_cr10': 'creality_cr10', 'creawsome_cr10mini': 'creality_cr10mini', 'creawsome_cr10s': 'creality_cr10s', 'creawsome_cr10s4': 'creality_cr10s4', 'creawsome_cr10s5': 'creality_cr10s5', 'creawsome_cr10spro': 'creality_cr10spro', 'creawsome_cr20': 'creality_cr20', 'creawsome_cr20pro': 'creality_cr20pro', 'creawsome_ender2': 'creality_ender2', 'creawsome_ender3': 'creality_ender3', 'creawsome_ender4': 'creality_ender4', 'creawsome_ender5': 'creality_ender5', 'creawsome_base_extruder_0': 'creality_base_extruder_0', 'creawsome_base_0.2': 'creality_base_0.2', 'creawsome_base_0.3': 'creality_base_0.3', 'creawsome_base_0.4': 'creality_base_0.4', 'creawsome_base_0.5': 'creality_base_0.5', 'creawsome_base_0.6': 'creality_base_0.6', 'creawsome_base_0.8': 'creality_base_0.8', 'creawsome_base_1.0': 'creality_base_1.0', 'creawsome_cr10_0.2': 'creality_cr10_0.2', 'creawsome_cr10_0.3': 'creality_cr10_0.3', 'creawsome_cr10_0.4': 'creality_cr10_0.4', 'creawsome_cr10_0.5': 'creality_cr10_0.5', 'creawsome_cr10_0.6': 'creality_cr10_0.6', 'creawsome_cr10_0.8': 'creality_cr10_0.8', 'creawsome_cr10_1.0': 'creality_cr10_1.0', 'creawsome_cr10mini_0.2': 'creality_cr10mini_0.2', 'creawsome_cr10mini_0.3': 'creality_cr10mini_0.3', 'creawsome_cr10mini_0.4': 'creality_cr10mini_0.4', 'creawsome_cr10mini_0.5': 'creality_cr10mini_0.5', 'creawsome_cr10mini_0.6': 'creality_cr10mini_0.6', 'creawsome_cr10mini_0.8': 'creality_cr10mini_0.8', 'creawsome_cr10mini_1.0': 'creality_cr10mini_1.0', 'creawsome_cr10s4_0.2': 'creality_cr10s4_0.2', 'creawsome_cr10s4_0.3': 'creality_cr10s4_0.3', 'creawsome_cr10s4_0.4': 'creality_cr10s4_0.4', 'creawsome_cr10s4_0.5': 'creality_cr10s4_0.5', 'creawsome_cr10s4_0.6': 'creality_cr10s4_0.6', 'creawsome_cr10s4_0.8': 'creality_cr10s4_0.8', 'creawsome_cr10s4_1.0': 'creality_cr10s4_1.0', 'creawsome_cr10s5_0.2': 'creality_cr10s5_0.2', 'creawsome_cr10s5_0.3': 'creality_cr10s5_0.3', 'creawsome_cr10s5_0.4': 'creality_cr10s5_0.4', 'creawsome_cr10s5_0.5': 'creality_cr10s5_0.5', 'creawsome_cr10s5_0.6': 'creality_cr10s5_0.6', 'creawsome_cr10s5_0.8': 'creality_cr10s5_0.8', 'creawsome_cr10s5_1.0': 'creality_cr10s5_1.0', 'creawsome_cr10s_0.2': 'creality_cr10s_0.2', 'creawsome_cr10s_0.3': 'creality_cr10s_0.3', 'creawsome_cr10s_0.4': 'creality_cr10s_0.4', 'creawsome_cr10s_0.5': 'creality_cr10s_0.5', 'creawsome_cr10s_0.6': 'creality_cr10s_0.6', 'creawsome_cr10s_0.8': 'creality_cr10s_0.8', 'creawsome_cr10s_1.0': 'creality_cr10s_1.0', 'creawsome_cr10spro_0.2': 'creality_cr10spro_0.2', 'creawsome_cr10spro_0.3': 'creality_cr10spro_0.3', 'creawsome_cr10spro_0.4': 'creality_cr10spro_0.4', 'creawsome_cr10spro_0.5': 'creality_cr10spro_0.5', 'creawsome_cr10spro_0.6': 'creality_cr10spro_0.6', 'creawsome_cr10spro_0.8': 'creality_cr10spro_0.8', 'creawsome_cr10spro_1.0': 'creality_cr10spro_1.0', 'creawsome_cr20_0.2': 'creality_cr20_0.2', 'creawsome_cr20_0.3': 'creality_cr20_0.3', 'creawsome_cr20_0.4': 'creality_cr20_0.4', 'creawsome_cr20_0.5': 'creality_cr20_0.5', 'creawsome_cr20_0.6': 'creality_cr20_0.6', 'creawsome_cr20_0.8': 'creality_cr20_0.8', 'creawsome_cr20_1.0': 'creality_cr20_1.0', 'creawsome_cr20pro_0.2': 'creality_cr20pro_0.2', 'creawsome_cr20pro_0.3': 'creality_cr20pro_0.3', 'creawsome_cr20pro_0.4': 'creality_cr20pro_0.4', 'creawsome_cr20pro_0.5': 'creality_cr20pro_0.5', 'creawsome_cr20pro_0.6': 'creality_cr20pro_0.6', 'creawsome_cr20pro_0.8': 'creality_cr20pro_0.8', 'creawsome_cr20pro_1.0': 'creality_cr20pro_1.0', 'creawsome_ender2_0.2': 'creality_ender2_0.2', 'creawsome_ender2_0.3': 'creality_ender2_0.3', 'creawsome_ender2_0.4': 'creality_ender2_0.4', 'creawsome_ender2_0.5': 'creality_ender2_0.5', 'creawsome_ender2_0.6': 'creality_ender2_0.6', 'creawsome_ender2_0.8': 'creality_ender2_0.8', 'creawsome_ender2_1.0': 'creality_ender2_1.0', 'creawsome_ender3_0.2': 'creality_ender3_0.2', 'creawsome_ender3_0.3': 'creality_ender3_0.3', 'creawsome_ender3_0.4': 'creality_ender3_0.4', 'creawsome_ender3_0.5': 'creality_ender3_0.5', 'creawsome_ender3_0.6': 'creality_ender3_0.6', 'creawsome_ender3_0.8': 'creality_ender3_0.8', 'creawsome_ender3_1.0': 'creality_ender3_1.0', 'creawsome_ender4_0.2': 'creality_ender4_0.2', 'creawsome_ender4_0.3': 'creality_ender4_0.3', 'creawsome_ender4_0.4': 'creality_ender4_0.4', 'creawsome_ender4_0.5': 'creality_ender4_0.5', 'creawsome_ender4_0.6': 'creality_ender4_0.6', 'creawsome_ender4_0.8': 'creality_ender4_0.8', 'creawsome_ender4_1.0': 'creality_ender4_1.0', 'creawsome_ender5_0.2': 'creality_ender5_0.2', 'creawsome_ender5_0.3': 'creality_ender5_0.3', 'creawsome_ender5_0.4': 'creality_ender5_0.4', 'creawsome_ender5_0.5': 'creality_ender5_0.5', 'creawsome_ender5_0.6': 'creality_ender5_0.6', 'creawsome_ender5_0.8': 'creality_ender5_0.8', 'creawsome_ender5_1.0': 'creality_ender5_1.0', 'creality_cr10_extruder_0': 'creality_base_extruder_0', 'creality_cr10s4_extruder_0': 'creality_base_extruder_0', 'creality_cr10s5_extruder_0': 'creality_base_extruder_0', 'creality_ender3_extruder_0': 'creality_base_extruder_0'}
_creality_quality_per_material = {'generic_abs_175': {'high': 'base_0.4_ABS_super', 'normal': 'base_0.4_ABS_super', 'fast': 'base_0.4_ABS_super', 'draft': 'base_0.4_ABS_standard', 'extra_fast': 'base_0.4_ABS_low', 'coarse': 'base_0.4_ABS_low', 'extra_coarse': 'base_0.4_ABS_low'}, 'generic_petg_175': {'high': 'base_0.4_PETG_super', 'normal': 'base_0.4_PETG_super', 'fast': 'base_0.4_PETG_super', 'draft': 'base_0.4_PETG_standard', 'extra_fast': 'base_0.4_PETG_low', 'coarse': 'base_0.4_PETG_low', 'extra_coarse': 'base_0.4_PETG_low'}, 'generic_pla_175': {'high': 'base_0.4_PLA_super', 'normal': 'base_0.4_PLA_super', 'fast': 'base_0.4_PLA_super', 'draft': 'base_0.4_PLA_standard', 'extra_fast': 'base_0.4_PLA_low', 'coarse': 'base_0.4_PLA_low', 'extra_coarse': 'base_0.4_PLA_low'}, 'generic_tpu_175': {'high': 'base_0.4_TPU_super', 'normal': 'base_0.4_TPU_super', 'fast': 'base_0.4_TPU_super', 'draft': 'base_0.4_TPU_standard', 'extra_fast': 'base_0.4_TPU_standard', 'coarse': 'base_0.4_TPU_standard', 'extra_coarse': 'base_0.4_TPU_standard'}, 'empty_material': {'high': 'base_global_super', 'normal': 'base_global_super', 'fast': 'base_global_super', 'draft': 'base_global_standard', 'extra_fast': 'base_global_low', 'coarse': 'base_global_low', 'extra_coarse': 'base_global_low'}}
_default_variants = {'creality_cr10_extruder_0': 'creality_cr10_0.4', 'creality_cr10s4_extruder_0': 'creality_cr10s4_0.4', 'creality_cr10s5_extruder_0': 'creality_cr10s5_0.4', 'creality_ender3_extruder_0': 'creality_ender3_0.4'}
_quality_changes_to_creality_base = {'creality_cr10_extruder_0', 'creality_cr10s4_extruder_0', 'creality_cr10s5_extruder_0', 'creality_ender3_extruder_0', 'creality_cr10', 'creality_cr10s', 'creality_cr10s4', 'creality_cr10s5', 'creality_ender3'}
_creality_limited_quality_type = {'high': 'super', 'normal': 'super', 'fast': 'super', 'draft': 'draft', 'extra_fast': 'draft', 'coarse': 'draft', 'extra_coarse': 'draft'}

class VersionUpgrade41to42(VersionUpgrade):
    """Upgrades configurations from the state they were in at version 4.1 to the

    state they should be in at version 4.2.
    """

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        'Upgrades instance containers to have the new version number.\n\n        This renames the renamed settings in the containers.\n        '
        parser = configparser.ConfigParser(interpolation=None, comment_prefixes=())
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '8'
        old_definition = parser['general']['definition']
        if old_definition in _renamed_profiles:
            parser['general']['definition'] = _renamed_profiles[old_definition]
        if 'values' in parser:
            for (old_name, new_name) in _renamed_settings.items():
                if old_name in parser['values']:
                    parser['values'][new_name] = parser['values'][old_name]
                    del parser['values'][old_name]
            for key in _removed_settings:
                if key in parser['values']:
                    del parser['values'][key]
        if parser['metadata'].get('type', '') == 'quality_changes':
            for possible_printer in _quality_changes_to_creality_base:
                if os.path.basename(filename).startswith(possible_printer + '_'):
                    parser['general']['definition'] = 'creality_base'
                    parser['metadata']['quality_type'] = _creality_limited_quality_type.get(parser['metadata']['quality_type'], 'draft')
                    break
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Upgrades Preferences to have the new version number.\n\n        This renames the renamed settings in the list of visible settings.\n        '
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['metadata']['setting_version'] = '8'
        if 'visible_settings' in parser['general']:
            visible_settings = parser['general']['visible_settings']
            visible_setting_set = set(visible_settings.split(';'))
            for (old_name, new_name) in _renamed_settings.items():
                if old_name in visible_setting_set:
                    visible_setting_set.remove(old_name)
                    visible_setting_set.add(new_name)
            for removed_key in _removed_settings:
                if removed_key in visible_setting_set:
                    visible_setting_set.remove(removed_key)
            parser['general']['visible_settings'] = ';'.join(visible_setting_set)
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
        parser['metadata']['setting_version'] = '8'
        if 'containers' in parser:
            definition_id = parser['containers']['6']
            if parser['metadata'].get('type', 'machine') == 'extruder_train':
                if parser['containers']['4'] == 'empty_variant':
                    if definition_id in _default_variants:
                        parser['containers']['4'] = _default_variants[definition_id]
                        if definition_id == 'creality_cr10_extruder_0':
                            if 'cr-10s' in parser['metadata'].get('machine', 'Creality CR-10').lower():
                                parser['containers']['4'] = 'creality_cr10s_0.4'
            material_id = parser['containers']['3']
            old_quality_id = parser['containers']['2']
            if material_id in _creality_quality_per_material and old_quality_id in _creality_quality_per_material[material_id]:
                if definition_id == 'creality_cr10_extruder_0':
                    if 'cr-10s' in parser['metadata'].get('machine', 'Creality CR-10').lower():
                        parser['containers']['2'] = _creality_quality_per_material[material_id][old_quality_id]
            stack_copy = {}
            stack_copy.update(parser['containers'])
            for (position, profile_id) in stack_copy.items():
                if profile_id in _renamed_profiles:
                    parser['containers'][position] = _renamed_profiles[profile_id]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])