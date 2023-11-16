import configparser
import io
from typing import Dict, List, Set, Tuple
from UM.VersionUpgrade import VersionUpgrade
deleted_settings = {'prime_tower_wall_thickness', 'dual_pre_wipe', 'prime_tower_purge_volume'}
changed_settings = {'retraction_combing': 'noskin'}
updated_settings = {'retraction_combing': 'infill'}
_RENAMED_MATERIAL_PROFILES = {'dsm_arnitel2045_175_cartesio_0.25_mm': 'dsm_arnitel2045_175_cartesio_0.25mm_thermoplastic_extruder', 'dsm_arnitel2045_175_cartesio_0.4_mm': 'dsm_arnitel2045_175_cartesio_0.4mm_thermoplastic_extruder', 'dsm_arnitel2045_175_cartesio_0.8_mm': 'dsm_arnitel2045_175_cartesio_0.8mm_thermoplastic_extruder', 'dsm_novamid1070_175_cartesio_0.25_mm': 'dsm_novamid1070_175_cartesio_0.25mm_thermoplastic_extruder', 'dsm_novamid1070_175_cartesio_0.4_mm': 'dsm_novamid1070_175_cartesio_0.4mm_thermoplastic_extruder', 'dsm_novamid1070_175_cartesio_0.8_mm': 'dsm_novamid1070_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_abs_175_cartesio_0.25_mm': 'generic_abs_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_abs_175_cartesio_0.4_mm': 'generic_abs_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_abs_175_cartesio_0.8_mm': 'generic_abs_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_hips_175_cartesio_0.25_mm': 'generic_hips_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_hips_175_cartesio_0.4_mm': 'generic_hips_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_hips_175_cartesio_0.8_mm': 'generic_hips_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_nylon_175_cartesio_0.25_mm': 'generic_nylon_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_nylon_175_cartesio_0.4_mm': 'generic_nylon_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_nylon_175_cartesio_0.8_mm': 'generic_nylon_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_pc_cartesio_0.25_mm': 'generic_pc_cartesio_0.25mm_thermoplastic_extruder', 'generic_pc_cartesio_0.4_mm': 'generic_pc_cartesio_0.4mm_thermoplastic_extruder', 'generic_pc_cartesio_0.8_mm': 'generic_pc_cartesio_0.8mm_thermoplastic_extruder', 'generic_pc_175_cartesio_0.25_mm': 'generic_pc_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_pc_175_cartesio_0.4_mm': 'generic_pc_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_pc_175_cartesio_0.8_mm': 'generic_pc_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_petg_175_cartesio_0.25_mm': 'generic_petg_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_petg_175_cartesio_0.4_mm': 'generic_petg_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_petg_175_cartesio_0.8_mm': 'generic_petg_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_pla_175_cartesio_0.25_mm': 'generic_pla_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_pla_175_cartesio_0.4_mm': 'generic_pla_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_pla_175_cartesio_0.8_mm': 'generic_pla_175_cartesio_0.8mm_thermoplastic_extruder', 'generic_pva_cartesio_0.25_mm': 'generic_pva_cartesio_0.25mm_thermoplastic_extruder', 'generic_pva_cartesio_0.4_mm': 'generic_pva_cartesio_0.4mm_thermoplastic_extruder', 'generic_pva_cartesio_0.8_mm': 'generic_pva_cartesio_0.8mm_thermoplastic_extruder', 'generic_pva_175_cartesio_0.25_mm': 'generic_pva_175_cartesio_0.25mm_thermoplastic_extruder', 'generic_pva_175_cartesio_0.4_mm': 'generic_pva_175_cartesio_0.4mm_thermoplastic_extruder', 'generic_pva_175_cartesio_0.8_mm': 'generic_pva_175_cartesio_0.8mm_thermoplastic_extruder', 'ultimaker_pc_black_cartesio_0.25_mm': 'ultimaker_pc_black_cartesio_0.25mm_thermoplastic_extruder', 'ultimaker_pc_black_cartesio_0.4_mm': 'ultimaker_pc_black_cartesio_0.4mm_thermoplastic_extruder', 'ultimaker_pc_black_cartesio_0.8_mm': 'ultimaker_pc_black_cartesio_0.8mm_thermoplastic_extruder', 'ultimaker_pc_transparent_cartesio_0.25_mm': 'ultimaker_pc_transparent_cartesio_0.25mm_thermoplastic_extruder', 'ultimaker_pc_transparent_cartesio_0.4_mm': 'ultimaker_pc_transparent_cartesio_0.4mm_thermoplastic_extruder', 'ultimaker_pc_transparent_cartesio_0.8_mm': 'ultimaker_pc_transparent_cartesio_0.8mm_thermoplastic_extruder', 'ultimaker_pc_white_cartesio_0.25_mm': 'ultimaker_pc_white_cartesio_0.25mm_thermoplastic_extruder', 'ultimaker_pc_white_cartesio_0.4_mm': 'ultimaker_pc_white_cartesio_0.4mm_thermoplastic_extruder', 'ultimaker_pc_white_cartesio_0.8_mm': 'ultimaker_pc_white_cartesio_0.8mm_thermoplastic_extruder', 'ultimaker_pva_cartesio_0.25_mm': 'ultimaker_pva_cartesio_0.25mm_thermoplastic_extruder', 'ultimaker_pva_cartesio_0.4_mm': 'ultimaker_pva_cartesio_0.4mm_thermoplastic_extruder', 'ultimaker_pva_cartesio_0.8_mm': 'ultimaker_pva_cartesio_0.8mm_thermoplastic_extruder'}

class VersionUpgrade34to35(VersionUpgrade):

    def upgradePreferences(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        if parser.has_option('info', 'asked_send_slice_info'):
            parser.set('info', 'asked_send_slice_info', 'False')
        if parser.has_option('info', 'send_slice_info'):
            parser.set('info', 'send_slice_info', 'True')
        parser['general']['version'] = '6'
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '5'
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeStack(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '4'
        parser['metadata']['setting_version'] = '5'
        if parser['containers']['3'] in _RENAMED_MATERIAL_PROFILES:
            parser['containers']['3'] = _RENAMED_MATERIAL_PROFILES[parser['containers']['3']]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '4'
        parser['metadata']['setting_version'] = '5'
        self._resetConcentric3DInfillPattern(parser)
        if 'values' in parser:
            for deleted_setting in deleted_settings:
                if deleted_setting not in parser['values']:
                    continue
                del parser['values'][deleted_setting]
            for setting_key in changed_settings:
                if setting_key not in parser['values']:
                    continue
                if parser['values'][setting_key] == changed_settings[setting_key]:
                    parser['values'][setting_key] = updated_settings[setting_key]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])

    def _resetConcentric3DInfillPattern(self, parser: configparser.ConfigParser) -> None:
        if False:
            return 10
        if 'values' not in parser:
            return
        for key in ('infill_pattern', 'support_pattern', 'support_interface_pattern', 'support_roof_pattern', 'support_bottom_pattern'):
            if key not in parser['values']:
                continue
            if parser['values'][key] == 'concentric_3d':
                del parser['values'][key]