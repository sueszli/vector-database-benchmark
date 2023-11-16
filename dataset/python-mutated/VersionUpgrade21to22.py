import configparser
from typing import Dict, Iterable, List, Optional, Set, Tuple
from UM.VersionUpgrade import VersionUpgrade
from . import MachineInstance
from . import Preferences
from . import Profile
_machines_with_machine_quality = {'ultimaker2plus': {'materials': {'generic_abs', 'generic_cpe', 'generic_pla', 'generic_pva', 'generic_cpe_plus', 'generic_nylon', 'generic_pc', 'generic_tpu'}, 'variants': {'0.25 mm', '0.4 mm', '0.6 mm', '0.8 mm'}}, 'ultimaker2_extended_plus': {'materials': {'generic_abs', 'generic_cpe', 'generic_pla', 'generic_pva', 'generic_cpe_plus', 'generic_nylon', 'generic_pc', 'generic_tpu'}, 'variants': {'0.25 mm', '0.4 mm', '0.6 mm', '0.8 mm'}}}
_material_translations = {'PLA': 'generic_pla', 'ABS': 'generic_abs', 'CPE': 'generic_cpe', 'CPE+': 'generic_cpe_plus', 'Nylon': 'generic_nylon', 'PC': 'generic_pc', 'TPU': 'generic_tpu'}
_material_translations_profiles = {'PLA': 'pla', 'ABS': 'abs', 'CPE': 'cpe', 'CPE+': 'cpep', 'Nylon': 'nylon', 'PC': 'pc', 'TPU': 'tpu'}
_printer_translations = {'ultimaker2plus': 'ultimaker2_plus'}
_printer_translations_profiles = {'ultimaker2plus': 'um2p', 'ultimaker2_extended_plus': 'um2ep'}
_profile_translations = {'Low Quality': 'low', 'Normal Quality': 'normal', 'High Quality': 'high', 'Ulti Quality': 'high', 'abs_0.25_normal': 'um2p_abs_0.25_normal', 'abs_0.4_fast': 'um2p_abs_0.4_fast', 'abs_0.4_high': 'um2p_abs_0.4_high', 'abs_0.4_normal': 'um2p_abs_0.4_normal', 'abs_0.6_normal': 'um2p_abs_0.6_normal', 'abs_0.8_normal': 'um2p_abs_0.8_normal', 'cpe_0.25_normal': 'um2p_cpe_0.25_normal', 'cpe_0.4_fast': 'um2p_cpe_0.4_fast', 'cpe_0.4_high': 'um2p_cpe_0.4_high', 'cpe_0.4_normal': 'um2p_cpe_0.4_normal', 'cpe_0.6_normal': 'um2p_cpe_0.6_normal', 'cpe_0.8_normal': 'um2p_cpe_0.8_normal', 'cpep_0.4_draft': 'um2p_cpep_0.4_draft', 'cpep_0.4_normal': 'um2p_cpep_0.4_normal', 'cpep_0.6_draft': 'um2p_cpep_0.6_draft', 'cpep_0.6_normal': 'um2p_cpep_0.6_normal', 'cpep_0.8_draft': 'um2p_cpep_0.8_draft', 'cpep_0.8_normal': 'um2p_cpep_0.8_normal', 'nylon_0.25_high': 'um2p_nylon_0.25_high', 'nylon_0.25_normal': 'um2p_nylon_0.25_normal', 'nylon_0.4_fast': 'um2p_nylon_0.4_fast', 'nylon_0.4_normal': 'um2p_nylon_0.4_normal', 'nylon_0.6_fast': 'um2p_nylon_0.6_fast', 'nylon_0.6_normal': 'um2p_nylon_0.6_normal', 'nylon_0.8_draft': 'um2p_nylon_0.8_draft', 'nylon_0.8_normal': 'um2p_nylon_0.8_normal', 'pc_0.25_high': 'um2p_pc_0.25_high', 'pc_0.25_normal': 'um2p_pc_0.25_normal', 'pc_0.4_fast': 'um2p_pc_0.4_fast', 'pc_0.4_normal': 'um2p_pc_0.4_normal', 'pc_0.6_fast': 'um2p_pc_0.6_fast', 'pc_0.6_normal': 'um2p_pc_0.6_normal', 'pc_0.8_draft': 'um2p_pc_0.8_draft', 'pc_0.8_normal': 'um2p_pc_0.8_normal', 'pla_0.25_normal': 'pla_0.25_normal', 'pla_0.4_fast': 'pla_0.4_fast', 'pla_0.4_high': 'pla_0.4_high', 'pla_0.4_normal': 'pla_0.4_normal', 'pla_0.6_normal': 'pla_0.6_normal', 'pla_0.8_normal': 'pla_0.8_normal', 'tpu_0.25_high': 'um2p_tpu_0.25_high', 'tpu_0.4_normal': 'um2p_tpu_0.4_normal', 'tpu_0.6_fast': 'um2p_tpu_0.6_fast'}
_removed_settings = {'fill_perimeter_gaps', 'support_area_smoothing'}
_setting_name_translations = {'remove_overlapping_walls_0_enabled': 'travel_compensate_overlapping_walls_0_enabled', 'remove_overlapping_walls_enabled': 'travel_compensate_overlapping_walls_enabled', 'remove_overlapping_walls_x_enabled': 'travel_compensate_overlapping_walls_x_enabled', 'retraction_hop': 'retraction_hop_enabled', 'skin_overlap': 'infill_overlap', 'skirt_line_width': 'skirt_brim_line_width', 'skirt_minimal_length': 'skirt_brim_minimal_length', 'skirt_speed': 'skirt_brim_speed', 'speed_support_lines': 'speed_support_infill', 'speed_support_roof': 'speed_support_interface', 'support_roof_density': 'support_interface_density', 'support_roof_enable': 'support_interface_enable', 'support_roof_extruder_nr': 'support_interface_extruder_nr', 'support_roof_line_distance': 'support_interface_line_distance', 'support_roof_line_width': 'support_interface_line_width', 'support_roof_pattern': 'support_interface_pattern'}
_quality_fallbacks = {'ultimaker2_plus': {'ultimaker2_plus_0.25': {'generic_abs': 'um2p_abs_0.25_normal', 'generic_cpe': 'um2p_cpe_0.25_normal', 'generic_nylon': 'um2p_nylon_0.25_normal', 'generic_pc': 'um2p_pc_0.25_normal', 'generic_pla': 'pla_0.25_normal', 'generic_tpu': 'um2p_tpu_0.25_high'}, 'ultimaker2_plus_0.4': {'generic_abs': 'um2p_abs_0.4_normal', 'generic_cpe': 'um2p_cpe_0.4_normal', 'generic_cpep': 'um2p_cpep_0.4_normal', 'generic_nylon': 'um2p_nylon_0.4_normal', 'generic_pc': 'um2p_pc_0.4_normal', 'generic_pla': 'pla_0.4_normal', 'generic_tpu': 'um2p_tpu_0.4_normal'}, 'ultimaker2_plus_0.6': {'generic_abs': 'um2p_abs_0.6_normal', 'generic_cpe': 'um2p_cpe_0.6_normal', 'generic_cpep': 'um2p_cpep_0.6_normal', 'generic_nylon': 'um2p_nylon_0.6_normal', 'generic_pc': 'um2p_pc_0.6_normal', 'generic_pla': 'pla_0.6_normal', 'generic_tpu': 'um2p_tpu_0.6_fast'}, 'ultimaker2_plus_0.8': {'generic_abs': 'um2p_abs_0.8_normal', 'generic_cpe': 'um2p_cpe_0.8_normal', 'generic_cpep': 'um2p_cpep_0.8_normal', 'generic_nylon': 'um2p_nylon_0.8_normal', 'generic_pc': 'um2p_pc_0.8_normal', 'generic_pla': 'pla_0.8_normal'}}}
_variant_translations = {'ultimaker2_plus': {'0.25 mm': 'ultimaker2_plus_0.25', '0.4 mm': 'ultimaker2_plus_0.4', '0.6 mm': 'ultimaker2_plus_0.6', '0.8 mm': 'ultimaker2_plus_0.8'}, 'ultimaker2_extended_plus': {'0.25 mm': 'ultimaker2_extended_plus_0.25', '0.4 mm': 'ultimaker2_extended_plus_0.4', '0.6 mm': 'ultimaker2_extended_plus_0.6', '0.8 mm': 'ultimaker2_extended_plus_0.8'}}
_variant_translations_profiles = {'0.25 mm': '0.25', '0.4 mm': '0.4', '0.6 mm': '0.6', '0.8 mm': '0.8'}
_variant_translations_materials = {'ultimaker2_plus': {'0.25 mm': 'ultimaker2_plus_0.25_mm', '0.4 mm': 'ultimaker2_plus_0.4_mm', '0.6 mm': 'ultimaker2_plus_0.6_mm', '0.8 mm': 'ultimaker2_plus_0.8_mm'}, 'ultimaker2_extended_plus': {'0.25 mm': 'ultimaker2_plus_0.25_mm', '0.4 mm': 'ultimaker2_plus_0.4_mm', '0.6 mm': 'ultimaker2_plus_0.6_mm', '0.8 mm': 'ultimaker2_plus_0.8_mm'}}

class VersionUpgrade21to22(VersionUpgrade):

    @staticmethod
    def getQualityFallback(machine: str, variant: str, material: str) -> str:
        if False:
            while True:
                i = 10
        if machine not in _quality_fallbacks:
            return 'normal'
        if variant not in _quality_fallbacks[machine]:
            return 'normal'
        if material not in _quality_fallbacks[machine][variant]:
            return 'normal'
        return _quality_fallbacks[machine][variant][material]

    @staticmethod
    def builtInProfiles() -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        return _profile_translations.keys()

    @staticmethod
    def machinesWithMachineQuality() -> Dict[str, Dict[str, Set[str]]]:
        if False:
            print('Hello World!')
        return _machines_with_machine_quality

    def upgradeMachineInstance(self, serialised: str, filename: str) -> Optional[Tuple[List[str], List[str]]]:
        if False:
            return 10
        machine_instance = MachineInstance.importFrom(serialised, filename)
        if not machine_instance:
            return None
        return machine_instance.export()

    def upgradePreferences(self, serialised: str, filename: str) -> Optional[Tuple[List[str], List[str]]]:
        if False:
            i = 10
            return i + 15
        preferences = Preferences.importFrom(serialised, filename)
        if not preferences:
            return None
        return preferences.export()

    def upgradeProfile(self, serialised: str, filename: str) -> Optional[Tuple[List[str], List[str]]]:
        if False:
            print('Hello World!')
        profile = Profile.importFrom(serialised, filename)
        if not profile:
            return None
        return profile.export()

    @staticmethod
    def translateMaterial(material: str) -> str:
        if False:
            return 10
        if material in _material_translations:
            return _material_translations[material]
        return material

    @staticmethod
    def translateMaterialForProfiles(material: str) -> str:
        if False:
            return 10
        if material in _material_translations_profiles:
            return _material_translations_profiles[material]
        return material

    @staticmethod
    def translatePrinter(printer: str) -> str:
        if False:
            i = 10
            return i + 15
        if printer in _printer_translations:
            return _printer_translations[printer]
        return printer

    @staticmethod
    def translatePrinterForProfile(printer: str) -> str:
        if False:
            return 10
        if printer in _printer_translations_profiles:
            return _printer_translations_profiles[printer]
        return printer

    @staticmethod
    def translateProfile(profile: str) -> str:
        if False:
            print('Hello World!')
        if profile in _profile_translations:
            return _profile_translations[profile]
        return profile

    @staticmethod
    def translateSettings(settings: Dict[str, str]) -> Dict[str, str]:
        if False:
            print('Hello World!')
        new_settings = {}
        for (key, value) in settings.items():
            if key in _removed_settings:
                continue
            if key == 'retraction_combing':
                new_settings[key] = 'off' if value == 'False' else 'all'
                continue
            if key == 'cool_fan_full_layer':
                new_settings[key] = str(int(value) + 1)
                continue
            if key in _setting_name_translations:
                new_settings[_setting_name_translations[key]] = value
                continue
            new_settings[key] = value
        return new_settings

    @staticmethod
    def translateSettingName(setting: str) -> str:
        if False:
            while True:
                i = 10
        if setting in _setting_name_translations:
            return _setting_name_translations[setting]
        return setting

    @staticmethod
    def translateVariant(variant: str, machine: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if machine in _variant_translations and variant in _variant_translations[machine]:
            return _variant_translations[machine][variant]
        return variant

    @staticmethod
    def translateVariantForMaterials(variant: str, machine: str) -> str:
        if False:
            while True:
                i = 10
        if machine in _variant_translations_materials and variant in _variant_translations_materials[machine]:
            return _variant_translations_materials[machine][variant]
        return variant

    @staticmethod
    def translateVariantForProfiles(variant: str) -> str:
        if False:
            return 10
        if variant in _variant_translations_profiles:
            return _variant_translations_profiles[variant]
        return variant