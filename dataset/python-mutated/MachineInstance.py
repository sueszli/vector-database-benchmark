import configparser
import io
import os.path
from typing import Dict, List, Optional, Set, Tuple
import urllib
import urllib.parse
import UM.VersionUpgrade
import UM.VersionUpgradeManager
from UM.Resources import Resources

def importFrom(serialised: str, filename: str) -> Optional['MachineInstance']:
    if False:
        return 10
    try:
        return MachineInstance(serialised, filename)
    except (configparser.Error, UM.VersionUpgrade.FormatException, UM.VersionUpgrade.InvalidVersionException):
        return None

class MachineInstance:

    def __init__(self, serialised: str, filename: str) -> None:
        if False:
            print('Hello World!')
        self._filename = filename
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(serialised)
        if not config.has_section('general'):
            raise UM.VersionUpgrade.FormatException('No "general" section.')
        if not config.has_option('general', 'version'):
            raise UM.VersionUpgrade.FormatException('No "version" in "general" section.')
        if not config.has_option('general', 'name'):
            raise UM.VersionUpgrade.FormatException('No "name" in "general" section.')
        if not config.has_option('general', 'type'):
            raise UM.VersionUpgrade.FormatException('No "type" in "general" section.')
        if int(config.get('general', 'version')) != 1:
            raise UM.VersionUpgrade.InvalidVersionException('The version of this machine instance is wrong. It must be 1.')
        self._type_name = config.get('general', 'type')
        self._variant_name = config.get('general', 'variant', fallback='empty_variant')
        self._name = config.get('general', 'name', fallback='')
        self._key = config.get('general', 'key', fallback='')
        self._active_profile_name = config.get('general', 'active_profile', fallback='empty_quality')
        self._active_material_name = config.get('general', 'material', fallback='empty_material')
        self._machine_setting_overrides = {}
        for (key, value) in config['machine_settings'].items():
            self._machine_setting_overrides[key] = value

    def export(self) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        config = configparser.ConfigParser(interpolation=None)
        config.add_section('general')
        config.set('general', 'name', self._name)
        config.set('general', 'id', self._name)
        config.set('general', 'version', '2')
        import VersionUpgrade21to22
        has_machine_qualities = self._type_name in VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.machinesWithMachineQuality()
        type_name = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translatePrinter(self._type_name)
        active_material = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateMaterial(self._active_material_name)
        variant = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateVariant(self._variant_name, type_name)
        variant_materials = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateVariantForMaterials(self._variant_name, type_name)
        if self._active_profile_name in VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.builtInProfiles():
            active_quality = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateProfile(self._active_profile_name)
            active_quality_changes = 'empty_quality_changes'
        else:
            active_quality = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.getQualityFallback(type_name, variant, active_material)
            active_quality_changes = self._active_profile_name
        if has_machine_qualities:
            active_material += '_' + variant_materials
        user_profile = configparser.ConfigParser(interpolation=None)
        user_profile['general'] = {'version': '2', 'name': 'Current settings', 'definition': type_name}
        user_profile['metadata'] = {'type': 'user', 'machine': self._name}
        user_profile['values'] = {}
        version_upgrade_manager = UM.VersionUpgradeManager.VersionUpgradeManager.getInstance()
        user_version_to_paths_dict = version_upgrade_manager.getStoragePaths('user')
        paths_set = set()
        for paths in user_version_to_paths_dict.values():
            paths_set |= paths
        user_storage = os.path.join(Resources.getDataStoragePath(), next(iter(paths_set)))
        user_profile_file = os.path.join(user_storage, urllib.parse.quote_plus(self._name) + '_current_settings.inst.cfg')
        if not os.path.exists(user_storage):
            os.makedirs(user_storage)
        with open(user_profile_file, 'w', encoding='utf-8') as file_handle:
            user_profile.write(file_handle)
        version_upgrade_manager.upgradeExtraFile(user_storage, urllib.parse.quote_plus(self._name), 'user')
        containers = [self._name + '_current_settings', active_quality_changes, active_quality, active_material, variant, type_name]
        config.set('general', 'containers', ','.join(containers))
        config.add_section('metadata')
        config.set('metadata', 'type', 'machine')
        VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateSettings(self._machine_setting_overrides)
        config.add_section('values')
        for (key, value) in self._machine_setting_overrides.items():
            config.set('values', key, str(value))
        output = io.StringIO()
        config.write(output)
        return ([self._filename], [output.getvalue()])