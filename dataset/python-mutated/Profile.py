import configparser
import io
from typing import Dict, List, Optional, Tuple
import UM.VersionUpgrade

def importFrom(serialised: str, filename: str) -> Optional['Profile']:
    if False:
        return 10
    try:
        return Profile(serialised, filename)
    except (configparser.Error, UM.VersionUpgrade.FormatException, UM.VersionUpgrade.InvalidVersionException):
        return None

class Profile:

    def __init__(self, serialised: str, filename: str) -> None:
        if False:
            while True:
                i = 10
        self._filename = filename
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if not parser.has_section('general'):
            raise UM.VersionUpgrade.FormatException('No "general" section.')
        if not parser.has_option('general', 'version'):
            raise UM.VersionUpgrade.FormatException('No "version" in the "general" section.')
        if int(parser.get('general', 'version')) != 1:
            raise UM.VersionUpgrade.InvalidVersionException('The version of this profile is wrong. It must be 1.')
        self._name = parser.get('general', 'name')
        self._type = parser.get('general', 'type')
        self._weight = None
        if 'weight' in parser['general']:
            self._weight = int(parser.get('general', 'weight'))
        self._machine_type_id = parser.get('general', 'machine_type')
        self._machine_variant_name = parser.get('general', 'machine_variant')
        self._machine_instance_name = parser.get('general', 'machine_instance')
        self._material_name = None
        if 'material' in parser['general']:
            self._material_name = parser.get('general', 'material')
        elif self._type == 'material':
            self._material_name = parser.get('general', 'name')
        self._settings = {}
        if parser.has_section('settings'):
            for (key, value) in parser['settings'].items():
                self._settings[key] = value
        self._changed_settings_defaults = {}
        if parser.has_section('defaults'):
            for (key, value) in parser['defaults'].items():
                self._changed_settings_defaults[key] = value
        self._disabled_settings_defaults = []
        if parser.has_section('disabled_defaults'):
            disabled_defaults_string = parser.get('disabled_defaults', 'values')
            self._disabled_settings_defaults = [item for item in disabled_defaults_string.split(',') if item != '']

    def export(self) -> Optional[Tuple[List[str], List[str]]]:
        if False:
            for i in range(10):
                print('nop')
        import VersionUpgrade21to22
        if self._name == 'Current settings':
            return None
        config = configparser.ConfigParser(interpolation=None)
        config.add_section('general')
        config.set('general', 'version', '2')
        config.set('general', 'name', self._name)
        if self._machine_type_id:
            translated_machine = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translatePrinter(self._machine_type_id)
            config.set('general', 'definition', translated_machine)
        else:
            config.set('general', 'definition', 'fdmprinter')
        config.add_section('metadata')
        config.set('metadata', 'quality_type', 'normal')
        config.set('metadata', 'type', 'quality')
        if self._weight:
            config.set('metadata', 'weight', str(self._weight))
        if self._machine_variant_name:
            if self._machine_type_id:
                config.set('metadata', 'variant', VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateVariant(self._machine_variant_name, self._machine_type_id))
            else:
                config.set('metadata', 'variant', self._machine_variant_name)
        if self._settings:
            self._settings = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateSettings(self._settings)
            config.add_section('values')
            for (key, value) in self._settings.items():
                config.set('values', key, str(value))
        if self._changed_settings_defaults:
            self._changed_settings_defaults = VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateSettings(self._changed_settings_defaults)
            config.add_section('defaults')
            for (key, value) in self._changed_settings_defaults.items():
                config.set('defaults', key, str(value))
        if self._disabled_settings_defaults:
            disabled_settings_defaults = [VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateSettingName(setting) for setting in self._disabled_settings_defaults]
            config.add_section('disabled_defaults')
            disabled_defaults_string = str(disabled_settings_defaults[0])
            for item in disabled_settings_defaults[1:]:
                disabled_defaults_string += ',' + str(item)
        output = io.StringIO()
        config.write(output)
        return ([self._filename], [output.getvalue()])