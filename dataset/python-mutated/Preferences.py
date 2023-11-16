import configparser
import io
from typing import List, Optional, Tuple
import UM.VersionUpgrade

def importFrom(serialised: str, filename: str) -> Optional['Preferences']:
    if False:
        print('Hello World!')
    try:
        return Preferences(serialised, filename)
    except (configparser.Error, UM.VersionUpgrade.FormatException, UM.VersionUpgrade.InvalidVersionException):
        return None

class Preferences:

    def __init__(self, serialised: str, filename: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._filename = filename
        self._config = configparser.ConfigParser(interpolation=None)
        self._config.read_string(serialised)
        if not self._config.has_section('general'):
            raise UM.VersionUpgrade.FormatException('No "general" section.')
        if not self._config.has_option('general', 'version'):
            raise UM.VersionUpgrade.FormatException('No "version" in "general" section.')
        if int(self._config.get('general', 'version')) != 2:
            raise UM.VersionUpgrade.InvalidVersionException('The version of this preferences file is wrong. It must be 2.')
        if self._config.has_option('general', 'name'):
            raise UM.VersionUpgrade.FormatException('There is a "name" field in this configuration file. I suspect it is not a preferences file.')

    def export(self) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        if self._config.has_section('cura') and self._config.has_option('cura', 'categories_expanded'):
            self._config.remove_option('cura', 'categories_expanded')
        if self._config.has_section('machines') and self._config.has_option('machines', 'setting_visibility'):
            visible_settings = self._config.get('machines', 'setting_visibility')
            visible_settings_list = visible_settings.split(',')
            import VersionUpgrade21to22
            visible_settings_list = [VersionUpgrade21to22.VersionUpgrade21to22.VersionUpgrade21to22.translateSettingName(setting_name) for setting_name in visible_settings_list]
            visible_settings = ','.join(visible_settings_list)
            self._config.set('machines', 'setting_visibility', value=visible_settings)
        if self._config.has_section('machines') and self._config.has_option('machines', 'active_instance'):
            active_machine = self._config.get('machines', 'active_instance')
            self._config.remove_option('machines', 'active_instance')
            self._config.set('cura', 'active_machine', active_machine)
        self._config.set('general', 'version', value='3')
        output = io.StringIO()
        self._config.write(output)
        return ([self._filename], [output.getvalue()])