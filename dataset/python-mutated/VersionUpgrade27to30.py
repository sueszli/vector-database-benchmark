import configparser
import io
import os
from typing import Dict, List, Tuple
import urllib.parse
import re
from UM.VersionUpgrade import VersionUpgrade
_renamed_themes = {'cura': 'cura-light'}
_renamed_i18n = {'7s': 'en_7S', 'de': 'de_DE', 'en': 'en_US', 'es': 'es_ES', 'fi': 'fi_FI', 'fr': 'fr_FR', 'hu': 'hu_HU', 'it': 'it_IT', 'jp': 'ja_JP', 'ko': 'ko_KR', 'nl': 'nl_NL', 'pl': 'pl_PL', 'ptbr': 'pt_BR', 'ru': 'ru_RU', 'tr': 'tr_TR'}

class VersionUpgrade27to30(VersionUpgrade):

    def getCfgVersion(self, serialised: str) -> int:
        if False:
            return 10
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        format_version = int(parser.get('general', 'version'))
        setting_version = int(parser.get('metadata', 'setting_version', fallback='0'))
        return format_version * 1000000 + setting_version

    def upgradePreferences(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if 'general' not in parser:
            parser['general'] = {}
        parser['general']['version'] = '5'
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '3'
        if 'theme' in parser['general']:
            if parser['general']['theme'] in _renamed_themes:
                parser['general']['theme'] = _renamed_themes[parser['general']['theme']]
        if 'language' in parser['general']:
            if parser['general']['language'] in _renamed_i18n:
                parser['general']['language'] = _renamed_i18n[parser['general']['language']]
        if parser.has_section('general') and 'visible_settings' in parser['general']:
            visible_settings = parser['general']['visible_settings'].split(';')
            new_visible_settings = []
            renamed_skin_preshrink_names = {'expand_upper_skins': 'top_skin_expand_distance', 'expand_lower_skins': 'bottom_skin_expand_distance'}
            for setting in visible_settings:
                if setting == 'expand_skins_into_infill':
                    continue
                if setting in renamed_skin_preshrink_names:
                    new_visible_settings.append(renamed_skin_preshrink_names[setting])
                    continue
                new_visible_settings.append(setting)
            parser['general']['visible_settings'] = ';'.join(new_visible_settings)
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def upgradeQualityChangesContainer(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if parser.has_section('values'):
            for remove_key in ['expand_skins_into_infill', 'expand_upper_skins', 'expand_lower_skins']:
                if remove_key in parser['values']:
                    del parser['values'][remove_key]
        for each_section in ('general', 'metadata'):
            if not parser.has_section(each_section):
                parser.add_section(each_section)
        if not parser.has_section('general'):
            parser.add_section('general')
        file_base_name = os.path.basename(filename)
        file_base_name = urllib.parse.unquote_plus(file_base_name)
        um2_pattern = re.compile('^ultimaker[^a-zA-Z\\\\d\\\\s:]2_.*$')
        ultimaker2_prefix_list = ['ultimaker2_extended_', 'ultimaker2_go_', 'ultimaker2_']
        exclude_prefix_list = ['ultimaker2_extended_plus_', 'ultimaker2_plus_']
        is_ultimaker2_family = um2_pattern.match(file_base_name) is not None
        if not is_ultimaker2_family and (not any((file_base_name.startswith(ep) for ep in exclude_prefix_list))):
            is_ultimaker2_family = any((file_base_name.startswith(ep) for ep in ultimaker2_prefix_list))
        if is_ultimaker2_family and parser['general']['definition'] == 'fdmprinter':
            parser['general']['definition'] = 'ultimaker2'
        parser['general']['version'] = '2'
        parser['metadata']['setting_version'] = '3'
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def upgradeOtherContainer(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            print('Hello World!')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if parser.has_section('values'):
            for remove_key in ['expand_skins_into_infill', 'expand_upper_skins', 'expand_lower_skins']:
                if remove_key in parser['values']:
                    del parser['values'][remove_key]
        for each_section in ('general', 'metadata'):
            if not parser.has_section(each_section):
                parser.add_section(each_section)
        parser['general']['version'] = '2'
        parser['metadata']['setting_version'] = '3'
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def upgradeStack(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        for each_section in ('general', 'metadata'):
            if not parser.has_section(each_section):
                parser.add_section(each_section)
        if 'general' not in parser:
            parser['general'] = {}
        parser['general']['version'] = '3'
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser['metadata']['setting_version'] = '3'
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])