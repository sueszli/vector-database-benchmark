import configparser
import io
import os
from typing import Dict, List, Set, Tuple
from urllib.parse import quote_plus
from UM.Resources import Resources
from UM.VersionUpgrade import VersionUpgrade
_removed_settings = {'start_layers_at_same_position', 'sub_div_rad_mult'}
_split_settings = {'support_interface_line_distance': {'support_roof_line_distance', 'support_bottom_line_distance'}}

class VersionUpgrade25to26(VersionUpgrade):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self._current_fdm_printer_count = 2

    def upgradePreferences(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if parser.has_section('general') and 'visible_settings' in parser['general']:
            visible_settings = parser['general']['visible_settings'].split(';')
            new_visible_settings = []
            for setting in visible_settings:
                if setting in _removed_settings:
                    continue
                if setting in _split_settings:
                    for replaced_setting in _split_settings[setting]:
                        new_visible_settings.append(replaced_setting)
                    continue
                new_visible_settings.append(setting)
            parser['general']['visible_settings'] = ';'.join(new_visible_settings)
        if 'general' not in parser:
            parser['general'] = {}
        parser.set('general', 'version', '4')
        if 'metadata' not in parser:
            parser['metadata'] = {}
        parser.set('metadata', 'setting_version', '1')
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def upgradeInstanceContainer(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        if parser.has_section('values'):
            for removed_setting in _removed_settings & parser['values'].keys():
                del parser['values'][removed_setting]
            for replaced_setting in _split_settings.keys() & parser['values'].keys():
                for replacement in _split_settings[replaced_setting]:
                    parser['values'][replacement] = parser['values'][replaced_setting]
                del replaced_setting
        for each_section in ('general', 'metadata'):
            if not parser.has_section(each_section):
                parser.add_section(each_section)
        parser['general']['version'] = '2'
        parser['metadata']['setting_version'] = '1'
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def upgradeMachineStack(self, serialised: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialised)
        machine_id = parser['general']['id']
        quality_container_id = parser['containers']['2']
        material_container_id = parser['containers']['3']
        if '6' in parser['containers']:
            definition_container_id = parser['containers']['6']
        else:
            definition_container_id = parser['containers']['5']
        if definition_container_id == 'custom' and (not self._checkCustomFdmPrinterHasExtruderStack(machine_id)):
            self._acquireNextUniqueCustomFdmPrinterExtruderStackIdIndex()
            for position in range(8):
                self._createCustomFdmPrinterExtruderStack(machine_id, position, quality_container_id, material_container_id)
        parser['general']['version'] = '3'
        parser['metadata']['setting_version'] = '1'
        output = io.StringIO()
        parser.write(output)
        return ([filename], [output.getvalue()])

    def _acquireNextUniqueCustomFdmPrinterExtruderStackIdIndex(self) -> int:
        if False:
            print('Hello World!')
        extruder_stack_dir = os.path.join(Resources.getDataStoragePath(), 'extruders')
        file_name_list = os.listdir(extruder_stack_dir)
        file_name_list = [os.path.basename(file_name) for file_name in file_name_list]
        while True:
            self._current_fdm_printer_count += 1
            stack_id_exists = False
            for position in range(8):
                stack_id = 'custom_extruder_%s' % (position + 1)
                if self._current_fdm_printer_count > 1:
                    stack_id += ' #%s' % self._current_fdm_printer_count
                if stack_id in file_name_list:
                    stack_id_exists = True
                    break
            if not stack_id_exists:
                break
        return self._current_fdm_printer_count

    def _checkCustomFdmPrinterHasExtruderStack(self, machine_id: str) -> bool:
        if False:
            print('Hello World!')
        extruder_stack_dir = os.path.join(Resources.getDataStoragePath(), 'extruders')
        has_extruders = False
        for item in os.listdir(extruder_stack_dir):
            file_path = os.path.join(extruder_stack_dir, item)
            if not os.path.isfile(file_path):
                continue
            parser = configparser.ConfigParser()
            try:
                parser.read([file_path])
            except:
                continue
            if 'metadata' not in parser:
                continue
            if 'machine' not in parser['metadata']:
                continue
            if machine_id != parser['metadata']['machine']:
                continue
            has_extruders = True
            break
        return has_extruders

    def _createCustomFdmPrinterExtruderStack(self, machine_id: str, position: int, quality_id: str, material_id: str) -> None:
        if False:
            while True:
                i = 10
        stack_id = 'custom_extruder_%s' % (position + 1)
        if self._current_fdm_printer_count > 1:
            stack_id += ' #%s' % self._current_fdm_printer_count
        definition_id = 'custom_extruder_%s' % (position + 1)
        definition_changes_parser = self._getCustomFdmPrinterDefinitionChanges(stack_id)
        definition_changes_id = definition_changes_parser['general']['name']
        user_settings_parser = self._getCustomFdmPrinterUserSettings(stack_id)
        user_settings_id = user_settings_parser['general']['name']
        parser = configparser.ConfigParser()
        parser.add_section('general')
        parser['general']['version'] = str(2)
        parser['general']['name'] = 'Extruder %s' % (position + 1)
        parser['general']['id'] = stack_id
        parser.add_section('metadata')
        parser['metadata']['type'] = 'extruder_train'
        parser['metadata']['machine'] = machine_id
        parser['metadata']['position'] = str(position)
        parser.add_section('containers')
        parser['containers']['0'] = user_settings_id
        parser['containers']['1'] = 'empty_quality_changes'
        parser['containers']['2'] = quality_id
        parser['containers']['3'] = material_id
        parser['containers']['4'] = 'empty_variant'
        parser['containers']['5'] = definition_changes_id
        parser['containers']['6'] = definition_id
        definition_changes_output = io.StringIO()
        definition_changes_parser.write(definition_changes_output)
        definition_changes_filename = quote_plus(definition_changes_id) + '.inst.cfg'
        user_settings_output = io.StringIO()
        user_settings_parser.write(user_settings_output)
        user_settings_filename = quote_plus(user_settings_id) + '.inst.cfg'
        extruder_output = io.StringIO()
        parser.write(extruder_output)
        extruder_filename = quote_plus(stack_id) + '.extruder.cfg'
        extruder_stack_dir = os.path.join(Resources.getDataStoragePath(), 'extruders')
        definition_changes_dir = os.path.join(Resources.getDataStoragePath(), 'definition_changes')
        user_settings_dir = os.path.join(Resources.getDataStoragePath(), 'user')
        with open(os.path.join(definition_changes_dir, definition_changes_filename), 'w', encoding='utf-8') as f:
            f.write(definition_changes_output.getvalue())
        with open(os.path.join(user_settings_dir, user_settings_filename), 'w', encoding='utf-8') as f:
            f.write(user_settings_output.getvalue())
        with open(os.path.join(extruder_stack_dir, extruder_filename), 'w', encoding='utf-8') as f:
            f.write(extruder_output.getvalue())

    def _getCustomFdmPrinterDefinitionChanges(self, stack_id: str) -> configparser.ConfigParser:
        if False:
            i = 10
            return i + 15
        definition_changes_id = stack_id + '_settings'
        parser = configparser.ConfigParser()
        parser.add_section('general')
        parser['general']['version'] = str(2)
        parser['general']['name'] = definition_changes_id
        parser['general']['definition'] = 'custom'
        parser.add_section('metadata')
        parser['metadata']['type'] = 'definition_changes'
        parser['metadata']['setting_version'] = str(1)
        parser.add_section('values')
        return parser

    def _getCustomFdmPrinterUserSettings(self, stack_id: str) -> configparser.ConfigParser:
        if False:
            i = 10
            return i + 15
        user_settings_id = stack_id + '_user'
        parser = configparser.ConfigParser()
        parser.add_section('general')
        parser['general']['version'] = str(2)
        parser['general']['name'] = user_settings_id
        parser['general']['definition'] = 'custom'
        parser.add_section('metadata')
        parser['metadata']['extruder'] = stack_id
        parser['metadata']['type'] = 'user'
        parser['metadata']['setting_version'] = str(1)
        parser.add_section('values')
        return parser