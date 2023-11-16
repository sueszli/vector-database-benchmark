import configparser
import io
import json
import math
import os.path
from typing import Dict
from UM.Application import Application
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.InstanceContainer import InstanceContainer
from cura.ReaderWriters.ProfileReader import ProfileReader

class LegacyProfileReader(ProfileReader):
    """A plugin that reads profile data from legacy Cura versions.

    It reads a profile from an .ini file, and performs some translations on it.
    Not all translations are correct, mind you, but it is a best effort.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialises the legacy profile reader.\n\n        This does nothing since the only other function is basically stateless.\n        '
        super().__init__()

    def prepareDefaults(self, json: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'Prepares the default values of all legacy settings.\n\n        These are loaded from the Dictionary of Doom.\n\n        :param json: The JSON file to load the default setting values from. This\n            should not be a URL but a pre-loaded JSON handle.\n        :return: A dictionary of the default values of the legacy Cura version.\n        '
        defaults = {}
        if 'defaults' in json:
            for key in json['defaults']:
                defaults[key] = json['defaults'][key]
        return defaults

    def prepareLocals(self, config_parser, config_section, defaults):
        if False:
            i = 10
            return i + 15
        'Prepares the local variables that can be used in evaluation of computing\n\n        new setting values from the old ones.\n\n        This fills a dictionary with all settings from the legacy Cura version\n        and their values, so that they can be used in evaluating the new setting\n        values as Python code.\n\n        :param config_parser: The ConfigParser that finds the settings in the\n            legacy profile.\n        :param config_section: The section in the profile where the settings\n            should be found.\n        :param defaults: The default values for all settings in the legacy Cura.\n        :return: A set of local variables, one for each setting in the legacy\n            profile.\n        '
        copied_locals = defaults.copy()
        for option in config_parser.options(config_section):
            copied_locals[option] = config_parser.get(config_section, option)
        return copied_locals

    def read(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        "Reads a legacy Cura profile from a file and returns it.\n\n        :param file_name: The file to read the legacy Cura profile from.\n        :return: The legacy Cura profile that was in the file, if any. If the\n            file could not be read or didn't contain a valid profile,  None is returned.\n        "
        if file_name.split('.')[-1] != 'ini':
            return None
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if not global_container_stack:
            return None
        multi_extrusion = global_container_stack.getProperty('machine_extruder_count', 'value') > 1
        if multi_extrusion:
            Logger.log('e', 'Unable to import legacy profile %s. Multi extrusion is not supported', file_name)
            raise Exception('Unable to import legacy profile. Multi extrusion is not supported')
        Logger.log('i', 'Importing legacy profile from file ' + file_name + '.')
        container_registry = ContainerRegistry.getInstance()
        profile_id = container_registry.uniqueName('Imported Legacy Profile')
        input_parser = configparser.ConfigParser(interpolation=None)
        try:
            input_parser.read([file_name])
        except Exception as e:
            Logger.log('e', 'Unable to open legacy profile %s: %s', file_name, str(e))
            return None
        section = ''
        for found_section in input_parser.sections():
            if found_section.startswith('profile'):
                section = found_section
                break
        if not section:
            return None
        try:
            with open(os.path.join(PluginRegistry.getInstance().getPluginPath('LegacyProfileReader'), 'DictionaryOfDoom.json'), 'r', encoding='utf-8') as f:
                dict_of_doom = json.load(f)
        except IOError as e:
            Logger.log('e', 'Could not open DictionaryOfDoom.json for reading: %s', str(e))
            return None
        except Exception as e:
            Logger.log('e', 'Could not parse DictionaryOfDoom.json: %s', str(e))
            return None
        defaults = self.prepareDefaults(dict_of_doom)
        legacy_settings = self.prepareLocals(input_parser, section, defaults)
        output_parser = configparser.ConfigParser(interpolation=None)
        output_parser.add_section('general')
        output_parser.add_section('metadata')
        output_parser.add_section('values')
        if 'translation' not in dict_of_doom:
            Logger.log('e', 'Dictionary of Doom has no translation. Is it the correct JSON file?')
            return None
        current_printer_definition = global_container_stack.definition
        quality_definition = current_printer_definition.getMetaDataEntry('quality_definition')
        if not quality_definition:
            quality_definition = current_printer_definition.getId()
        output_parser['general']['definition'] = quality_definition
        for new_setting in dict_of_doom['translation']:
            old_setting_expression = dict_of_doom['translation'][new_setting]
            compiled = compile(old_setting_expression, new_setting, 'eval')
            try:
                new_value = eval(compiled, {'math': math}, legacy_settings)
                value_using_defaults = eval(compiled, {'math': math}, defaults)
            except Exception:
                Logger.log('w', 'Setting ' + new_setting + ' could not be set because the evaluation failed. Something is probably missing from the imported legacy profile.')
                continue
            definitions = current_printer_definition.findDefinitions(key=new_setting)
            if definitions:
                if new_value != value_using_defaults and definitions[0].default_value != new_value:
                    output_parser['values'][new_setting] = str(new_value)
        if len(output_parser['values']) == 0:
            Logger.log('i', 'A legacy profile was imported but everything evaluates to the defaults, creating an empty profile.')
        output_parser['general']['version'] = '4'
        output_parser['general']['name'] = profile_id
        output_parser['metadata']['type'] = 'quality_changes'
        output_parser['metadata']['quality_type'] = 'normal'
        output_parser['metadata']['position'] = '0'
        output_parser['metadata']['setting_version'] = '5'
        stream = io.StringIO()
        output_parser.write(stream)
        data = stream.getvalue()
        profile = InstanceContainer(profile_id)
        profile.deserialize(data, file_name)
        profile.setDirty(True)
        global_container_id = container_registry.uniqueName('Global Imported Legacy Profile')
        global_profile = profile.duplicate(new_id=global_container_id, new_name=profile_id)
        del global_profile.getMetaData()['position']
        global_profile.setDirty(True)
        profile_definition = 'fdmprinter'
        from UM.Util import parseBool
        if parseBool(global_container_stack.getMetaDataEntry('has_machine_quality', 'False')):
            profile_definition = global_container_stack.getMetaDataEntry('quality_definition')
            if not profile_definition:
                profile_definition = global_container_stack.definition.getId()
        global_profile.setDefinition(profile_definition)
        return [global_profile]