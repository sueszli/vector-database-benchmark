import json
import os
import pytest
from typing import Any, Dict
import uuid
from unittest.mock import patch, MagicMock
import UM.Settings.ContainerRegistry
import UM.Settings.ContainerStack
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.VersionUpgradeManager import FilesDataUpdateResult
from UM.Resources import Resources
Resources.addSearchPath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources')))
machine_filepaths = sorted(os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'definitions')))
machine_filepaths = [os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'definitions', filename) for filename in machine_filepaths]
extruder_filepaths = sorted(os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'extruders')))
extruder_filepaths = [os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'extruders', filename) for filename in extruder_filepaths]
definition_filepaths = machine_filepaths + extruder_filepaths
all_meshes = os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'meshes'))
all_images = os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'images'))
cr = UM.Settings.ContainerRegistry.ContainerRegistry(None)

@pytest.fixture
def definition_container():
    if False:
        i = 10
        return i + 15
    uid = str(uuid.uuid4())
    result = UM.Settings.DefinitionContainer.DefinitionContainer(uid)
    assert result.getId() == uid
    return result

@pytest.mark.parametrize('file_path', definition_filepaths)
def test_definitionIds(file_path):
    if False:
        i = 10
        return i + 15
    '\n    Test the validity of the definition IDs.\n    :param file_path: The path of the machine definition to test.\n    '
    definition_id = os.path.basename(file_path).split('.')[0]
    assert ' ' not in definition_id, 'Definition located at [%s] contains spaces, this is now allowed!' % file_path

@pytest.mark.parametrize('file_path', definition_filepaths)
def test_noCategory(file_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Categories for definition files have been deprecated. Test that they are not\n    present.\n    :param file_path: The path of the machine definition to test.\n    '
    with open(file_path, encoding='utf-8') as f:
        json = f.read()
        metadata = DefinitionContainer.deserializeMetadata(json, 'test_container_id')
        assert 'category' not in metadata[0], 'Definition located at [%s] referenced a category, which is no longer allowed' % file_path

@pytest.mark.parametrize('file_path', machine_filepaths)
def test_validateMachineDefinitionContainer(file_path, definition_container):
    if False:
        while True:
            i = 10
    'Tests all definition containers'
    file_name = os.path.basename(file_path)
    if file_name == 'fdmprinter.def.json' or file_name == 'fdmextruder.def.json':
        return
    mocked_vum = MagicMock()
    mocked_vum.updateFilesData = lambda ct, v, fdl, fnl: FilesDataUpdateResult(ct, v, fdl, fnl)
    with patch('UM.VersionUpgradeManager.VersionUpgradeManager.getInstance', MagicMock(return_value=mocked_vum)):
        assertIsDefinitionValid(definition_container, file_path)

def assertIsDefinitionValid(definition_container, file_path):
    if False:
        i = 10
        return i + 15
    with open(file_path, encoding='utf-8') as data:
        json = data.read()
        (parser, is_valid) = definition_container.readAndValidateSerialized(json)
        assert is_valid
        metadata = DefinitionContainer.deserializeMetadata(json, 'whatever')
        if 'platform' in metadata[0]:
            assert metadata[0]['platform'] in all_meshes, 'Definition located at [%s] references a platform that could not be found' % file_path
        if 'platform_texture' in metadata[0]:
            assert metadata[0]['platform_texture'] in all_images, 'Definition located at [%s] references a platform_texture that could not be found' % file_path

@pytest.mark.parametrize('file_path', definition_filepaths)
def test_validateOverridingDefaultValue(file_path: str):
    if False:
        i = 10
        return i + 15
    'Tests whether setting values are not being hidden by parent containers.\n\n    When a definition container defines a "default_value" but inherits from a\n    definition that defines a "value", the "default_value" is ineffective. This\n    test fails on those things.\n    '
    with open(file_path, encoding='utf-8') as f:
        doc = json.load(f)
    if 'inherits' not in doc:
        return
    if 'overrides' not in doc:
        return
    parent_settings = getInheritedSettings(doc['inherits'])
    faulty_keys = set()
    for (key, val) in doc['overrides'].items():
        if key in parent_settings and 'value' in parent_settings[key]:
            if 'default_value' in val:
                faulty_keys.add(key)
    assert not faulty_keys, 'Unnecessary default_values for {faulty_keys} in {file_name}'.format(faulty_keys=sorted(faulty_keys), file_name=file_path)

def getInheritedSettings(definition_id: str) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    "Get all settings and their properties from a definition we're inheriting from.\n\n    :param definition_id: The definition we're inheriting from.\n    :return: A dictionary of settings by key. Each setting is a dictionary of properties.\n    "
    definition_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'definitions', definition_id + '.def.json')
    with open(definition_path, encoding='utf-8') as f:
        doc = json.load(f)
    result = {}
    if 'inherits' in doc:
        result.update(getInheritedSettings(doc['inherits']))
    if 'settings' in doc:
        result.update(flattenSettings(doc['settings']))
    if 'overrides' in doc:
        result = merge_dicts(result, doc['overrides'])
    return result

def flattenSettings(settings: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Put all settings in the main dictionary rather than in children dicts.\n\n    :param settings: Nested settings. The keys are the setting IDs. The values\n    are dictionaries of properties per setting, including the "children" property.\n    :return: A dictionary of settings by key. Each setting is a dictionary of properties.\n    '
    result = {}
    for (entry, contents) in settings.items():
        if 'children' in contents:
            result.update(flattenSettings(contents['children']))
            del contents['children']
        result[entry] = contents
    return result

def merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Make one dictionary override the other. Nested dictionaries override each\n\n    other in the same way.\n    :param base: A dictionary of settings that will get overridden by the other.\n    :param overrides: A dictionary of settings that will override the other.\n    :return: Combined setting data.\n    '
    result = {}
    result.update(base)
    for (key, val) in overrides.items():
        if key not in result:
            result[key] = val
            continue
        if isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result

@pytest.mark.parametrize('file_path', definition_filepaths)
def test_noId(file_path: str):
    if False:
        return 10
    "Verifies that definition contains don't have an ID field.\n\n    ID fields are legacy. They should not be used any more. This is legacy that\n    people don't seem to be able to get used to.\n    "
    with open(file_path, encoding='utf-8') as f:
        doc = json.load(f)
    assert 'id' not in doc, 'Definitions should not have an ID field.'

@pytest.mark.parametrize('file_path', extruder_filepaths)
def test_extruderMatch(file_path: str):
    if False:
        i = 10
        return i + 15
    '\n    Verifies that extruders say that they work on the same extruder_nr as what is listed in their machine definition.\n    '
    extruder_id = os.path.basename(file_path).split('.')[0]
    with open(file_path, encoding='utf-8') as f:
        doc = json.load(f)
    if 'metadata' not in doc:
        return
    if 'machine' not in doc['metadata'] or 'position' not in doc['metadata']:
        return
    machine = doc['metadata']['machine']
    position = doc['metadata']['position']
    for machine_filepath in machine_filepaths:
        machine_id = os.path.basename(machine_filepath).split('.')[0]
        if machine_id == machine:
            break
    else:
        assert False, 'The machine ID {machine} is not found.'.format(machine=machine)
    with open(machine_filepath, encoding='utf-8') as f:
        machine_doc = json.load(f)
    assert 'metadata' in machine_doc, 'Machine definition missing metadata entry.'
    assert 'machine_extruder_trains' in machine_doc['metadata'], 'Machine must define extruder trains.'
    extruder_trains = machine_doc['metadata']['machine_extruder_trains']
    assert position in extruder_trains, 'There must be a reference to the extruder in the machine definition.'
    assert extruder_trains[position] == extruder_id, 'The extruder referenced in the machine definition must match up.'
    if 'overrides' not in doc or 'extruder_nr' not in doc['overrides'] or 'default_value' not in doc['overrides']['extruder_nr']:
        assert position == '0'
    assert doc['overrides']['extruder_nr']['default_value'] == int(position)

@pytest.mark.parametrize('file_path', definition_filepaths)
def test_noNewSettings(file_path: str):
    if False:
        return 10
    "\n    Tests that a printer definition doesn't define any new settings.\n\n    Settings that are not common to all printers can cause Cura to crash, for instance when the setting is saved in a\n    profile and that profile is then used in a different printer.\n    :param file_path: A path to a definition file to test.\n    "
    filename = os.path.basename(file_path)
    if filename == 'fdmprinter.def.json' or filename == 'fdmextruder.def.json':
        return
    with open(file_path, encoding='utf-8') as f:
        doc = json.load(f)
    assert 'settings' not in doc