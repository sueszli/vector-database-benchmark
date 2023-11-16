import configparser
import os.path
import pytest
import unittest.mock
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import UM.Application
import UM.PluginRegistry
import UM.Settings.ContainerRegistry
import UM.Settings.InstanceContainer
import LegacyProfileReader as LegacyProfileReaderModule

@pytest.fixture
def legacy_profile_reader():
    if False:
        i = 10
        return i + 15
    try:
        return LegacyProfileReaderModule.LegacyProfileReader()
    except TypeError:
        return LegacyProfileReaderModule.LegacyProfileReader.LegacyProfileReader()
test_prepareDefaultsData = [{'defaults': {'foo': 'bar'}, 'cheese': 'delicious'}, {'cat': 'fluffy', 'dog': 'floofy'}]

@pytest.mark.parametrize('input', test_prepareDefaultsData)
def test_prepareDefaults(legacy_profile_reader, input):
    if False:
        return 10
    output = legacy_profile_reader.prepareDefaults(input)
    if 'defaults' in input:
        assert input['defaults'] == output
    else:
        assert output == {}
test_prepareLocalsData = [({'profile': {'layer_height': '0.2', 'infill_density': '30'}}, {'layer_height': '0.1', 'infill_density': '20', 'line_width': '0.4'}), ({'profile': {}}, {}), ({'profile': {}}, {'foo': 'bar', 'boo': 'far'}), ({'some_other_name': {'foo': 'bar'}, 'profile': {'foo': 'baz'}}, {'foo': 'bla'})]

@pytest.mark.parametrize('parser_data, defaults', test_prepareLocalsData)
def test_prepareLocals(legacy_profile_reader, parser_data, defaults):
    if False:
        for i in range(10):
            print('nop')
    parser = configparser.ConfigParser()
    parser.read_dict(parser_data)
    output = legacy_profile_reader.prepareLocals(parser, 'profile', defaults)
    assert set(defaults.keys()) <= set(output.keys())
    assert set(parser_data['profile']) <= set(output.keys())
    for key in output:
        if key in parser_data['profile']:
            assert output[key] == parser_data['profile'][key]
        else:
            assert output[key] == defaults[key]
test_prepareLocalsNoSectionErrorData = [({'some_other_name': {'foo': 'bar'}}, {'foo': 'baz'})]

@pytest.mark.parametrize('parser_data, defaults', test_prepareLocalsNoSectionErrorData)
def test_prepareLocalsNoSectionError(legacy_profile_reader, parser_data, defaults):
    if False:
        return 10
    'Test cases where a key error is expected.'
    parser = configparser.ConfigParser()
    parser.read_dict(parser_data)
    with pytest.raises(configparser.NoSectionError):
        legacy_profile_reader.prepareLocals(parser, 'profile', defaults)
intercepted_data = ''

@pytest.mark.parametrize('file_name', ['normal_case.ini'])
def test_read(legacy_profile_reader, file_name):
    if False:
        print('Hello World!')
    global_stack = unittest.mock.MagicMock()
    global_stack.getProperty = unittest.mock.MagicMock(return_value=1)

    def getMetaDataEntry(key, default_value=''):
        if False:
            i = 10
            return i + 15
        if key == 'quality_definition':
            return 'mocked_quality_definition'
        if key == 'has_machine_quality':
            return 'True'
    global_stack.definition.getMetaDataEntry = getMetaDataEntry
    global_stack.definition.getId = unittest.mock.MagicMock(return_value='mocked_global_definition')
    application = unittest.mock.MagicMock()
    application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=global_stack)
    application_getInstance = unittest.mock.MagicMock(return_value=application)
    container_registry = unittest.mock.MagicMock()
    container_registry_getInstance = unittest.mock.MagicMock(return_value=container_registry)
    container_registry.uniqueName = unittest.mock.MagicMock(return_value='Imported Legacy Profile')
    container_registry.findDefinitionContainers = unittest.mock.MagicMock(return_value=[global_stack.definition])
    UM.Settings.InstanceContainer.setContainerRegistry(container_registry)
    plugin_registry = unittest.mock.MagicMock()
    plugin_registry_getInstance = unittest.mock.MagicMock(return_value=plugin_registry)
    plugin_registry.getPluginPath = unittest.mock.MagicMock(return_value=os.path.dirname(LegacyProfileReaderModule.__file__))

    def deserialize(self, data, filename):
        if False:
            print('Hello World!')
        global intercepted_data
        intercepted_data = data
        parser = configparser.ConfigParser()
        parser.read_string(data)
        self._metadata['position'] = parser['metadata']['position']

    def duplicate(self, new_id, new_name):
        if False:
            while True:
                i = 10
        self._metadata['id'] = new_id
        self._metadata['name'] = new_name
        return self
    with unittest.mock.patch.object(UM.Application.Application, 'getInstance', application_getInstance):
        with unittest.mock.patch.object(UM.Settings.ContainerRegistry.ContainerRegistry, 'getInstance', container_registry_getInstance):
            with unittest.mock.patch.object(UM.PluginRegistry.PluginRegistry, 'getInstance', plugin_registry_getInstance):
                with unittest.mock.patch.object(UM.Settings.InstanceContainer.InstanceContainer, 'deserialize', deserialize):
                    with unittest.mock.patch.object(UM.Settings.InstanceContainer.InstanceContainer, 'duplicate', duplicate):
                        result = legacy_profile_reader.read(os.path.join(os.path.dirname(__file__), file_name))
    assert len(result) == 1
    parser = configparser.ConfigParser()
    parser.read_string(intercepted_data)
    assert parser['general']['definition'] == 'mocked_quality_definition'
    assert parser['general']['version'] == '4'
    assert parser['general']['name'] == 'Imported Legacy Profile'
    assert parser['metadata']['type'] == 'quality_changes'
    assert parser['metadata']['quality_type'] == 'normal'
    assert parser['metadata']['position'] == '0'
    assert parser['metadata']['setting_version'] == '5'