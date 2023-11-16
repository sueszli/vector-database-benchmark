import os
import pytest
import unittest.mock
from cura.ReaderWriters.ProfileReader import NoProfileException
from cura.Settings.ExtruderStack import ExtruderStack
from cura.Settings.GlobalStack import GlobalStack
import UM.Settings.InstanceContainer
import UM.Settings.ContainerRegistry
import UM.Settings.ContainerStack
import cura.CuraApplication

def teardown():
    if False:
        for i in range(10):
            print('nop')
    temporary_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stacks', 'temporary.stack.cfg')
    if os.path.isfile(temporary_file):
        os.remove(temporary_file)

def test_createUniqueName(container_registry):
    if False:
        for i in range(10):
            print('nop')
    from cura.CuraApplication import CuraApplication
    assert container_registry.createUniqueName('user', 'test', 'test2', 'nope') == 'test2'
    instance = UM.Settings.InstanceContainer.InstanceContainer(container_id='test2')
    instance.setMetaDataEntry('type', 'user')
    instance.setMetaDataEntry('setting_version', CuraApplication.SettingVersion)
    container_registry.addContainer(instance)
    assert container_registry.createUniqueName('user', 'test', 'test2', 'nope') == 'test2 #2'
    assert container_registry.createUniqueName('user', 'test', 'test2 #2', 'nope') == 'test2 #2'
    assert container_registry.createUniqueName('user', 'test', '', 'nope') == 'nope'

def test_addContainerExtruderStack(container_registry, definition_container, definition_changes_container):
    if False:
        return 10
    'Tests whether addContainer properly converts to ExtruderStack.'
    container_registry.addContainer(definition_container)
    container_registry.addContainer(definition_changes_container)
    container_stack = ExtruderStack('Test Extruder Stack')
    container_stack.setMetaDataEntry('type', 'extruder_train')
    container_stack.setDefinition(definition_container)
    container_stack.setDefinitionChanges(definition_changes_container)
    mock_super_add_container = unittest.mock.MagicMock()
    with unittest.mock.patch('UM.Settings.ContainerRegistry.ContainerRegistry.addContainer', mock_super_add_container):
        container_registry.addContainer(container_stack)
    assert len(mock_super_add_container.call_args_list) == 1
    assert len(mock_super_add_container.call_args_list[0][0]) == 1
    assert type(mock_super_add_container.call_args_list[0][0][0]) == ExtruderStack

def test_addContainerGlobalStack(container_registry, definition_container, definition_changes_container):
    if False:
        for i in range(10):
            print('nop')
    'Tests whether addContainer properly converts to GlobalStack.'
    container_registry.addContainer(definition_container)
    container_registry.addContainer(definition_changes_container)
    container_stack = GlobalStack('Test Global Stack')
    container_stack.setMetaDataEntry('type', 'machine')
    container_stack.setDefinition(definition_container)
    container_stack.setDefinitionChanges(definition_changes_container)
    mock_super_add_container = unittest.mock.MagicMock()
    with unittest.mock.patch('UM.Settings.ContainerRegistry.ContainerRegistry.addContainer', mock_super_add_container):
        container_registry.addContainer(container_stack)
    assert len(mock_super_add_container.call_args_list) == 1
    assert len(mock_super_add_container.call_args_list[0][0]) == 1
    assert type(mock_super_add_container.call_args_list[0][0][0]) == GlobalStack

def test_addContainerGoodSettingVersion(container_registry, definition_container):
    if False:
        while True:
            i = 10
    from cura.CuraApplication import CuraApplication
    definition_container.getMetaData()['setting_version'] = CuraApplication.SettingVersion
    container_registry.addContainer(definition_container)
    instance = UM.Settings.InstanceContainer.InstanceContainer(container_id='Test Instance Right Version')
    instance.setMetaDataEntry('setting_version', CuraApplication.SettingVersion)
    instance.setDefinition(definition_container.getId())
    mock_super_add_container = unittest.mock.MagicMock()
    with unittest.mock.patch('UM.Settings.ContainerRegistry.ContainerRegistry.addContainer', mock_super_add_container):
        container_registry.addContainer(instance)
    mock_super_add_container.assert_called_once_with(instance)

def test_addContainerNoSettingVersion(container_registry, definition_container):
    if False:
        print('Hello World!')
    from cura.CuraApplication import CuraApplication
    definition_container.getMetaData()['setting_version'] = CuraApplication.SettingVersion
    container_registry.addContainer(definition_container)
    instance = UM.Settings.InstanceContainer.InstanceContainer(container_id='Test Instance No Version')
    instance.setDefinition(definition_container.getId())
    mock_super_add_container = unittest.mock.MagicMock()
    with unittest.mock.patch('UM.Settings.ContainerRegistry.ContainerRegistry.addContainer', mock_super_add_container):
        container_registry.addContainer(instance)
    mock_super_add_container.assert_not_called()

def test_addContainerBadSettingVersion(container_registry, definition_container):
    if False:
        while True:
            i = 10
    from cura.CuraApplication import CuraApplication
    definition_container.getMetaData()['setting_version'] = CuraApplication.SettingVersion
    container_registry.addContainer(definition_container)
    instance = UM.Settings.InstanceContainer.InstanceContainer(container_id='Test Instance Wrong Version')
    instance.setMetaDataEntry('setting_version', 9001)
    instance.setDefinition(definition_container.getId())
    mock_super_add_container = unittest.mock.MagicMock()
    with unittest.mock.patch('UM.Settings.ContainerRegistry.ContainerRegistry.addContainer', mock_super_add_container):
        container_registry.addContainer(instance)
    mock_super_add_container.assert_not_called()
test_loadMetaDataValidation_data = [{'id': 'valid_container', 'is_valid': True, 'metadata': {'id': 'valid_container', 'setting_version': None, 'foo': 'bar'}}, {'id': 'wrong_setting_version', 'is_valid': False, 'metadata': {'id': 'wrong_setting_version', 'setting_version': '5', 'foo': 'bar'}}, {'id': 'missing_setting_version', 'is_valid': False, 'metadata': {'id': 'missing_setting_version', 'foo': 'bar'}}, {'id': 'unparsable_setting_version', 'is_valid': False, 'metadata': {'id': 'unparsable_setting_version', 'setting_version': 'Not an integer!', 'foo': 'bar'}}]

@pytest.mark.parametrize('parameters', test_loadMetaDataValidation_data)
def test_loadMetadataValidation(container_registry, definition_container, parameters):
    if False:
        i = 10
        return i + 15
    from cura.CuraApplication import CuraApplication
    definition_container.getMetaData()['setting_version'] = CuraApplication.SettingVersion
    container_registry.addContainer(definition_container)
    if 'setting_version' in parameters['metadata'] and parameters['metadata']['setting_version'] is None:
        parameters['metadata']['setting_version'] = CuraApplication.SettingVersion
    mock_provider = unittest.mock.MagicMock()
    mock_provider.getAllIds = unittest.mock.MagicMock(return_value=[parameters['id']])
    mock_provider.loadMetadata = unittest.mock.MagicMock(return_value=parameters['metadata'])
    container_registry._providers = [mock_provider]
    container_registry.loadAllMetadata()
    if parameters['is_valid']:
        assert parameters['id'] in container_registry.metadata
        assert container_registry.metadata[parameters['id']] == parameters['metadata']
    else:
        assert parameters['id'] not in container_registry.metadata

class TestExportQualityProfile:

    def test_exportQualityProfileInvalidFileType(self, container_registry):
        if False:
            return 10
        assert not container_registry.exportQualityProfile([], 'zomg', 'invalid')

    def test_exportQualityProfileFailedWriter(self, container_registry):
        if False:
            for i in range(10):
                print('nop')
        mocked_writer = unittest.mock.MagicMock(name='mocked_writer')
        mocked_writer.write = unittest.mock.MagicMock(return_value=False)
        container_registry._findProfileWriter = unittest.mock.MagicMock('findProfileWriter', return_value=mocked_writer)
        with unittest.mock.patch('UM.Application.Application.getInstance'):
            assert not container_registry.exportQualityProfile([], 'zomg', 'test files (*.tst)')

    def test_exportQualityProfileExceptionWriter(self, container_registry):
        if False:
            i = 10
            return i + 15
        mocked_writer = unittest.mock.MagicMock(name='mocked_writer')
        mocked_writer.write = unittest.mock.MagicMock(return_value=True, side_effect=Exception('Failed :('))
        container_registry._findProfileWriter = unittest.mock.MagicMock('findProfileWriter', return_value=mocked_writer)
        with unittest.mock.patch('UM.Application.Application.getInstance'):
            assert not container_registry.exportQualityProfile([], 'zomg', 'test files (*.tst)')

    def test_exportQualityProfileSuccessWriter(self, container_registry):
        if False:
            print('Hello World!')
        mocked_writer = unittest.mock.MagicMock(name='mocked_writer')
        mocked_writer.write = unittest.mock.MagicMock(return_value=True)
        container_registry._findProfileWriter = unittest.mock.MagicMock('findProfileWriter', return_value=mocked_writer)
        with unittest.mock.patch('UM.Application.Application.getInstance'):
            assert container_registry.exportQualityProfile([], 'zomg', 'test files (*.tst)')

def test__findProfileWriterNoPlugins(container_registry):
    if False:
        for i in range(10):
            print('nop')
    container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[])
    with unittest.mock.patch('UM.PluginRegistry.PluginRegistry.getInstance'):
        assert container_registry._findProfileWriter('.zomg', 'dunno') is None

def test__findProfileWriter(container_registry):
    if False:
        i = 10
        return i + 15
    container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[('writer_id', {'profile_writer': [{'extension': '.zomg', 'description': 'dunno'}]})])
    with unittest.mock.patch('UM.PluginRegistry.PluginRegistry.getInstance'):
        assert container_registry._findProfileWriter('.zomg', 'dunno') is not None

def test_importProfileEmptyFileName(container_registry):
    if False:
        i = 10
        return i + 15
    result = container_registry.importProfile('')
    assert result['status'] == 'error'
mocked_application = unittest.mock.MagicMock(name='application')
mocked_plugin_registry = unittest.mock.MagicMock(name='mocked_plugin_registry')

@unittest.mock.patch('UM.Application.Application.getInstance', unittest.mock.MagicMock(return_value=mocked_application))
@unittest.mock.patch('UM.PluginRegistry.PluginRegistry.getInstance', unittest.mock.MagicMock(return_value=mocked_plugin_registry))
class TestImportProfile:
    mocked_global_stack = unittest.mock.MagicMock(name='global stack')
    mocked_global_stack.getId = unittest.mock.MagicMock(return_value='blarg')
    mocked_profile_reader = unittest.mock.MagicMock()
    mocked_plugin_registry.getPluginObject = unittest.mock.MagicMock(return_value=mocked_profile_reader)

    def test_importProfileWithoutGlobalStack(self, container_registry):
        if False:
            i = 10
            return i + 15
        mocked_application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=None)
        result = container_registry.importProfile('non_empty')
        assert result['status'] == 'error'

    def test_importProfileNoProfileException(self, container_registry):
        if False:
            while True:
                i = 10
        container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[('reader_id', {'profile_reader': [{'extension': 'zomg', 'description': 'dunno'}]})])
        mocked_application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=self.mocked_global_stack)
        self.mocked_profile_reader.read = unittest.mock.MagicMock(side_effect=NoProfileException)
        result = container_registry.importProfile('test.zomg')
        assert result['status'] == 'ok'

    def test_importProfileGenericException(self, container_registry):
        if False:
            i = 10
            return i + 15
        container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[('reader_id', {'profile_reader': [{'extension': 'zomg', 'description': 'dunno'}]})])
        mocked_application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=self.mocked_global_stack)
        self.mocked_profile_reader.read = unittest.mock.MagicMock(side_effect=Exception)
        result = container_registry.importProfile('test.zomg')
        assert result['status'] == 'error'

    def test_importProfileNoDefinitionFound(self, container_registry):
        if False:
            i = 10
            return i + 15
        container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[('reader_id', {'profile_reader': [{'extension': 'zomg', 'description': 'dunno'}]})])
        mocked_application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=self.mocked_global_stack)
        container_registry.findDefinitionContainers = unittest.mock.MagicMock(return_value=[])
        mocked_profile = unittest.mock.MagicMock(name='Mocked_global_profile')
        self.mocked_profile_reader.read = unittest.mock.MagicMock(return_value=[mocked_profile])
        result = container_registry.importProfile('test.zomg')
        assert result['status'] == 'error'

    @pytest.mark.skip
    def test_importProfileSuccess(self, container_registry):
        if False:
            i = 10
            return i + 15
        container_registry._getIOPlugins = unittest.mock.MagicMock(return_value=[('reader_id', {'profile_reader': [{'extension': 'zomg', 'description': 'dunno'}]})])
        mocked_application.getGlobalContainerStack = unittest.mock.MagicMock(return_value=self.mocked_global_stack)
        mocked_definition = unittest.mock.MagicMock(name='definition')
        container_registry.findContainers = unittest.mock.MagicMock(return_value=[mocked_definition])
        container_registry.findDefinitionContainers = unittest.mock.MagicMock(return_value=[mocked_definition])
        mocked_profile = unittest.mock.MagicMock(name='Mocked_global_profile')
        self.mocked_profile_reader.read = unittest.mock.MagicMock(return_value=[mocked_profile])
        with unittest.mock.patch.object(container_registry, 'createUniqueName', return_value='derp'):
            with unittest.mock.patch.object(container_registry, '_configureProfile', return_value=None):
                result = container_registry.importProfile('test.zomg')
        assert result['status'] == 'ok'

@pytest.mark.parametrize('metadata,result', [(None, False), ({}, False), ({'setting_version': cura.CuraApplication.CuraApplication.SettingVersion, 'type': 'some_type', 'name': 'some_name'}, True), ({'setting_version': 0, 'type': 'some_type', 'name': 'some_name'}, False)])
def test_isMetaDataValid(container_registry, metadata, result):
    if False:
        return 10
    assert container_registry._isMetadataValid(metadata) == result

def test_getIOPlugins(container_registry):
    if False:
        while True:
            i = 10
    plugin_registry = unittest.mock.MagicMock()
    plugin_registry.getActivePlugins = unittest.mock.MagicMock(return_value=['lizard'])
    plugin_registry.getMetaData = unittest.mock.MagicMock(return_value={'zomg': {'test': 'test'}})
    with unittest.mock.patch('UM.PluginRegistry.PluginRegistry.getInstance', unittest.mock.MagicMock(return_value=plugin_registry)):
        assert container_registry._getIOPlugins('zomg') == [('lizard', {'zomg': {'test': 'test'}})]