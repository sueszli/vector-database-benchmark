import copy
from unittest import mock
import graphene
import pytest
from ....plugins.error_codes import PluginErrorCode
from ....plugins.manager import get_plugins_manager
from ....plugins.models import PluginConfiguration
from ....plugins.tests.sample_plugins import ChannelPluginSample, PluginSample
from ....plugins.tests.utils import get_config_value
from ...tests.utils import assert_no_permission, get_graphql_content
PLUGIN_UPDATE_MUTATION = '\n    mutation pluginUpdate(\n        $id: ID!\n        $active: Boolean\n        $channel: ID\n        $configuration: [ConfigurationItemInput!]\n    ) {\n        pluginUpdate(\n            id: $id\n            channelId: $channel\n            input: { active: $active, configuration: $configuration }\n        ) {\n            plugin {\n                name\n                description\n                globalConfiguration{\n                  active\n                  configuration{\n                    name\n                    value\n                    helpText\n                    type\n                    label\n                  }\n                  channel{\n                    id\n                    slug\n                  }\n                }\n                channelConfigurations{\n                  active\n                  channel{\n                    id\n                    slug\n                  }\n                  configuration{\n                    name\n                    value\n                    helpText\n                    type\n                    label\n                  }\n                }\n            }\n            errors {\n                field\n                message\n            }\n            pluginsErrors {\n                field\n                code\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('active', 'updated_configuration_item'), [(True, {'name': 'Username', 'value': 'user'}), (False, {'name': 'Username', 'value': 'admin@example.com'})])
def test_plugin_configuration_update(staff_api_client_can_manage_plugins, settings, active, updated_configuration_item):
    if False:
        return 10
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(PluginSample.PLUGIN_ID)
    old_configuration = copy.deepcopy(plugin.configuration)
    variables = {'id': plugin.PLUGIN_ID, 'active': active, 'channel': None, 'configuration': [updated_configuration_item]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    plugin_data = content['data']['pluginUpdate']['plugin']
    assert plugin_data['name'] == plugin.PLUGIN_NAME
    assert plugin_data['description'] == plugin.PLUGIN_DESCRIPTION
    plugin = PluginConfiguration.objects.get(identifier=PluginSample.PLUGIN_ID)
    assert plugin.active == active
    first_configuration_item = plugin.configuration[0]
    assert first_configuration_item['name'] == updated_configuration_item['name']
    assert first_configuration_item['value'] == updated_configuration_item['value']
    second_configuration_item = plugin.configuration[1]
    assert second_configuration_item['name'] == old_configuration[1]['name']
    assert second_configuration_item['value'] == old_configuration[1]['value']
    configuration = plugin_data['globalConfiguration']['configuration']
    assert configuration is not None
    assert configuration[0]['name'] == updated_configuration_item['name']
    assert configuration[0]['value'] == updated_configuration_item['value']

def test_plugin_configuration_update_value_not_given(staff_api_client_can_manage_plugins, settings):
    if False:
        while True:
            i = 10
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(PluginSample.PLUGIN_ID)
    old_configuration = copy.deepcopy(plugin.configuration)
    configuration_item = {'name': old_configuration[0]['name']}
    active = True
    variables = {'id': plugin.PLUGIN_ID, 'active': active, 'channel': None, 'configuration': [configuration_item]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    plugin_data = content['data']['pluginUpdate']['plugin']
    assert plugin_data['name'] == plugin.PLUGIN_NAME
    assert plugin_data['description'] == plugin.PLUGIN_DESCRIPTION
    plugin = PluginConfiguration.objects.get(identifier=PluginSample.PLUGIN_ID)
    assert plugin.active == active
    first_configuration_item = plugin.configuration[0]
    assert first_configuration_item['name'] == configuration_item['name']
    assert first_configuration_item['value'] == old_configuration[0]['value']
    second_configuration_item = plugin.configuration[1]
    assert second_configuration_item['name'] == old_configuration[1]['name']
    assert second_configuration_item['value'] == old_configuration[1]['value']
    configuration = plugin_data['globalConfiguration']['configuration']
    assert configuration is not None
    assert configuration[0]['name'] == configuration_item['name']
    assert configuration[0]['value'] == old_configuration[0]['value']

@pytest.mark.parametrize('active', [True, False])
def test_plugin_configuration_update_for_channel_configurations(staff_api_client_can_manage_plugins, settings, active, channel_PLN):
    if False:
        return 10
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.ChannelPluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(ChannelPluginSample.PLUGIN_ID, channel_slug=channel_PLN.slug)
    variables = {'id': plugin.PLUGIN_ID, 'active': active, 'channel': graphene.Node.to_global_id('Channel', channel_PLN.id), 'configuration': [{'name': 'input-per-channel', 'value': 'update-value'}]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    plugin_data = content['data']['pluginUpdate']['plugin']
    assert plugin_data['name'] == plugin.PLUGIN_NAME
    assert plugin_data['description'] == plugin.PLUGIN_DESCRIPTION
    assert len(plugin_data['channelConfigurations']) == 1
    api_configuration = plugin_data['channelConfigurations'][0]
    plugin = PluginConfiguration.objects.get(identifier=ChannelPluginSample.PLUGIN_ID)
    assert plugin.active == active == api_configuration['active']
    configuration_item = plugin.configuration[0]
    assert configuration_item['name'] == 'input-per-channel'
    assert configuration_item['value'] == 'update-value'
    configuration = api_configuration['configuration']
    assert len(configuration) == 1
    assert configuration[0]['name'] == configuration_item['name']
    assert configuration[0]['value'] == configuration_item['value']

def test_plugin_configuration_update_channel_slug_required(staff_api_client_can_manage_plugins, settings, channel_PLN):
    if False:
        print('Hello World!')
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.ChannelPluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(ChannelPluginSample.PLUGIN_ID, channel_slug=channel_PLN.slug)
    variables = {'id': plugin.PLUGIN_ID, 'active': True, 'channel': None, 'configuration': [{'name': 'input-per-channel', 'value': 'update-value'}]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert not content['data']['pluginUpdate']['plugin']
    assert len(content['data']['pluginUpdate']['pluginsErrors']) == 1
    error = content['data']['pluginUpdate']['pluginsErrors'][0]
    assert error['field'] == 'id'
    assert error['code'] == PluginErrorCode.NOT_FOUND.name

def test_plugin_configuration_update_unneeded_channel_slug(staff_api_client_can_manage_plugins, settings, channel_PLN):
    if False:
        for i in range(10):
            print('nop')
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(PluginSample.PLUGIN_ID, channel_slug=channel_PLN.slug)
    variables = {'id': plugin.PLUGIN_ID, 'active': True, 'channel': graphene.Node.to_global_id('Channel', channel_PLN.id), 'configuration': [{'name': 'input-per-channel', 'value': 'update-value'}]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert not content['data']['pluginUpdate']['plugin']
    assert len(content['data']['pluginUpdate']['pluginsErrors']) == 1
    error = content['data']['pluginUpdate']['pluginsErrors'][0]
    assert error['field'] == 'id'
    assert error['code'] == PluginErrorCode.INVALID.name

def test_plugin_configuration_update_containing_invalid_plugin_id(staff_api_client_can_manage_plugins):
    if False:
        while True:
            i = 10
    variables = {'id': 'fake-id', 'active': True, 'channel': None, 'configuration': [{'name': 'Username', 'value': 'user'}]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['pluginUpdate']['pluginsErrors'][0] == {'field': 'id', 'code': PluginErrorCode.NOT_FOUND.name}

def test_plugin_update_saves_boolean_as_boolean(staff_api_client_can_manage_plugins, settings):
    if False:
        for i in range(10):
            print('nop')
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(PluginSample.PLUGIN_ID)
    use_sandbox = get_config_value('Use sandbox', plugin.configuration)
    variables = {'id': plugin.PLUGIN_ID, 'active': plugin.active, 'channel': None, 'configuration': [{'name': 'Use sandbox', 'value': True}]}
    response = staff_api_client_can_manage_plugins.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert len(content['data']['pluginUpdate']['errors']) == 0
    use_sandbox_new_value = get_config_value('Use sandbox', plugin.configuration)
    assert type(use_sandbox) == type(use_sandbox_new_value)

def test_plugin_configuration_update_as_customer_user(user_api_client, settings):
    if False:
        while True:
            i = 10
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    manager = get_plugins_manager()
    plugin = manager.get_plugin(PluginSample.PLUGIN_ID)
    variables = {'id': plugin.PLUGIN_ID, 'active': True, 'channel': None, 'configuration': [{'name': 'Username', 'value': 'user'}]}
    response = user_api_client.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    assert_no_permission(response)

def test_cannot_update_configuration_of_hidden_plugin(settings, staff_api_client_can_manage_plugins):
    if False:
        i = 10
        return i + 15
    client = staff_api_client_can_manage_plugins
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.PluginSample']
    plugin_id = PluginSample.PLUGIN_ID
    original_config = get_plugins_manager().get_plugin(plugin_id).configuration
    variables = {'id': plugin_id, 'active': False, 'channel': None, 'configuration': [{'name': 'Username', 'value': 'MyNewUsername'}]}
    with mock.patch.object(PluginSample, 'HIDDEN', new=True):
        response = client.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    assert response.status_code == 200
    content = get_graphql_content(response)
    assert content['data']['pluginUpdate']['pluginsErrors'] == [{'code': 'NOT_FOUND', 'field': 'id'}]
    plugin = get_plugins_manager().get_plugin(plugin_id)
    assert plugin.active is True
    assert plugin.configuration == original_config
    response = client.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    assert response.status_code == 200
    content = get_graphql_content(response)
    assert content['data']['pluginUpdate']['pluginsErrors'] == []
    plugin = get_plugins_manager().get_plugin(plugin_id)
    assert plugin.active is False
    assert plugin.configuration != original_config

def test_cannot_update_configuration_of_hidden_multichannel_plugin(settings, staff_api_client_can_manage_plugins, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    client = staff_api_client_can_manage_plugins
    settings.PLUGINS = ['saleor.plugins.tests.sample_plugins.ChannelPluginSample']
    plugin_id = ChannelPluginSample.PLUGIN_ID
    original_config = get_plugins_manager().get_plugin(plugin_id, channel_slug=channel_USD.slug).configuration
    variables = {'id': plugin_id, 'active': False, 'channel': graphene.Node.to_global_id('Channel', channel_USD.id), 'configuration': [{'name': 'input-per-channel', 'value': 'NewValue'}]}
    with mock.patch.object(PluginSample, 'HIDDEN', new=True):
        response = client.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    assert response.status_code == 200
    content = get_graphql_content(response)
    assert content['data']['pluginUpdate']['pluginsErrors'] == [{'code': 'NOT_FOUND', 'field': 'id'}]
    plugin = get_plugins_manager().get_plugin(plugin_id, channel_slug=channel_USD.slug)
    assert plugin.active is True
    assert plugin.configuration == original_config
    response = client.post_graphql(PLUGIN_UPDATE_MUTATION, variables)
    assert response.status_code == 200
    content = get_graphql_content(response)
    assert content['data']['pluginUpdate']['pluginsErrors'] == []
    plugin = get_plugins_manager().get_plugin(plugin_id, channel_slug=channel_USD.slug)
    assert plugin.active is False
    assert plugin.configuration != original_config