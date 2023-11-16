"""Tests for the ckanext.example_iauthfunctions extension.

"""
import pytest
import ckan.plugins
import ckan.tests.factories as factories
import ckan.tests.helpers as helpers

@pytest.mark.ckan_config('ckan.plugins', 'example_iresourcecontroller')
@pytest.mark.usefixtures('non_clean_db', 'with_plugins')
class TestExampleIResourceController(object):
    """Tests for the plugin that uses IResourceController.

    """

    def test_resource_controller_plugin_create(self):
        if False:
            while True:
                i = 10
        user = factories.Sysadmin()
        package = factories.Dataset(user=user)
        plugin = ckan.plugins.get_plugin('example_iresourcecontroller')
        helpers.call_action('resource_create', package_id=package['id'], name='test-resource', url='http://resource.create/', apikey=user['apikey'])
        assert plugin.counter['before_resource_create'] == 1, plugin.counter
        assert plugin.counter['after_resource_create'] == 1, plugin.counter
        assert plugin.counter['before_resource_update'] == 0, plugin.counter
        assert plugin.counter['after_resource_update'] == 0, plugin.counter
        assert plugin.counter['before_resource_delete'] == 0, plugin.counter
        assert plugin.counter['after_resource_delete'] == 0, plugin.counter

    def test_resource_controller_plugin_update(self):
        if False:
            for i in range(10):
                print('nop')
        user = factories.Sysadmin()
        resource = factories.Resource(user=user)
        plugin = ckan.plugins.get_plugin('example_iresourcecontroller')
        helpers.call_action('resource_update', id=resource['id'], url='http://resource.updated/', apikey=user['apikey'])
        assert plugin.counter['before_resource_create'] == 1, plugin.counter
        assert plugin.counter['after_resource_create'] == 1, plugin.counter
        assert plugin.counter['before_resource_update'] == 1, plugin.counter
        assert plugin.counter['after_resource_update'] == 1, plugin.counter
        assert plugin.counter['before_resource_delete'] == 0, plugin.counter
        assert plugin.counter['after_resource_delete'] == 0, plugin.counter

    def test_resource_controller_plugin_delete(self):
        if False:
            print('Hello World!')
        user = factories.Sysadmin()
        resource = factories.Resource(user=user)
        plugin = ckan.plugins.get_plugin('example_iresourcecontroller')
        helpers.call_action('resource_delete', id=resource['id'], apikey=user['apikey'])
        assert plugin.counter['before_resource_create'] == 1, plugin.counter
        assert plugin.counter['after_resource_create'] == 1, plugin.counter
        assert plugin.counter['before_resource_update'] == 0, plugin.counter
        assert plugin.counter['after_resource_update'] == 0, plugin.counter
        assert plugin.counter['before_resource_delete'] == 1, plugin.counter
        assert plugin.counter['after_resource_delete'] == 1, plugin.counter

    def test_resource_controller_plugin_show(self):
        if False:
            print('Hello World!')
        "\n        Before show gets called by the other methods but we test it\n        separately here and make sure that it doesn't call the other\n        methods.\n        "
        user = factories.Sysadmin()
        package = factories.Dataset(user=user)
        factories.Resource(user=user, package_id=package['id'])
        plugin = ckan.plugins.get_plugin('example_iresourcecontroller')
        helpers.call_action('package_show', name_or_id=package['id'])
        assert plugin.counter['before_resource_create'] == 1, plugin.counter
        assert plugin.counter['after_resource_create'] == 1, plugin.counter
        assert plugin.counter['before_resource_update'] == 0, plugin.counter
        assert plugin.counter['after_resource_update'] == 0, plugin.counter
        assert plugin.counter['before_resource_delete'] == 0, plugin.counter
        assert plugin.counter['after_resource_delete'] == 0, plugin.counter
        assert plugin.counter['before_resource_show'] == 4, plugin.counter