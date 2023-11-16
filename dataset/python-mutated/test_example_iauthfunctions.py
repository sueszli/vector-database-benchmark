"""Tests for the ckanext.example_iauthfunctions extension.

"""
import pytest
import ckan.logic as logic
import ckan.tests.factories as factories
import ckan.tests.helpers as helpers
from ckan.plugins.toolkit import NotAuthorized, ObjectNotFound

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v6_parent_auth_functions')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
class TestAuthV6(object):

    def test_resource_delete_editor(self):
        if False:
            print('Hello World!')
        'Normally organization admins can delete resources\n        Our plugin prevents this by blocking delete organization.\n\n        Ensure the delete button is not displayed (as only resource delete\n        is checked for showing this)\n\n        '
        user = factories.User()
        owner_org = factories.Organization(users=[{'name': user['id'], 'capacity': 'admin'}])
        dataset = factories.Dataset(owner_org=owner_org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        with pytest.raises(logic.NotAuthorized) as e:
            logic.check_access('resource_delete', {'user': user['name']}, {'id': resource['id']})
        assert e.value.message == 'User %s not authorized to delete resource %s' % (user['name'], resource['id'])

    def test_resource_delete_sysadmin(self):
        if False:
            for i in range(10):
                print('nop')
        'Normally organization admins can delete resources\n        Our plugin prevents this by blocking delete organization.\n\n        Ensure the delete button is not displayed (as only resource delete\n        is checked for showing this)\n\n        '
        user = factories.Sysadmin()
        owner_org = factories.Organization(users=[{'name': user['id'], 'capacity': 'admin'}])
        dataset = factories.Dataset(owner_org=owner_org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        assert logic.check_access('resource_delete', {'user': user['name']}, {'id': resource['id']})

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v5_custom_config_setting')
@pytest.mark.ckan_config('ckan.iauthfunctions.users_can_create_groups', False)
@pytest.mark.usefixtures('clean_db', 'with_plugins')
class TestAuthV5(object):

    def test_sysadmin_can_create_group_when_config_is_false(self):
        if False:
            return 10
        sysadmin = factories.Sysadmin()
        context = {'ignore_auth': False, 'user': sysadmin['name']}
        helpers.call_action('group_create', context, name='test-group')

    def test_user_cannot_create_group_when_config_is_false(self):
        if False:
            while True:
                i = 10
        user = factories.User()
        context = {'ignore_auth': False, 'user': user['name']}
        with pytest.raises(NotAuthorized):
            helpers.call_action('group_create', context, name='test-group')

    def test_visitor_cannot_create_group_when_config_is_false(self):
        if False:
            while True:
                i = 10
        context = {'ignore_auth': False, 'user': None}
        with pytest.raises(NotAuthorized):
            helpers.call_action('group_create', context, name='test-group')

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v5_custom_config_setting')
@pytest.mark.ckan_config('ckan.iauthfunctions.users_can_create_groups', True)
@pytest.mark.usefixtures('clean_db', 'with_plugins')
class TestAuthV5WithUserCreateGroup(object):

    def test_sysadmin_can_create_group_when_config_is_true(self):
        if False:
            print('Hello World!')
        sysadmin = factories.Sysadmin()
        context = {'ignore_auth': False, 'user': sysadmin['name']}
        helpers.call_action('group_create', context, name='test-group')

    def test_user_can_create_group_when_config_is_true(self):
        if False:
            print('Hello World!')
        user = factories.User()
        context = {'ignore_auth': False, 'user': user['name']}
        helpers.call_action('group_create', context, name='test-group')

    def test_visitor_cannot_create_group_when_config_is_true(self):
        if False:
            for i in range(10):
                print('nop')
        context = {'ignore_auth': False, 'user': None}
        with pytest.raises(NotAuthorized):
            helpers.call_action('group_create', context, name='test-group')

@pytest.fixture
def curators_group():
    if False:
        for i in range(10):
            print('nop')
    "This is a helper method for test methods to call when they want\n    the 'curators' group to be created.\n    "
    sysadmin = factories.Sysadmin()
    noncurator = factories.User()
    curator = factories.User()
    users = [{'name': curator['name'], 'capacity': 'member'}]
    context = {'ignore_auth': False, 'user': sysadmin['name']}
    group = helpers.call_action('group_create', context, name='curators', users=users)
    return (noncurator, curator, group)

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v4')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
def test_group_create_with_no_curators_group():
    if False:
        for i in range(10):
            print('nop')
    "Test that group_create doesn't crash when there's no curators group.\n    "
    sysadmin = factories.Sysadmin()
    assert 'curators' not in helpers.call_action('group_list', {})
    context = {'ignore_auth': False, 'user': sysadmin['name']}
    helpers.call_action('group_create', context, name='test-group')

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v4')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
def test_group_create_with_visitor(curators_group):
    if False:
        for i in range(10):
            print('nop')
    "A visitor (not logged in) should not be able to create a group.\n\n    Note: this also tests that the group_create auth function doesn't\n    crash when the user isn't logged in.\n    "
    context = {'ignore_auth': False, 'user': None}
    with pytest.raises(NotAuthorized):
        helpers.call_action('group_create', context, name='this_group_should_not_be_created')

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v4')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
def test_group_create_with_non_curator(curators_group):
    if False:
        while True:
            i = 10
    "A user who isn't a member of the curators group should not be able\n    to create a group.\n    "
    (noncurator, _, _) = curators_group
    context = {'ignore_auth': False, 'user': noncurator['name']}
    with pytest.raises(NotAuthorized):
        helpers.call_action('group_create', context, name='this_group_should_not_be_created')

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v4')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
def test_group_create_with_curator(curators_group):
    if False:
        return 10
    'A member of the curators group should be able to create a group.\n    '
    (_, curator, _) = curators_group
    name = 'my-new-group'
    context = {'ignore_auth': False, 'user': curator['name']}
    result = helpers.call_action('group_create', context, name=name)
    assert result['name'] == name

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v3')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
class TestExampleIAuthFunctionsPluginV3(object):

    def test_group_create_with_no_curators_group_v3(self):
        if False:
            while True:
                i = 10
        "Test that group_create returns a 404 when there's no curators group.\n\n        With this version of the plugin group_create returns a spurious 404\n        when a user _is_ logged-in but the site has no curators group.\n        "
        assert 'curators' not in helpers.call_action('group_list', {})
        user = factories.User()
        context = {'ignore_auth': False, 'user': user['name']}
        with pytest.raises(ObjectNotFound):
            helpers.call_action('group_create', context, name='this_group_should_not_be_created')

    def test_group_create_with_visitor_v3(self, curators_group):
        if False:
            while True:
                i = 10
        'Test that group_create returns 403 when no one is logged in.\n\n        Since #1210 non-logged in requests are automatically rejected, unless\n        the auth function has the appropiate decorator\n        '
        context = {'ignore_auth': False, 'user': None}
        with pytest.raises(NotAuthorized):
            helpers.call_action('group_create', context, name='this_group_should_not_be_created')

@pytest.mark.ckan_config('ckan.plugins', 'example_iauthfunctions_v2')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
def test_group_create_with_curator_v2(curators_group):
    if False:
        return 10
    'Test that a curator can*not* create a group.\n\n    In this version of the plugin, even users who are members of the\n    curators group cannot create groups.\n    '
    (_, curator, _) = curators_group
    context = {'ignore_auth': False, 'user': curator['name']}
    with pytest.raises(NotAuthorized):
        helpers.call_action('group_create', context, name='this_group_should_not_be_created')