import pytest
from ckan import authz as auth, model, logic
from ckan.tests import factories, helpers
_check = auth.check_config_permission

@pytest.mark.ckan_config('ckan.auth.anon_create_dataset', True)
def test_config_overrides_default():
    if False:
        for i in range(10):
            print('nop')
    assert _check('anon_create_dataset') is True

@pytest.mark.ckan_config('ckan.auth.anon_create_dataset', True)
def test_config_override_also_works_with_prefix():
    if False:
        return 10
    assert _check('ckan.auth.anon_create_dataset') is True

@pytest.mark.ckan_config('ckan.auth.unknown_permission', True)
def test_unknown_permission_returns_false():
    if False:
        print('Hello World!')
    assert _check('unknown_permission') is False

def test_unknown_permission_not_in_config_returns_false():
    if False:
        i = 10
        return i + 15
    assert _check('unknown_permission') is False

def test_default_roles_that_cascade_to_sub_groups_is_a_list():
    if False:
        print('Hello World!')
    assert isinstance(_check('roles_that_cascade_to_sub_groups'), list)

@pytest.mark.ckan_config('ckan.auth.roles_that_cascade_to_sub_groups', ['admin', 'editor'])
def test_roles_that_cascade_to_sub_groups_is_a_list():
    if False:
        print('Hello World!')
    assert sorted(_check('roles_that_cascade_to_sub_groups')) == sorted(['admin', 'editor'])

@pytest.mark.usefixtures('non_clean_db')
def test_get_user_returns_user_obj():
    if False:
        print('Hello World!')
    user = factories.User()
    assert auth._get_user(user['name']).name == user['name']

def test_get_user_not_found():
    if False:
        i = 10
        return i + 15
    name = factories.User.stub().name
    assert auth._get_user(name) is None

def test_no_attributes_set_on_imported_auth_members():
    if False:
        print('Hello World!')
    import ckan.logic.auth.get as auth_get
    logic.check_access('package_search', {})
    assert hasattr(auth_get.package_search, 'auth_allow_anonymous_access')
    assert not hasattr(auth_get.config, 'auth_allow_anonymous_access')

@pytest.mark.usefixtures('non_clean_db')
class TestAuthOrgHierarchy(object):

    def test_parent_admin_auth(self):
        if False:
            return 10
        user = factories.User()
        parent = factories.Organization(users=[{'capacity': 'admin', 'name': user['name']}])
        child = factories.Organization()
        helpers.call_action('member_create', id=child['id'], object=parent['id'], object_type='group', capacity='parent')
        context = {'model': model, 'user': user['name']}
        helpers.call_auth('organization_member_create', context, id=parent['id'])
        helpers.call_auth('organization_member_create', context, id=child['id'])
        helpers.call_auth('package_create', context, owner_org=parent['id'])
        helpers.call_auth('package_create', context, owner_org=child['id'])

    def test_child_admin_auth(self):
        if False:
            while True:
                i = 10
        user = factories.User()
        parent = factories.Organization()
        child = factories.Organization(users=[{'capacity': 'admin', 'name': user['name']}])
        helpers.call_action('member_create', id=child['id'], object=parent['id'], object_type='group', capacity='parent')
        context = {'model': model, 'user': user['name']}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('organization_member_create', context, id=parent['id'])
        helpers.call_auth('organization_member_create', context, id=child['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_create', context, owner_org=parent['id'])
        helpers.call_auth('package_create', context, owner_org=child['id'])

    def test_parent_editor_auth(self):
        if False:
            while True:
                i = 10
        user = factories.User()
        parent = factories.Organization(users=[{'capacity': 'editor', 'name': user['name']}])
        child = factories.Organization()
        helpers.call_action('member_create', id=child['id'], object=parent['id'], object_type='group', capacity='parent')
        context = {'model': model, 'user': user['name']}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('organization_member_create', context, id=parent['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('organization_member_create', context, id=child['id'])
        helpers.call_auth('package_create', context, owner_org=parent['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_create', context, owner_org=child['id'])

    def test_child_editor_auth(self):
        if False:
            i = 10
            return i + 15
        user = factories.User()
        parent = factories.Organization()
        child = factories.Organization(users=[{'capacity': 'editor', 'name': user['name']}])
        helpers.call_action('member_create', id=child['id'], object=parent['id'], object_type='group', capacity='parent')
        context = {'model': model, 'user': user['name']}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('organization_member_create', context, id=parent['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('organization_member_create', context, id=child['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_create', context, owner_org=parent['id'])
        helpers.call_auth('package_create', context, owner_org=child['id'])