"""Unit tests for ckan/logic/auth/get.py.

"""
import pytest
import ckan.tests.helpers as helpers
import ckan.tests.factories as factories
import ckan.logic as logic
from ckan import model
from unittest import mock

@pytest.mark.ckan_config(u'ckan.auth.public_user_details', False)
@mock.patch('flask_login.utils._get_user')
def test_auth_user_list(current_user):
    if False:
        while True:
            i = 10
    current_user.return_value = mock.Mock(is_anonymous=True)
    context = {'user': None, 'model': model}
    with pytest.raises(logic.NotAuthorized):
        helpers.call_auth('user_list', context=context)

def test_authed_user_list():
    if False:
        while True:
            i = 10
    context = {'user': None, 'model': model}
    assert helpers.call_auth('user_list', context=context)

def test_user_list_email_parameter():
    if False:
        print('Hello World!')
    context = {'user': None, 'model': model}
    with pytest.raises(logic.NotAuthorized):
        helpers.call_auth('user_list', email='a@example.com', context=context)

@pytest.mark.usefixtures(u'non_clean_db')
class TestGetAuth(object):

    @pytest.mark.ckan_config(u'ckan.auth.public_user_details', False)
    @mock.patch('flask_login.utils._get_user')
    def test_restrict_anon_auth_when_user_is_anonymouus(self, current_user):
        if False:
            i = 10
            return i + 15
        fred = factories.User()
        fred['capacity'] = 'editor'
        current_user.return_value = mock.Mock(is_anonymous=True)
        context = {'user': None, 'model': model}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('user_show', context=context, id=fred['id'])

    @pytest.mark.ckan_config(u'ckan.auth.public_user_details', False)
    @mock.patch('flask_login.utils._get_user')
    def test_restrict_anon_auth_when_user_is_in_context(self, current_user):
        if False:
            for i in range(10):
                print('nop')
        fred = factories.User()
        fred['capacity'] = 'editor'
        current_user.return_value = mock.Mock(is_anonymous=True)
        context = {'user': fred['id'], 'model': model}
        assert helpers.call_auth('user_show', context=context, id=fred['id'])

    def test_authed_user_show(self):
        if False:
            return 10
        fred = factories.User()
        fred['capacity'] = 'editor'
        context = {'user': None, 'model': model}
        assert helpers.call_auth('user_show', context=context, id=fred['id'])

    def test_package_show__deleted_dataset_is_hidden_to_public(self):
        if False:
            print('Hello World!')
        dataset = factories.Dataset(state='deleted')
        context = {'model': model}
        context['user'] = ''
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_show', context=context, id=dataset['name'])

    def test_package_show__deleted_dataset_is_visible_to_editor(self):
        if False:
            while True:
                i = 10
        fred = factories.User()
        fred['capacity'] = 'editor'
        org = factories.Organization(users=[fred])
        dataset = factories.Dataset(owner_org=org['id'], state='deleted')
        context = {'model': model}
        context['user'] = fred['name']
        ret = helpers.call_auth('package_show', context=context, id=dataset['name'])
        assert ret

    def test_group_show__deleted_group_is_hidden_to_public(self):
        if False:
            for i in range(10):
                print('nop')
        group = factories.Group(state='deleted')
        context = {'model': model}
        context['user'] = ''
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('group_show', context=context, id=group['name'])

    def test_group_show__deleted_group_is_visible_to_its_member(self):
        if False:
            i = 10
            return i + 15
        fred = factories.User()
        fred['capacity'] = 'editor'
        org = factories.Group(users=[fred], state='deleted')
        context = {'model': model}
        context['user'] = fred['name']
        ret = helpers.call_auth('group_show', context=context, id=org['name'])
        assert ret

    def test_group_show__deleted_org_is_visible_to_its_member(self):
        if False:
            for i in range(10):
                print('nop')
        fred = factories.User()
        fred['capacity'] = 'editor'
        org = factories.Organization(users=[fred], state='deleted')
        context = {'model': model}
        context['user'] = fred['name']
        ret = helpers.call_auth('group_show', context=context, id=org['name'])
        assert ret

    @pytest.mark.ckan_config(u'ckan.auth.public_user_details', False)
    def test_group_show__user_is_hidden_to_public(self):
        if False:
            for i in range(10):
                print('nop')
        group = factories.Group()
        context = {'model': model}
        context['user'] = ''
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('group_show', context=context, id=group['name'], include_users=True)

    def test_group_show__user_is_avail_to_public(self):
        if False:
            while True:
                i = 10
        group = factories.Group()
        context = {'model': model}
        context['user'] = ''
        assert helpers.call_auth('group_show', context=context, id=group['name'])

    def test_config_option_show_anon_user(self):
        if False:
            for i in range(10):
                print('nop')
        'An anon user is not authorized to use config_option_show action.'
        context = {'user': None, 'model': None}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('config_option_show', context=context)

    def test_config_option_show_normal_user(self):
        if False:
            for i in range(10):
                print('nop')
        'A normal logged in user is not authorized to use config_option_show\n        action.'
        fred = factories.User()
        context = {'user': fred['name'], 'model': None}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('config_option_show', context=context)

    def test_config_option_show_sysadmin(self):
        if False:
            while True:
                i = 10
        'A sysadmin is authorized to use config_option_show action.'
        fred = factories.Sysadmin()
        context = {'user': fred['name'], 'model': None}
        assert helpers.call_auth('config_option_show', context=context)

    def test_config_option_list_anon_user(self):
        if False:
            while True:
                i = 10
        'An anon user is not authorized to use config_option_list action.'
        context = {'user': None, 'model': None}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('config_option_list', context=context)

    def test_config_option_list_normal_user(self):
        if False:
            for i in range(10):
                print('nop')
        'A normal logged in user is not authorized to use config_option_list\n        action.'
        fred = factories.User()
        context = {'user': fred['name'], 'model': None}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('config_option_list', context=context)

    def test_config_option_list_sysadmin(self):
        if False:
            for i in range(10):
                print('nop')
        'A sysadmin is authorized to use config_option_list action.'
        fred = factories.Sysadmin()
        context = {'user': fred['name'], 'model': None}
        assert helpers.call_auth('config_option_list', context=context)

@pytest.mark.usefixtures('non_clean_db')
class TestApiToken(object):

    def test_anon_is_not_allowed_to_get_tokens(self):
        if False:
            return 10
        user = factories.User()
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth(u'api_token_list', {u'user': None, u'model': model}, user_id=user['name'])

    def test_auth_user_is_allowed_to_list_tokens(self):
        if False:
            while True:
                i = 10
        user = factories.User()
        helpers.call_auth(u'api_token_list', {u'model': model, u'user': user[u'name']}, user_id=user[u'name'])

@pytest.mark.usefixtures('non_clean_db', 'with_plugins')
@pytest.mark.ckan_config('ckan.plugins', 'image_view')
@pytest.mark.ckan_config(u'ckan.auth.allow_dataset_collaborators', True)
class TestGetAuthWithCollaborators(object):

    def _get_context(self, user):
        if False:
            for i in range(10):
                print('nop')
        return {'model': model, 'user': user if isinstance(user, str) else user.get('name')}

    def test_dataset_show_private_editor(self):
        if False:
            print('Hello World!')
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_show', context=context, id=dataset['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='editor')
        assert helpers.call_auth('package_show', context=context, id=dataset['id'])

    def test_dataset_show_private_member(self):
        if False:
            return 10
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_show', context=context, id=dataset['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='member')
        assert helpers.call_auth('package_show', context=context, id=dataset['id'])

    def test_resource_show_private_editor(self):
        if False:
            for i in range(10):
                print('nop')
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_show', context=context, id=resource['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='editor')
        assert helpers.call_auth('resource_show', context=context, id=resource['id'])

    def test_resource_show_private_member(self):
        if False:
            return 10
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_show', context=context, id=resource['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='member')
        assert helpers.call_auth('resource_show', context=context, id=resource['id'])

    def test_resource_view_list_private_editor(self):
        if False:
            i = 10
            return i + 15
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_view_list', context=context, id=resource['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='editor')
        assert helpers.call_auth('resource_view_list', context=context, id=resource['id'])

    def test_resource_view_list_private_member(self):
        if False:
            i = 10
            return i + 15
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        user = factories.User()
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_view_list', context=context, id=resource['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='member')
        assert helpers.call_auth('resource_view_list', context=context, id=resource['id'])

    def test_resource_view_show_private_editor(self):
        if False:
            i = 10
            return i + 15
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        resource_view = factories.ResourceView(resource_id=resource['id'])
        user = factories.User()
        context = self._get_context(user)
        context['resource'] = model.Resource.get(resource['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_view_show', context=context, id=resource_view['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='editor')
        assert helpers.call_auth('resource_view_show', context=context, id=resource_view['id'])

    def test_resource_view_show_private_member(self):
        if False:
            return 10
        org = factories.Organization()
        dataset = factories.Dataset(private=True, owner_org=org['id'])
        resource = factories.Resource(package_id=dataset['id'])
        resource_view = factories.ResourceView(resource_id=resource['id'])
        user = factories.User()
        context = self._get_context(user)
        context['resource'] = model.Resource.get(resource['id'])
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('resource_view_show', context=context, id=resource_view['id'])
        helpers.call_action('package_collaborator_create', id=dataset['id'], user_id=user['id'], capacity='member')
        assert helpers.call_auth('resource_view_show', context=context, id=resource_view['id'])

@pytest.mark.usefixtures('non_clean_db')
@pytest.mark.ckan_config(u'ckan.auth.allow_dataset_collaborators', True)
class TestPackageMemberList(object):

    def _get_context(self, user):
        if False:
            while True:
                i = 10
        return {'model': model, 'user': user if isinstance(user, str) else user.get('name')}

    def setup(self):
        if False:
            return 10
        self.org_admin = factories.User()
        self.org_editor = factories.User()
        self.org_member = factories.User()
        self.normal_user = factories.User()
        self.org = factories.Organization(users=[{'name': self.org_admin['name'], 'capacity': 'admin'}, {'name': self.org_editor['name'], 'capacity': 'editor'}, {'name': self.org_member['name'], 'capacity': 'member'}])
        self.dataset = factories.Dataset(owner_org=self.org['id'])

    def test_list_org_admin_is_authorized(self):
        if False:
            for i in range(10):
                print('nop')
        context = self._get_context(self.org_admin)
        assert helpers.call_auth('package_collaborator_list', context=context, id=self.dataset['id'])

    def test_list_org_editor_is_not_authorized(self):
        if False:
            for i in range(10):
                print('nop')
        context = self._get_context(self.org_editor)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list', context=context, id=self.dataset['id'])

    def test_list_org_member_is_not_authorized(self):
        if False:
            print('Hello World!')
        context = self._get_context(self.org_member)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list', context=context, id=self.dataset['id'])

    def test_list_org_admin_from_other_org_is_not_authorized(self):
        if False:
            i = 10
            return i + 15
        org_admin2 = factories.User()
        factories.Organization(users=[{'name': org_admin2['name'], 'capacity': 'admin'}])
        context = self._get_context(org_admin2)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.dataset['id'])

    @pytest.mark.ckan_config('ckan.auth.allow_admin_collaborators', True)
    def test_list_collaborator_admin_is_authorized(self):
        if False:
            return 10
        user = factories.User()
        helpers.call_action('package_collaborator_create', id=self.dataset['id'], user_id=user['id'], capacity='admin')
        context = self._get_context(user)
        assert helpers.call_auth('package_collaborator_list', context=context, id=self.dataset['id'])

    @pytest.mark.parametrize('role', ['editor', 'member'])
    def test_list_collaborator_editor_and_member_are_not_authorized(self, role):
        if False:
            return 10
        user = factories.User()
        helpers.call_action('package_collaborator_create', id=self.dataset['id'], user_id=user['id'], capacity=role)
        context = self._get_context(user)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list', context=context, id=self.dataset['id'])

    def test_user_list_own_user_is_authorized(self):
        if False:
            print('Hello World!')
        context = self._get_context(self.normal_user)
        assert helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.normal_user['id'])

    def test_user_list_org_admin_is_not_authorized(self):
        if False:
            while True:
                i = 10
        context = self._get_context(self.org_admin)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.normal_user['id'])

    def test_user_list_org_editor_is_not_authorized(self):
        if False:
            return 10
        context = self._get_context(self.org_editor)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.normal_user['id'])

    def test_user_list_org_member_is_not_authorized(self):
        if False:
            while True:
                i = 10
        context = self._get_context(self.org_member)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.normal_user['id'])

    def test_user_list_org_admin_from_other_org_is_not_authorized(self):
        if False:
            return 10
        org_admin2 = factories.User()
        factories.Organization(users=[{'name': org_admin2['name'], 'capacity': 'admin'}])
        context = self._get_context(org_admin2)
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth('package_collaborator_list_for_user', context=context, id=self.normal_user['id'])

    @pytest.mark.ckan_config('ckan.auth.create_dataset_if_not_in_organization', True)
    @pytest.mark.ckan_config('ckan.auth.create_unowned_dataset', True)
    def test_list_unowned_datasets(self):
        if False:
            print('Hello World!')
        user = factories.User()
        dataset = factories.Dataset(user=user)
        assert dataset['owner_org'] is None
        assert dataset['creator_user_id'] == user['id']
        context = self._get_context(user)
        assert helpers.call_auth('package_collaborator_list', context=context, id=dataset['id'])

class TestFollower:
    functions = ['user_follower_list', 'dataset_follower_list', 'group_follower_list', 'organization_follower_list']

    @pytest.mark.parametrize('func', functions)
    def test_anon_cannot_list_followers(self, func):
        if False:
            for i in range(10):
                print('nop')
        context = {'user': '', 'model': model}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth(func, context=context)

    @pytest.mark.usefixtures('non_clean_db')
    @pytest.mark.parametrize('func', functions)
    def test_user_cannot_list_followers(self, func):
        if False:
            return 10
        user = factories.User()
        context = {'user': user['name'], 'model': model}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth(func, context=context)

    @pytest.mark.usefixtures('non_clean_db')
    @pytest.mark.parametrize('func', functions)
    def test_sysadmin_can_list_followers(self, func):
        if False:
            i = 10
            return i + 15
        sysadmin = factories.Sysadmin()
        context = {'user': sysadmin['name'], 'model': model}
        assert helpers.call_auth(func, context=context)

class TestFollowee:
    functions = ['user_followee_list', 'dataset_followee_list', 'group_followee_list', 'organization_followee_list']

    @pytest.mark.parametrize('func', functions)
    def test_anon_cannot_list_followees(self, func):
        if False:
            i = 10
            return i + 15
        context = {'user': '', 'model': model}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth(func, context=context)

    @pytest.mark.usefixtures('non_clean_db')
    @pytest.mark.parametrize('func', functions)
    def test_user_cannot_list_followees_of_another_user(self, func):
        if False:
            return 10
        user = factories.User()
        context = {'user': user['name'], 'model': model}
        with pytest.raises(logic.NotAuthorized):
            helpers.call_auth(func, context=context)

    @pytest.mark.usefixtures('non_clean_db')
    @pytest.mark.parametrize('func', functions)
    def test_user_can_list_own_followees(self, func):
        if False:
            i = 10
            return i + 15
        user = factories.User()
        context = {'user': user['name'], 'model': model}
        assert helpers.call_auth(func, context=context, id=user['id'])

    @pytest.mark.usefixtures('non_clean_db')
    @pytest.mark.parametrize('func', functions)
    def test_sysadmin_can_list_followees(self, func):
        if False:
            i = 10
            return i + 15
        sysadmin = factories.Sysadmin()
        context = {'user': sysadmin['name'], 'model': model}
        assert helpers.call_auth(func, context=context)

@pytest.mark.usefixtures('non_clean_db')
class TestStatusShow:

    def test_status_show_is_visible_to_anonymous(self):
        if False:
            while True:
                i = 10
        context = {'user': '', 'model': model}
        assert helpers.call_auth('status_show', context)

@pytest.mark.usefixtures('non_clean_db')
class TestFolloweeCount:
    functions = ['dataset_followee_count', 'followee_count', 'group_followee_count', 'organization_followee_count', 'user_followee_count']

    @pytest.mark.parametrize('func', functions)
    def test_anonymous_can_see_followee_count(self, func):
        if False:
            return 10
        user = factories.User()
        context = {'user': '', 'model': model}
        assert helpers.call_auth(func, context, id=user['id'])

@pytest.mark.usefixtures('non_clean_db')
class TestFollowerCount:
    functions = ['dataset_follower_count', 'group_follower_count', 'organization_follower_count', 'user_follower_count']

    @pytest.mark.parametrize('func', functions)
    def test_anonymous_can_see_follower_count(self, func):
        if False:
            while True:
                i = 10
        user = factories.User()
        context = {'user': '', 'model': model}
        assert helpers.call_auth(func, context, id=user['id'])

@pytest.mark.usefixtures('non_clean_db')
class TestAmFollowing:
    functions = ['am_following_dataset', 'am_following_group', 'am_following_user']

    @pytest.mark.parametrize('func', functions)
    def test_anonymous_can_see_am_following(self, func):
        if False:
            while True:
                i = 10
        user = factories.User()
        context = {'user': '', 'model': model}
        assert helpers.call_auth(func, context, id=user['id'])

class TestVariousGetMethods:
    functions = ['group_package_show', 'member_list', 'resource_search', 'tag_search', 'term_translation_show']

    @pytest.mark.parametrize('func', functions)
    def test_anonymous_can_call_get_method(self, func):
        if False:
            print('Hello World!')
        user = factories.User()
        context = {'user': '', 'model': model}
        assert helpers.call_auth(func, context, id=user['id'])