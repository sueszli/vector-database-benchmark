import pytest
import re
from django.test.utils import override_settings
from awx.main.models import User, Organization, Team
from awx.sso.saml_pipeline import _update_m2m_from_expression, _update_user_orgs, _update_user_teams, _update_user_orgs_by_saml_attr, _update_user_teams_by_saml_attr, _check_flag

@pytest.fixture
def users():
    if False:
        while True:
            i = 10
    u1 = User.objects.create(username='user1@foo.com', last_name='foo', first_name='bar', email='user1@foo.com')
    u2 = User.objects.create(username='user2@foo.com', last_name='foo', first_name='bar', email='user2@foo.com')
    u3 = User.objects.create(username='user3@foo.com', last_name='foo', first_name='bar', email='user3@foo.com')
    return (u1, u2, u3)

@pytest.mark.django_db
class TestSAMLPopulateUser:

    def test_populate_user(self):
        if False:
            print('Hello World!')
        assert True

@pytest.mark.django_db
class TestSAMLSimpleMaps:

    @pytest.fixture
    def backend(self):
        if False:
            print('Hello World!')

        class Backend:
            s = {'ORGANIZATION_MAP': {'Default': {'remove': True, 'admins': 'foobar', 'remove_admins': True, 'users': 'foo', 'remove_users': True, 'organization_alias': ''}}, 'TEAM_MAP': {'Blue': {'organization': 'Default', 'remove': True, 'users': ''}, 'Red': {'organization': 'Default', 'remove': True, 'users': ''}}}

            def setting(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return self.s[key]
        return Backend()

    def test__update_user_orgs(self, backend, users):
        if False:
            return 10
        (u1, u2, u3) = users
        backend.setting('ORGANIZATION_MAP')['Default']['admins'] = re.compile('.*')
        backend.setting('ORGANIZATION_MAP')['Default']['users'] = re.compile('.*')
        desired_org_state = {}
        orgs_to_create = []
        _update_user_orgs(backend, desired_org_state, orgs_to_create, u1)
        _update_user_orgs(backend, desired_org_state, orgs_to_create, u2)
        _update_user_orgs(backend, desired_org_state, orgs_to_create, u3)
        assert desired_org_state == {'Default': {'member_role': True, 'admin_role': True, 'auditor_role': False}}
        assert orgs_to_create == ['Default']
        backend.setting('ORGANIZATION_MAP')['Default']['admins'] = ''
        backend.setting('ORGANIZATION_MAP')['Default']['users'] = ''
        backend.setting('ORGANIZATION_MAP')['Default']['remove_admins'] = True
        backend.setting('ORGANIZATION_MAP')['Default']['remove_users'] = True
        desired_org_state = {}
        orgs_to_create = []
        _update_user_orgs(backend, desired_org_state, orgs_to_create, u1)
        assert desired_org_state == {'Default': {'member_role': False, 'admin_role': False, 'auditor_role': False}}
        assert orgs_to_create == ['Default']
        backend.setting('ORGANIZATION_MAP')['Default']['remove_admins'] = False
        backend.setting('ORGANIZATION_MAP')['Default']['remove_users'] = False
        desired_org_state = {}
        orgs_to_create = []
        _update_user_orgs(backend, desired_org_state, orgs_to_create, u2)
        assert desired_org_state == {'Default': {'member_role': None, 'admin_role': None, 'auditor_role': False}}
        assert orgs_to_create == ['Default']
        backend.setting('ORGANIZATION_MAP')['Default']['organization_alias'] = 'Default_Alias'
        orgs_to_create = []
        _update_user_orgs(backend, {}, orgs_to_create, u1)
        assert orgs_to_create == ['Default_Alias']

    def test__update_user_teams(self, backend, users):
        if False:
            while True:
                i = 10
        (u1, u2, u3) = users
        backend.setting('TEAM_MAP')['Blue']['users'] = re.compile('.*')
        backend.setting('TEAM_MAP')['Red']['users'] = re.compile('.*')
        desired_team_state = {}
        teams_to_create = {}
        _update_user_teams(backend, desired_team_state, teams_to_create, u1)
        assert teams_to_create == {'Red': 'Default', 'Blue': 'Default'}
        assert desired_team_state == {'Default': {'Blue': {'member_role': True}, 'Red': {'member_role': True}}}
        backend.setting('TEAM_MAP')['Blue']['remove'] = True
        backend.setting('TEAM_MAP')['Red']['remove'] = True
        backend.setting('TEAM_MAP')['Blue']['users'] = ''
        backend.setting('TEAM_MAP')['Red']['users'] = ''
        desired_team_state = {}
        teams_to_create = {}
        _update_user_teams(backend, desired_team_state, teams_to_create, u1)
        assert teams_to_create == {'Red': 'Default', 'Blue': 'Default'}
        assert desired_team_state == {'Default': {'Blue': {'member_role': False}, 'Red': {'member_role': False}}}
        backend.setting('TEAM_MAP')['Blue']['remove'] = False
        backend.setting('TEAM_MAP')['Red']['remove'] = False
        desired_team_state = {}
        teams_to_create = {}
        _update_user_teams(backend, desired_team_state, teams_to_create, u2)
        assert teams_to_create == {'Red': 'Default', 'Blue': 'Default'}
        assert desired_team_state == {}

@pytest.mark.django_db
class TestSAMLM2M:

    @pytest.mark.parametrize('expression, remove, expected_return', [(None, False, None), ('', False, None), (None, True, False), (True, False, True), (True, True, True), ('user1', False, True), ('user1@foo.com', False, True), ('user27', False, None), ('user27', True, False), (['user1'], False, True), (['user1@foo.com'], False, True), (['user27'], False, None), (['user27'], True, False), (['user27', 'user28'], False, None), (['user27', 'user28'], True, False), (['user1', 'user1@foo.com'], False, True), (['user1', 'user28', 'user27'], False, True), (re.compile('^user.*'), False, True), (re.compile('^user.*'), True, True), (re.compile('.*@foo.com$'), False, True), (re.compile('.*@foo.com$'), True, True), (re.compile('^$'), False, None), (re.compile('^$'), True, False), ([re.compile('^user.*')], False, True), ([re.compile('^user.*')], True, True), ([re.compile('.*@foo.com$')], False, True), ([re.compile('.*@foo.com$')], True, True), ([re.compile('^$')], False, None), ([re.compile('^$')], True, False), ([re.compile('^user.*'), re.compile('.*@bar.com$')], False, True), ([re.compile('^user27$'), re.compile('.*@foo.com$')], False, True), ([re.compile('^user27$'), re.compile('.*@bar.com$')], False, None), ([re.compile('^user27$'), re.compile('.*@bar.com$')], True, False), (['user1', re.compile('.*@bar.com$')], False, True), (['user27', re.compile('.*@foo.com$')], False, True), (['user27', re.compile('.*@bar.com$')], False, None), (['user27', re.compile('.*@bar.com$')], True, False)])
    def test__update_m2m_from_expression(self, expression, remove, expected_return):
        if False:
            i = 10
            return i + 15
        user = User.objects.create(username='user1', last_name='foo', first_name='bar', email='user1@foo.com')
        return_val = _update_m2m_from_expression(user, expression, remove)
        assert return_val == expected_return

@pytest.mark.django_db
class TestSAMLAttrMaps:

    @pytest.fixture
    def backend(self):
        if False:
            return 10

        class Backend:
            s = {'ORGANIZATION_MAP': {'Default1': {'remove': True, 'admins': 'foobar', 'remove_admins': True, 'users': 'foo', 'remove_users': True, 'organization_alias': 'o1_alias'}}}

            def setting(self, key):
                if False:
                    i = 10
                    return i + 15
                return self.s[key]
        return Backend()

    @pytest.mark.parametrize('setting, expected_state, expected_orgs_to_create, kwargs_member_of_mods', [({'saml_attr': 'memberOf', 'saml_admin_attr': 'admins', 'saml_auditor_attr': 'auditors', 'remove': True, 'remove_admins': True}, {'Default2': {'member_role': True}, 'Default3': {'admin_role': True}, 'Default4': {'auditor_role': True}, 'o1_alias': {'member_role': True}, 'Rando1': {'admin_role': False, 'auditor_role': False, 'member_role': False}}, ['o1_alias', 'Default2', 'Default3', 'Default4'], None), ({'saml_attr': 'memberOf', 'saml_admin_attr': 'admins', 'saml_auditor_attr': 'auditors', 'remove': True, 'remove_admins': True}, {'Default3': {'admin_role': True, 'member_role': True}, 'Default4': {'auditor_role': True}, 'Rando1': {'admin_role': False, 'auditor_role': False, 'member_role': False}}, ['Default3', 'Default4'], ['Default3']), ({'saml_attr': 'memberOf', 'saml_admin_attr': 'admins', 'saml_auditor_attr': 'auditors', 'remove': False, 'remove_admins': False, 'remove_auditors': False}, {'Default2': {'member_role': True}, 'Default3': {'admin_role': True}, 'Default4': {'auditor_role': True}, 'o1_alias': {'member_role': True}}, ['o1_alias', 'Default2', 'Default3', 'Default4'], ['Default1', 'Default2'])])
    def test__update_user_orgs_by_saml_attr(self, backend, setting, expected_state, expected_orgs_to_create, kwargs_member_of_mods):
        if False:
            print('Hello World!')
        kwargs = {'username': u'cmeyers@redhat.com', 'uid': 'idp:cmeyers@redhat.com', 'request': {u'SAMLResponse': [], u'RelayState': [u'idp']}, 'is_new': False, 'response': {'session_index': '_0728f0e0-b766-0135-75fa-02842b07c044', 'idp_name': u'idp', 'attributes': {'memberOf': ['Default1', 'Default2'], 'admins': ['Default3'], 'auditors': ['Default4'], 'groups': ['Blue', 'Red'], 'User.email': ['cmeyers@redhat.com'], 'User.LastName': ['Meyers'], 'name_id': 'cmeyers@redhat.com', 'User.FirstName': ['Chris'], 'PersonImmutableID': []}}, 'social': None, 'strategy': None, 'new_association': False}
        if kwargs_member_of_mods:
            kwargs['response']['attributes']['memberOf'] = kwargs_member_of_mods
        Organization.objects.create(name='Rando1')
        with override_settings(SOCIAL_AUTH_SAML_ORGANIZATION_ATTR=setting):
            desired_org_state = {}
            orgs_to_create = []
            _update_user_orgs_by_saml_attr(backend, desired_org_state, orgs_to_create, **kwargs)
            assert desired_org_state == expected_state
            assert orgs_to_create == expected_orgs_to_create

    @pytest.mark.parametrize('setting, expected_team_state, expected_teams_to_create, kwargs_group_override', [({'saml_attr': 'groups', 'remove': False, 'team_org_map': [{'team': 'Blue', 'organization': 'Default1'}, {'team': 'Blue', 'organization': 'Default2'}, {'team': 'Blue', 'organization': 'Default3'}, {'team': 'Red', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default3'}, {'team': 'Yellow', 'team_alias': 'Yellow_Alias', 'organization': 'Default4', 'organization_alias': 'Default4_Alias'}]}, {'Default1': {'Blue': {'member_role': True}, 'Green': {'member_role': False}, 'Red': {'member_role': True}}, 'Default2': {'Blue': {'member_role': True}}, 'Default3': {'Blue': {'member_role': True}, 'Green': {'member_role': False}}, 'Default4': {'Yellow': {'member_role': False}}}, {'Blue': 'Default3', 'Red': 'Default1'}, None), ({'saml_attr': 'groups', 'remove': False, 'team_org_map': [{'team': 'Blue', 'organization': 'Default1'}, {'team': 'Blue', 'organization': 'Default2'}, {'team': 'Blue', 'organization': 'Default3'}, {'team': 'Red', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default3'}, {'team': 'Yellow', 'team_alias': 'Yellow_Alias', 'organization': 'Default4', 'organization_alias': 'Default4_Alias'}]}, {'Default1': {'Blue': {'member_role': True}, 'Green': {'member_role': True}, 'Red': {'member_role': True}}, 'Default2': {'Blue': {'member_role': True}}, 'Default3': {'Blue': {'member_role': True}, 'Green': {'member_role': True}}, 'Default4': {'Yellow': {'member_role': False}}}, {'Blue': 'Default3', 'Red': 'Default1', 'Green': 'Default3'}, ['Blue', 'Red', 'Green']), ({'saml_attr': 'groups', 'remove': True, 'team_org_map': [{'team': 'Blue', 'organization': 'Default1'}, {'team': 'Blue', 'organization': 'Default2'}, {'team': 'Blue', 'organization': 'Default3'}, {'team': 'Red', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default1'}, {'team': 'Green', 'organization': 'Default3'}, {'team': 'Yellow', 'team_alias': 'Yellow_Alias', 'organization': 'Default4', 'organization_alias': 'Default4_Alias'}]}, {'Default1': {'Blue': {'member_role': False}, 'Green': {'member_role': True}, 'Red': {'member_role': False}}, 'Default2': {'Blue': {'member_role': False}}, 'Default3': {'Blue': {'member_role': False}, 'Green': {'member_role': True}}, 'Default4': {'Yellow': {'member_role': False}}, 'Rando1': {'Rando1': {'member_role': False}}}, {'Green': 'Default3'}, ['Green'])])
    def test__update_user_teams_by_saml_attr(self, setting, expected_team_state, expected_teams_to_create, kwargs_group_override):
        if False:
            print('Hello World!')
        kwargs = {'username': u'cmeyers@redhat.com', 'uid': 'idp:cmeyers@redhat.com', 'request': {u'SAMLResponse': [], u'RelayState': [u'idp']}, 'is_new': False, 'response': {'session_index': '_0728f0e0-b766-0135-75fa-02842b07c044', 'idp_name': u'idp', 'attributes': {'memberOf': ['Default1', 'Default2'], 'admins': ['Default3'], 'auditors': ['Default4'], 'groups': ['Blue', 'Red'], 'User.email': ['cmeyers@redhat.com'], 'User.LastName': ['Meyers'], 'name_id': 'cmeyers@redhat.com', 'User.FirstName': ['Chris'], 'PersonImmutableID': []}}, 'social': None, 'strategy': None, 'new_association': False}
        if kwargs_group_override:
            kwargs['response']['attributes']['groups'] = kwargs_group_override
        o = Organization.objects.create(name='Rando1')
        Team.objects.create(name='Rando1', organization_id=o.id)
        with override_settings(SOCIAL_AUTH_SAML_TEAM_ATTR=setting):
            desired_team_state = {}
            teams_to_create = {}
            _update_user_teams_by_saml_attr(desired_team_state, teams_to_create, **kwargs)
            assert desired_team_state == expected_team_state
            assert teams_to_create == expected_teams_to_create

@pytest.mark.django_db
class TestSAMLUserFlags:

    @pytest.mark.parametrize('user_flags_settings, expected, is_superuser', [({}, (False, False), False), ({'is_superuser_role': 'test-role-1'}, (True, True), False), ({'is_superuser_attr': 'is_superuser'}, (True, True), False), ({'is_superuser_attr': 'is_superuser', 'is_superuser_value': 'junk'}, (False, False), False), ({'is_superuser_attr': 'is_superuser', 'is_superuser_value': 'true'}, (True, True), False), ({'is_superuser_role': 'test-role-1', 'is_superuser_attr': 'gibberish', 'is_superuser_value': 'true'}, (True, True), False), ({'is_superuser_role': 'test-role-1', 'is_superuser_attr': 'test-role-1'}, (True, True), False), ({'is_superuser_role': 'test-role-1', 'is_superuser_attr': 'is_superuser', 'is_superuser_value': 'junk'}, (False, False), False), ({'is_superuser_role': 'test-role-1', 'is_superuser_attr': 'is_superuser', 'is_superuser_value': 'true'}, (True, True), False), ({'is_superuser_attr': 'name_id', 'is_superuser_value': 'test_id'}, (True, True), False), ({'is_superuser_attr': 'name_id', 'is_superuser_value': 'junk'}, (False, False), False), ({'is_superuser_attr': 'name_id', 'is_superuser_value': 'junk', 'remove_superusers': True}, (False, True), True), ({'is_superuser_attr': 'name_id', 'is_superuser_value': 'junk', 'remove_superusers': False}, (True, False), True), ({'is_superuser_attr': 'is_superuser', 'is_superuser_value': ['junk', 'junk2', 'else', 'junk']}, (True, True), False), ({'is_superuser_attr': 'is_superuser', 'is_superuser_value': ['junk', 'junk2', 'junk']}, (False, True), True), ({'is_superuser_role': ['junk', 'junk2', 'something', 'junk']}, (True, True), False), ({'is_superuser_role': ['junk', 'junk2', 'junk']}, (False, True), True)])
    def test__check_flag(self, user_flags_settings, expected, is_superuser):
        if False:
            for i in range(10):
                print('nop')
        user = User()
        user.username = 'John'
        user.is_superuser = is_superuser
        attributes = {'email': ['noone@nowhere.com'], 'last_name': ['Westcott'], 'is_superuser': ['something', 'else', 'true'], 'username': ['test_id'], 'first_name': ['John'], 'Role': ['test-role-1', 'something', 'different'], 'name_id': 'test_id'}
        assert expected == _check_flag(user, 'superuser', attributes, user_flags_settings)

@pytest.mark.django_db
def test__update_user_orgs_org_map_and_saml_attr():
    if False:
        i = 10
        return i + 15
    '\n    This combines the action of two other tests where an org membership is defined both by\n    the ORGANIZATION_MAP and the SOCIAL_AUTH_SAML_ORGANIZATION_ATTR at the same time\n    '

    class BackendClass:
        s = {'ORGANIZATION_MAP': {'Default1': {'remove': True, 'remove_admins': True, 'users': 'foobar', 'remove_users': True, 'organization_alias': 'o1_alias'}}}

        def setting(self, key):
            if False:
                for i in range(10):
                    print('nop')
            return self.s[key]
    backend = BackendClass()
    setting = {'saml_attr': 'memberOf', 'saml_admin_attr': 'admins', 'saml_auditor_attr': 'auditors', 'remove': True, 'remove_admins': True}
    kwargs = {'username': 'foobar', 'uid': 'idp:cmeyers@redhat.com', 'request': {u'SAMLResponse': [], u'RelayState': [u'idp']}, 'is_new': False, 'response': {'session_index': '_0728f0e0-b766-0135-75fa-02842b07c044', 'idp_name': u'idp', 'attributes': {'admins': ['Default1']}}, 'social': None, 'strategy': None, 'new_association': False}
    this_user = User.objects.create(username='foobar')
    with override_settings(SOCIAL_AUTH_SAML_ORGANIZATION_ATTR=setting):
        desired_org_state = {}
        orgs_to_create = []
        _update_user_orgs_by_saml_attr(backend, desired_org_state, orgs_to_create, **kwargs)
        assert desired_org_state['o1_alias']['admin_role'] is True
        assert set(orgs_to_create) == set(['o1_alias'])
        _update_user_orgs(backend, desired_org_state, orgs_to_create, this_user)
        assert desired_org_state['o1_alias']['member_role'] is True
        assert desired_org_state['o1_alias']['admin_role'] is True
        assert set(orgs_to_create) == set(['o1_alias'])