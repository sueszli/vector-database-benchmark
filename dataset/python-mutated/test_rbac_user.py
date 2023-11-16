import pytest
from unittest import mock
from django.test import TransactionTestCase
from awx.main.access import UserAccess, RoleAccess, TeamAccess
from awx.main.models import User, Organization, Inventory, Role

class TestSysAuditorTransactional(TransactionTestCase):

    def rando(self):
        if False:
            for i in range(10):
                print('nop')
        return User.objects.create(username='rando', password='rando', email='rando@com.com')

    def inventory(self):
        if False:
            for i in range(10):
                print('nop')
        org = Organization.objects.create(name='org')
        inv = Inventory.objects.create(name='inv', organization=org)
        return inv

    def test_auditor_caching(self):
        if False:
            for i in range(10):
                print('nop')
        rando = self.rando()
        with self.assertNumQueries(1):
            v = rando.is_system_auditor
        assert not v
        with self.assertNumQueries(0):
            v = rando.is_system_auditor
        assert not v

    def test_auditor_setter(self):
        if False:
            i = 10
            return i + 15
        rando = self.rando()
        inventory = self.inventory()
        rando.is_system_auditor = True
        assert rando in inventory.read_role

    def test_refresh_with_set(self):
        if False:
            for i in range(10):
                print('nop')
        rando = self.rando()
        rando.is_system_auditor = True
        assert rando.is_system_auditor
        rando.is_system_auditor = False
        assert not rando.is_system_auditor

@pytest.mark.django_db
def test_system_auditor_is_system_auditor(system_auditor):
    if False:
        for i in range(10):
            print('nop')
    assert system_auditor.is_system_auditor

@pytest.mark.django_db
def test_system_auditor_can_modify_self(system_auditor):
    if False:
        i = 10
        return i + 15
    access = UserAccess(system_auditor)
    assert access.can_change(obj=system_auditor, data=dict(is_system_auditor='true'))

@pytest.mark.django_db
def test_user_queryset(user):
    if False:
        for i in range(10):
            print('nop')
    u = user('pete', False)
    access = UserAccess(u)
    qs = access.get_queryset()
    assert qs.count() == 1

@pytest.mark.django_db
@pytest.mark.parametrize('ext_auth,superuser,expect', [(True, True, True), (False, True, True), (True, False, True), (False, False, False)], ids=['superuser', 'superuser-off', 'org', 'org-off'])
def test_manage_org_auth_setting(ext_auth, superuser, expect, organization, rando, user, team):
    if False:
        for i in range(10):
            print('nop')
    u = user('foo-user', is_superuser=superuser)
    if not superuser:
        organization.admin_role.members.add(u)
    with mock.patch('awx.main.access.settings') as settings_mock:
        settings_mock.MANAGE_ORGANIZATION_AUTH = ext_auth
        assert [UserAccess(u).can_attach(rando, organization.admin_role, 'roles'), UserAccess(u).can_attach(rando, organization.member_role, 'roles'), UserAccess(u).can_attach(rando, team.admin_role, 'roles'), UserAccess(u).can_attach(rando, team.member_role, 'roles'), RoleAccess(u).can_attach(organization.admin_role, rando, 'members'), RoleAccess(u).can_attach(organization.member_role, rando, 'members'), RoleAccess(u).can_attach(team.admin_role, rando, 'members'), RoleAccess(u).can_attach(team.member_role, rando, 'members')] == [expect for i in range(8)]
        assert [UserAccess(u).can_unattach(rando, organization.admin_role, 'roles'), UserAccess(u).can_unattach(rando, organization.member_role, 'roles'), UserAccess(u).can_unattach(rando, team.admin_role, 'roles'), UserAccess(u).can_unattach(rando, team.member_role, 'roles'), RoleAccess(u).can_unattach(organization.admin_role, rando, 'members'), RoleAccess(u).can_unattach(organization.member_role, rando, 'members'), RoleAccess(u).can_unattach(team.admin_role, rando, 'members'), RoleAccess(u).can_unattach(team.member_role, rando, 'members')] == [expect for i in range(8)]

@pytest.mark.django_db
@pytest.mark.parametrize('ext_auth', [True, False])
def test_team_org_resource_role(ext_auth, organization, rando, org_admin, team):
    if False:
        while True:
            i = 10
    with mock.patch('awx.main.access.settings') as settings_mock:
        settings_mock.MANAGE_ORGANIZATION_AUTH = ext_auth
        assert [TeamAccess(org_admin).can_attach(team, organization.workflow_admin_role, 'roles'), RoleAccess(org_admin).can_attach(organization.workflow_admin_role, team, 'member_role.parents')] == [True for i in range(2)]
        assert [TeamAccess(org_admin).can_unattach(team, organization.workflow_admin_role, 'roles'), RoleAccess(org_admin).can_unattach(organization.workflow_admin_role, team, 'member_role.parents')] == [True for i in range(2)]

@pytest.mark.django_db
def test_user_accessible_objects(user, organization):
    if False:
        return 10
    '\n    We cannot directly use accessible_objects for User model because\n    both editing and read permissions are obligated to complex business logic\n    '
    admin = user('admin', False)
    u = user('john', False)
    access = UserAccess(admin)
    assert access.get_queryset().count() == 1
    organization.member_role.members.add(u)
    organization.member_role.members.add(admin)
    assert access.get_queryset().count() == 2
    organization.member_role.members.remove(u)
    assert access.get_queryset().count() == 1

@pytest.mark.django_db
def test_org_admin_create_sys_auditor(org_admin):
    if False:
        print('Hello World!')
    access = UserAccess(org_admin)
    assert not access.can_add(data=dict(username='new_user', password='pa$$sowrd', email='asdf@redhat.com', is_system_auditor='true'))

@pytest.mark.django_db
def test_org_admin_edit_sys_auditor(org_admin, alice, organization):
    if False:
        return 10
    organization.member_role.members.add(alice)
    access = UserAccess(org_admin)
    assert not access.can_change(obj=alice, data=dict(is_system_auditor='true'))

@pytest.mark.django_db
def test_org_admin_can_delete_orphan(org_admin, alice):
    if False:
        print('Hello World!')
    access = UserAccess(org_admin)
    assert access.can_delete(alice)

@pytest.mark.django_db
def test_org_admin_can_delete_group_member(org_admin, org_member):
    if False:
        i = 10
        return i + 15
    access = UserAccess(org_admin)
    assert access.can_delete(org_member)

@pytest.mark.django_db
def test_org_admin_cannot_delete_member_attached_to_other_group(org_admin, org_member):
    if False:
        return 10
    other_org = Organization.objects.create(name='other-org', description='other-org-desc')
    access = UserAccess(org_admin)
    other_org.member_role.members.add(org_member)
    assert not access.can_delete(org_member)

@pytest.mark.parametrize('reverse', (True, False))
@pytest.mark.django_db
def test_consistency_of_is_superuser_flag(reverse):
    if False:
        return 10
    users = [User.objects.create(username='rando_{}'.format(i)) for i in range(2)]
    for u in users:
        assert u.is_superuser is False
    system_admin = Role.singleton('system_administrator')
    if reverse:
        for u in users:
            u.roles.add(system_admin)
    else:
        system_admin.members.add(*[u.id for u in users])
    for u in users:
        u.refresh_from_db()
        assert u.is_superuser is True
    users[0].roles.clear()
    for u in users:
        u.refresh_from_db()
    assert users[0].is_superuser is False
    assert users[1].is_superuser is True
    system_admin.members.clear()
    for u in users:
        u.refresh_from_db()
        assert u.is_superuser is False