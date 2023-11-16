import pytest
from awx.main.access import RoleAccess, UserAccess, OrganizationAccess, TeamAccess
from awx.main.models import Role, Organization

@pytest.mark.django_db
def test_team_access_attach(rando, team, inventory):
    if False:
        for i in range(10):
            print('nop')
    team.admin_role.members.add(rando)
    inventory.read_role.members.add(rando)
    team.member_role.children.add(inventory.read_role)
    team_access = TeamAccess(rando)
    role_access = RoleAccess(rando)
    data = {'id': inventory.admin_role.pk}
    assert not team_access.can_attach(team, inventory.admin_role, 'member_role.children', data, False)
    assert not role_access.can_attach(inventory.admin_role, team, 'member_role.parents', data, False)

@pytest.mark.django_db
def test_user_access_attach(rando, inventory):
    if False:
        i = 10
        return i + 15
    inventory.read_role.members.add(rando)
    user_access = UserAccess(rando)
    role_access = RoleAccess(rando)
    data = {'id': inventory.admin_role.pk}
    assert not user_access.can_attach(rando, inventory.admin_role, 'roles', data, False)
    assert not role_access.can_attach(inventory.admin_role, rando, 'members', data, False)

@pytest.mark.django_db
def test_visible_roles(admin_user, system_auditor, rando, organization, project):
    if False:
        return 10
    '\n    system admin & system auditor fixtures needed to create system roles\n    '
    organization.auditor_role.members.add(rando)
    access = RoleAccess(rando)
    assert rando not in organization.admin_role
    assert access.can_read(organization.admin_role)
    assert organization.admin_role in Role.visible_roles(rando)
    assert rando not in project.admin_role
    assert access.can_read(project.admin_role)
    assert project.admin_role in Role.visible_roles(rando)

@pytest.mark.django_db
def test_org_user_role_attach(user, organization, inventory):
    if False:
        for i in range(10):
            print('nop')
    '\n    Org admins must not be able to add arbitrary users to their\n    organization, because that would give them admin permission to that user\n    '
    admin = user('admin')
    nonmember = user('nonmember')
    other_org = Organization.objects.create(name='other_org')
    other_org.member_role.members.add(nonmember)
    inventory.admin_role.members.add(nonmember)
    organization.admin_role.members.add(admin)
    role_access = RoleAccess(admin)
    org_access = OrganizationAccess(admin)
    assert not role_access.can_attach(organization.member_role, nonmember, 'members', None)
    assert not role_access.can_attach(organization.admin_role, nonmember, 'members', None)
    assert not org_access.can_attach(organization, nonmember, 'member_role.members', None)
    assert not org_access.can_attach(organization, nonmember, 'admin_role.members', None)

@pytest.mark.django_db
def test_user_org_object_roles(organization, org_admin, org_member):
    if False:
        while True:
            i = 10
    '\n    Unlike admin & member roles, the special-purpose organization roles do not\n    confer any permissions related to user management,\n    Normal rules about role delegation should apply, only admin to org needed.\n    '
    assert RoleAccess(org_admin).can_attach(organization.notification_admin_role, org_member, 'members', None)
    assert OrganizationAccess(org_admin).can_attach(organization, org_member, 'notification_admin_role.members', None)
    assert not RoleAccess(org_member).can_attach(organization.notification_admin_role, org_member, 'members', None)
    assert not OrganizationAccess(org_member).can_attach(organization, org_member, 'notification_admin_role.members', None)

@pytest.mark.django_db
def test_team_org_object_roles(organization, team, org_admin, org_member):
    if False:
        for i in range(10):
            print('nop')
    '\n    the special-purpose organization roles are not ancestors of any\n    team roles, and can be delegated en masse through teams,\n    following normal admin rules\n    '
    assert RoleAccess(org_admin).can_attach(organization.notification_admin_role, team, 'member_role.parents', {'id': 68})
    team.admin_role.members.add(org_member)
    assert not RoleAccess(org_member).can_attach(organization.notification_admin_role, team, 'member_role.parents', {'id': 68})
    assert not RoleAccess(org_admin).can_attach(organization.member_role, team, 'member_role.parents', {'id': 68})

@pytest.mark.django_db
def test_org_superuser_role_attach(admin_user, org_admin, organization):
    if False:
        return 10
    '\n    Ideally, you would not add superusers to roles (particularly member_role)\n    but it has historically been possible\n    this checks that the situation does not grant unexpected permissions\n    '
    organization.member_role.members.add(admin_user)
    role_access = RoleAccess(org_admin)
    org_access = OrganizationAccess(org_admin)
    assert not role_access.can_attach(organization.member_role, admin_user, 'members', None)
    assert not role_access.can_attach(organization.admin_role, admin_user, 'members', None)
    assert not org_access.can_attach(organization, admin_user, 'member_role.members', None)
    assert not org_access.can_attach(organization, admin_user, 'admin_role.members', None)
    user_access = UserAccess(org_admin)
    assert not user_access.can_change(admin_user, {'last_name': 'Witzel'})

@pytest.mark.django_db
def test_org_object_role_not_sufficient(user, organization):
    if False:
        while True:
            i = 10
    member = user('amember')
    obj_admin = user('icontrolallworkflows')
    organization.member_role.members.add(member)
    organization.workflow_admin_role.members.add(obj_admin)
    user_access = UserAccess(obj_admin)
    assert not user_access.can_change(member, {'last_name': 'Witzel'})

@pytest.mark.django_db
def test_need_all_orgs_to_admin_user(user):
    if False:
        for i in range(10):
            print('nop')
    '\n    Old behavior - org admin to ANY organization that a user is member of\n        grants permission to admin that user\n    New behavior enforced here - org admin to ALL organizations that a\n        user is member of grants permission to admin that user\n    '
    org1 = Organization.objects.create(name='org1')
    org2 = Organization.objects.create(name='org2')
    org1_admin = user('org1-admin')
    org1.admin_role.members.add(org1_admin)
    org12_member = user('org12-member')
    org1.member_role.members.add(org12_member)
    org2.member_role.members.add(org12_member)
    user_access = UserAccess(org1_admin)
    assert not user_access.can_change(org12_member, {'last_name': 'Witzel'})
    role_access = RoleAccess(org1_admin)
    org_access = OrganizationAccess(org1_admin)
    assert not role_access.can_attach(org1.admin_role, org12_member, 'members', None)
    assert not role_access.can_attach(org1.member_role, org12_member, 'members', None)
    assert not org_access.can_attach(org1, org12_member, 'admin_role.members')
    assert not org_access.can_attach(org1, org12_member, 'member_role.members')
    org2.admin_role.members.add(org1_admin)
    assert role_access.can_attach(org1.admin_role, org12_member, 'members', None)
    assert role_access.can_attach(org1.member_role, org12_member, 'members', None)
    assert org_access.can_attach(org1, org12_member, 'admin_role.members')
    assert org_access.can_attach(org1, org12_member, 'member_role.members')

@pytest.mark.django_db
def test_orphaned_user_allowed(org_admin, rando, organization, org_credential):
    if False:
        return 10
    "\n    We still allow adoption of orphaned* users by assigning them to\n    organization member role, but only in the situation where the\n    org admin already posesses indirect access to all of the user's roles\n    *orphaned means user is not a member of any organization\n    "
    org_credential.admin_role.members.add(rando)
    role_access = RoleAccess(org_admin)
    org_access = OrganizationAccess(org_admin)
    assert role_access.can_attach(organization.member_role, rando, 'members', None)
    assert org_access.can_attach(organization, rando, 'member_role.members', None)
    user_access = UserAccess(org_admin)
    assert not user_access.can_change(rando, {'last_name': 'Witzel'})