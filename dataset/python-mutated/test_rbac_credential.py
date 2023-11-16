import pytest
from unittest import mock
from awx.main.access import CredentialAccess
from awx.main.models.credential import Credential
from django.contrib.auth.models import User

@pytest.mark.django_db
def test_credential_use_role(credential, user, permissions):
    if False:
        return 10
    u = user('user', False)
    credential.use_role.members.add(u)
    assert u in credential.use_role

def test_credential_access_superuser():
    if False:
        for i in range(10):
            print('nop')
    u = User(username='admin', is_superuser=True)
    access = CredentialAccess(u)
    credential = Credential()
    assert access.can_add(None)
    assert access.can_change(credential, None)
    assert access.can_delete(credential)

@pytest.mark.django_db
def test_credential_access_self(rando):
    if False:
        while True:
            i = 10
    access = CredentialAccess(rando)
    assert access.can_add({'user': rando.pk})

@pytest.mark.django_db
@pytest.mark.parametrize('ext_auth', [True, False])
def test_credential_access_org_user(org_member, org_admin, ext_auth):
    if False:
        i = 10
        return i + 15
    access = CredentialAccess(org_admin)
    with mock.patch('awx.main.access.settings') as settings_mock:
        settings_mock.MANAGE_ORGANIZATION_AUTH = ext_auth
        assert access.can_add({'user': org_member.pk})

@pytest.mark.django_db
def test_credential_access_auditor(credential, organization_factory):
    if False:
        i = 10
        return i + 15
    objects = organization_factory('org_cred_auditor', users=['user1'], roles=['org_cred_auditor.auditor_role:user1'])
    credential.organization = objects.organization
    credential.save()
    access = CredentialAccess(objects.users.user1)
    assert access.can_read(credential)

@pytest.mark.django_db
def test_credential_access_member(alice, credential):
    if False:
        while True:
            i = 10
    credential.admin_role.members.add(alice)
    access = CredentialAccess(alice)
    assert access.can_change(credential, {'description': 'New description.', 'organization': None})

@pytest.mark.django_db
@pytest.mark.parametrize('role_name', ['admin_role', 'credential_admin_role'])
def test_org_credential_access_admin(role_name, alice, org_credential):
    if False:
        return 10
    role = getattr(org_credential.organization, role_name)
    role.members.add(alice)
    access = CredentialAccess(alice)
    assert access.can_change(org_credential, {'description': 'New description.', 'organization': org_credential.organization.pk})

@pytest.mark.django_db
def test_org_and_user_credential_access(alice, organization):
    if False:
        i = 10
        return i + 15
    'Address specific bug where any user could make an org credential\n    in another org without any permissions to that org\n    '
    assert not CredentialAccess(alice).can_add({'name': 'New credential.', 'user': alice.pk, 'organization': organization.pk})

@pytest.mark.django_db
def test_org_credential_access_member(alice, org_credential):
    if False:
        i = 10
        return i + 15
    org_credential.admin_role.members.add(alice)
    access = CredentialAccess(alice)
    assert access.can_change(org_credential, {'description': 'New description.', 'organization': org_credential.organization.pk})
    assert access.can_change(org_credential, {'description': 'New description.'})

@pytest.mark.django_db
def test_cred_no_org(user, credential):
    if False:
        while True:
            i = 10
    su = user('su', True)
    access = CredentialAccess(su)
    assert access.can_change(credential, {'user': su.pk})