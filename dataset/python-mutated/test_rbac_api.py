from unittest import mock
import pytest
from django.db import transaction
from awx.api.versioning import reverse
from awx.main.models.rbac import Role, ROLE_SINGLETON_SYSTEM_ADMINISTRATOR

@pytest.fixture
def role():
    if False:
        print('Hello World!')
    return Role.objects.create(role_field='admin_role')

@pytest.mark.django_db
def test_get_roles_list_admin(organization, get, admin):
    if False:
        return 10
    'Admin can see list of all roles'
    url = reverse('api:role_list')
    response = get(url, admin)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] > 0

@pytest.mark.django_db
def test_get_roles_list_user(organization, inventory, team, get, user):
    if False:
        i = 10
        return i + 15
    'Users can see all roles they have access to, but not all roles'
    this_user = user('user-test_get_roles_list_user')
    organization.member_role.members.add(this_user)
    custom_role = Role.objects.create(role_field='custom_role-test_get_roles_list_user')
    organization.member_role.children.add(custom_role)
    url = reverse('api:role_list')
    response = get(url, this_user)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] > 0
    assert roles['count'] == len(roles['results'])
    role_hash = {}
    for r in roles['results']:
        role_hash[r['id']] = r
    assert Role.singleton(ROLE_SINGLETON_SYSTEM_ADMINISTRATOR).id in role_hash
    assert organization.admin_role.id in role_hash
    assert organization.member_role.id in role_hash
    assert custom_role.id in role_hash
    assert inventory.admin_role.id not in role_hash
    assert team.member_role.id not in role_hash

@pytest.mark.django_db
def test_roles_visibility(get, organization, project, admin, alice, bob):
    if False:
        for i in range(10):
            print('nop')
    Role.singleton('system_auditor').members.add(alice)
    assert get(reverse('api:role_list') + '?id=%d' % project.update_role.id, user=admin).data['count'] == 1
    assert get(reverse('api:role_list') + '?id=%d' % project.update_role.id, user=alice).data['count'] == 1
    assert get(reverse('api:role_list') + '?id=%d' % project.update_role.id, user=bob).data['count'] == 0
    organization.auditor_role.members.add(bob)
    assert get(reverse('api:role_list') + '?id=%d' % project.update_role.id, user=bob).data['count'] == 1

@pytest.mark.django_db
def test_roles_filter_visibility(get, organization, project, admin, alice, bob):
    if False:
        for i in range(10):
            print('nop')
    Role.singleton('system_auditor').members.add(alice)
    project.update_role.members.add(admin)
    assert get(reverse('api:user_roles_list', kwargs={'pk': admin.id}) + '?id=%d' % project.update_role.id, user=admin).data['count'] == 1
    assert get(reverse('api:user_roles_list', kwargs={'pk': admin.id}) + '?id=%d' % project.update_role.id, user=alice).data['count'] == 1
    assert get(reverse('api:user_roles_list', kwargs={'pk': admin.id}) + '?id=%d' % project.update_role.id, user=bob).data['count'] == 0
    organization.auditor_role.members.add(bob)
    assert get(reverse('api:user_roles_list', kwargs={'pk': admin.id}) + '?id=%d' % project.update_role.id, user=bob).data['count'] == 1
    organization.auditor_role.members.remove(bob)
    project.use_role.members.add(bob)
    assert get(reverse('api:user_roles_list', kwargs={'pk': admin.id}) + '?id=%d' % project.update_role.id, user=bob).data['count'] == 1

@pytest.mark.django_db
def test_cant_create_role(post, admin):
    if False:
        for i in range(10):
            print('nop')
    "Ensure we can't create new roles through the api"
    response = post(reverse('api:role_list'), {'name': 'New Role'}, admin)
    assert response.status_code == 405

@pytest.mark.django_db
def test_cant_delete_role(delete, admin, inventory):
    if False:
        print('Hello World!')
    "Ensure we can't delete roles through the api"
    response = delete(reverse('api:role_detail', kwargs={'pk': inventory.admin_role.id}), admin)
    assert response.status_code == 405

@pytest.mark.django_db
def test_get_user_roles_list(get, admin):
    if False:
        while True:
            i = 10
    url = reverse('api:user_roles_list', kwargs={'pk': admin.id})
    response = get(url, admin)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] > 0

@pytest.mark.django_db
def test_user_view_other_user_roles(organization, inventory, team, get, alice, bob):
    if False:
        for i in range(10):
            print('nop')
    'Users can see roles for other users, but only the roles that that user has access to see as well'
    organization.member_role.members.add(alice)
    organization.admin_role.members.add(bob)
    organization.member_role.members.add(bob)
    custom_role = Role.objects.create(role_field='custom_role-test_user_view_admin_roles_list')
    organization.member_role.children.add(custom_role)
    team.member_role.members.add(bob)
    url = reverse('api:user_roles_list', kwargs={'pk': bob.id})
    response = get(url, alice)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] > 0
    assert roles['count'] == len(roles['results'])
    role_hash = {}
    for r in roles['results']:
        role_hash[r['id']] = r['name']
    assert organization.admin_role.id in role_hash
    assert custom_role.id not in role_hash
    assert Role.singleton(ROLE_SINGLETON_SYSTEM_ADMINISTRATOR).id not in role_hash
    assert inventory.admin_role.id not in role_hash
    assert team.member_role.id not in role_hash
    team.member_role.members.add(alice)
    response = get(url, alice)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] > 0
    assert roles['count'] == len(roles['results'])
    role_hash = {}
    for r in roles['results']:
        role_hash[r['id']] = r['name']
    assert team.member_role.id in role_hash

@pytest.mark.django_db
def test_add_role_to_user(role, post, admin):
    if False:
        print('Hello World!')
    assert admin.roles.filter(id=role.id).count() == 0
    url = reverse('api:user_roles_list', kwargs={'pk': admin.id})
    response = post(url, {'id': role.id}, admin)
    assert response.status_code == 204
    assert admin.roles.filter(id=role.id).count() == 1
    response = post(url, {'id': role.id}, admin)
    assert response.status_code == 204
    assert admin.roles.filter(id=role.id).count() == 1
    response = post(url, {}, admin)
    assert response.status_code == 400
    assert admin.roles.filter(id=role.id).count() == 1

@pytest.mark.django_db
def test_remove_role_from_user(role, post, admin):
    if False:
        return 10
    assert admin.roles.filter(id=role.id).count() == 0
    url = reverse('api:user_roles_list', kwargs={'pk': admin.id})
    response = post(url, {'id': role.id}, admin)
    assert response.status_code == 204
    assert admin.roles.filter(id=role.id).count() == 1
    response = post(url, {'disassociate': role.id, 'id': role.id}, admin)
    assert response.status_code == 204
    assert admin.roles.filter(id=role.id).count() == 0

@pytest.mark.django_db
def test_get_teams_roles_list(get, team, organization, admin):
    if False:
        while True:
            i = 10
    team.member_role.children.add(organization.admin_role)
    url = reverse('api:team_roles_list', kwargs={'pk': team.id})
    response = get(url, admin)
    assert response.status_code == 200
    roles = response.data
    assert roles['count'] == 1
    assert roles['results'][0]['id'] == organization.admin_role.id or roles['results'][1]['id'] == organization.admin_role.id

@pytest.mark.django_db
def test_add_role_to_teams(team, post, admin):
    if False:
        print('Hello World!')
    assert team.member_role.children.filter(id=team.member_role.id).count() == 0
    url = reverse('api:team_roles_list', kwargs={'pk': team.id})
    response = post(url, {'id': team.member_role.id}, admin)
    assert response.status_code == 204
    assert team.member_role.children.filter(id=team.member_role.id).count() == 1
    response = post(url, {'id': team.member_role.id}, admin)
    assert response.status_code == 204
    assert team.member_role.children.filter(id=team.member_role.id).count() == 1
    response = post(url, {}, admin)
    assert response.status_code == 400
    assert team.member_role.children.filter(id=team.member_role.id).count() == 1

@pytest.mark.django_db
def test_remove_role_from_teams(team, post, admin):
    if False:
        while True:
            i = 10
    assert team.member_role.children.filter(id=team.member_role.id).count() == 0
    url = reverse('api:team_roles_list', kwargs={'pk': team.id})
    response = post(url, {'id': team.member_role.id}, admin)
    assert response.status_code == 204
    assert team.member_role.children.filter(id=team.member_role.id).count() == 1
    response = post(url, {'disassociate': team.member_role.id, 'id': team.member_role.id}, admin)
    assert response.status_code == 204
    assert team.member_role.children.filter(id=team.member_role.id).count() == 0

@pytest.mark.django_db
def test_get_role(get, admin, role):
    if False:
        while True:
            i = 10
    url = reverse('api:role_detail', kwargs={'pk': role.id})
    response = get(url, admin)
    assert response.status_code == 200
    assert response.data['id'] == role.id

@pytest.mark.django_db
def test_put_role_405(put, admin, role):
    if False:
        return 10
    url = reverse('api:role_detail', kwargs={'pk': role.id})
    response = put(url, {'name': 'Some new name'}, admin)
    assert response.status_code == 405

@pytest.mark.django_db
def test_put_role_access_denied(put, alice, role):
    if False:
        return 10
    url = reverse('api:role_detail', kwargs={'pk': role.id})
    response = put(url, {'name': 'Some new name'}, alice)
    assert response.status_code == 403 or response.status_code == 405

@pytest.mark.django_db
def test_get_role_users(get, admin, role):
    if False:
        return 10
    role.members.add(admin)
    url = reverse('api:role_users_list', kwargs={'pk': role.id})
    response = get(url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 1
    assert response.data['results'][0]['id'] == admin.id

@pytest.mark.django_db
def test_add_user_to_role(post, admin, role):
    if False:
        while True:
            i = 10
    url = reverse('api:role_users_list', kwargs={'pk': role.id})
    assert role.members.filter(id=admin.id).count() == 0
    post(url, {'id': admin.id}, admin)
    assert role.members.filter(id=admin.id).count() == 1

@pytest.mark.django_db
def test_remove_user_to_role(post, admin, role):
    if False:
        return 10
    role.members.add(admin)
    url = reverse('api:role_users_list', kwargs={'pk': role.id})
    assert role.members.filter(id=admin.id).count() == 1
    post(url, {'disassociate': True, 'id': admin.id}, admin)
    assert role.members.filter(id=admin.id).count() == 0

@pytest.mark.django_db
def test_org_admin_add_user_to_job_template(post, organization, check_jobtemplate, user):
    if False:
        while True:
            i = 10
    'Tests that a user with permissions to assign/revoke membership to a particular role can do so'
    org_admin = user('org-admin')
    joe = user('joe')
    organization.admin_role.members.add(org_admin)
    assert org_admin in check_jobtemplate.admin_role
    assert joe not in check_jobtemplate.execute_role
    post(reverse('api:role_users_list', kwargs={'pk': check_jobtemplate.execute_role.id}), {'id': joe.id}, org_admin)
    assert joe in check_jobtemplate.execute_role

@pytest.mark.django_db
def test_org_admin_remove_user_from_job_template(post, organization, check_jobtemplate, user):
    if False:
        i = 10
        return i + 15
    'Tests that a user with permissions to assign/revoke membership to a particular role can do so'
    org_admin = user('org-admin')
    joe = user('joe')
    organization.admin_role.members.add(org_admin)
    check_jobtemplate.execute_role.members.add(joe)
    assert org_admin in check_jobtemplate.admin_role
    assert joe in check_jobtemplate.execute_role
    post(reverse('api:role_users_list', kwargs={'pk': check_jobtemplate.execute_role.id}), {'disassociate': True, 'id': joe.id}, org_admin)
    assert joe not in check_jobtemplate.execute_role

@pytest.mark.django_db
def test_user_fail_to_add_user_to_job_template(post, organization, check_jobtemplate, user):
    if False:
        while True:
            i = 10
    'Tests that a user without permissions to assign/revoke membership to a particular role cannot do so'
    rando = user('rando')
    joe = user('joe')
    assert rando not in check_jobtemplate.admin_role
    assert joe not in check_jobtemplate.execute_role
    with transaction.atomic():
        res = post(reverse('api:role_users_list', kwargs={'pk': check_jobtemplate.execute_role.id}), {'id': joe.id}, rando)
    assert res.status_code == 403
    assert joe not in check_jobtemplate.execute_role

@pytest.mark.django_db
def test_user_fail_to_remove_user_to_job_template(post, organization, check_jobtemplate, user):
    if False:
        i = 10
        return i + 15
    'Tests that a user without permissions to assign/revoke membership to a particular role cannot do so'
    rando = user('rando')
    joe = user('joe')
    check_jobtemplate.execute_role.members.add(joe)
    assert rando not in check_jobtemplate.admin_role
    assert joe in check_jobtemplate.execute_role
    with transaction.atomic():
        res = post(reverse('api:role_users_list', kwargs={'pk': check_jobtemplate.execute_role.id}), {'disassociate': True, 'id': joe.id}, rando)
    assert res.status_code == 403
    assert joe in check_jobtemplate.execute_role

@pytest.mark.django_db
def test_get_role_teams(get, team, admin, role):
    if False:
        for i in range(10):
            print('nop')
    role.parents.add(team.member_role)
    url = reverse('api:role_teams_list', kwargs={'pk': role.id})
    response = get(url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 1
    assert response.data['results'][0]['id'] == team.id

@pytest.mark.django_db
def test_add_team_to_role(post, team, admin, role):
    if False:
        while True:
            i = 10
    url = reverse('api:role_teams_list', kwargs={'pk': role.id})
    assert role.members.filter(id=admin.id).count() == 0
    res = post(url, {'id': team.id}, admin)
    assert res.status_code == 204
    assert role.parents.filter(id=team.member_role.id).count() == 1

@pytest.mark.django_db
def test_remove_team_from_role(post, team, admin, role):
    if False:
        i = 10
        return i + 15
    role.members.add(admin)
    url = reverse('api:role_teams_list', kwargs={'pk': role.id})
    assert role.members.filter(id=admin.id).count() == 1
    res = post(url, {'disassociate': True, 'id': team.id}, admin)
    assert res.status_code == 204
    assert role.parents.filter(id=team.member_role.id).count() == 0

@pytest.mark.django_db
def test_role_parents(get, team, admin, role):
    if False:
        i = 10
        return i + 15
    role.parents.add(team.member_role)
    url = reverse('api:role_parents_list', kwargs={'pk': role.id})
    response = get(url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 1
    assert response.data['results'][0]['id'] == team.member_role.id

@pytest.mark.django_db
def test_role_children(get, team, admin, role):
    if False:
        return 10
    role.parents.add(team.member_role)
    url = reverse('api:role_children_list', kwargs={'pk': team.member_role.id})
    response = get(url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 2
    assert response.data['results'][0]['id'] == role.id or response.data['results'][1]['id'] == role.id

@pytest.mark.django_db
def test_ensure_rbac_fields_are_present(organization, get, admin):
    if False:
        while True:
            i = 10
    url = reverse('api:organization_detail', kwargs={'pk': organization.id})
    response = get(url, admin)
    assert response.status_code == 200
    org = response.data
    assert 'summary_fields' in org
    assert 'object_roles' in org['summary_fields']
    role_pk = org['summary_fields']['object_roles']['admin_role']['id']
    role_url = reverse('api:role_detail', kwargs={'pk': role_pk})
    org_role_response = get(role_url, admin)
    assert org_role_response.status_code == 200
    role = org_role_response.data
    assert role['related']['organization'] == url

@pytest.mark.django_db
def test_ensure_role_summary_is_present(organization, get, user):
    if False:
        print('Hello World!')
    url = reverse('api:organization_detail', kwargs={'pk': organization.id})
    response = get(url, user('admin', True))
    assert response.status_code == 200
    org = response.data
    assert 'summary_fields' in org
    assert 'object_roles' in org['summary_fields']
    assert org['summary_fields']['object_roles']['admin_role']['id'] > 0