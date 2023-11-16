import pytest
from awx.main.models import CredentialInputSource
from awx.api.versioning import reverse

@pytest.mark.django_db
def test_associate_credential_input_source(get, post, delete, admin, vault_credential, external_credential):
    if False:
        i = 10
        return i + 15
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_example_key'}}
    response = post(list_url, params, admin)
    assert response.status_code == 201
    detail = get(response.data['url'], admin)
    assert detail.status_code == 200
    response = get(list_url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 1
    assert CredentialInputSource.objects.count() == 1
    input_source = CredentialInputSource.objects.first()
    assert input_source.metadata == {'key': 'some_example_key'}
    response = delete(reverse('api:credential_input_source_detail', kwargs={'pk': detail.data['id']}), admin)
    assert response.status_code == 204
    response = get(list_url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 0
    assert CredentialInputSource.objects.count() == 0

@pytest.mark.django_db
@pytest.mark.parametrize('metadata', [{}, {'key': None}, {'key': 123}, {'extraneous': 'foo'}])
def test_associate_credential_input_source_with_invalid_metadata(get, post, admin, vault_credential, external_credential, metadata):
    if False:
        while True:
            i = 10
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': metadata}
    response = post(list_url, params, admin)
    assert response.status_code == 400
    assert b'metadata' in response.content

@pytest.mark.django_db
def test_create_from_list(get, post, admin, vault_credential, external_credential):
    if False:
        for i in range(10):
            print('nop')
    params = {'source_credential': external_credential.pk, 'target_credential': vault_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_example_key'}}
    assert post(reverse('api:credential_input_source_list'), params, admin).status_code == 201
    assert CredentialInputSource.objects.count() == 1

@pytest.mark.django_db
def test_create_credential_input_source_with_external_target_returns_400(post, admin, external_credential, other_external_credential):
    if False:
        return 10
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': other_external_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'token', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, admin)
    assert response.status_code == 400
    assert response.data['target_credential'] == ['Target must be a non-external credential']

@pytest.mark.django_db
def test_input_source_rbac_associate(get, post, delete, alice, vault_credential, external_credential):
    if False:
        for i in range(10):
            print('nop')
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, alice)
    assert response.status_code == 403
    vault_credential.admin_role.members.add(alice)
    response = post(list_url, params, alice)
    assert response.status_code == 403
    external_credential.use_role.members.add(alice)
    response = post(list_url, params, alice)
    assert response.status_code == 201
    detail = get(response.data['url'], alice)
    assert detail.status_code == 200
    vault_credential.admin_role.members.remove(alice)
    external_credential.use_role.members.remove(alice)
    assert get(response.data['url'], alice).status_code == 403
    delete_url = reverse('api:credential_input_source_detail', kwargs={'pk': detail.data['id']})
    response = delete(delete_url, alice)
    assert response.status_code == 403
    vault_credential.admin_role.members.add(alice)
    response = delete(delete_url, alice)
    assert response.status_code == 204

@pytest.mark.django_db
def test_input_source_detail_rbac(get, post, patch, delete, admin, alice, vault_credential, external_credential, other_external_credential):
    if False:
        for i in range(10):
            print('nop')
    sublist_url = reverse('api:credential_input_source_sublist', kwargs={'pk': vault_credential.pk})
    params = {'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_key'}}
    response = post(sublist_url, params, admin)
    assert response.status_code == 201
    url = response.data['url']
    detail = get(url, alice)
    assert detail.status_code == 403
    vault_credential.read_role.members.add(alice)
    detail = get(url, alice)
    assert detail.status_code == 200
    response = get(sublist_url, admin)
    assert response.status_code == 200
    assert response.data['count'] == 1
    assert patch(url, {'input_field_name': 'vault_id'}, alice).status_code == 403
    assert delete(url, alice).status_code == 403
    vault_credential.admin_role.members.add(alice)
    assert patch(url, {'input_field_name': 'vault_id'}, alice).status_code == 403
    external_credential.use_role.members.add(alice)
    assert patch(url, {'input_field_name': 'vault_id'}, alice).status_code == 200
    assert CredentialInputSource.objects.first().input_field_name == 'vault_id'
    assert patch(url, {'source_credential': other_external_credential.pk}, alice).status_code == 403
    assert delete(url, alice).status_code == 204
    assert CredentialInputSource.objects.count() == 0

@pytest.mark.django_db
def test_input_source_create_rbac(get, post, patch, delete, alice, vault_credential, external_credential, other_external_credential):
    if False:
        print('Hello World!')
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, alice)
    assert response.status_code == 403
    vault_credential.admin_role.members.add(alice)
    response = post(list_url, params, alice)
    assert response.status_code == 403
    external_credential.use_role.members.add(alice)
    response = post(list_url, params, alice)
    assert response.status_code == 201
    assert CredentialInputSource.objects.count() == 1

@pytest.mark.django_db
def test_input_source_rbac_swap_target_credential(get, post, put, patch, admin, alice, machine_credential, vault_credential, external_credential):
    if False:
        return 10
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, admin)
    assert response.status_code == 201
    url = response.data['url']
    external_credential.admin_role.members.add(alice)
    assert patch(url, {'target_credential': machine_credential.pk, 'input_field_name': 'password'}, alice).status_code == 403
    vault_credential.admin_role.members.add(alice)
    assert patch(url, {'target_credential': machine_credential.pk, 'input_field_name': 'password'}, alice).status_code == 403
    machine_credential.admin_role.members.add(alice)
    assert patch(url, {'target_credential': machine_credential.pk, 'input_field_name': 'password'}, alice).status_code == 200

@pytest.mark.django_db
def test_input_source_rbac_change_metadata(get, post, put, patch, admin, alice, machine_credential, external_credential):
    if False:
        i = 10
        return i + 15
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': machine_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'password', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, admin)
    assert response.status_code == 201
    url = response.data['url']
    assert patch(url, {'metadata': {'key': 'some_other_key'}}, alice).status_code == 403
    machine_credential.admin_role.members.add(alice)
    assert patch(url, {'metadata': {'key': 'some_other_key'}}, alice).status_code == 403
    external_credential.use_role.members.add(alice)
    assert patch(url, {'metadata': {'key': 'some_other_key'}}, alice).status_code == 200

@pytest.mark.django_db
def test_create_credential_input_source_with_non_external_source_returns_400(post, admin, credential, vault_credential):
    if False:
        while True:
            i = 10
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': credential.pk, 'input_field_name': 'vault_password'}
    response = post(list_url, params, admin)
    assert response.status_code == 400
    assert response.data['source_credential'] == ['Source must be an external credential']

@pytest.mark.django_db
def test_create_credential_input_source_with_undefined_input_returns_400(post, admin, vault_credential, external_credential):
    if False:
        i = 10
        return i + 15
    list_url = reverse('api:credential_input_source_list')
    params = {'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'not_defined_for_credential_type', 'metadata': {'key': 'some_key'}}
    response = post(list_url, params, admin)
    assert response.status_code == 400
    assert response.data['input_field_name'] == ['Input field must be defined on target credential (options are vault_id, vault_password).']

@pytest.mark.django_db
def test_create_credential_input_source_with_already_used_input_returns_400(post, admin, vault_credential, external_credential, other_external_credential):
    if False:
        for i in range(10):
            print('nop')
    list_url = reverse('api:credential_input_source_list')
    all_params = [{'target_credential': vault_credential.pk, 'source_credential': external_credential.pk, 'input_field_name': 'vault_password'}, {'target_credential': vault_credential.pk, 'source_credential': other_external_credential.pk, 'input_field_name': 'vault_password'}]
    all_responses = [post(list_url, params, admin) for params in all_params]
    assert all_responses.pop().status_code == 400