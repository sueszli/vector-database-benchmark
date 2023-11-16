import json
import pytest
from django.urls import reverse
from pytest_lazyfixture import lazy_fixture
from rest_framework import status

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_state_for_identity_override(client, environment, identity, feature):
    if False:
        i = 10
        return i + 15
    create_url = reverse('api-v1:features:featurestates-list')
    data = {'enabled': True, 'feature_state_value': {'type': 'unicode', 'string_value': 'test value'}, 'identity': identity, 'environment': environment, 'feature': feature}
    response = client.post(create_url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_state_for_identity_with_identifier(client, environment, identity, feature, identity_identifier):
    if False:
        print('Hello World!')
    create_url = reverse('api-v1:features:featurestates-list')
    data = {'enabled': True, 'feature_state_value': {'type': 'unicode', 'string_value': 'test value'}, 'identifier': identity_identifier, 'environment': environment, 'feature': feature}
    response = client.post(create_url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_list_feature_states_for_environment(client, environment, feature):
    if False:
        return 10
    base_url = reverse('api-v1:features:featurestates-list')
    url = f'{base_url}?environment={environment}'
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 1
    assert response_json['results'][0]['environment'] == environment

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_feature_state(client, environment, feature_state, feature, identity):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:features:featurestates-detail', args=[feature_state])
    feature_state_value = 'New value'
    data = {'enabled': True, 'feature_state_value': {'type': 'unicode', 'string_value': feature_state_value}, 'environment': environment, 'feature': feature}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['feature_state_value']['string_value'] == feature_state_value

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_feature_state_for_identity_with_identifier(client, environment, identity_featurestate, feature, identity, identity_identifier):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:features:featurestates-detail', args=[identity_featurestate])
    feature_state_value = 'New value'
    data = {'enabled': True, 'feature_state_value': {'type': 'unicode', 'string_value': feature_state_value}, 'identifier': identity_identifier, 'environment': environment, 'feature': feature}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['feature_state_value']['string_value'] == feature_state_value