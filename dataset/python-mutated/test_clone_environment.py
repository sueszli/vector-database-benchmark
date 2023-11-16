import json
import pytest
from django.urls import reverse
from pytest_lazyfixture import lazy_fixture
from rest_framework import status
from rest_framework.test import APIClient
from tests.integration.helpers import get_env_feature_states_list_with_api, get_feature_segement_list_with_api

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_clone_environment_clones_feature_states_with_value(client, project, environment, environment_api_key, feature):
    if False:
        print('Hello World!')
    feature_state = get_env_feature_states_list_with_api(client, {'environment': environment, 'feature': feature})['results'][0]['id']
    fs_update_url = reverse('api-v1:features:featurestates-detail', args=[feature_state])
    data = {'id': feature_state, 'feature_state_value': 'new_value', 'enabled': False, 'feature': feature, 'environment': environment, 'identity': None, 'feature_segment': None}
    client.put(fs_update_url, data=json.dumps(data), content_type='application/json')
    env_name = 'Cloned env'
    url = reverse('api-v1:environments:environment-clone', args=[environment_api_key])
    res = client.post(url, {'name': env_name})
    assert res.status_code == status.HTTP_200_OK
    source_env_feature_states = get_env_feature_states_list_with_api(client, {'environment': environment})
    clone_env_feature_states = get_env_feature_states_list_with_api(client, {'environment': res.json()['id']})
    assert source_env_feature_states['count'] == 1
    assert source_env_feature_states['results'][0]['id'] != clone_env_feature_states['results'][0]['id']
    assert source_env_feature_states['results'][0]['environment'] != clone_env_feature_states['results'][0]['environment']
    assert source_env_feature_states['results'][0]['feature_state_value'] == clone_env_feature_states['results'][0]['feature_state_value']
    assert source_env_feature_states['results'][0]['enabled'] == clone_env_feature_states['results'][0]['enabled']

def test_clone_environment_creates_admin_permission_with_the_current_user(admin_user, admin_client, environment, environment_api_key):
    if False:
        while True:
            i = 10
    env_name = 'Cloned env'
    url = reverse('api-v1:environments:environment-clone', args=[environment_api_key])
    res = admin_client.post(url, {'name': env_name})
    clone_env_api_key = res.json()['api_key']
    perm_url = reverse('api-v1:environments:environment-user-permissions-list', args=[clone_env_api_key])
    response = admin_client.get(perm_url)
    assert response.json()[0]['admin'] is True

def test_env_clone_creates_feature_segment(admin_client: APIClient, environment: int, environment_api_key: str, feature: int, feature_segment: int, segment_featurestate: int):
    if False:
        for i in range(10):
            print('nop')
    env_name = 'Cloned env'
    url = reverse('api-v1:environments:environment-clone', args=[environment_api_key])
    response = admin_client.post(url, {'name': env_name})
    clone_env_id = response.json()['id']
    base_url = reverse('api-v1:features:feature-segment-list')
    url = f'{base_url}?environment={clone_env_id}&feature={feature}'
    response = admin_client.get(url)
    json_response = response.json()
    assert json_response['count'] == 1
    assert json_response['results'][0]['environment'] == clone_env_id
    assert json_response['results'][0]['id'] != feature_segment

def test_env_clone_clones_segments_overrides(admin_client, environment, environment_api_key, feature, feature_segment, segment):
    if False:
        return 10
    create_url = reverse('api-v1:features:featurestates-list')
    data = {'feature_state_value': {'type': 'unicode', 'boolean_value': None, 'integer_value': None, 'string_value': 'dumb'}, 'multivariate_feature_state_values': [], 'enabled': False, 'feature': feature, 'environment': environment, 'identity': None, 'feature_segment': feature_segment}
    seg_override_response = admin_client.post(create_url, data=json.dumps(data), content_type='application/json')
    assert seg_override_response.status_code == status.HTTP_201_CREATED
    env_name = 'Cloned env'
    url = reverse('api-v1:environments:environment-clone', args=[environment_api_key])
    res = admin_client.post(url, {'name': env_name})
    clone_env_id = res.json()['id']
    source_env_feature_states = get_env_feature_states_list_with_api(admin_client, {'environment': environment, 'feature': feature, 'feature_segment': feature_segment})
    source_feature_segment_id = source_env_feature_states['results'][0]['feature_segment']
    clone_feature_segment_id = get_feature_segement_list_with_api(admin_client, {'environment': res.json()['id'], 'feature': feature, 'segment': segment})['results'][0]['id']
    clone_env_feature_states = get_env_feature_states_list_with_api(admin_client, {'environment': clone_env_id, 'feature': feature, 'feature_segment': clone_feature_segment_id})
    assert source_env_feature_states['count'] == 1
    assert source_env_feature_states['results'][0]['id'] != clone_env_feature_states['results'][0]['id']
    assert source_env_feature_states['results'][0]['environment'] != clone_env_feature_states['results'][0]['environment']
    assert source_env_feature_states['results'][0]['feature_state_value'] == clone_env_feature_states['results'][0]['feature_state_value']
    assert source_env_feature_states['results'][0]['enabled'] == clone_env_feature_states['results'][0]['enabled']
    assert clone_env_feature_states['results'][0]['feature_segment'] == clone_feature_segment_id
    assert clone_feature_segment_id != source_feature_segment_id