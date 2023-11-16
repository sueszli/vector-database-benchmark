import json
import pytest
from django.urls import reverse
from pytest_lazyfixture import lazy_fixture
from rest_framework import status

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_feature_state_value_updates_feature_state_value(client, environment, environment_api_key, feature, feature_state):
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-featurestates-detail', args=[environment_api_key, feature_state])
    new_value = 'new-value'
    data = {'id': feature_state, 'feature_state_value': new_value, 'enabled': False, 'feature': feature, 'environment': environment, 'identity': None, 'feature_segment': None}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    response.json()['feature_state_value'] == new_value