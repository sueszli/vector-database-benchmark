import json
from unittest import mock
import pytest
from django.urls import reverse
from rest_framework import status
from tests.integration.helpers import create_feature_with_api, create_mv_option_with_api
from features.feature_types import MULTIVARIATE
variant_1_value = 'variant-1-value'
variant_2_value = 'variant-2-value'
control_value = 'control'
variant_1_percentage_allocation = 20
variant_2_percentage_allocation = 30
total_variance_percentage = variant_1_percentage_allocation + variant_2_percentage_allocation

@pytest.mark.parametrize('hashed_percentage, expected_mv_value', ((variant_1_percentage_allocation / 100 - 0.01, variant_1_value), (total_variance_percentage / 100 - 0.01, variant_2_value), (total_variance_percentage / 100 + 0.01, control_value)))
@mock.patch('features.models.get_hashed_percentage_for_object_ids')
def test_get_feature_states_for_identity(mock_get_hashed_percentage_value, hashed_percentage, expected_mv_value, sdk_client, admin_client, project, environment_api_key, environment, identity, identity_identifier):
    if False:
        while True:
            i = 10
    standard_feature_initial_value = 'control'
    standard_feature_id = create_feature_with_api(client=admin_client, project_id=project, feature_name='standard_feature', initial_value=standard_feature_initial_value)
    multivariate_feature_id = create_feature_with_api(client=admin_client, project_id=project, feature_name='multivariate_feature', initial_value=control_value, feature_type=MULTIVARIATE)
    create_mv_option_with_api(admin_client, project, multivariate_feature_id, variant_1_percentage_allocation, variant_1_value)
    create_mv_option_with_api(admin_client, project, multivariate_feature_id, variant_2_percentage_allocation, variant_2_value)
    mock_get_hashed_percentage_value.return_value = hashed_percentage
    base_identity_flags_url = reverse('api-v1:sdk-identities')
    identity_flags_url = f'{base_identity_flags_url}?identifier={identity_identifier}'
    identity_response_1 = sdk_client.get(identity_flags_url)
    assert identity_response_1.status_code == status.HTTP_200_OK
    identity_response_json = identity_response_1.json()
    assert len(identity_response_json['flags']) == 2
    values_dict = {flag['feature']['id']: flag['feature_state_value'] for flag in identity_response_json['flags']}
    assert values_dict[standard_feature_id] == standard_feature_initial_value
    assert values_dict[multivariate_feature_id] == expected_mv_value
    feature_state_id = next(filter(lambda flag: flag['feature']['id'] == multivariate_feature_id, identity_response_json['flags']))['id']
    feature_state_detail_url = reverse('api-v1:environments:environment-featurestates-detail', args=[environment_api_key, feature_state_id])
    retrieve_feature_state_response = admin_client.get(feature_state_detail_url)
    feature_state_data = retrieve_feature_state_response.json()
    mv_values = feature_state_data['multivariate_feature_state_values']
    mv_values[0]['percentage_allocation'] = 0
    mv_values[1]['percentage_allocation'] = 100
    update_feature_state_response = admin_client.put(feature_state_detail_url, data=json.dumps(feature_state_data), content_type='application/json')
    assert update_feature_state_response.status_code == status.HTTP_200_OK
    identity_response_2 = sdk_client.get(identity_flags_url)
    values_dict = {flag['feature']['id']: flag['feature_state_value'] for flag in identity_response_2.json()['flags']}
    assert values_dict[multivariate_feature_id] == variant_2_value

def test_get_feature_states_for_identity_only_makes_one_query_to_get_mv_feature_states(sdk_client, admin_client, project, environment, identity, identity_identifier, django_assert_num_queries):
    if False:
        return 10
    for i in range(2):
        feature_id = create_feature_with_api(client=admin_client, project_id=project, feature_name=f'multivariate_feature_{i}', initial_value=control_value, feature_type=MULTIVARIATE)
        create_mv_option_with_api(admin_client, project, feature_id, variant_1_percentage_allocation, variant_1_value)
        create_mv_option_with_api(admin_client, project, feature_id, variant_2_percentage_allocation, variant_2_value)
    base_url = reverse('api-v1:sdk-identities')
    url = f'{base_url}?identifier={identity_identifier}'
    with django_assert_num_queries(6):
        first_identity_response = sdk_client.get(url)
    feature_id = create_feature_with_api(client=admin_client, project_id=project, feature_name='another_multivariate_feature', initial_value=control_value, feature_type=MULTIVARIATE)
    create_mv_option_with_api(admin_client, project, feature_id, variant_1_percentage_allocation, variant_1_value)
    create_mv_option_with_api(admin_client, project, feature_id, variant_2_percentage_allocation, variant_2_value)
    with django_assert_num_queries(5):
        second_identity_response = sdk_client.get(url)
    assert first_identity_response.status_code == status.HTTP_200_OK
    assert second_identity_response.status_code == status.HTTP_200_OK
    first_identity_response_json = first_identity_response.json()
    assert len(first_identity_response_json['flags']) == 2
    second_identity_response_json = second_identity_response.json()
    assert len(second_identity_response_json['flags']) == 3