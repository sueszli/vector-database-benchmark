from core.constants import FLAGSMITH_UPDATED_AT_HEADER
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from environments.models import Environment, EnvironmentAPIKey
from features.models import Feature
from segments.models import EQUAL, Condition, Segment, SegmentRule

def test_get_environment_document(organisation_one, organisation_one_project_one, django_assert_num_queries):
    if False:
        while True:
            i = 10
    project = organisation_one_project_one
    environment = Environment.objects.create(name='Test Environment', project=project)
    api_key = EnvironmentAPIKey.objects.create(environment=environment)
    client = APIClient()
    client.credentials(HTTP_X_ENVIRONMENT_KEY=api_key.key)
    Feature.objects.create(name='test_feature', project=project)
    for i in range(10):
        segment = Segment.objects.create(project=project)
        segment_rule = SegmentRule.objects.create(segment=segment, type=SegmentRule.ALL_RULE)
        Condition.objects.create(operator=EQUAL, property=f'property_{i}', value=f'value_{i}', rule=segment_rule)
        nested_rule = SegmentRule.objects.create(segment=segment, rule=segment_rule, type=SegmentRule.ALL_RULE)
        Condition.objects.create(operator=EQUAL, property=f'nested_prop_{i}', value=f'nested_value_{i}', rule=nested_rule)
    url = reverse('api-v1:environment-document')
    with django_assert_num_queries(11):
        response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()
    assert response.headers[FLAGSMITH_UPDATED_AT_HEADER] == str(environment.updated_at.timestamp())

def test_get_environment_document_fails_with_invalid_key(organisation_one, organisation_one_project_one):
    if False:
        print('Hello World!')
    project = organisation_one_project_one
    environment = Environment.objects.create(name='Test Environment', project=project)
    client = APIClient()
    client.credentials(HTTP_X_ENVIRONMENT_KEY=environment.api_key)
    url = reverse('api-v1:environment-document')
    response = client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_get_environment_document_is_not_throttled_by_user_throttle(environment, feature, settings, environment_api_key):
    if False:
        i = 10
        return i + 15
    settings.REST_FRAMEWORK = {'DEFAULT_THROTTLE_RATES': {'user': '1/minute'}}
    client = APIClient()
    client.credentials(HTTP_X_ENVIRONMENT_KEY=environment_api_key.key)
    url = reverse('api-v1:environment-document')
    for _ in range(10):
        response = client.get(url)
        assert response.status_code == status.HTTP_200_OK