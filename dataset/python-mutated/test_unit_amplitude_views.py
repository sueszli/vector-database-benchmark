import json
from django.urls import reverse
from rest_framework import status

def test_create_amplitude_integration(environment, admin_client):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:environments:integrations-amplitude-list', args=[environment.api_key])
    response = admin_client.post(path=url, data=json.dumps({'api_key': 'some-key'}), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED

def test_create_amplitude_integration_in_environment_with_deleted_integration(environment, admin_client, deleted_amplitude_integration):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:environments:integrations-amplitude-list', args=[environment.api_key])
    response = admin_client.post(path=url, data=json.dumps({'api_key': 'some-key'}), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED