from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

def test_get_version_info(api_client: APIClient) -> None:
    if False:
        return 10
    url = reverse('version-info')
    response = api_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'ci_commit_sha': 'unknown', 'image_tag': 'unknown', 'is_enterprise': False}