import pytest
from django.contrib.sites.models import Site
from django.urls import reverse
from rest_framework import status
from users.models import FFAdminUser

def test_returns_404_when_user_exists(admin_user, django_client):
    if False:
        return 10
    url = reverse('api-v1:users:config-init')
    get_response = django_client.get(url)
    post_response = django_client.post(url)
    assert get_response.status_code == status.HTTP_404_NOT_FOUND
    assert post_response.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.django_db()
def test_returns_200_when_no_user_exists(django_client):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:users:config-init')
    response = django_client.get(url)
    assert response.status_code == status.HTTP_200_OK

def test_valid_request_creates_admin_and_updates_site(db, django_client):
    if False:
        print('Hello World!')
    url = reverse('api-v1:users:config-init')
    form_data = {'username': 'test-admin', 'email': 'test@email.com', 'password': 'test123', 'site_name': 'test_site', 'site_domain': 'test.com'}
    response = django_client.post(url, data=form_data)
    assert response.status_code == status.HTTP_200_OK
    assert FFAdminUser.objects.filter(email=form_data.get('email')).count() == 1
    assert Site.objects.filter(name=form_data.get('site_name')).count() == 1

def test_invalid_form_does_not_change_anything(db, django_client):
    if False:
        while True:
            i = 10
    url = reverse('api-v1:users:config-init')
    form_data = {'username': 'test-admin', 'email': 'invalid_email', 'password': 'test123', 'site_name': 'test_site', 'site_domain': 'test.com'}
    response = django_client.post(url, data=form_data)
    assert response.status_code == status.HTTP_200_OK
    assert Site.objects.filter(name=form_data.get('site_name')).count() == 0
    assert FFAdminUser.objects.filter(email=form_data.get('email')).count() == 0