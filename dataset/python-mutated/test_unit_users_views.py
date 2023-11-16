import json
import pytest
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.urls import reverse
from djoser import utils
from djoser.email import PasswordResetEmail
from rest_framework import status
from rest_framework.test import APIClient
from organisations.models import Organisation
from users.models import FFAdminUser

@pytest.mark.django_db
def test_delete_user():
    if False:
        print('Hello World!')

    def delete_user(user: FFAdminUser, password: str, delete_orphan_organisations: bool=True):
        if False:
            while True:
                i = 10
        client = APIClient()
        client.force_authenticate(user)
        data = {'current_password': password, 'delete_orphan_organisations': delete_orphan_organisations}
        url = '/api/v1/auth/users/me/'
        return client.delete(url, data=json.dumps(data), content_type='application/json')
    email1 = 'test1@example.com'
    email2 = 'test2@example.com'
    email3 = 'test3@example.com'
    password = 'password'
    user1 = FFAdminUser.objects.create_user(email=email1, password=password)
    user2 = FFAdminUser.objects.create_user(email=email2, password=password)
    user3 = FFAdminUser.objects.create_user(email=email3, password=password)
    org1 = Organisation.objects.create(name='org1')
    org2 = Organisation.objects.create(name='org2')
    org3 = Organisation.objects.create(name='org3')
    org1.users.add(user1)
    org2.users.add(user1)
    org3.users.add(user1)
    org2.users.add(user2)
    org1.users.add(user3)
    response = delete_user(user2, password)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert not FFAdminUser.objects.filter(email=email2).exists()
    assert Organisation.objects.filter(name='org3').count() == 1
    assert Organisation.objects.filter(name='org1').count() == 1
    assert Organisation.objects.filter(name='org2').count() == 1
    response = delete_user(user1, password)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert not FFAdminUser.objects.filter(email=email1).exists()
    assert Organisation.objects.filter(name='org3').count() == 0
    assert Organisation.objects.filter(name='org2').count() == 0
    assert Organisation.objects.filter(name='org1').count() == 1
    assert FFAdminUser.objects.filter(email=email3).exists()
    response = delete_user(user3, password, False)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert not FFAdminUser.objects.filter(email=email3).exists()
    assert Organisation.objects.filter(name='org1').count() == 1

@pytest.mark.django_db
def test_change_email_address_api(mocker):
    if False:
        print('Hello World!')
    mocked_task = mocker.patch('users.signals.send_email_changed_notification_email')
    old_email = 'test_user@test.com'
    first_name = 'firstname'
    user = FFAdminUser.objects.create_user(username='test_user', email=old_email, first_name=first_name, last_name='user', password='password')
    client = APIClient()
    client.force_authenticate(user)
    new_email = 'test_user1@test.com'
    data = {'new_email': new_email, 'current_password': 'password'}
    url = reverse('api-v1:custom_auth:ffadminuser-set-username')
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert user.email == new_email
    (args, kwargs) = mocked_task.delay.call_args
    assert len(args) == 0
    assert len(kwargs) == 1
    assert kwargs['args'] == (first_name, settings.DEFAULT_FROM_EMAIL, old_email)

@pytest.mark.django_db
def test_send_reset_password_emails_rate_limit(settings, client, test_user):
    if False:
        while True:
            i = 10
    settings.MAX_PASSWORD_RESET_EMAILS = 2
    settings.PASSWORD_RESET_EMAIL_COOLDOWN = 60
    url = reverse('api-v1:custom_auth:ffadminuser-reset-password')
    data = {'email': test_user.email}
    for _ in range(5):
        response = client.post(url, data=json.dumps(data), content_type='application/json')
        assert response.status_code == status.HTTP_204_NO_CONTENT
    assert len(mail.outbox) == 2
    isinstance(mail.outbox[0], PasswordResetEmail)
    isinstance(mail.outbox[1], PasswordResetEmail)
    mail.outbox.clear()
    settings.PASSWORD_RESET_EMAIL_COOLDOWN = 0.001
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert len(mail.outbox) == 1
    isinstance(mail.outbox[0], PasswordResetEmail)

@pytest.mark.django_db
def test_send_reset_password_emails_rate_limit_resets_after_password_reset(settings, client, test_user):
    if False:
        return 10
    settings.MAX_PASSWORD_RESET_EMAILS = 2
    settings.PASSWORD_RESET_EMAIL_COOLDOWN = 60 * 60 * 24
    url = reverse('api-v1:custom_auth:ffadminuser-reset-password')
    data = {'email': test_user.email}
    for _ in range(5):
        response = client.post(url, data=json.dumps(data), content_type='application/json')
        assert response.status_code == status.HTTP_204_NO_CONTENT
    assert len(mail.outbox) == 2
    mail.outbox.clear()
    reset_password_data = {'new_password': 'new_password', 're_new_password': 'new_password', 'uid': utils.encode_uid(test_user.pk), 'token': default_token_generator.make_token(test_user)}
    reset_password_confirm_url = reverse('api-v1:custom_auth:ffadminuser-reset-password-confirm')
    response = client.post(reset_password_confirm_url, data=json.dumps(reset_password_data), content_type='application/json')
    assert response.status_code == status.HTTP_204_NO_CONTENT
    client.post(url, data=json.dumps(data), content_type='application/json')
    assert len(mail.outbox) == 1