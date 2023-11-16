from datetime import datetime
from allauth.account.admin import EmailAddress
from django.contrib.auth.models import User
from django.core import mail
from django.utils.timezone import make_aware
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase
from apps.accounts.models import Membership, Secret, Site
from apps.notebooks.models import Notebook
from apps.workers.models import Worker

class AccountsTestCase(APITestCase):
    register_url = '/api/v1/auth/register/'
    verify_email_url = '/api/v1/auth/register/verify-email/'
    login_url = '/api/v1/auth/login/'
    user_details_url = '/api/v1/auth/user/'
    logout_url = '/api/v1/auth/logout/'
    delete_account_url = '/api/v1/auth/delete-account/'
    change_password_url = '/api/v1/auth/password/change/'
    reset_password_url = '/api/v1/auth/password/reset/'
    reset_password_confirm_url = '/api/v1/auth/password/reset/confirm/'
    sites_url = '/api/v1/sites/'
    members_url = '/api/v1/{}/members/'
    invite_url = '/api/v1/{}/invite/'

    def setUp(self):
        if False:
            return 10
        self.user1_params = {'username': 'user1', 'email': 'piotr@example.com', 'password': 'verysecret'}
        self.user = User.objects.create_user(username=self.user1_params['username'], email=self.user1_params['email'], password=self.user1_params['password'])
        EmailAddress.objects.create(user=self.user, email=self.user.email, verified=True, primary=True)
        self.user2_params = {'username': 'user2', 'email': 'piotr2@example.com', 'password': 'verysecret2'}
        self.user2 = User.objects.create_user(username=self.user2_params['username'], email=self.user2_params['email'], password=self.user2_params['password'])
        EmailAddress.objects.create(user=self.user2, email=self.user2.email, verified=True, primary=True)

    def test_list_members(self):
        if False:
            while True:
                i = 10
        site = Site.objects.create(title='First site', slug='first-site', created_by=self.user)
        response = self.client.post(self.login_url, self.user1_params)
        token = response.json()['key']
        headers = {'HTTP_AUTHORIZATION': 'Token ' + token}
        response = self.client.get(self.members_url.format(site.id), **headers)
        self.assertEqual(len(response.json()), 0)
        member = Membership.objects.create(user=self.user2, host=site, rights=Membership.VIEW, created_by=self.user)
        response = self.client.get(self.members_url.format(site.id), **headers)
        self.assertEqual(len(response.json()), 1)
        response = self.client.post(self.login_url, self.user2_params)
        token = response.json()['key']
        headers = {'HTTP_AUTHORIZATION': 'Token ' + token}
        response = self.client.get(self.members_url.format(site.id), **headers)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()['detail'], 'You do not have permission to perform this action.')
        member.rights = Membership.EDIT
        member.save()
        response = self.client.get(self.members_url.format(site.id), **headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)

    def test_create_member(self):
        if False:
            return 10
        site = Site.objects.create(title='First site', slug='first-site', created_by=self.user)
        response = self.client.post(self.login_url, self.user1_params)
        token = response.json()['key']
        headers = {'HTTP_AUTHORIZATION': 'Token ' + token}
        new_data = {'user_id': 2, 'rights': Membership.VIEW}
        response = self.client.post(self.members_url.format(site.id), new_data, **headers)
        self.assertEqual(response.status_code, 201)

    def test_register(self):
        if False:
            return 10
        data = {'email': 'user2@example-email.com', 'password1': 'verysecret', 'password2': 'verysecret'}
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.json()['detail'], 'Verification e-mail sent.')
        login_data = {'email': data['email'], 'password': data['password1']}
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertTrue('E-mail is not verified.' in response.json()['non_field_errors'])
        self.assertEqual(len(mail.outbox), 1)
        email_lines = mail.outbox[0].body.splitlines()
        activation_line = [l for l in email_lines if 'verify-email' in l][0]
        activation_link = activation_line.split('go to ')[1]
        activation_key = activation_link.split('/')[4]
        response = self.client.post(self.verify_email_url, {'key': activation_key})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['detail'], 'ok')
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue('key' in response.json())

    def test_get_user_details(self):
        if False:
            return 10
        response = self.client.post(self.login_url, self.user1_params)
        token = response.json()['key']
        headers = {'HTTP_AUTHORIZATION': 'Token ' + token}
        response = self.client.get(self.user_details_url, **headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for k in ['username', 'email', 'profile']:
            self.assertTrue(k in data)
        self.assertTrue('info' in data['profile'])