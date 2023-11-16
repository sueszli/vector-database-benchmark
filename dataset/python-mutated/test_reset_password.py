import json
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.core import mail
from django.test.utils import override_settings
from django.urls import reverse
from allauth.account import app_settings
from allauth.account.forms import ResetPasswordForm
from allauth.account.models import EmailAddress
from allauth.tests import TestCase

@override_settings(ACCOUNT_PREVENT_ENUMERATION=False, ACCOUNT_DEFAULT_HTTP_PROTOCOL='https', ACCOUNT_EMAIL_VERIFICATION=app_settings.EmailVerificationMethod.MANDATORY, ACCOUNT_AUTHENTICATION_METHOD=app_settings.AuthenticationMethod.USERNAME, ACCOUNT_SIGNUP_FORM_CLASS=None, ACCOUNT_EMAIL_SUBJECT_PREFIX=None, LOGIN_REDIRECT_URL='/accounts/profile/', ACCOUNT_SIGNUP_REDIRECT_URL='/accounts/welcome/', ACCOUNT_ADAPTER='allauth.account.adapter.DefaultAccountAdapter', ACCOUNT_USERNAME_REQUIRED=True)
class ResetPasswordTests(TestCase):

    def test_user_email_not_sent_inactive_user(self):
        if False:
            for i in range(10):
                print('nop')
        User = get_user_model()
        User.objects.create_user('mike123', 'mike@ixample.org', 'test123', is_active=False)
        data = {'email': 'mike@ixample.org'}
        form = ResetPasswordForm(data)
        self.assertFalse(form.is_valid())

    def test_password_reset_get(self):
        if False:
            for i in range(10):
                print('nop')
        resp = self.client.get(reverse('account_reset_password'))
        self.assertTemplateUsed(resp, 'account/password_reset.html')

    def test_password_set_redirect(self):
        if False:
            return 10
        resp = self._password_set_or_change_redirect('account_set_password', True)
        self.assertRedirects(resp, reverse('account_change_password'), fetch_redirect_response=False)

    def test_set_password_not_allowed(self):
        if False:
            i = 10
            return i + 15
        user = self._create_user_and_login(True)
        pwd = '!*123i1uwn12W23'
        self.assertFalse(user.check_password(pwd))
        resp = self.client.post(reverse('account_set_password'), data={'password1': pwd, 'password2': pwd})
        user.refresh_from_db()
        self.assertFalse(user.check_password(pwd))
        self.assertTrue(user.has_usable_password())
        self.assertEqual(resp.status_code, 302)

    def test_password_change_no_redirect(self):
        if False:
            print('Hello World!')
        resp = self._password_set_or_change_redirect('account_change_password', True)
        self.assertEqual(resp.status_code, 200)

    def test_password_set_no_redirect(self):
        if False:
            print('Hello World!')
        resp = self._password_set_or_change_redirect('account_set_password', False)
        self.assertEqual(resp.status_code, 200)

    def test_password_change_redirect(self):
        if False:
            i = 10
            return i + 15
        resp = self._password_set_or_change_redirect('account_change_password', False)
        self.assertRedirects(resp, reverse('account_set_password'), fetch_redirect_response=False)

    def test_password_forgotten_username_hint(self):
        if False:
            while True:
                i = 10
        user = self._request_new_password()
        body = mail.outbox[0].body
        assert user.username in body

    @override_settings(ACCOUNT_AUTHENTICATION_METHOD=app_settings.AuthenticationMethod.EMAIL)
    def test_password_forgotten_no_username_hint(self):
        if False:
            for i in range(10):
                print('nop')
        user = self._request_new_password()
        body = mail.outbox[0].body
        assert user.username not in body

    def _request_new_password(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.create(username='john', email='john@example.org', is_active=True)
        user.set_password('doe')
        user.save()
        self.client.post(reverse('account_reset_password'), data={'email': 'john@example.org'})
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['john@example.org'])
        return user

    def test_password_reset_flow_with_empty_session(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the password reset flow when the session is empty:\n        requesting a new password, receiving the reset link via email,\n        following the link, getting redirected to the\n        new link (without the token)\n        Copying the link and using it in a DIFFERENT client (Browser/Device).\n        '
        self._request_new_password()
        body = mail.outbox[0].body
        self.assertGreater(body.find('https://'), 0)
        url = body[body.find('/password/reset/'):].split()[0]
        resp = self.client.get(url)
        reset_pass_url = resp.url
        resp = self.client_class().get(reset_pass_url)
        self.assertTemplateUsed(resp, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertTrue(resp.context_data['token_fail'])

    def test_password_reset_flow(self):
        if False:
            return 10
        '\n        Tests the password reset flow: requesting a new password,\n        receiving the reset link via email and finally resetting the\n        password to a new value.\n        '
        user = self._request_new_password()
        body = mail.outbox[0].body
        self.assertGreater(body.find('https://'), 0)
        url = body[body.find('/password/reset/'):].split()[0]
        resp = self.client.get(url)
        url = resp.url
        resp = self.client.get(url)
        self.assertTemplateUsed(resp, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertFalse('token_fail' in resp.context_data)
        resp = self.client.post(url, {'password1': 'newpass123', 'password2': 'newpass123'})
        self.assertRedirects(resp, reverse('account_reset_password_from_key_done'))
        user = get_user_model().objects.get(pk=user.pk)
        self.assertTrue(user.check_password('newpass123'))
        resp = self.client.post(url, {'password1': 'newpass123', 'password2': 'newpass123'})
        self.assertTemplateUsed(resp, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertTrue(resp.context_data['token_fail'])
        response = self.client.get(url)
        self.assertTemplateUsed(response, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertTrue(response.context_data['token_fail'])
        response = self.client.post(url, {'password1': 'newpass123', 'password2': 'newpass123'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content.decode('utf8'))
        assert 'invalid' in data['form']['errors'][0]

    @override_settings(ACCOUNT_AUTHENTICATION_METHOD=app_settings.AuthenticationMethod.EMAIL)
    def test_password_reset_flow_with_another_user_logged_in(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests the password reset flow: if User B requested a password\n        reset earlier and now User A is logged in, User B now clicks on\n        the link, ensure User A is logged out before continuing.\n        '
        self._request_new_password()
        body = mail.outbox[0].body
        self.assertGreater(body.find('https://'), 0)
        user2 = self._create_user(username='john2', email='john2@example.com')
        EmailAddress.objects.create(user=user2, email=user2.email, primary=True, verified=True)
        resp = self.client.post(reverse('account_login'), {'login': user2.email, 'password': 'doe'})
        self.assertEqual(user2, resp.context['user'])
        url = body[body.find('/password/reset/'):].split()[0]
        resp = self.client.get(url)
        url = resp.url
        resp = self.client.get(url)
        self.assertTemplateUsed(resp, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertFalse('token_fail' in resp.context_data)
        resp = self.client.post(url, {'password1': 'newpass123', 'password2': 'newpass123'}, follow=True)
        self.assertRedirects(resp, reverse('account_reset_password_from_key_done'))
        self.assertNotEqual(user2, resp.context['user'])
        self.assertEqual(AnonymousUser(), resp.context['user'])

    def test_password_reset_flow_with_email_changed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the password reset token is invalidated if\n        the user email address was changed.\n        '
        user = self._request_new_password()
        body = mail.outbox[0].body
        self.assertGreater(body.find('https://'), 0)
        EmailAddress.objects.create(user=user, email='other@email.org')
        url = body[body.find('/password/reset/'):].split()[0]
        resp = self.client.get(url)
        self.assertTemplateUsed(resp, 'account/password_reset_from_key.%s' % app_settings.TEMPLATE_EXTENSION)
        self.assertTrue('token_fail' in resp.context_data)

    @override_settings(ACCOUNT_LOGIN_ON_PASSWORD_RESET=True)
    def test_password_reset_ACCOUNT_LOGIN_ON_PASSWORD_RESET(self):
        if False:
            while True:
                i = 10
        user = self._request_new_password()
        body = mail.outbox[0].body
        url = body[body.find('/password/reset/'):].split()[0]
        resp = self.client.get(url)
        resp = self.client.post(resp.url, {'password1': 'newpass123', 'password2': 'newpass123'})
        self.assertTrue(user.is_authenticated)
        self.assertRedirects(resp, '/confirm-email/')

    def _create_user(self, username='john', password='doe', **kwargs):
        if False:
            print('Hello World!')
        user = get_user_model().objects.create(username=username, is_active=True, **kwargs)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save()
        return user

    def _create_user_and_login(self, usable_password=True):
        if False:
            i = 10
            return i + 15
        password = 'doe' if usable_password else False
        user = self._create_user(password=password)
        self.client.force_login(user)
        return user

    def _password_set_or_change_redirect(self, urlname, usable_password):
        if False:
            print('Hello World!')
        self._create_user_and_login(usable_password)
        return self.client.get(reverse(urlname))