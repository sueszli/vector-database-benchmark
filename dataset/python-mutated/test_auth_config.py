import pytest
from django.conf import settings
from django.test.utils import override_settings
from sentry import newsletter
from sentry.receivers import create_default_projects
from sentry.silo import SiloMode
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import assume_test_silo_mode, control_silo_test

@control_silo_test(stable=True)
class AuthConfigEndpointTest(APITestCase):
    path = '/api/0/auth/config/'

    def test_logged_in(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user('foo@example.com')
        self.login_as(user)
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['nextUri'] == '/organizations/new/'

    def test_logged_in_active_org(self):
        if False:
            return 10
        user = self.create_user('foo@example.com')
        self.create_organization(owner=user, slug='ricks-org')
        self.login_as(user)
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['nextUri'] == '/organizations/ricks-org/issues/'

    @override_settings(SENTRY_SINGLE_ORGANIZATION=True)
    @assume_test_silo_mode(SiloMode.MONOLITH)
    def test_single_org(self):
        if False:
            i = 10
            return i + 15
        create_default_projects()
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['nextUri'] == '/auth/login/sentry/'

    def test_superuser_is_not_redirected(self):
        if False:
            return 10
        user = self.create_user('foo@example.com', is_superuser=True)
        self.login_as(user)
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['nextUri'] == '/organizations/new/'

    def test_unauthenticated(self):
        if False:
            return 10
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert not response.data['canRegister']
        assert not response.data['hasNewsletter']
        assert response.data['serverHostname'] == 'testserver'

    @pytest.mark.skipif(settings.SENTRY_NEWSLETTER != 'sentry.newsletter.dummy.DummyNewsletter', reason='Requires DummyNewsletter.')
    def test_has_newsletter(self):
        if False:
            print('Hello World!')
        newsletter.backend.enable()
        response = self.client.get(self.path)
        newsletter.backend.disable()
        assert response.status_code == 200
        assert response.data['hasNewsletter']

    def test_can_register(self):
        if False:
            print('Hello World!')
        with self.options({'auth.allow-registration': True}):
            with self.feature('auth:register'):
                response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['canRegister']

    def test_session_expired(self):
        if False:
            return 10
        self.client.cookies['session_expired'] = '1'
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert response.data['warning'] == 'Your session has expired.'