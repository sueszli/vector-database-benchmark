from django.urls import reverse
from sentry.auth.providers.fly.provider import FlyOAuth2Provider
from sentry.models.authprovider import AuthProvider
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class AuthOrganizationChannelLoginTest(TestCase):

    def create_auth_provider(self, partner_org_id, sentry_org_id):
        if False:
            print('Hello World!')
        config_data = FlyOAuth2Provider.build_config(resource={'id': partner_org_id})
        AuthProvider.objects.create(organization_id=sentry_org_id, provider='fly', config=config_data)

    def setup(self):
        if False:
            return 10
        self.organization = self.create_organization(name='test org', owner=self.user)
        self.create_auth_provider('fly-test-org', self.organization.id)
        self.path = reverse('sentry-auth-channel', args=['fly', 'fly-test-org'])

    def test_redirect_for_logged_in_user(self):
        if False:
            return 10
        self.setup()
        self.login_as(self.user)
        response = self.client.get(self.path, follow=True)
        assert response.status_code == 200
        assert response.redirect_chain == [(f'/organizations/{self.organization.slug}/issues/', 302)]

    def test_redirect_for_logged_in_user_with_different_active_org(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup()
        self.login_as(self.user)
        another_org = self.create_organization(name='another org', owner=self.user)
        self.create_auth_provider('another-fly-org', another_org.id)
        path = reverse('sentry-auth-channel', args=['fly', 'another-fly-org'])
        response = self.client.get(path + '?next=/projects/', follow=True)
        assert response.status_code == 200
        assert response.redirect_chain == [(f'/auth/login/{another_org.slug}/?next=/projects/', 302)]

    def test_redirect_for_logged_out_user(self):
        if False:
            print('Hello World!')
        self.setup()
        response = self.client.get(self.path, follow=True)
        assert response.status_code == 200
        assert response.redirect_chain == [(f'/auth/login/{self.organization.slug}/', 302)]

    def test_with_next_uri(self):
        if False:
            print('Hello World!')
        self.setup()
        self.login_as(self.user)
        response = self.client.get(self.path + '?next=/projects/', follow=True)
        assert response.status_code == 200
        assert response.redirect_chain == [('/projects/', 302)]

    def test_subdomain_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup()
        another_org = self.create_organization(name='another org')
        path = reverse('sentry-auth-channel', args=['fly', another_org.id])
        response = self.client.get(path, HTTP_HOST=f'{self.organization.slug}.testserver', follow=True)
        assert response.status_code == 200
        assert response.redirect_chain == [(f'/auth/login/{self.organization.slug}/', 302)]