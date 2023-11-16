from urllib.parse import parse_qsl, urlparse
from django.test import override_settings
from django.urls import reverse
from sentry.models.organizationmember import OrganizationMember
from sentry.testutils.cases import TestCase

class VstsExtensionConfigurationTest(TestCase):

    @property
    def path(self):
        if False:
            for i in range(10):
                print('nop')
        return reverse('vsts-extension-configuration')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.create_user()
        self.org = self.create_organization()
        OrganizationMember.objects.create(user_id=self.user.id, organization=self.org, role='admin')

    def test_logged_in_one_org(self):
        if False:
            i = 10
            return i + 15
        self.login_as(self.user)
        resp = self.client.get(self.path, {'targetId': '1', 'targetName': 'foo'})
        assert resp.status_code == 302
        assert resp.headers['Location'].startswith('https://app.vssps.visualstudio.com/oauth2/authorize')

    def test_logged_in_many_orgs(self):
        if False:
            print('Hello World!')
        self.login_as(self.user)
        org = self.create_organization()
        OrganizationMember.objects.create(user_id=self.user.id, organization=org)
        resp = self.client.get(self.path, {'targetId': '1', 'targetName': 'foo'})
        assert resp.status_code == 302
        assert '/extensions/vsts/link/' in resp.headers['Location']

    def test_choose_org(self):
        if False:
            while True:
                i = 10
        self.login_as(self.user)
        resp = self.client.get(self.path, {'targetId': '1', 'targetName': 'foo', 'orgSlug': self.org.slug})
        assert resp.status_code == 302
        assert resp.headers['Location'].startswith('https://app.vssps.visualstudio.com/oauth2/authorize')

    def test_logged_out(self):
        if False:
            print('Hello World!')
        query = {'targetId': '1', 'targetName': 'foo'}
        resp = self.client.get(self.path, query)
        assert resp.status_code == 302
        assert '/auth/login/' in resp.headers['Location']
        next_parts = urlparse(dict(parse_qsl(urlparse(resp.headers['Location']).query))['next'])
        assert next_parts.path == '/extensions/vsts/configure/'
        assert dict(parse_qsl(next_parts.query)) == query

    @override_settings(SENTRY_FEATURES={})
    def test_goes_to_setup_unregisted_feature(self):
        if False:
            return 10
        self.login_as(self.user)
        resp = self.client.get(self.path, {'targetId': '1', 'targetName': 'foo'})
        assert resp.status_code == 302
        assert resp.headers['Location'].startswith('https://app.vssps.visualstudio.com/oauth2/authorize')