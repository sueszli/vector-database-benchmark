from django.db.models import F
from sentry.models.authprovider import AuthProvider
from sentry.models.organization import Organization
from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class AcceptOrganizationInviteTest(AcceptanceTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.create_user('foo@example.com')
        self.org = self.create_organization(name='Rowdy Tiger', owner=None)
        self.team = self.create_team(organization=self.org, name='Mariachi Band')
        self.member = self.create_member(user=None, email='bar@example.com', organization=self.org, role='owner', teams=[self.team])

    def test_invite_simple(self):
        if False:
            return 10
        self.login_as(self.user)
        self.browser.get(self.member.get_invite_link().split('/', 3)[-1])
        self.browser.wait_until('[data-test-id="accept-invite"]')
        assert self.browser.element_exists('[data-test-id="join-organization"]')

    def test_invite_not_authenticated(self):
        if False:
            print('Hello World!')
        self.browser.get(self.member.get_invite_link().split('/', 3)[-1])
        self.browser.wait_until('[data-test-id="accept-invite"]')
        assert self.browser.element_exists('[data-test-id="create-account"]')

    def test_invite_2fa_enforced_org(self):
        if False:
            i = 10
            return i + 15
        self.org.update(flags=F('flags').bitor(Organization.flags.require_2fa))
        self.browser.get(self.member.get_invite_link().split('/', 3)[-1])
        self.browser.wait_until('[data-test-id="accept-invite"]')
        assert not self.browser.element_exists_by_test_id('2fa-warning')
        self.login_as(self.user)
        self.org.update(flags=F('flags').bitor(Organization.flags.require_2fa))
        self.browser.get(self.member.get_invite_link().split('/', 3)[-1])
        self.browser.wait_until('[data-test-id="accept-invite"]')
        assert self.browser.element_exists_by_test_id('2fa-warning')

    def test_invite_sso_org(self):
        if False:
            for i in range(10):
                print('nop')
        AuthProvider.objects.create(organization_id=self.org.id, provider='google')
        self.browser.get(self.member.get_invite_link().split('/', 3)[-1])
        self.browser.wait_until('[data-test-id="accept-invite"]')
        assert self.browser.element_exists_by_test_id('action-info-sso')
        assert self.browser.element_exists('[data-test-id="sso-login"]')