from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class OrganizationJoinRequestTest(AcceptanceTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.user = self.create_user('foo@example.com')
        self.org = self.create_organization(name='Rowdy Tiger', owner=self.user)

    def test_view(self):
        if False:
            while True:
                i = 10
        self.browser.get(f'/join-request/{self.org.slug}/')
        self.browser.wait_until('[data-test-id="join-request"]')
        assert self.browser.element_exists('[data-test-id="join-request"]')