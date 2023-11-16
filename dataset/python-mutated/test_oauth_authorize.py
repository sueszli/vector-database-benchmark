from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class OAuthAuthorizeTest(AcceptanceTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.user = self.create_user('foo@example.com', is_superuser=True)
        self.login_as(self.user)

    def test_simple(self):
        if False:
            return 10
        self.browser.get('/debug/oauth/authorize/')
        self.browser.wait_until_not('.loading')
        self.browser.get('/debug/oauth/authorize/error/')
        self.browser.wait_until_not('.loading')