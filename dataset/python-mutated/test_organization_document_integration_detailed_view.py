from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class OrganizationDocumentIntegrationDetailView(AcceptanceTestCase):
    """
    As a developer, I can view an document-based integration, and learn more about it with the linked resources.
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.organization = self.create_organization(owner=self.user, name='Walter Mitty')
        self.doc = self.create_doc_integration(name='Quintessence of Life', features=[1, 2, 3], is_draft=False)
        self.login_as(self.user)

    def load_page(self, slug):
        if False:
            print('Hello World!')
        url = f'/settings/{self.organization.slug}/document-integrations/{slug}/'
        self.browser.get(url)
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')

    def test_view_doc(self):
        if False:
            for i in range(10):
                print('nop')
        self.load_page(self.doc.slug)
        assert self.browser.element_exists('[data-test-id="learn-more"]')