from django.test.client import RequestFactory
from django.urls import reverse
from fixtures.apidocs_test_case import APIDocsTestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class OrganizationDetailsDocs(APIDocsTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        organization = self.create_organization(owner=self.user, name='Rowdy Tiger')
        self.url = reverse('sentry-api-0-organization-details', kwargs={'organization_slug': organization.slug})
        self.login_as(user=self.user)

    def test_get(self):
        if False:
            return 10
        response = self.client.get(self.url)
        request = RequestFactory().get(self.url)
        self.validate_schema(request, response)

    def test_put(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'name': 'foo'}
        response = self.client.put(self.url, data)
        request = RequestFactory().put(self.url, data)
        self.validate_schema(request, response)