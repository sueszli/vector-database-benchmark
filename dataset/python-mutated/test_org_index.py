from django.test.client import RequestFactory
from django.urls import reverse
from fixtures.apidocs_test_case import APIDocsTestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class OrganizationIndexDocs(APIDocsTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_organization(owner=self.user, name='Rowdy Tiger')
        self.url = reverse('sentry-api-0-organizations')
        self.login_as(user=self.user)

    def test_get(self):
        if False:
            while True:
                i = 10
        response = self.client.get(self.url)
        request = RequestFactory().get(self.url)
        self.validate_schema(request, response)