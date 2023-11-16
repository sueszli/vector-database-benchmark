from django.test.client import RequestFactory
from django.urls import reverse
from fixtures.apidocs_test_case import APIDocsTestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test
class ProjectServiceHookDetailsDocs(APIDocsTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        hook = self.create_service_hook(project=self.project, events=('event.created',))
        self.url = reverse('sentry-api-0-project-service-hook-details', kwargs={'organization_slug': self.organization.slug, 'project_slug': self.project.slug, 'hook_id': hook.guid})
        self.login_as(user=self.user)

    def test_get(self):
        if False:
            print('Hello World!')
        response = self.client.get(self.url)
        request = RequestFactory().get(self.url)
        self.validate_schema(request, response)

    def test_put(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'url': 'https://example.com/other-sentry-hook', 'events': ['event.created']}
        response = self.client.put(self.url, data)
        request = RequestFactory().put(self.url, data)
        self.validate_schema(request, response)