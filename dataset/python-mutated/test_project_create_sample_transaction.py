from django.urls import reverse
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

@region_silo_test(stable=True)
class ProjectCreateSampleTransactionTest(APITestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.login_as(user=self.user)
        self.team = self.create_team()

    def test_no_platform(self):
        if False:
            while True:
                i = 10
        project = self.create_project(teams=[self.team], name='foo', platform=None)
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == '/productstore'
        project.refresh_from_db()
        assert not project.flags.has_transactions

    def test_react(self):
        if False:
            while True:
                i = 10
        project = self.create_project(teams=[self.team], name='foo', platform='javascript-react')
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == '/productstore'

    def test_django(self):
        if False:
            while True:
                i = 10
        project = self.create_project(teams=[self.team], name='foo', platform='python-django')
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == 'getProductList'

    def test_ios(self):
        if False:
            return 10
        project = self.create_project(teams=[self.team], name='foo', platform='apple-ios')
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == 'iOS_Swift.ViewController'

    def test_other_platform(self):
        if False:
            while True:
                i = 10
        project = self.create_project(teams=[self.team], name='foo', platform='other')
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == '/productstore'

    def test_path_traversal_attempt(self):
        if False:
            for i in range(10):
                print('nop')
        project = self.create_project(teams=[self.team], name='foo', platform='../../../etc/passwd')
        url = reverse('sentry-api-0-project-create-sample-transaction', kwargs={'organization_slug': project.organization.slug, 'project_slug': project.slug})
        response = self.client.post(url, format='json')
        assert response.status_code == 200
        assert response.data['title'] == '/productstore'