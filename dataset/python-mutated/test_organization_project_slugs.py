from fixtures.apidocs_test_case import APIDocsTestCase
from sentry.models.project import Project
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class OrganizationIndexDocs(APIDocsTestCase):
    endpoint = 'sentry-api-0-short-ids-update'
    method = 'put'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.project_one = self.create_project(organization=self.organization, slug='old-one')
        self.project_two = self.create_project(organization=self.organization, slug='old-two')
        self.slugs = {self.project_one.id: 'new-one', self.project_two.id: 'new-two'}
        self.login_as(user=self.user)

    def test_updates_project_slugs(self):
        if False:
            print('Hello World!')
        response = self.get_success_response(self.organization.slug, slugs=self.slugs, status_code=200)
        (project_one, project_two) = Project.objects.all().order_by('slug')
        assert project_one.slug == 'new-one'
        assert project_two.slug == 'new-two'
        assert response.data['updated_slugs'] == {str(project_one.id): 'new-one', str(project_two.id): 'new-two'}

    def test_invalid_numeric_slug(self):
        if False:
            for i in range(10):
                print('nop')
        invalid_slugs = {**self.slugs, self.project_two.id: '1234'}
        response = self.get_error_response(self.organization.slug, slugs=invalid_slugs, status_code=400)
        assert response.data['detail'] == 'Invalid slug "1234".'

    def test_duplicate_slug(self):
        if False:
            return 10
        self.create_project(organization=self.organization, slug='new-one')
        response = self.get_error_response(self.organization.slug, slugs=self.slugs, status_code=409)
        assert response.data['detail'] == 'A project with slug "new-one" already exists.'