from sentry.api.fields.sentry_slug import DEFAULT_SLUG_ERROR_MESSAGE
from sentry.models.project import Project
from sentry.models.rule import Rule
from sentry.notifications.types import FallthroughChoiceType
from sentry.testutils.cases import APITestCase
from sentry.testutils.helpers import with_feature
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class TeamProjectsListTest(APITestCase):
    endpoint = 'sentry-api-0-team-project-index'
    method = 'get'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.team = self.create_team(members=[self.user])
        self.proj1 = self.create_project(teams=[self.team])
        self.proj2 = self.create_project(teams=[self.team])
        self.login_as(user=self.user)

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get_success_response(self.organization.slug, self.team.slug, status_code=200)
        project_ids = {item['id'] for item in response.data}
        assert len(response.data) == 2
        assert project_ids == {str(self.proj1.id), str(self.proj2.id)}

    def test_excludes_project(self):
        if False:
            i = 10
            return i + 15
        proj3 = self.create_project()
        response = self.get_success_response(self.organization.slug, self.team.slug, status_code=200)
        assert str(proj3.id) not in response.data

@region_silo_test(stable=True)
class TeamProjectsCreateTest(APITestCase):
    endpoint = 'sentry-api-0-team-project-index'
    method = 'post'

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.team = self.create_team(members=[self.user])
        self.data = {'name': 'foo', 'slug': 'bar', 'platform': 'python'}
        self.login_as(user=self.user)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get_success_response(self.organization.slug, self.team.slug, **self.data, status_code=201)
        project = Project.objects.get(id=response.data['id'])
        assert project.name == 'foo'
        assert project.slug == 'bar'
        assert project.platform == 'python'
        assert project.teams.first() == self.team

    def test_invalid_numeric_slug(self):
        if False:
            print('Hello World!')
        response = self.get_error_response(self.organization.slug, self.team.slug, name='fake name', slug='12345', status_code=400)
        assert response.data['slug'][0] == DEFAULT_SLUG_ERROR_MESSAGE

    def test_generated_slug_not_entirely_numeric(self):
        if False:
            i = 10
            return i + 15
        response = self.get_success_response(self.organization.slug, self.team.slug, name='1234', status_code=201)
        slug = response.data['slug']
        assert slug.startswith('1234-')
        assert not slug.isdecimal()

    def test_invalid_platform(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_error_response(self.organization.slug, self.team.slug, name='fake name', platform='fake platform', status_code=400)
        assert response.data['platform'][0] == 'Invalid platform'

    def test_duplicate_slug(self):
        if False:
            while True:
                i = 10
        self.create_project(slug='bar')
        response = self.get_error_response(self.organization.slug, self.team.slug, **self.data, status_code=409)
        assert response.data['detail'] == 'A project with this slug already exists.'

    def test_default_rules(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_success_response(self.organization.slug, self.team.slug, **self.data, default_rules=True, status_code=201)
        project = Project.objects.get(id=response.data['id'])
        assert Rule.objects.filter(project=project).exists()

    @with_feature('organizations:issue-alert-fallback-targeting')
    def test_default_rule_fallback_targeting(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_success_response(self.organization.slug, self.team.slug, **self.data, default_rules=True, status_code=201)
        project = Project.objects.get(id=response.data['id'])
        rule = Rule.objects.filter(project=project).first()
        assert rule.data['actions'][0]['fallthroughType'] == FallthroughChoiceType.ACTIVE_MEMBERS.value

    def test_without_default_rules(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_success_response(self.organization.slug, self.team.slug, **self.data, default_rules=False, status_code=201)
        project = Project.objects.get(id=response.data['id'])
        assert not Rule.objects.filter(project=project).exists()