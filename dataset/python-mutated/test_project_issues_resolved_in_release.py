from uuid import uuid1
from sentry.models.commit import Commit
from sentry.models.grouplink import GroupLink
from sentry.models.groupresolution import GroupResolution
from sentry.models.releasecommit import ReleaseCommit
from sentry.models.repository import Repository
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

@region_silo_test(stable=True)
class ProjectIssuesResolvedInReleaseEndpointTest(APITestCase):
    endpoint = 'sentry-api-0-project-release-resolved'
    method = 'get'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.user = self.create_user()
        self.org = self.create_organization()
        self.team = self.create_team(organization=self.org)
        self.create_member(organization=self.org, user=self.user, teams=[self.team])
        self.project = self.create_project(teams=[self.team])
        self.release = self.create_release(project=self.project)
        self.group = self.create_group(project=self.project)
        self.login_as(self.user)

    def build_grouplink(self):
        if False:
            print('Hello World!')
        repo = Repository.objects.create(organization_id=self.org.id, name=self.project.name)
        commit = Commit.objects.create(organization_id=self.org.id, repository_id=repo.id, key=uuid1().hex)
        commit2 = Commit.objects.create(organization_id=self.org.id, repository_id=repo.id, key=uuid1().hex)
        ReleaseCommit.objects.create(organization_id=self.org.id, release=self.release, commit=commit, order=1)
        ReleaseCommit.objects.create(organization_id=self.org.id, release=self.release, commit=commit2, order=0)
        GroupLink.objects.create(group_id=self.group.id, project_id=self.group.project_id, linked_type=GroupLink.LinkedType.commit, relationship=GroupLink.Relationship.resolves, linked_id=commit.id)

    def build_group_resolution(self, group=None):
        if False:
            while True:
                i = 10
        return GroupResolution.objects.create(group=self.group if group is None else group, release=self.release, type=GroupResolution.Type.in_release)

    def run_test(self, expected_groups):
        if False:
            print('Hello World!')
        response = self.get_success_response(self.org.slug, self.project.slug, self.release.version)
        assert len(response.data) == len(expected_groups)
        expected = set(map(str, [g.id for g in expected_groups]))
        assert {item['id'] for item in response.data} == expected

    def test_shows_issues_from_groupresolution(self):
        if False:
            return 10
        '\n        tests that the endpoint will correctly retrieve issues resolved\n        in a release from the GroupResolution model\n        '
        self.build_group_resolution()
        self.run_test([self.group])

    def test_shows_issues_from_grouplink(self):
        if False:
            i = 10
            return i + 15
        '\n        tests that the endpoint will correctly retrieve issues resolved\n        in a release from the GroupLink model\n        '
        self.build_grouplink()
        self.run_test([self.group])

    def test_does_not_return_duplicate_groups(self):
        if False:
            return 10
        '\n        tests that the endpoint will correctly retrieve issues resolved\n        in a release from the GroupLink and GroupResolution model\n        but will not return the groups twice if they appear in both\n        '
        self.build_grouplink()
        self.build_group_resolution()
        self.run_test([self.group])

    def test_return_groups_from_both_types(self):
        if False:
            print('Hello World!')
        '\n        tests that the endpoint will correctly retrieve issues resolved\n        in a release from both the GroupLink and GroupResolution model\n        '
        self.build_grouplink()
        group_2 = self.create_group(project=self.project)
        self.build_group_resolution(group_2)
        self.run_test([self.group, group_2])