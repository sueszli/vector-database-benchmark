from datetime import datetime, timezone
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class OrganizationMemberUnreleasedCommitsTest(APITestCase):
    endpoint = 'sentry-api-0-organization-member-unreleased-commits'

    def test_simple(self):
        if False:
            while True:
                i = 10
        repo = self.create_repo(self.project)
        repo2 = self.create_repo(self.project)
        release = self.create_release(self.project)
        author = self.create_commit_author(project=self.project, user=self.user)
        self.create_commit(project=self.project, repo=repo, release=release, author=author, date_added=datetime(2015, 1, 1, tzinfo=timezone.utc))
        self.create_commit(project=self.project, repo=repo2, author=author, date_added=datetime(2015, 1, 2, tzinfo=timezone.utc))
        unreleased_commit = self.create_commit(project=self.project, repo=repo, author=author, date_added=datetime(2015, 1, 2, tzinfo=timezone.utc))
        unreleased_commit2 = self.create_commit(project=self.project, repo=repo, author=author, date_added=datetime(2015, 1, 3, tzinfo=timezone.utc))
        self.create_commit(project=self.project, repo=repo, date_added=datetime(2015, 1, 3, tzinfo=timezone.utc))
        self.login_as(self.user)
        response = self.get_success_response(self.organization.slug, 'me')
        assert len(response.data['commits']) == 2
        assert response.data['commits'][0]['id'] == unreleased_commit2.key
        assert response.data['commits'][1]['id'] == unreleased_commit.key
        assert len(response.data['repositories']) == 1
        assert str(repo.id) in response.data['repositories']