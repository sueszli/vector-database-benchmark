from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class OrganizationUserTeamsTest(APITestCase):
    endpoint = 'sentry-api-0-organization-user-teams'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.foo = self.create_user('foo@example.com')
        self.bar = self.create_user('bar@example.com', is_superuser=True)
        self.org = self.create_organization(owner=self.user)
        self.team1 = self.create_team(organization=self.org)
        self.team2 = self.create_team(organization=self.org)
        self.team3 = self.create_team(organization=self.org)
        self.project1 = self.create_project(teams=[self.team1])
        self.project2 = self.create_project(teams=[self.team2])
        self.create_member(organization=self.org, user=self.foo, teams=[self.team1, self.team2])
        self.create_member(organization=self.org, user=self.bar, teams=[self.team2])

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        self.login_as(user=self.foo)
        response = self.get_success_response(self.org.slug)
        assert len(response.data) == 2
        response.data.sort(key=lambda x: x['id'])
        assert response.data[0]['id'] == str(self.team1.id)
        assert response.data[0]['isMember']
        assert response.data[0]['projects'][0]['id'] == str(self.project1.id)
        assert response.data[1]['id'] == str(self.team2.id)
        assert response.data[1]['isMember']
        assert response.data[1]['projects'][0]['id'] == str(self.project2.id)

    def test_super_user(self):
        if False:
            return 10
        self.login_as(user=self.bar, superuser=True)
        response = self.get_success_response(self.org.slug)
        assert len(response.data) == 3
        response.data.sort(key=lambda x: x['id'])
        assert response.data[0]['id'] == str(self.team1.id)
        assert not response.data[0]['isMember']
        assert response.data[0]['projects'][0]['id'] == str(self.project1.id)
        assert response.data[1]['id'] == str(self.team2.id)
        assert response.data[1]['isMember']
        assert response.data[1]['projects'][0]['id'] == str(self.project2.id)
        assert response.data[2]['id'] == str(self.team3.id)
        assert not response.data[2]['isMember']