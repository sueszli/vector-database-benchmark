from sentry.models.groupsubscription import GroupSubscription
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class GroupParticipantsTest(APITestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(self.user)

    def _get_path_functions(self):
        if False:
            for i in range(10):
                print('nop')
        return (lambda group: f'/api/0/issues/{group.id}/participants/', lambda group: f'/api/0/organizations/{self.organization.slug}/issues/{group.id}/participants/')

    def test_simple(self):
        if False:
            return 10
        group = self.create_group()
        GroupSubscription.objects.create(user_id=self.user.id, group=group, project=group.project, is_active=True)
        for path_func in self._get_path_functions():
            path = path_func(group)
            response = self.client.get(path)
            assert len(response.data) == 1, response
            assert response.data[0]['id'] == str(self.user.id)