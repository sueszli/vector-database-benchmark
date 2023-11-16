from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

class UserPermissionsConfigTest(APITestCase):
    endpoint = 'sentry-api-0-user-permissions-config'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.user = self.create_user(is_superuser=True)
        self.login_as(user=self.user, superuser=True)
        self.add_user_permission(self.user, 'users.admin')

@control_silo_test(stable=True)
class UserPermissionsConfigGetTest(UserPermissionsConfigTest):

    def test_lookup_self(self):
        if False:
            i = 10
            return i + 15
        resp = self.get_response('me')
        assert resp.status_code == 200
        assert len(resp.data) == 3, resp.data
        assert 'broadcasts.admin' in resp.data
        assert 'users.admin' in resp.data
        assert 'options.admin' in resp.data