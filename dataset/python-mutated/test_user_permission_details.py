from sentry.models.userpermission import UserPermission
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

class UserDetailsTest(APITestCase):
    endpoint = 'sentry-api-0-user-permission-details'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.user = self.create_user(is_superuser=True)
        self.login_as(user=self.user, superuser=True)
        self.add_user_permission(self.user, 'users.admin')

    def test_fails_without_superuser(self):
        if False:
            print('Hello World!')
        self.user = self.create_user(is_superuser=False)
        self.login_as(self.user)
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 403
        self.user.update(is_superuser=True)
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 403

    def test_fails_without_users_admin_permission(self):
        if False:
            i = 10
            return i + 15
        self.user = self.create_user(is_superuser=True)
        self.login_as(self.user, superuser=True)
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 403

@control_silo_test(stable=True)
class UserPermissionDetailsGetTest(UserDetailsTest):

    def test_with_permission(self):
        if False:
            for i in range(10):
                print('nop')
        UserPermission.objects.create(user=self.user, permission='broadcasts.admin')
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 204

    def test_without_permission(self):
        if False:
            i = 10
            return i + 15
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 404

@control_silo_test(stable=True)
class UserPermissionDetailsPostTest(UserDetailsTest):
    method = 'POST'

    def test_with_permission(self):
        if False:
            while True:
                i = 10
        UserPermission.objects.create(user=self.user, permission='broadcasts.admin')
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 410
        assert UserPermission.objects.filter(user=self.user, permission='broadcasts.admin').exists()

    def test_without_permission(self):
        if False:
            print('Hello World!')
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 201
        assert UserPermission.objects.filter(user=self.user, permission='broadcasts.admin').exists()

@control_silo_test(stable=True)
class UserPermissionDetailsDeleteTest(UserDetailsTest):
    method = 'DELETE'

    def test_with_permission(self):
        if False:
            print('Hello World!')
        UserPermission.objects.create(user=self.user, permission='broadcasts.admin')
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 204
        assert not UserPermission.objects.filter(user=self.user, permission='broadcasts.admin').exists()

    def test_without_permission(self):
        if False:
            return 10
        resp = self.get_response('me', 'broadcasts.admin')
        assert resp.status_code == 404
        assert not UserPermission.objects.filter(user=self.user, permission='broadcasts.admin').exists()