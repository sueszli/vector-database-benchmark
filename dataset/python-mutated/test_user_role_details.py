from sentry.models.userrole import UserRole
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

class UserUserRolesTest(APITestCase):
    endpoint = 'sentry-api-0-user-userrole-details'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.user = self.create_user(is_superuser=True)
        self.login_as(user=self.user, superuser=True)
        self.add_user_permission(self.user, 'users.admin')

    def test_fails_without_superuser(self):
        if False:
            while True:
                i = 10
        self.user = self.create_user(is_superuser=False)
        self.login_as(self.user)
        UserRole.objects.create(name='test-role')
        resp = self.get_response('me', 'test-role')
        assert resp.status_code == 403
        self.user.update(is_superuser=True)
        resp = self.get_response('me', 'test-role')
        assert resp.status_code == 403

    def test_fails_without_users_admin_permission(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.create_user(is_superuser=True)
        self.login_as(self.user, superuser=True)
        resp = self.get_response('me', 'test-role')
        assert resp.status_code == 403

@control_silo_test(stable=True)
class UserUserRolesDetailsTest(UserUserRolesTest):

    def test_lookup_self(self):
        if False:
            while True:
                i = 10
        role = UserRole.objects.create(name='support', permissions=['broadcasts.admin'])
        role.users.add(self.user)
        role2 = UserRole.objects.create(name='admin', permissions=['users.admin'])
        role2.users.add(self.user)
        resp = self.get_response('me', 'support')
        assert resp.status_code == 200
        assert resp.data['name'] == 'support'

@control_silo_test(stable=True)
class UserUserRolesCreateTest(UserUserRolesTest):
    method = 'POST'

    def test_adds_role(self):
        if False:
            i = 10
            return i + 15
        UserRole.objects.create(name='support', permissions=['broadcasts.admin'])
        UserRole.objects.create(name='admin', permissions=['users.admin'])
        resp = self.get_response('me', 'support')
        assert resp.status_code == 201
        assert UserRole.objects.filter(users=self.user, name='support').exists()
        assert not UserRole.objects.filter(users=self.user, name='admin').exists()

    def test_invalid_role(self):
        if False:
            return 10
        UserRole.objects.create(name='other', permissions=['users.edit'])
        resp = self.get_response('me', 'blah')
        assert resp.status_code == 404

    def test_existing_role(self):
        if False:
            return 10
        role = UserRole.objects.create(name='support', permissions=['broadcasts.admin'])
        role.users.add(self.user)
        resp = self.get_response('me', 'support')
        assert resp.status_code == 410

@control_silo_test(stable=True)
class UserUserRolesDeleteTest(UserUserRolesTest):
    method = 'DELETE'

    def test_removes_role(self):
        if False:
            i = 10
            return i + 15
        role = UserRole.objects.create(name='support', permissions=['broadcasts.admin'])
        role.users.add(self.user)
        role2 = UserRole.objects.create(name='admin', permissions=['users.admin'])
        role2.users.add(self.user)
        resp = self.get_response('me', 'support')
        assert resp.status_code == 204
        assert not UserRole.objects.filter(users=self.user, name='support').exists()
        assert UserRole.objects.filter(users=self.user, name='admin').exists()

    def test_invalid_role(self):
        if False:
            while True:
                i = 10
        UserRole.objects.create(name='other', permissions=['users.edit'])
        resp = self.get_response('me', 'blah')
        assert resp.status_code == 404

    def test_nonexistant_role(self):
        if False:
            for i in range(10):
                print('nop')
        UserRole.objects.create(name='support', permissions=['broadcasts.admin'])
        resp = self.get_response('me', 'support')
        assert resp.status_code == 404