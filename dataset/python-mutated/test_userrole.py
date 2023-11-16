from django.conf import settings
from sentry.models.userrole import UserRole, manage_default_super_admin_role
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class UserRoleTest(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        manage_default_super_admin_role()

    def test_permissions_for_user(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user(email='a@example.com')
        user2 = self.create_user(email='b@example.com')
        role = UserRole.objects.create(name='test', permissions=['test1', 'test2'])
        role.users.add(user)
        role2 = UserRole.objects.create(name='test2', permissions=['test2', 'test3'])
        role2.users.add(user)
        assert sorted(UserRole.permissions_for_user(user.id)) == ['test1', 'test2', 'test3']
        assert sorted(UserRole.permissions_for_user(user2.id)) == []

    def test_creates_super_admin_role(self):
        if False:
            i = 10
            return i + 15
        role = UserRole.objects.get(name='Super Admin')
        assert sorted(role.permissions) == sorted(settings.SENTRY_USER_PERMISSIONS)