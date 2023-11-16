from django.contrib.auth.models import AnonymousUser, Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase
from wagtail.images.models import Image
from wagtail.images.tests.utils import get_test_image_file
from wagtail.permission_policies import AuthenticationOnlyPermissionPolicy, BlanketPermissionPolicy, ModelPermissionPolicy, OwnershipPermissionPolicy
from wagtail.test.utils import WagtailTestUtils

class PermissionPolicyTestUtils:

    def assertResultSetEqual(self, actual, expected):
        if False:
            while True:
                i = 10
        self.assertEqual(set(actual), set(expected))

    def assertUserPermissionMatrix(self, test_cases, actions=()):
        if False:
            i = 10
            return i + 15
        "\n        Given a list of (user, can_add, can_change, can_delete, can_frobnicate) tuples\n        (where 'frobnicate' is an unrecognised action not defined on the model),\n        confirm that all tuples correctly represent permissions for that user as\n        returned by user_has_permission\n        "
        if not actions:
            actions = ['add', 'change', 'delete', 'frobnicate']
        for test_case in test_cases:
            user = test_case[0]
            expected_results = zip(actions, test_case[1:])
            for (action, expected_result) in expected_results:
                if expected_result:
                    self.assertTrue(self.policy.user_has_permission(user, action), msg="User {} should be able to {}, but can't".format(user, action))
                else:
                    self.assertFalse(self.policy.user_has_permission(user, action), msg='User %s should not be able to %s, but can' % (user, action))

    def assertUserInstancePermissionMatrix(self, instance, test_cases, actions=()):
        if False:
            return 10
        "\n        Given a list of (user, can_change, can_delete, can_frobnicate) tuples\n        (where 'frobnicate' is an unrecognised action not defined on the model),\n        confirm that all tuples correctly represent permissions for that user on\n        the given instance, as returned by user_has_permission_for_instance\n        "
        if not actions:
            actions = ['change', 'delete', 'frobnicate']
        for test_case in test_cases:
            user = test_case[0]
            expected_results = zip(actions, test_case[1:])
            for (action, expected_result) in expected_results:
                if expected_result:
                    self.assertTrue(self.policy.user_has_permission_for_instance(user, action, instance), msg="User %s should be able to %s instance %s, but can't" % (user, action, instance))
                else:
                    self.assertFalse(self.policy.user_has_permission_for_instance(user, action, instance), msg='User %s should not be able to %s instance %s, but can' % (user, action, instance))

class PermissionPolicyTestCase(PermissionPolicyTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        image_content_type = ContentType.objects.get_for_model(Image)
        add_image_permission = Permission.objects.get(content_type=image_content_type, codename='add_image')
        change_image_permission = Permission.objects.get(content_type=image_content_type, codename='change_image')
        delete_image_permission = Permission.objects.get(content_type=image_content_type, codename='delete_image')
        image_adders_group = Group.objects.create(name='Image adders')
        image_adders_group.permissions.add(add_image_permission)
        image_changers_group = Group.objects.create(name='Image changers')
        image_changers_group.permissions.add(change_image_permission)
        self.superuser = self.create_superuser('superuser', 'superuser@example.com', 'password')
        self.inactive_superuser = self.create_superuser('inactivesuperuser', 'inactivesuperuser@example.com', 'password')
        self.inactive_superuser.is_active = False
        self.inactive_superuser.save()
        self.image_adder = self.create_user('imageadder', 'imageadder@example.com', 'password')
        self.image_adder.groups.add(image_adders_group)
        self.oneoff_image_adder = self.create_user('oneoffimageadder', 'oneoffimageadder@example.com', 'password')
        self.oneoff_image_adder.user_permissions.add(add_image_permission)
        self.inactive_image_adder = self.create_user('inactiveimageadder', 'inactiveimageadder@example.com', 'password')
        self.inactive_image_adder.groups.add(image_adders_group)
        self.inactive_image_adder.is_active = False
        self.inactive_image_adder.save()
        self.image_changer = self.create_user('imagechanger', 'imagechanger@example.com', 'password')
        self.image_changer.groups.add(image_changers_group)
        self.oneoff_image_changer = self.create_user('oneoffimagechanger', 'oneoffimagechanger@example.com', 'password')
        self.oneoff_image_changer.user_permissions.add(change_image_permission)
        self.inactive_image_changer = self.create_user('inactiveimagechanger', 'inactiveimagechanger@example.com', 'password')
        self.inactive_image_changer.groups.add(image_changers_group)
        self.inactive_image_changer.is_active = False
        self.inactive_image_changer.save()
        self.oneoff_image_deleter = self.create_user('oneoffimagedeleter', 'oneoffimagedeleter@example.com', 'password')
        self.oneoff_image_deleter.user_permissions.add(delete_image_permission)
        self.useless_user = self.create_user('uselessuser', 'uselessuser@example.com', 'password')
        self.anonymous_user = AnonymousUser()
        self.adder_image = Image.objects.create(title="imageadder's image", file=get_test_image_file(), uploaded_by_user=self.image_adder)
        self.useless_image = Image.objects.create(title="uselessuser's image", file=get_test_image_file(), uploaded_by_user=self.useless_user)
        self.anonymous_image = Image.objects.create(title='anonymous image', file=get_test_image_file())

class TestBlanketPermissionPolicy(PermissionPolicyTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.policy = BlanketPermissionPolicy(Image)
        self.active_users = [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter, self.useless_user]
        self.all_users = self.active_users + [self.inactive_superuser, self.inactive_image_adder, self.inactive_image_changer, self.anonymous_user]

    def test_user_has_permission(self):
        if False:
            print('Hello World!')
        self.assertUserPermissionMatrix([(user, True, True, True, True) for user in self.all_users])

    def test_user_has_any_permission(self):
        if False:
            for i in range(10):
                print('nop')
        for user in self.all_users:
            self.assertTrue(self.policy.user_has_any_permission(user, ['add', 'change']))

    def test_users_with_permission(self):
        if False:
            while True:
                i = 10
        users_with_add_permission = self.policy.users_with_permission('add')
        self.assertResultSetEqual(users_with_add_permission, self.active_users)

    def test_users_with_any_permission(self):
        if False:
            return 10
        users_with_add_or_change_permission = self.policy.users_with_any_permission(['add', 'change'])
        self.assertResultSetEqual(users_with_add_or_change_permission, self.active_users)

    def test_user_has_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertUserInstancePermissionMatrix(self.adder_image, [(user, True, True, True) for user in self.all_users])

    def test_user_has_any_permission_for_instance(self):
        if False:
            for i in range(10):
                print('nop')
        for user in self.all_users:
            self.assertTrue(self.policy.user_has_any_permission_for_instance(user, ['change', 'delete'], self.adder_image))

    def test_instances_user_has_permission_for(self):
        if False:
            print('Hello World!')
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        for user in self.all_users:
            self.assertResultSetEqual(self.policy.instances_user_has_permission_for(user, 'change'), all_images)

    def test_instances_user_has_any_permission_for(self):
        if False:
            while True:
                i = 10
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        for user in self.all_users:
            self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(user, ['change', 'delete']), all_images)

    def test_users_with_permission_for_instance(self):
        if False:
            while True:
                i = 10
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.useless_image)
        self.assertResultSetEqual(users_with_change_permission, self.active_users)

    def test_users_with_any_permission_for_instance(self):
        if False:
            print('Hello World!')
        users_with_change_or_del_permission = self.policy.users_with_any_permission_for_instance(['change', 'delete'], self.useless_image)
        self.assertResultSetEqual(users_with_change_or_del_permission, self.active_users)

class TestAuthenticationOnlyPermissionPolicy(PermissionPolicyTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.policy = AuthenticationOnlyPermissionPolicy(Image)

    def test_user_has_permission(self):
        if False:
            return 10
        self.assertUserPermissionMatrix([(self.superuser, True, True, True, True), (self.inactive_superuser, False, False, False, False), (self.image_adder, True, True, True, True), (self.oneoff_image_adder, True, True, True, True), (self.inactive_image_adder, False, False, False, False), (self.image_changer, True, True, True, True), (self.oneoff_image_changer, True, True, True, True), (self.inactive_image_changer, False, False, False, False), (self.oneoff_image_deleter, True, True, True, True), (self.useless_user, True, True, True, True), (self.anonymous_user, False, False, False, False)])

    def test_user_has_any_permission(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.policy.user_has_any_permission(self.superuser, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.inactive_superuser, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.useless_user, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.anonymous_user, ['add', 'change']))

    def test_users_with_permission(self):
        if False:
            i = 10
            return i + 15
        users_with_add_permission = self.policy.users_with_permission('add')
        self.assertResultSetEqual(users_with_add_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter, self.useless_user])

    def test_users_with_any_permission(self):
        if False:
            i = 10
            return i + 15
        users_with_add_or_change_permission = self.policy.users_with_any_permission(['add', 'change'])
        self.assertResultSetEqual(users_with_add_or_change_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter, self.useless_user])

    def test_user_has_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertUserInstancePermissionMatrix(self.adder_image, [(self.superuser, True, True, True), (self.inactive_superuser, False, False, False), (self.image_adder, True, True, True), (self.oneoff_image_adder, True, True, True), (self.inactive_image_adder, False, False, False), (self.image_changer, True, True, True), (self.oneoff_image_changer, True, True, True), (self.inactive_image_changer, False, False, False), (self.oneoff_image_deleter, True, True, True), (self.useless_user, True, True, True), (self.anonymous_user, False, False, False)])

    def test_user_has_any_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.superuser, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.inactive_superuser, ['change', 'delete'], self.adder_image))
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.useless_user, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.anonymous_user, ['change', 'delete'], self.adder_image))

    def test_instances_user_has_permission_for(self):
        if False:
            i = 10
            return i + 15
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.superuser, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.inactive_superuser, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.useless_user, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.anonymous_user, 'change'), no_images)

    def test_instances_user_has_any_permission_for(self):
        if False:
            print('Hello World!')
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.superuser, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.inactive_superuser, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.useless_user, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.anonymous_user, ['change', 'delete']), no_images)

    def test_users_with_permission_for_instance(self):
        if False:
            print('Hello World!')
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.useless_image)
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter, self.useless_user])

    def test_users_with_any_permission_for_instance(self):
        if False:
            while True:
                i = 10
        users_with_change_or_del_permission = self.policy.users_with_any_permission_for_instance(['change', 'delete'], self.useless_image)
        self.assertResultSetEqual(users_with_change_or_del_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter, self.useless_user])

class TestModelPermissionPolicy(PermissionPolicyTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.policy = ModelPermissionPolicy(Image)

    def test_user_has_permission(self):
        if False:
            while True:
                i = 10
        self.assertUserPermissionMatrix([(self.superuser, True, True, True, True), (self.inactive_superuser, False, False, False, False), (self.image_adder, True, False, False, False), (self.oneoff_image_adder, True, False, False, False), (self.inactive_image_adder, False, False, False, False), (self.image_changer, False, True, False, False), (self.oneoff_image_changer, False, True, False, False), (self.inactive_image_changer, False, False, False, False), (self.oneoff_image_deleter, False, False, True, False), (self.useless_user, False, False, False, False), (self.anonymous_user, False, False, False, False)])

    def test_user_has_any_permission(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.policy.user_has_any_permission(self.superuser, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.inactive_superuser, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.image_adder, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.oneoff_image_adder, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.image_changer, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.image_changer, ['add', 'delete']))
        self.assertFalse(self.policy.user_has_any_permission(self.inactive_image_adder, ['add', 'delete']))
        self.assertFalse(self.policy.user_has_any_permission(self.useless_user, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.anonymous_user, ['add', 'change']))

    def test_users_with_permission(self):
        if False:
            return 10
        users_with_add_permission = self.policy.users_with_permission('add')
        self.assertResultSetEqual(users_with_add_permission, [self.superuser, self.image_adder, self.oneoff_image_adder])
        users_with_change_permission = self.policy.users_with_permission('change')
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_changer, self.oneoff_image_changer])

    def test_users_with_any_permission(self):
        if False:
            while True:
                i = 10
        users_with_add_or_change_permission = self.policy.users_with_any_permission(['add', 'change'])
        self.assertResultSetEqual(users_with_add_or_change_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_change_or_delete_permission = self.policy.users_with_any_permission(['change', 'delete'])
        self.assertResultSetEqual(users_with_change_or_delete_permission, [self.superuser, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter])

    def test_user_has_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertUserInstancePermissionMatrix(self.adder_image, [(self.superuser, True, True, True), (self.inactive_superuser, False, False, False), (self.image_adder, False, False, False), (self.oneoff_image_adder, False, False, False), (self.inactive_image_adder, False, False, False), (self.image_changer, True, False, False), (self.oneoff_image_changer, True, False, False), (self.inactive_image_changer, False, False, False), (self.oneoff_image_deleter, False, True, False), (self.useless_user, False, False, False), (self.anonymous_user, False, False, False)])

    def test_user_has_any_permission_for_instance(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.superuser, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.inactive_superuser, ['change', 'delete'], self.adder_image))
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.image_changer, ['change', 'delete'], self.adder_image))
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.oneoff_image_changer, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.image_adder, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.inactive_image_changer, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.useless_user, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.anonymous_user, ['change', 'delete'], self.adder_image))

    def test_instances_user_has_permission_for(self):
        if False:
            i = 10
            return i + 15
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.superuser, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.inactive_superuser, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.image_changer, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.oneoff_image_changer, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.useless_user, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.inactive_image_changer, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.image_changer, 'delete'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.anonymous_user, 'change'), no_images)

    def test_instances_user_has_any_permission_for(self):
        if False:
            print('Hello World!')
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.superuser, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.inactive_superuser, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.image_changer, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.oneoff_image_changer, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.useless_user, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.inactive_image_changer, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.image_adder, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.anonymous_user, ['change', 'delete']), no_images)

    def test_users_with_permission_for_instance(self):
        if False:
            print('Hello World!')
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.useless_image)
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_changer, self.oneoff_image_changer])
        users_with_delete_permission = self.policy.users_with_permission_for_instance('delete', self.useless_image)
        self.assertResultSetEqual(users_with_delete_permission, [self.superuser, self.oneoff_image_deleter])

    def test_users_with_any_permission_for_instance(self):
        if False:
            print('Hello World!')
        users_with_change_or_del_permission = self.policy.users_with_any_permission_for_instance(['change', 'delete'], self.useless_image)
        self.assertResultSetEqual(users_with_change_or_del_permission, [self.superuser, self.image_changer, self.oneoff_image_changer, self.oneoff_image_deleter])

class TestOwnershipPermissionPolicy(PermissionPolicyTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.policy = OwnershipPermissionPolicy(Image, owner_field_name='uploaded_by_user')

    def test_user_has_permission(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertUserPermissionMatrix([(self.superuser, True, True, True, True), (self.inactive_superuser, False, False, False, False), (self.image_adder, True, True, True, False), (self.oneoff_image_adder, True, True, True, False), (self.inactive_image_adder, False, False, False, False), (self.image_changer, False, True, True, False), (self.oneoff_image_changer, False, True, True, False), (self.inactive_image_changer, False, False, False, False), (self.oneoff_image_deleter, False, False, False, False), (self.useless_user, False, False, False, False), (self.anonymous_user, False, False, False, False)])

    def test_user_has_any_permission(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.policy.user_has_any_permission(self.superuser, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.inactive_superuser, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.image_changer, ['add', 'change']))
        self.assertTrue(self.policy.user_has_any_permission(self.oneoff_image_changer, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.oneoff_image_deleter, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.inactive_image_changer, ['add', 'delete']))
        self.assertFalse(self.policy.user_has_any_permission(self.useless_user, ['add', 'change']))
        self.assertFalse(self.policy.user_has_any_permission(self.anonymous_user, ['add', 'change']))

    def test_users_with_permission(self):
        if False:
            while True:
                i = 10
        users_with_add_permission = self.policy.users_with_permission('add')
        self.assertResultSetEqual(users_with_add_permission, [self.superuser, self.image_adder, self.oneoff_image_adder])
        users_with_change_permission = self.policy.users_with_permission('change')
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_delete_permission = self.policy.users_with_permission('delete')
        self.assertResultSetEqual(users_with_delete_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_frobnicate_permission = self.policy.users_with_permission('frobnicate')
        self.assertResultSetEqual(users_with_frobnicate_permission, [self.superuser])

    def test_users_with_any_permission(self):
        if False:
            for i in range(10):
                print('nop')
        users_with_add_or_change_permission = self.policy.users_with_any_permission(['add', 'change'])
        self.assertResultSetEqual(users_with_add_or_change_permission, [self.superuser, self.image_adder, self.oneoff_image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_add_or_frobnicate_permission = self.policy.users_with_any_permission(['add', 'frobnicate'])
        self.assertResultSetEqual(users_with_add_or_frobnicate_permission, [self.superuser, self.image_adder, self.oneoff_image_adder])

    def test_user_has_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertUserInstancePermissionMatrix(self.adder_image, [(self.superuser, True, True, True), (self.inactive_superuser, False, False, False), (self.image_adder, True, True, False), (self.oneoff_image_adder, False, False, False), (self.inactive_image_adder, False, False, False), (self.image_changer, True, True, False), (self.oneoff_image_changer, True, True, False), (self.inactive_image_changer, False, False, False), (self.oneoff_image_deleter, False, False, False), (self.useless_user, False, False, False), (self.anonymous_user, False, False, False)])
        self.assertUserInstancePermissionMatrix(self.useless_image, [(self.superuser, True, True, True), (self.image_adder, False, False, False), (self.oneoff_image_adder, False, False, False), (self.image_changer, True, True, False), (self.oneoff_image_changer, True, True, False), (self.inactive_superuser, False, False, False), (self.inactive_image_adder, False, False, False), (self.inactive_image_changer, False, False, False), (self.oneoff_image_deleter, False, False, False), (self.useless_user, False, False, False), (self.anonymous_user, False, False, False)])
        self.assertUserInstancePermissionMatrix(self.anonymous_image, [(self.superuser, True, True, True), (self.image_adder, False, False, False), (self.oneoff_image_adder, False, False, False), (self.image_changer, True, True, False), (self.oneoff_image_changer, True, True, False), (self.inactive_superuser, False, False, False), (self.inactive_image_adder, False, False, False), (self.inactive_image_changer, False, False, False), (self.oneoff_image_deleter, False, False, False), (self.useless_user, False, False, False), (self.anonymous_user, False, False, False)])

    def test_user_has_any_permission_for_instance(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.superuser, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.inactive_superuser, ['change', 'delete'], self.adder_image))
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.image_changer, ['change', 'frobnicate'], self.adder_image))
        self.assertTrue(self.policy.user_has_any_permission_for_instance(self.oneoff_image_changer, ['change', 'frobnicate'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.oneoff_image_deleter, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.inactive_image_changer, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.useless_user, ['change', 'delete'], self.adder_image))
        self.assertFalse(self.policy.user_has_any_permission_for_instance(self.anonymous_user, ['change', 'delete'], self.adder_image))

    def test_instances_user_has_permission_for(self):
        if False:
            while True:
                i = 10
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.superuser, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.inactive_superuser, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.image_adder, 'change'), [self.adder_image])
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.image_adder, 'delete'), [self.adder_image])
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.image_changer, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.oneoff_image_changer, 'change'), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.inactive_image_changer, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.useless_user, 'change'), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_permission_for(self.anonymous_user, 'change'), no_images)

    def test_instances_user_has_any_permission_for(self):
        if False:
            return 10
        all_images = [self.adder_image, self.useless_image, self.anonymous_image]
        no_images = []
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.superuser, ['change', 'delete']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.inactive_superuser, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.image_adder, ['delete', 'frobnicate']), [self.adder_image])
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.oneoff_image_changer, ['delete', 'frobnicate']), all_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.useless_user, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.inactive_image_changer, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.oneoff_image_deleter, ['change', 'delete']), no_images)
        self.assertResultSetEqual(self.policy.instances_user_has_any_permission_for(self.anonymous_user, ['change', 'delete']), no_images)

    def test_users_with_permission_for_instance(self):
        if False:
            for i in range(10):
                print('nop')
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.adder_image)
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_delete_permission = self.policy.users_with_permission_for_instance('delete', self.adder_image)
        self.assertResultSetEqual(users_with_delete_permission, [self.superuser, self.image_adder, self.image_changer, self.oneoff_image_changer])
        users_with_delete_permission = self.policy.users_with_permission_for_instance('frobnicate', self.adder_image)
        self.assertResultSetEqual(users_with_delete_permission, [self.superuser])
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.useless_image)
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_changer, self.oneoff_image_changer])
        users_with_change_permission = self.policy.users_with_permission_for_instance('change', self.anonymous_image)
        self.assertResultSetEqual(users_with_change_permission, [self.superuser, self.image_changer, self.oneoff_image_changer])

    def test_users_with_any_permission_for_instance(self):
        if False:
            print('Hello World!')
        users_with_change_or_frob_permission = self.policy.users_with_any_permission_for_instance(['change', 'frobnicate'], self.adder_image)
        self.assertResultSetEqual(users_with_change_or_frob_permission, [self.superuser, self.image_adder, self.image_changer, self.oneoff_image_changer])