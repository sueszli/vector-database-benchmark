from django.contrib.auth import authenticate
from django.contrib.auth.context_processors import PermLookupDict, PermWrapper
from django.contrib.auth.models import Permission, User
from django.contrib.contenttypes.models import ContentType
from django.db.models import Q
from django.test import SimpleTestCase, TestCase, override_settings
from .settings import AUTH_MIDDLEWARE, AUTH_TEMPLATES

class MockUser:

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'MockUser()'

    def has_module_perms(self, perm):
        if False:
            print('Hello World!')
        return perm == 'mockapp'

    def has_perm(self, perm, obj=None):
        if False:
            i = 10
            return i + 15
        return perm == 'mockapp.someperm'

class PermWrapperTests(SimpleTestCase):
    """
    Test some details of the PermWrapper implementation.
    """

    class EQLimiterObject:
        """
        This object makes sure __eq__ will not be called endlessly.
        """

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.eq_calls = 0

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            if self.eq_calls > 0:
                return True
            self.eq_calls += 1
            return False

    def test_repr(self):
        if False:
            while True:
                i = 10
        perms = PermWrapper(MockUser())
        self.assertEqual(repr(perms), 'PermWrapper(MockUser())')

    def test_permwrapper_in(self):
        if False:
            print('Hello World!')
        "\n        'something' in PermWrapper works as expected.\n        "
        perms = PermWrapper(MockUser())
        self.assertIn('mockapp', perms)
        self.assertNotIn('nonexistent', perms)
        self.assertIn('mockapp.someperm', perms)
        self.assertNotIn('mockapp.nonexistent', perms)

    def test_permlookupdict_in(self):
        if False:
            while True:
                i = 10
        "\n        No endless loops if accessed with 'in' - refs #18979.\n        "
        pldict = PermLookupDict(MockUser(), 'mockapp')
        with self.assertRaises(TypeError):
            self.EQLimiterObject() in pldict

    def test_iter(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesMessage(TypeError, 'PermWrapper is not iterable.'):
            iter(PermWrapper(MockUser()))

@override_settings(ROOT_URLCONF='auth_tests.urls', TEMPLATES=AUTH_TEMPLATES)
class AuthContextProcessorTests(TestCase):
    """
    Tests for the ``django.contrib.auth.context_processors.auth`` processor
    """

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        cls.superuser = User.objects.create_superuser(username='super', password='secret', email='super@example.com')

    @override_settings(MIDDLEWARE=AUTH_MIDDLEWARE)
    def test_session_not_accessed(self):
        if False:
            print('Hello World!')
        '\n        The session is not accessed simply by including\n        the auth context processor\n        '
        response = self.client.get('/auth_processor_no_attr_access/')
        self.assertContains(response, 'Session not accessed')

    @override_settings(MIDDLEWARE=AUTH_MIDDLEWARE)
    def test_session_is_accessed(self):
        if False:
            print('Hello World!')
        '\n        The session is accessed if the auth context processor\n        is used and relevant attributes accessed.\n        '
        response = self.client.get('/auth_processor_attr_access/')
        self.assertContains(response, 'Session accessed')

    def test_perms_attrs(self):
        if False:
            return 10
        u = User.objects.create_user(username='normal', password='secret')
        u.user_permissions.add(Permission.objects.get(content_type=ContentType.objects.get_for_model(Permission), codename='add_permission'))
        self.client.force_login(u)
        response = self.client.get('/auth_processor_perms/')
        self.assertContains(response, 'Has auth permissions')
        self.assertContains(response, 'Has auth.add_permission permissions')
        self.assertNotContains(response, 'nonexistent')

    def test_perm_in_perms_attrs(self):
        if False:
            i = 10
            return i + 15
        u = User.objects.create_user(username='normal', password='secret')
        u.user_permissions.add(Permission.objects.get(content_type=ContentType.objects.get_for_model(Permission), codename='add_permission'))
        self.client.login(username='normal', password='secret')
        response = self.client.get('/auth_processor_perm_in_perms/')
        self.assertContains(response, 'Has auth permissions')
        self.assertContains(response, 'Has auth.add_permission permissions')
        self.assertNotContains(response, 'nonexistent')

    def test_message_attrs(self):
        if False:
            print('Hello World!')
        self.client.force_login(self.superuser)
        response = self.client.get('/auth_processor_messages/')
        self.assertContains(response, 'Message 1')

    def test_user_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The lazy objects returned behave just like the wrapped objects.\n        '
        self.client.login(username='super', password='secret')
        user = authenticate(username='super', password='secret')
        response = self.client.get('/auth_processor_user/')
        self.assertContains(response, 'unicode: super')
        self.assertContains(response, 'id: %d' % self.superuser.pk)
        self.assertContains(response, 'username: super')
        self.assertContains(response, 'url: /userpage/super/')
        Q(user=response.context['user']) & Q(someflag=True)
        self.assertEqual(response.context['user'], user)
        self.assertEqual(user, response.context['user'])