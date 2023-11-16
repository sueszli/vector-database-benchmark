from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.test import APIClient
from .views import BasicModelWithUsersViewSet, OrganizationPermissions

@override_settings(ROOT_URLCONF='tests.browsable_api.no_auth_urls')
class AnonymousUserTests(TestCase):
    """Tests correct handling of anonymous user request on endpoints with IsAuthenticated permission class."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.client = APIClient(enforce_csrf_checks=True)

    def tearDown(self):
        if False:
            return 10
        self.client.logout()

    def test_get_raises_typeerror_when_anonymous_user_in_queryset_filter(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.client.get('/basicviewset')

    def test_get_returns_http_forbidden_when_anonymous_user(self):
        if False:
            i = 10
            return i + 15
        old_permissions = BasicModelWithUsersViewSet.permission_classes
        BasicModelWithUsersViewSet.permission_classes = [IsAuthenticated, OrganizationPermissions]
        response = self.client.get('/basicviewset')
        BasicModelWithUsersViewSet.permission_classes = old_permissions
        self.assertEqual(response.status_code, 403)

@override_settings(ROOT_URLCONF='tests.browsable_api.auth_urls')
class DropdownWithAuthTests(TestCase):
    """Tests correct dropdown behaviour with Auth views enabled."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.client = APIClient(enforce_csrf_checks=True)
        self.username = 'john'
        self.email = 'lennon@thebeatles.com'
        self.password = 'password'
        self.user = User.objects.create_user(self.username, self.email, self.password)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.client.logout()

    def test_name_shown_when_logged_in(self):
        if False:
            while True:
                i = 10
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/')
        content = response.content.decode()
        assert 'john' in content

    def test_logout_shown_when_logged_in(self):
        if False:
            print('Hello World!')
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/')
        content = response.content.decode()
        assert '>Log out<' in content

    def test_login_shown_when_logged_out(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/')
        content = response.content.decode()
        assert '>Log in<' in content

@override_settings(ROOT_URLCONF='tests.browsable_api.no_auth_urls')
class NoDropdownWithoutAuthTests(TestCase):
    """Tests correct dropdown behaviour with Auth views NOT enabled."""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = APIClient(enforce_csrf_checks=True)
        self.username = 'john'
        self.email = 'lennon@thebeatles.com'
        self.password = 'password'
        self.user = User.objects.create_user(self.username, self.email, self.password)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.client.logout()

    def test_name_shown_when_logged_in(self):
        if False:
            i = 10
            return i + 15
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/')
        content = response.content.decode()
        assert 'john' in content

    def test_dropdown_not_shown_when_logged_in(self):
        if False:
            while True:
                i = 10
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/')
        content = response.content.decode()
        assert '<li class="dropdown">' not in content

    def test_dropdown_not_shown_when_logged_out(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/')
        content = response.content.decode()
        assert '<li class="dropdown">' not in content