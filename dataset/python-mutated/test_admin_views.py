import unittest.mock
from django import forms
from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.core.exceptions import ImproperlyConfigured
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail import hooks
from wagtail.admin.admin_url_finder import AdminURLFinder
from wagtail.compat import AUTH_USER_APP_LABEL, AUTH_USER_MODEL_NAME
from wagtail.models import Collection, GroupCollectionPermission, GroupPagePermission, Page
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.template_tests import AdminTemplateTestUtils
from wagtail.users.forms import UserCreationForm, UserEditForm
from wagtail.users.models import UserProfile
from wagtail.users.permission_order import register as register_permission_order
from wagtail.users.views.groups import GroupViewSet
from wagtail.users.views.users import get_user_creation_form, get_user_edit_form
from wagtail.users.wagtail_hooks import get_group_viewset_cls
delete_user_perm_codename = f'delete_{AUTH_USER_MODEL_NAME.lower()}'
change_user_perm_codename = f'change_{AUTH_USER_MODEL_NAME.lower()}'
User = get_user_model()

def test_avatar_provider(user, default, size=50):
    if False:
        return 10
    return '/nonexistent/path/to/avatar.png'

class CustomUserCreationForm(UserCreationForm):
    country = forms.CharField(required=True, label='Country')
    attachment = forms.FileField(required=True, label='Attachment')

class CustomUserEditForm(UserEditForm):
    country = forms.CharField(required=True, label='Country')
    attachment = forms.FileField(required=True, label='Attachment')

class CustomGroupViewSet(GroupViewSet):
    icon = 'custom-icon'

class TestUserFormHelpers(TestCase):

    def test_get_user_edit_form_with_default_form(self):
        if False:
            i = 10
            return i + 15
        user_form = get_user_edit_form()
        self.assertIs(user_form, UserEditForm)

    def test_get_user_creation_form_with_default_form(self):
        if False:
            i = 10
            return i + 15
        user_form = get_user_creation_form()
        self.assertIs(user_form, UserCreationForm)

    @override_settings(WAGTAIL_USER_CREATION_FORM='wagtail.users.tests.CustomUserCreationForm')
    def test_get_user_creation_form_with_custom_form(self):
        if False:
            i = 10
            return i + 15
        user_form = get_user_creation_form()
        self.assertIs(user_form, CustomUserCreationForm)

    @override_settings(WAGTAIL_USER_EDIT_FORM='wagtail.users.tests.CustomUserEditForm')
    def test_get_user_edit_form_with_custom_form(self):
        if False:
            return 10
        user_form = get_user_edit_form()
        self.assertIs(user_form, CustomUserEditForm)

    @override_settings(WAGTAIL_USER_CREATION_FORM='wagtail.users.tests.CustomUserCreationFormDoesNotExist')
    def test_get_user_creation_form_with_invalid_form(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ImproperlyConfigured, get_user_creation_form)

    @override_settings(WAGTAIL_USER_EDIT_FORM='wagtail.users.tests.CustomUserEditFormDoesNotExist')
    def test_get_user_edit_form_with_invalid_form(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ImproperlyConfigured, get_user_edit_form)

class TestGroupUsersView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password', first_name='First Name', last_name='Last Name')
        self.test_group = Group.objects.create(name='Test Group')
        self.test_user.groups.add(self.test_group)
        self.login()

    def get(self, params={}, group_id=None):
        if False:
            print('Hello World!')
        return self.client.get(reverse('wagtailusers_groups:users', args=(group_id or self.test_group.pk,)), params)

    def test_simple(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/index.html')
        self.assertContains(response, 'testuser')
        self.assertContains(response, 'Add a user')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_inexisting_group(self):
        if False:
            i = 10
            return i + 15
        response = self.get(group_id=9999)
        self.assertEqual(response.status_code, 404)

    def test_search(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['query_string'], 'Hello')

    def test_search_query_one_field(self):
        if False:
            while True:
                i = 10
        response = self.get({'q': 'first name'})
        self.assertEqual(response.status_code, 200)
        results = response.context['users']
        self.assertIn(self.test_user, results)

    def test_search_query_multiple_fields(self):
        if False:
            while True:
                i = 10
        response = self.get({'q': 'first name last name'})
        self.assertEqual(response.status_code, 200)
        results = response.context['users']
        self.assertIn(self.test_user, results)

    def test_pagination(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'p': 1})
        self.assertEqual(response.status_code, 200)
        response = self.get({'p': 9999})
        self.assertEqual(response.status_code, 404)

class TestGroupUsersResultsView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password', first_name='First Name', last_name='Last Name')
        self.test_group = Group.objects.create(name='Test Group')
        self.test_user.groups.add(self.test_group)
        self.login()

    def get(self, params={}, group_id=None):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailusers_groups:users_results', args=(group_id or self.test_group.pk,)), params)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/results.html')
        self.assertContains(response, 'testuser')
        self.assertNotContains(response, 'Add a user')

class TestUserIndexView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password', first_name='First Name', last_name='Last Name')
        self.login()

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_users:index'), params)

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/index.html')
        self.assertContains(response, 'testuser')
        self.assertContains(response, 'Add a user')
        self.assertBreadcrumbsNotRendered(response.content)

    @unittest.skipIf(settings.AUTH_USER_MODEL == 'emailuser.EmailUser', 'Negative UUID not possible')
    def test_allows_negative_ids(self):
        if False:
            print('Hello World!')
        self.create_user('guardian', 'guardian@example.com', 'gu@rd14n', pk=-1)
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'testuser')
        self.assertContains(response, 'guardian')

    def test_search(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['query_string'], 'Hello')

    def test_search_query_one_field(self):
        if False:
            return 10
        response = self.get({'q': 'first name'})
        self.assertEqual(response.status_code, 200)
        results = response.context['users']
        self.assertIn(self.test_user, results)

    def test_search_query_multiple_fields(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': 'first name last name'})
        self.assertEqual(response.status_code, 200)
        results = response.context['users']
        self.assertIn(self.test_user, results)

    def test_pagination(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'p': 1})
        self.assertEqual(response.status_code, 200)
        response = self.get({'p': 9999})
        self.assertEqual(response.status_code, 404)

    def test_valid_ordering(self):
        if False:
            return 10
        response = self.get({'ordering': 'email'})
        self.assertNotEqual(response.context_data['ordering'], 'email')
        self.assertEqual(response.context_data['ordering'], 'name')
        response = self.get({'ordering': 'username'})
        self.assertEqual(response.context_data['ordering'], 'username')

    def test_num_queries(self):
        if False:
            i = 10
            return i + 15
        self.get()
        num_queries = 9
        with self.assertNumQueries(num_queries):
            self.get()
        self.create_user('test', 'test@example.com', 'gu@rd14n')
        with self.assertNumQueries(num_queries):
            self.get()

class TestUserIndexResultsView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password', first_name='First Name', last_name='Last Name')
        self.login()

    def get(self, params={}):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailusers_users:index_results'), params)

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/results.html')
        self.assertContains(response, 'testuser')
        self.assertNotContains(response, 'Add a user')

class TestUserCreateView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.login()

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_users:add'), params)

    def post(self, post_data={}, follow=False):
        if False:
            i = 10
            return i + 15
        return self.client.post(reverse('wagtailusers_users:add'), post_data, follow=follow)

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertContains(response, 'Password')
        self.assertContains(response, 'Password confirmation')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_create(self):
        if False:
            return 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'password', 'password2': 'password'}, follow=True)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 1)
        if settings.AUTH_USER_MODEL == 'emailuser.EmailUser':
            self.assertContains(response, 'User &#x27;test@user.com&#x27; created.')
        else:
            self.assertContains(response, 'User &#x27;testuser&#x27; created.')

    @unittest.skipUnless(settings.AUTH_USER_MODEL == 'customuser.CustomUser', 'Only applicable to CustomUser')
    @override_settings(WAGTAIL_USER_CREATION_FORM='wagtail.users.tests.CustomUserCreationForm', WAGTAIL_USER_CUSTOM_FIELDS=['country', 'document'])
    def test_create_with_custom_form(self):
        if False:
            while True:
                i = 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'password', 'password2': 'password', 'country': 'testcountry', 'attachment': SimpleUploadedFile('test.txt', b'Uploaded file')})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 1)
        self.assertEqual(users.first().country, 'testcountry')
        self.assertEqual(users.first().attachment.read(), b'Uploaded file')

    def test_create_with_whitespaced_password(self):
        if False:
            for i in range(10):
                print('nop')
        'Password should not be stripped'
        self.post({'username': 'testuser2', 'email': 'test@user2.com', 'first_name': 'Test', 'last_name': 'User', 'password1': '  whitespaced_password  ', 'password2': '  whitespaced_password  '}, follow=True)
        self.client.logout()
        username = 'testuser2'
        if settings.AUTH_USER_MODEL == 'emailuser.EmailUser':
            username = 'test@user2.com'
        self.login(username=username, password='  whitespaced_password  ')

    def test_create_with_password_mismatch(self):
        if False:
            return 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'password1', 'password2': 'password2'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertTrue(response.context['form'].errors['password2'])
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 0)

    @override_settings(AUTH_PASSWORD_VALIDATORS=[{'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'}])
    def test_create_with_password_validation(self):
        if False:
            return 10
        '\n        Test that the Django password validators are run when creating a user.\n        Specifically test that the UserAttributeSimilarityValidator works,\n        which requires a full-populated user model before the validation works.\n        '
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Example', 'last_name': 'Name', 'password1': 'example name', 'password2': 'example name'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        errors = response.context['form'].errors.as_data()
        self.assertIn('password2', errors)
        self.assertEqual(errors['password2'][0].code, 'password_too_similar')
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 0)

    def test_create_with_missing_password(self):
        if False:
            return 10
        'Password should be required by default'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': '', 'password2': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertTrue(response.context['form'].errors['password1'])
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 0)

    @override_settings(WAGTAILUSERS_PASSWORD_REQUIRED=False)
    def test_password_fields_exist_when_not_required(self):
        if False:
            print('Hello World!')
        'Password fields should still be shown if WAGTAILUSERS_PASSWORD_REQUIRED is False'
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertContains(response, 'Password')
        self.assertContains(response, 'Password confirmation')

    @override_settings(WAGTAILUSERS_PASSWORD_REQUIRED=False)
    def test_create_with_password_not_required(self):
        if False:
            i = 10
            return i + 15
        'Password should not be required if WAGTAILUSERS_PASSWORD_REQUIRED is False'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': '', 'password2': ''})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 1)
        self.assertEqual(users.first().password, '')

    @override_settings(WAGTAILUSERS_PASSWORD_REQUIRED=False)
    def test_optional_password_is_still_validated(self):
        if False:
            for i in range(10):
                print('nop')
        'When WAGTAILUSERS_PASSWORD_REQUIRED is False, password validation should still apply if a password _is_ supplied'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'banana', 'password2': 'kumquat'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertTrue(response.context['form'].errors['password2'])
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 0)

    @override_settings(WAGTAILUSERS_PASSWORD_REQUIRED=False)
    def test_password_still_accepted_when_optional(self):
        if False:
            while True:
                i = 10
        'When WAGTAILUSERS_PASSWORD_REQUIRED is False, we should still allow a password to be set'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'banana', 'password2': 'banana'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 1)
        self.assertTrue(users.first().check_password('banana'))

    @override_settings(WAGTAILUSERS_PASSWORD_ENABLED=False)
    def test_password_fields_not_shown_when_disabled(self):
        if False:
            i = 10
            return i + 15
        'WAGTAILUSERS_PASSWORD_ENABLED=False should cause password fields to be removed'
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')
        self.assertNotContains(response, 'Password')
        self.assertNotContains(response, 'Password confirmation')

    @override_settings(WAGTAILUSERS_PASSWORD_ENABLED=False)
    def test_password_fields_ignored_when_disabled(self):
        if False:
            print('Hello World!')
        'When WAGTAILUSERS_PASSWORD_ENABLED is False, users should always be created without a usable password'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'banana', 'password2': 'kumquat'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='test@user.com')
        self.assertEqual(users.count(), 1)
        self.assertEqual(users.first().password, '')

    def test_before_create_user_hook(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_user', hook_func):
            response = self.client.get(reverse('wagtailusers_users:add'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_create_user_hook_post(self):
        if False:
            while True:
                i = 10

        def hook_func(request):
            if False:
                return 10
            self.assertIsInstance(request, HttpRequest)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_user', hook_func):
            post_data = {'username': 'testuser', 'email': 'testuser@test.com', 'password1': 'password12', 'password2': 'password12', 'first_name': 'test', 'last_name': 'user'}
            response = self.client.post(reverse('wagtailusers_users:add'), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_after_create_user_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, user):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(user, get_user_model())
            return HttpResponse('Overridden!')
        with self.register_hook('after_create_user', hook_func):
            post_data = {'username': 'testuser', 'email': 'testuser@test.com', 'password1': 'password12', 'password2': 'password12', 'first_name': 'test', 'last_name': 'user'}
            response = self.client.post(reverse('wagtailusers_users:add'), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

class TestUserDeleteView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password')
        self.superuser = self.create_superuser(username='testsuperuser', email='testsuperuser@email.com', password='password')
        self.current_user = self.login()

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)), params)

    def post(self, post_data={}, follow=False):
        if False:
            print('Hello World!')
        return self.client.post(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)), post_data, follow=follow)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/confirm_delete.html')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_delete(self):
        if False:
            while True:
                i = 10
        response = self.post(follow=True)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='testuser@email.com')
        self.assertEqual(users.count(), 0)
        if settings.AUTH_USER_MODEL == 'emailuser.EmailUser':
            self.assertContains(response, 'User &#x27;testuser@email.com&#x27; deleted.')
        else:
            self.assertContains(response, 'User &#x27;testuser&#x27; deleted.')

    def test_user_cannot_delete_self(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailusers_users:delete', args=(self.current_user.pk,)))
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertTrue(get_user_model().objects.filter(pk=self.current_user.pk).exists())

    def test_user_can_delete_other_superuser(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailusers_users:delete', args=(self.superuser.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/confirm_delete.html')
        response = self.client.post(reverse('wagtailusers_users:delete', args=(self.superuser.pk,)))
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='testsuperuser@email.com')
        self.assertEqual(users.count(), 0)

    def test_before_delete_user_hook(self):
        if False:
            while True:
                i = 10

        def hook_func(request, user):
            if False:
                return 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.pk, self.test_user.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_user', hook_func):
            response = self.client.get(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_delete_user_hook_post(self):
        if False:
            print('Hello World!')

        def hook_func(request, user):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.pk, self.test_user.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_user', hook_func):
            response = self.client.post(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_after_delete_user_hook(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, user):
            if False:
                print('Hello World!')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.email, self.test_user.email)
            return HttpResponse('Overridden!')
        with self.register_hook('after_delete_user', hook_func):
            response = self.client.post(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

class TestUserDeleteViewForNonSuperuser(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', password='password')
        self.deleter_user = self.create_user(username='deleter', password='password')
        deleters_group = Group.objects.create(name='User deleters')
        deleters_group.permissions.add(Permission.objects.get(codename='access_admin'))
        deleters_group.permissions.add(Permission.objects.get(content_type__app_label=AUTH_USER_APP_LABEL, codename=delete_user_perm_codename))
        self.deleter_user.groups.add(deleters_group)
        self.superuser = self.create_test_user()
        self.login(username='deleter', password='password')

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/confirm_delete.html')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_delete(self):
        if False:
            while True:
                i = 10
        response = self.client.post(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)))
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        users = get_user_model().objects.filter(email='testuser@email.com')
        self.assertEqual(users.count(), 0)

    def test_user_cannot_delete_self(self):
        if False:
            while True:
                i = 10
        response = self.client.post(reverse('wagtailusers_users:delete', args=(self.deleter_user.pk,)))
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertTrue(get_user_model().objects.filter(pk=self.deleter_user.pk).exists())

    def test_user_cannot_delete_superuser(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.post(reverse('wagtailusers_users:delete', args=(self.superuser.pk,)))
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertTrue(get_user_model().objects.filter(pk=self.superuser.pk).exists())

class TestUserEditView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', first_name='Original', last_name='User', password='password')
        self.current_user = self.login()

    def get(self, params={}, user_id=None):
        if False:
            print('Hello World!')
        return self.client.get(reverse('wagtailusers_users:edit', args=(user_id or self.test_user.pk,)), params)

    def post(self, post_data={}, user_id=None, follow=False):
        if False:
            while True:
                i = 10
        return self.client.post(reverse('wagtailusers_users:edit', args=(user_id or self.test_user.pk,)), post_data, follow=follow)

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/edit.html')
        self.assertContains(response, 'Password')
        self.assertContains(response, 'Password confirmation')
        self.assertBreadcrumbsNotRendered(response.content)
        url_finder = AdminURLFinder(self.current_user)
        expected_url = '/admin/users/%s/' % self.test_user.pk
        self.assertEqual(url_finder.get_edit_url(self.test_user), expected_url)

    def test_nonexistant_redirect(self):
        if False:
            return 10
        invalid_id = '99999999-9999-9999-9999-999999999999' if settings.AUTH_USER_MODEL == 'emailuser.EmailUser' else 100000
        self.assertEqual(self.get(user_id=invalid_id).status_code, 404)

    def test_simple_post(self):
        if False:
            print('Hello World!')
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'newpassword', 'password2': 'newpassword', 'is_active': 'on'}, follow=True)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertTrue(user.check_password('newpassword'))
        if settings.AUTH_USER_MODEL == 'emailuser.EmailUser':
            self.assertContains(response, 'User &#x27;test@user.com&#x27; updated.')
        else:
            self.assertContains(response, 'User &#x27;testuser&#x27; updated.')

    def test_password_optional(self):
        if False:
            print('Hello World!')
        'Leaving password fields blank should leave it unchanged'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': '', 'password2': '', 'is_active': 'on'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertTrue(user.check_password('password'))

    def test_passwords_match(self):
        if False:
            for i in range(10):
                print('nop')
        'Password fields should be validated if supplied'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'banana', 'password2': 'kumquat', 'is_active': 'on'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/edit.html')
        self.assertTrue(response.context['form'].errors['password2'])
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Original')
        self.assertTrue(user.check_password('password'))

    @override_settings(AUTH_PASSWORD_VALIDATORS=[{'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'}])
    def test_edit_with_password_validation(self):
        if False:
            while True:
                i = 10
        '\n        Test that the Django password validators are run when editing a user.\n        Specifically test that the UserAttributeSimilarityValidator works,\n        which requires a full-populated user model before the validation works.\n        '
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'Name', 'password1': 'edited name', 'password2': 'edited name'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/edit.html')
        errors = response.context['form'].errors.as_data()
        self.assertIn('password2', errors)
        self.assertEqual(errors['password2'][0].code, 'password_too_similar')
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Original')
        self.assertTrue(user.check_password('password'))

    def test_edit_and_deactivate(self):
        if False:
            i = 10
            return i + 15
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'password', 'password2': 'password'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertIs(user.is_superuser, False)
        self.assertIs(user.is_active, False)

    def test_edit_and_make_superuser(self):
        if False:
            while True:
                i = 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'password', 'password2': 'password', 'is_active': 'on', 'is_superuser': 'on'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertIs(user.is_superuser, True)
        self.assertIs(user.is_active, True)

    def test_edit_self(self):
        if False:
            print('Hello World!')
        response = self.post({'username': 'test@email.com', 'email': 'test@email.com', 'first_name': 'Edited Myself', 'last_name': 'User', 'is_active': 'on', 'is_superuser': 'on'}, self.current_user.pk)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.current_user.pk)
        self.assertEqual(user.first_name, 'Edited Myself')
        self.assertIs(user.is_superuser, True)
        self.assertIs(user.is_active, True)

    def test_editing_own_password_does_not_log_out(self):
        if False:
            return 10
        response = self.post({'username': 'test@email.com', 'email': 'test@email.com', 'first_name': 'Edited Myself', 'last_name': 'User', 'password1': 'c0rrecth0rse', 'password2': 'c0rrecth0rse', 'is_active': 'on', 'is_superuser': 'on'}, self.current_user.pk)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.current_user.pk)
        self.assertEqual(user.first_name, 'Edited Myself')
        response = self.client.get(reverse('wagtailusers_users:index'))
        self.assertEqual(response.status_code, 200)

    def test_cannot_demote_self(self):
        if False:
            i = 10
            return i + 15
        "\n        check that unsetting a user's own is_active or is_superuser flag has no effect\n        "
        response = self.post({'username': 'test@email.com', 'email': 'test@email.com', 'first_name': 'Edited Myself', 'last_name': 'User'}, self.current_user.pk)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.current_user.pk)
        self.assertEqual(user.first_name, 'Edited Myself')
        self.assertIs(user.is_superuser, True)
        self.assertIs(user.is_active, True)

    @unittest.skipUnless(settings.AUTH_USER_MODEL == 'customuser.CustomUser', 'Only applicable to CustomUser')
    @override_settings(WAGTAIL_USER_EDIT_FORM='wagtail.users.tests.CustomUserEditForm')
    def test_edit_with_custom_form(self):
        if False:
            print('Hello World!')
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'password', 'password2': 'password', 'country': 'testcountry', 'attachment': SimpleUploadedFile('test.txt', b'Uploaded file')})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertEqual(user.country, 'testcountry')
        self.assertEqual(user.attachment.read(), b'Uploaded file')

    @unittest.skipIf(settings.AUTH_USER_MODEL == 'emailuser.EmailUser', 'Not applicable to EmailUser')
    def test_edit_validation_error(self):
        if False:
            return 10
        response = self.post({'username': '', 'email': 'test@user.com', 'first_name': 'Teset', 'last_name': 'User', 'password1': 'password', 'password2': 'password'})
        self.assertEqual(response.status_code, 200)

    @override_settings(WAGTAILUSERS_PASSWORD_ENABLED=False)
    def test_password_fields_not_shown_when_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        'WAGTAILUSERS_PASSWORD_ENABLED=False should cause password fields to be removed'
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/edit.html')
        self.assertNotContains(response, 'Password')
        self.assertNotContains(response, 'Password confirmation')

    @override_settings(WAGTAILUSERS_PASSWORD_ENABLED=False)
    def test_password_fields_ignored_when_disabled(self):
        if False:
            return 10
        'When WAGTAILUSERS_PASSWORD_REQUIRED is False, existing password should be left unchanged'
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'is_active': 'on', 'password1': 'banana', 'password2': 'kumquat'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertTrue(user.check_password('password'))

    def test_before_edit_user_hook(self):
        if False:
            return 10

        def hook_func(request, user):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.pk, self.test_user.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_user', hook_func):
            response = self.client.get(reverse('wagtailusers_users:edit', args=(self.test_user.pk,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_edit_user_hook_post(self):
        if False:
            return 10

        def hook_func(request, user):
            if False:
                return 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.pk, self.test_user.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_user', hook_func):
            post_data = {'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'password', 'password2': 'password'}
            response = self.client.post(reverse('wagtailusers_users:edit', args=(self.test_user.pk,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_after_edit_user_hook_post(self):
        if False:
            while True:
                i = 10

        def hook_func(request, user):
            if False:
                print('Hello World!')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(user.pk, self.test_user.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('after_edit_user', hook_func):
            post_data = {'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'password', 'password2': 'password'}
            response = self.client.post(reverse('wagtailusers_users:edit', args=(self.test_user.pk,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

class TestUserProfileCreation(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_user = self.create_user(username='testuser', password='password')

    def test_user_created_without_profile(self):
        if False:
            return 10
        self.assertEqual(UserProfile.objects.filter(user=self.test_user).count(), 0)
        with self.assertRaises(UserProfile.DoesNotExist):
            self.test_user.wagtail_userprofile

    def test_user_profile_created_when_method_called(self):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(UserProfile.get_for_user(self.test_user), UserProfile)
        self.assertEqual(UserProfile.objects.filter(user=self.test_user).count(), 1)

    def test_avatar_empty_on_profile_creation(self):
        if False:
            for i in range(10):
                print('nop')
        user_profile = UserProfile.get_for_user(self.test_user)
        self.assertFalse(user_profile.avatar)

class TestUserEditViewForNonSuperuser(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.editor_user = self.create_user(username='editor', password='password')
        editors_group = Group.objects.create(name='User editors')
        editors_group.permissions.add(Permission.objects.get(codename='access_admin'))
        editors_group.permissions.add(Permission.objects.get(content_type__app_label=AUTH_USER_APP_LABEL, codename=change_user_perm_codename))
        self.editor_user.groups.add(editors_group)
        self.login(username='editor', password='password')

    def test_user_cannot_escalate_privileges(self):
        if False:
            return 10
        "\n        Check that a non-superuser cannot edit their own is_active or is_superuser flag.\n        (note: this doesn't necessarily guard against other routes to escalating privileges, such\n        as creating a new user with is_superuser=True or adding oneself to a group with additional\n        privileges - the latter will be dealt with by #537)\n        "
        editors_group = Group.objects.get(name='User editors')
        post_data = {'username': 'editor', 'email': 'editor@email.com', 'first_name': 'Escalating', 'last_name': 'User', 'password1': '', 'password2': '', 'groups': [editors_group.id], 'is_superuser': 'on', 'is_active': 'on'}
        response = self.client.post(reverse('wagtailusers_users:edit', args=(self.editor_user.pk,)), post_data)
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.editor_user.pk)
        self.assertTrue(user.groups.filter(name='User editors').exists())
        self.assertEqual(user.first_name, 'Escalating')
        self.assertIs(user.is_superuser, False)

class TestGroupIndexView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.login()

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_groups:index'), params)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/groups/index.html')
        self.assertTemplateUsed(response, 'wagtailadmin/generic/index.html')
        self.assertContains(response, 'Add a group')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_search(self):
        if False:
            print('Hello World!')
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['search_form']['q'].value(), 'Hello')

    def test_default_ordering(self):
        if False:
            i = 10
            return i + 15
        Group.objects.create(name='Photographers')
        response = self.get()
        names = [group.name for group in response.context_data['object_list']]
        self.assertEqual(names, ['Editors', 'Moderators', 'Photographers'])

class TestGroupIndexResultsView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()

    def get(self, params={}):
        if False:
            return 10
        return self.client.get(reverse('wagtailusers_groups:index_results'), params)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/listing_results.html')
        self.assertNotContains(response, 'Add a group')

    def test_search(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['search_form']['q'].value(), 'Hello')

class TestGroupCreateView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()
        self.add_doc_permission = Permission.objects.get(content_type__app_label='wagtaildocs', codename='add_document')
        self.change_doc_permission = Permission.objects.get(content_type__app_label='wagtaildocs', codename='change_document')

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_groups:add'), params)

    def post(self, post_data={}):
        if False:
            i = 10
            return i + 15
        post_defaults = {'page_permissions-TOTAL_FORMS': ['0'], 'page_permissions-MAX_NUM_FORMS': ['1000'], 'page_permissions-INITIAL_FORMS': ['0'], 'collection_permissions-TOTAL_FORMS': ['0'], 'collection_permissions-MAX_NUM_FORMS': ['1000'], 'collection_permissions-INITIAL_FORMS': ['0'], 'document_permissions-TOTAL_FORMS': ['0'], 'document_permissions-MAX_NUM_FORMS': ['1000'], 'document_permissions-INITIAL_FORMS': ['0'], 'image_permissions-TOTAL_FORMS': ['0'], 'image_permissions-MAX_NUM_FORMS': ['1000'], 'image_permissions-INITIAL_FORMS': ['0']}
        for (k, v) in post_defaults.items():
            post_data[k] = post_data.get(k, v)
        return self.client.post(reverse('wagtailusers_groups:add'), post_data)

    def test_simple(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/groups/create.html')
        self.assertBreadcrumbsNotRendered(response.content)

    def test_create_group(self):
        if False:
            i = 10
            return i + 15
        response = self.post({'name': 'test group'})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        groups = Group.objects.filter(name='test group')
        self.assertEqual(groups.count(), 1)

    def test_group_create_adding_permissions(self):
        if False:
            while True:
                i = 10
        response = self.post({'name': 'test group', 'page_permissions-0-page': ['1'], 'page_permissions-0-permissions': ['change_page', 'publish_page'], 'page_permissions-TOTAL_FORMS': ['1'], 'document_permissions-0-collection': [Collection.get_first_root_node().pk], 'document_permissions-0-permissions': [self.add_doc_permission.pk], 'document_permissions-TOTAL_FORMS': ['1']})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        new_group = Group.objects.get(name='test group')
        self.assertEqual(new_group.page_permissions.all().count(), 2)
        self.assertEqual(new_group.collection_permissions.filter(permission=self.add_doc_permission).count(), 1)

    def test_duplicate_page_permissions_error(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.post({'name': 'test group', 'page_permissions-0-page': ['1'], 'page_permissions-0-permissions': ['publish_page'], 'page_permissions-1-page': ['1'], 'page_permissions-1-permissions': ['change_page'], 'page_permissions-TOTAL_FORMS': ['2']})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['permission_panels'][0].non_form_errors)

    def test_duplicate_document_permissions_error(self):
        if False:
            for i in range(10):
                print('nop')
        root_collection = Collection.get_first_root_node()
        response = self.post({'name': 'test group', 'document_permissions-0-collection': [root_collection.pk], 'document_permissions-0-permissions': [self.add_doc_permission.pk], 'document_permissions-1-collection': [root_collection.pk], 'document_permissions-1-permissions': [self.change_doc_permission.pk], 'document_permissions-TOTAL_FORMS': ['2']})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any((hasattr(panel, 'non_form_errors') and panel.non_form_errors for panel in response.context['permission_panels'])))

    def test_can_submit_blank_permission_form(self):
        if False:
            while True:
                i = 10
        response = self.post({'name': 'test group', 'page_permissions-0-page': [''], 'page_permissions-TOTAL_FORMS': ['1'], 'document_permissions-0-collection': [''], 'document_permissions-TOTAL_FORMS': ['1']})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        new_group = Group.objects.get(name='test group')
        self.assertEqual(new_group.page_permissions.all().count(), 0)
        self.assertEqual(new_group.collection_permissions.filter(permission=self.add_doc_permission).count(), 0)

    def test_custom_permissions_hidden(self):
        if False:
            i = 10
            return i + 15
        Permission.objects.exclude(Q(codename__startswith='add') | Q(codename__startswith='change') | Q(codename__startswith='delete') | Q(codename__startswith='publish')).delete()
        response = self.get()
        self.assertInHTML('Custom permissions', response.content.decode(), count=0)

    def test_custom_permissions_shown(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertInHTML('Custom permissions', response.content.decode())

    def test_show_publish_permissions(self):
        if False:
            return 10
        response = self.get()
        html = response.content.decode()
        self.assertInHTML('<th>Publish</th>', html)
        self.assertInHTML('Can publish draft state model', html)
        self.assertInHTML('Can publish draft state custom primary key model', html)
        self.assertNotInHTML('Can publish advert', html)

    def test_hide_publish_permissions(self):
        if False:
            print('Hello World!')
        Permission.objects.filter(codename__startswith='publish').delete()
        response = self.get()
        html = response.content.decode()
        self.assertNotInHTML('<th>Publish</th>', html)
        self.assertNotInHTML('Can publish draft state model', html)
        self.assertNotInHTML('Can publish draft state custom primary key model', html)
        self.assertNotInHTML('Can publish advert', html)

class TestGroupEditView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_group = Group.objects.create(name='test group')
        self.root_page = Page.objects.get(pk=1)
        self.root_add_permission = GroupPagePermission.objects.create(page=self.root_page, permission_type='add', group=self.test_group)
        self.home_page = Page.objects.get(pk=2)
        self.registered_permissions = Permission.objects.none()
        for fn in hooks.get_hooks('register_permissions'):
            self.registered_permissions = self.registered_permissions | fn()
        self.existing_permission = self.registered_permissions.order_by('pk')[0]
        self.another_permission = self.registered_permissions.order_by('pk')[1]
        self.test_group.permissions.add(self.existing_permission)
        self.root_collection = Collection.get_first_root_node()
        self.evil_plans_collection = self.root_collection.add_child(name='Evil plans')
        self.add_doc_permission = Permission.objects.get(content_type__app_label='wagtaildocs', codename='add_document')
        self.change_doc_permission = Permission.objects.get(content_type__app_label='wagtaildocs', codename='change_document')
        GroupCollectionPermission.objects.create(group=self.test_group, collection=self.evil_plans_collection, permission=self.add_doc_permission)
        self.user = self.login()

    def get(self, params={}, group_id=None):
        if False:
            return 10
        return self.client.get(reverse('wagtailusers_groups:edit', args=(group_id or self.test_group.pk,)), params)

    def post(self, post_data={}, group_id=None):
        if False:
            i = 10
            return i + 15
        post_defaults = {'name': 'test group', 'permissions': [self.existing_permission.pk], 'page_permissions-TOTAL_FORMS': ['1'], 'page_permissions-MAX_NUM_FORMS': ['1000'], 'page_permissions-INITIAL_FORMS': ['1'], 'page_permissions-0-page': [self.root_page.pk], 'page_permissions-0-permissions': ['add_page'], 'document_permissions-TOTAL_FORMS': ['1'], 'document_permissions-MAX_NUM_FORMS': ['1000'], 'document_permissions-INITIAL_FORMS': ['1'], 'document_permissions-0-collection': [self.evil_plans_collection.pk], 'document_permissions-0-permissions': [self.add_doc_permission.pk], 'image_permissions-TOTAL_FORMS': ['0'], 'image_permissions-MAX_NUM_FORMS': ['1000'], 'image_permissions-INITIAL_FORMS': ['0'], 'collection_permissions-TOTAL_FORMS': ['0'], 'collection_permissions-MAX_NUM_FORMS': ['1000'], 'collection_permissions-INITIAL_FORMS': ['0']}
        for (k, v) in post_defaults.items():
            post_data[k] = post_data.get(k, v)
        return self.client.post(reverse('wagtailusers_groups:edit', args=(group_id or self.test_group.pk,)), post_data)

    def add_non_registered_perm(self):
        if False:
            while True:
                i = 10
        self.non_registered_perms = Permission.objects.exclude(pk__in=self.registered_permissions)
        self.non_registered_perm = self.non_registered_perms[0]
        self.test_group.permissions.add(self.non_registered_perm)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/groups/edit.html')
        self.assertBreadcrumbsNotRendered(response.content)
        url_finder = AdminURLFinder(self.user)
        expected_url = '/admin/groups/edit/%d/' % self.test_group.id
        self.assertEqual(url_finder.get_edit_url(self.test_group), expected_url)

    def test_nonexistant_group_redirect(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.get(group_id=100000).status_code, 404)

    def test_group_edit(self):
        if False:
            i = 10
            return i + 15
        response = self.post({'name': 'test group edited'})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        group = Group.objects.get(pk=self.test_group.pk)
        self.assertEqual(group.name, 'test group edited')

    def test_group_edit_validation_error(self):
        if False:
            while True:
                i = 10
        response = self.post({'name': ''})
        self.assertEqual(response.status_code, 200)

    def test_group_edit_adding_page_permissions_same_page(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.test_group.page_permissions.count(), 1)
        response = self.post({'page_permissions-0-permissions': ['add_page', 'publish_page', 'change_page']})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.page_permissions.count(), 3)

    def test_group_edit_adding_document_permissions_same_collection(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 1)
        response = self.post({'document_permissions-0-permissions': [self.add_doc_permission.pk, self.change_doc_permission.pk]})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 2)

    def test_group_edit_adding_document_permissions_different_collection(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 1)
        response = self.post({'document_permissions-TOTAL_FORMS': ['2'], 'document_permissions-1-collection': [self.root_collection.pk], 'document_permissions-1-permissions': [self.add_doc_permission.pk, self.change_doc_permission.pk]})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 3)

    def test_group_edit_deleting_page_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.test_group.page_permissions.count(), 1)
        response = self.post({'page_permissions-0-DELETE': ['1']})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.page_permissions.count(), 0)

    def test_group_edit_deleting_document_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 1)
        response = self.post({'document_permissions-0-DELETE': ['1']})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.collection_permissions.filter(permission__content_type__app_label='wagtaildocs').count(), 0)

    def test_group_edit_loads_with_django_permissions_shown(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertTagInHTML('<input name="permissions" type="checkbox" checked value="%s">' % self.existing_permission.id, response.content.decode(), allow_extra_attrs=True)

    def test_group_edit_displays_collection_nesting(self):
        if False:
            for i in range(10):
                print('nop')
        self.evil_plans_collection.add_child(instance=Collection(name='Eviler Plans'))
        response = self.get()
        self.assertContains(response, '>&nbsp;&nbsp;&nbsp;&nbsp;&#x21b3 Eviler Plans', count=4)

    def test_group_edit_loads_with_page_permissions_shown(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.test_group.page_permissions.count(), 1)
        response = self.get()
        page_permissions_formset = response.context['permission_panels'][0]
        self.assertEqual(page_permissions_formset.management_form['INITIAL_FORMS'].value(), 1)
        self.assertEqual(page_permissions_formset.forms[0]['page'].value(), self.root_page.pk)
        self.assertEqual(page_permissions_formset.forms[0]['permissions'].value(), ['add_page'])
        GroupPagePermission.objects.create(page=self.root_page, permission_type='change', group=self.test_group)
        self.assertEqual(self.test_group.page_permissions.count(), 2)
        response = self.get()
        page_permissions_formset = response.context['permission_panels'][0]
        self.assertEqual(page_permissions_formset.management_form['INITIAL_FORMS'].value(), 1)
        self.assertEqual(len(page_permissions_formset.forms), 1)
        self.assertEqual(page_permissions_formset.forms[0]['page'].value(), self.root_page.pk)
        self.assertEqual(set(page_permissions_formset.forms[0]['permissions'].value()), {'add_page', 'change_page'})
        GroupPagePermission.objects.create(page=self.home_page, permission_type='change', group=self.test_group)
        self.assertEqual(self.test_group.page_permissions.count(), 3)
        response = self.get()
        page_permissions_formset = response.context['permission_panels'][0]
        self.assertEqual(page_permissions_formset.management_form['INITIAL_FORMS'].value(), 2)
        self.assertEqual(page_permissions_formset.forms[0]['page'].value(), self.root_page.pk)
        self.assertEqual(set(page_permissions_formset.forms[0]['permissions'].value()), {'add_page', 'change_page'})
        self.assertEqual(page_permissions_formset.forms[1]['page'].value(), self.home_page.pk)
        self.assertEqual(page_permissions_formset.forms[1]['permissions'].value(), ['change_page'])

    def test_duplicate_page_permissions_error(self):
        if False:
            return 10
        response = self.post({'page_permissions-1-page': [self.root_page.pk], 'page_permissions-1-permissions': ['change_page'], 'page_permissions-TOTAL_FORMS': ['2']})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['permission_panels'][0].non_form_errors)

    def test_duplicate_document_permissions_error(self):
        if False:
            print('Hello World!')
        response = self.post({'document_permissions-1-page': [self.evil_plans_collection.pk], 'document_permissions-1-permissions': [self.change_doc_permission], 'document_permissions-TOTAL_FORMS': ['2']})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any((hasattr(panel, 'non_form_errors') and panel.non_form_errors for panel in response.context['permission_panels'])))

    def test_group_add_registered_django_permissions(self):
        if False:
            return 10
        self.assertEqual(self.test_group.permissions.count(), 1)
        response = self.post({'permissions': [self.existing_permission.pk, self.another_permission.pk]})
        self.assertRedirects(response, reverse('wagtailusers_groups:index'))
        self.assertEqual(self.test_group.permissions.count(), 2)

    def test_group_retains_non_registered_permissions_when_editing(self):
        if False:
            return 10
        self.add_non_registered_perm()
        original_permissions = list(self.test_group.permissions.all())
        self.post()
        self.assertEqual(list(self.test_group.permissions.all()), original_permissions)
        self.assertEqual(self.test_group.permissions.count(), 2)

    def test_group_retains_non_registered_permissions_when_adding(self):
        if False:
            i = 10
            return i + 15
        self.add_non_registered_perm()
        self.post({'permissions': [self.existing_permission.pk, self.another_permission.pk]})
        self.assertEqual(self.test_group.permissions.count(), 3)
        self.assertIn(self.non_registered_perm, self.test_group.permissions.all())

    def test_group_retains_non_registered_permissions_when_deleting(self):
        if False:
            i = 10
            return i + 15
        self.add_non_registered_perm()
        self.post({'permissions': []})
        self.assertEqual(self.test_group.permissions.count(), 1)
        self.assertEqual(self.test_group.permissions.all()[0], self.non_registered_perm)

    def test_is_custom_permission_checked(self):
        if False:
            for i in range(10):
                print('nop')
        custom_permission = Permission.objects.get(codename='view_fancysnippet')
        self.test_group.permissions.add(custom_permission)
        response = self.get()
        self.assertTagInHTML('<input type="checkbox" name="permissions" value="%s" checked>' % custom_permission.id, response.content.decode())

    def test_show_publish_permissions(self):
        if False:
            print('Hello World!')
        response = self.get()
        html = response.content.decode()
        self.assertInHTML('<th>Publish</th>', html)
        self.assertInHTML('Can publish draft state model', html)
        self.assertInHTML('Can publish draft state custom primary key model', html)
        self.assertNotInHTML('Can publish advert', html)

    def test_hide_publish_permissions(self):
        if False:
            return 10
        Permission.objects.filter(codename__startswith='publish').delete()
        response = self.get()
        html = response.content.decode()
        self.assertNotInHTML('<th>Publish</th>', html)
        self.assertNotInHTML('Can publish draft state model', html)
        self.assertNotInHTML('Can publish draft state custom primary key model', html)
        self.assertNotInHTML('Can publish advert', html)

    def test_group_edit_loads_with_django_permissions_in_order(self):
        if False:
            return 10

        def object_position(object_perms):
            if False:
                i = 10
                return i + 15

            def flatten(perm_set):
                if False:
                    for i in range(10):
                        print('nop')
                for v in perm_set.values():
                    if isinstance(v, list):
                        yield from v
                    else:
                        yield v
            return [(perm.content_type.app_label, perm.content_type.model) for perm_set in object_perms for perm in [next((v for v in flatten(perm_set) if 'perm' in v))['perm']]]
        register_permission_order('snippetstests.fancysnippet', order=100)
        register_permission_order('snippetstests.standardsnippet', order=110)
        response = self.get()
        object_positions = object_position(response.context['object_perms'])
        self.assertEqual(object_positions[0], ('snippetstests', 'fancysnippet'), msg='Configured object permission order is incorrect')
        self.assertEqual(object_positions[1], ('snippetstests', 'standardsnippet'), msg='Configured object permission order is incorrect')
        register_permission_order('snippetstests.standardsnippet', order=90)
        response = self.get()
        object_positions = object_position(response.context['object_perms'])
        self.assertEqual(object_positions[0], ('snippetstests', 'standardsnippet'), msg='Configured object permission order is incorrect')
        self.assertEqual(object_positions[1], ('snippetstests', 'fancysnippet'), msg='Configured object permission order is incorrect')
        self.assertEqual(object_positions[2:], sorted(object_positions[2:]), msg='Default object permission order is incorrect')

class TestGroupViewSet(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.app_config = apps.get_app_config('wagtailusers')

    def test_get_group_viewset_cls(self):
        if False:
            print('Hello World!')
        self.assertIs(get_group_viewset_cls(self.app_config), GroupViewSet)

    def test_get_group_viewset_cls_with_custom_form(self):
        if False:
            i = 10
            return i + 15
        with unittest.mock.patch.object(self.app_config, 'group_viewset', new='wagtail.users.tests.CustomGroupViewSet'):
            group_viewset = get_group_viewset_cls(self.app_config)
        self.assertIs(group_viewset, CustomGroupViewSet)
        self.assertEqual(group_viewset.icon, 'custom-icon')

    def test_get_group_viewset_cls_custom_form_invalid_value(self):
        if False:
            print('Hello World!')
        with unittest.mock.patch.object(self.app_config, 'group_viewset', new='asdfasdf'):
            with self.assertRaises(ImproperlyConfigured) as exc_info:
                get_group_viewset_cls(self.app_config)
            self.assertIn("asdfasdf doesn't look like a module path", str(exc_info.exception))

    def test_get_group_viewset_cls_custom_form_does_not_exist(self):
        if False:
            i = 10
            return i + 15
        with unittest.mock.patch.object(self.app_config, 'group_viewset', new='wagtail.users.tests.CustomClassDoesNotExist'):
            with self.assertRaises(ImproperlyConfigured) as exc_info:
                get_group_viewset_cls(self.app_config)
            self.assertIn('Module "wagtail.users.tests" does not define a "CustomClassDoesNotExist" attribute/class', str(exc_info.exception))

class TestAuthorisationIndexView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self._user = self.create_user(username='auth_user', password='password')
        self._user.user_permissions.add(Permission.objects.get(codename='access_admin'))
        self.login(username='auth_user', password='password')

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_users:index'))

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')

    def test_authorised(self):
        if False:
            print('Hello World!')
        for permission in ('add', 'change', 'delete'):
            permission_name = f'{permission}_{AUTH_USER_MODEL_NAME.lower()}'
            permission_object = Permission.objects.get(codename=permission_name)
            self._user.user_permissions.add(permission_object)
            response = self.get()
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'wagtailusers/users/index.html')
            self.assertContains(response, 'auth_user')
            self._user.user_permissions.remove(permission_object)

class TestAuthorisationCreateView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._user = self.create_user(username='auth_user', password='password')
        self._user.user_permissions.add(Permission.objects.get(codename='access_admin'))
        self.login(username='auth_user', password='password')

    def get(self, params={}):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailusers_users:add'), params)

    def post(self, post_data={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.post(reverse('wagtailusers_users:add'), post_data)

    def gain_permissions(self):
        if False:
            while True:
                i = 10
        self._user.user_permissions.add(Permission.objects.get(content_type__app_label=AUTH_USER_APP_LABEL, codename=f'add_{AUTH_USER_MODEL_NAME.lower()}'))

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')

    def test_authorised(self):
        if False:
            while True:
                i = 10
        self.gain_permissions()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/create.html')

    def test_unauthorised_post(self):
        if False:
            while True:
                i = 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'password', 'password2': 'password'})
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')
        user = get_user_model().objects.filter(email='test@user.com')
        self.assertFalse(user.exists())

    def test_authorised_post(self):
        if False:
            return 10
        self.gain_permissions()
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Test', 'last_name': 'User', 'password1': 'password', 'password2': 'password'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.filter(email='test@user.com')
        self.assertTrue(user.exists())

class TestAuthorisationEditView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._user = self.create_user(username='auth_user', password='password')
        self._user.user_permissions.add(Permission.objects.get(codename='access_admin'))
        self.login(username='auth_user', password='password')
        self.test_user = self.create_user(username='testuser', email='testuser@email.com', first_name='Original', last_name='User', password='password')

    def get(self, params={}, user_id=None):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailusers_users:edit', args=(user_id or self.test_user.pk,)), params)

    def post(self, post_data={}, user_id=None):
        if False:
            for i in range(10):
                print('nop')
        return self.client.post(reverse('wagtailusers_users:edit', args=(user_id or self.test_user.pk,)), post_data)

    def gain_permissions(self):
        if False:
            while True:
                i = 10
        self._user.user_permissions.add(Permission.objects.get(content_type__app_label=AUTH_USER_APP_LABEL, codename=f'change_{AUTH_USER_MODEL_NAME.lower()}'))

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')

    def test_authorised_get(self):
        if False:
            i = 10
            return i + 15
        self.gain_permissions()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/edit.html')

    def test_unauthorised_post(self):
        if False:
            while True:
                i = 10
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'newpassword', 'password2': 'newpassword', 'is_active': 'on'})
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertNotEqual(user.first_name, 'Edited')
        self.assertFalse(user.check_password('newpassword'))

    def test_authorised_post(self):
        if False:
            for i in range(10):
                print('nop')
        self.gain_permissions()
        response = self.post({'username': 'testuser', 'email': 'test@user.com', 'first_name': 'Edited', 'last_name': 'User', 'password1': 'newpassword', 'password2': 'newpassword', 'is_active': 'on'})
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.get(pk=self.test_user.pk)
        self.assertEqual(user.first_name, 'Edited')
        self.assertTrue(user.check_password('newpassword'))

class TestAuthorisationDeleteView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._user = self.create_user(username='auth_user', password='password')
        self._user.user_permissions.add(Permission.objects.get(codename='access_admin'))
        self.login(username='auth_user', password='password')
        self.test_user = self.create_user(username='test_user', email='test_user@email.com', password='password')

    def get(self, params={}):
        if False:
            print('Hello World!')
        return self.client.get(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)), params)

    def post(self, post_data={}):
        if False:
            return 10
        return self.client.post(reverse('wagtailusers_users:delete', args=(self.test_user.pk,)), post_data)

    def gain_permissions(self):
        if False:
            while True:
                i = 10
        self._user.user_permissions.add(Permission.objects.get(content_type__app_label=AUTH_USER_APP_LABEL, codename=f'delete_{AUTH_USER_MODEL_NAME.lower()}'))

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')

    def test_authorised_get(self):
        if False:
            print('Hello World!')
        self.gain_permissions()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/users/confirm_delete.html')

    def test_unauthorised_post(self):
        if False:
            i = 10
            return i + 15
        response = self.post()
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertEqual(response.context['message'], 'Sorry, you do not have permission to access this area.')
        user = get_user_model().objects.filter(email='test_user@email.com')
        self.assertTrue(user.exists())

    def test_authorised_post(self):
        if False:
            return 10
        self.gain_permissions()
        response = self.post()
        self.assertRedirects(response, reverse('wagtailusers_users:index'))
        user = get_user_model().objects.filter(email='test_user@email.com')
        self.assertFalse(user.exists())