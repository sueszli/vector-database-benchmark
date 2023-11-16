from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.test import TestCase
from django.urls import reverse
from wagtail.test.utils import WagtailTestUtils
from wagtail.users.views.bulk_actions.user_bulk_action import UserBulkAction
User = get_user_model()

class TestUserSetActiveStateView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_users = [self.create_user(username=f'testuser-{i}', email=f'testuser{i}@email.com', password=f'password-{i}') for i in range(1, 6)]
        for (i, user) in enumerate(self.test_users):
            user.is_active = i & 1
            user.save()
        self.current_user = self.login()
        self.url = reverse('wagtail_bulk_action', args=(User._meta.app_label, User._meta.model_name, 'set_active_state')) + '?'
        self.self_toggle_url = self.url + f'id={self.current_user.pk}'
        for user in self.test_users:
            self.url += f'id={user.pk}&'
        self.make_active_data = {'mark_as_active': True}
        self.make_inactive_data = {'mark_as_active': False}

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailusers/bulk_actions/confirm_bulk_set_active_state.html')

    def test_bulk_toggle(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.post(self.url, self.make_active_data)
        self.assertEqual(response.status_code, 302)
        for user in self.test_users:
            self.assertTrue(User.objects.get(email=user.email).is_active)

    def test_user_cannot_mark_self_as_inactive(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(self.self_toggle_url)
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertInHTML('<p>You cannot change your own active status</p>', html)
        needle = '<ul>'
        needle += f'<li>{self.current_user.email}</li>'
        needle += '</ul>'
        self.assertInHTML(needle, html)
        self.assertTrue(User.objects.get(pk=self.current_user.pk).is_active)

    def test_before_toggle_user_hook_post(self):
        if False:
            print('Hello World!')

        def hook_func(request, action_type, users, action_class_instance):
            if False:
                i = 10
                return i + 15
            self.assertEqual(action_type, 'set_active_state')
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(action_class_instance, UserBulkAction)
            self.assertCountEqual([user.pk for user in self.test_users], [user.pk for user in users])
            return HttpResponse('Overridden!')
        with self.register_hook('before_bulk_action', hook_func):
            response = self.client.post(self.url, self.make_active_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_after_toggle_user_hook(self):
        if False:
            return 10

        def hook_func(request, action_type, users, action_class_instance):
            if False:
                print('Hello World!')
            self.assertEqual(action_type, 'set_active_state')
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(action_class_instance, UserBulkAction)
            self.assertCountEqual([user.pk for user in self.test_users], [user.pk for user in users])
            return HttpResponse('Overridden!')
        with self.register_hook('after_bulk_action', hook_func):
            response = self.client.post(self.url, self.make_active_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        for user in self.test_users:
            self.assertTrue(User.objects.get(email=user.email).is_active)