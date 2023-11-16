from unittest import mock
from django.contrib.auth.models import Permission
from django.db.models.signals import post_delete, pre_delete
from django.http import HttpRequest, HttpResponse
from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail.models import Page
from wagtail.signals import page_unpublished
from wagtail.test.testapp.models import SimplePage, StandardChild, StandardIndex
from wagtail.test.utils import WagtailTestUtils

class TestPageDelete(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.root_page = Page.objects.get(id=2)
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.child_page)
        self.child_index = StandardIndex(title='Hello index', slug='hello-index')
        self.root_page.add_child(instance=self.child_index)
        self.grandchild_page = StandardChild(title='Hello Kitty', slug='hello-kitty')
        self.child_index.add_child(instance=self.grandchild_page)
        self.user = self.login()

    def test_page_delete(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(SimplePage.objects.filter(id=self.child_page.id).exists())

    @override_settings(WAGTAILADMIN_UNSAFE_PAGE_DELETION_LIMIT=10)
    def test_confirm_delete_scenario_1(self):
        if False:
            print('Hello World!')
        child_1 = SimplePage(title='child 1', slug='child-1', content='hello')
        self.child_page.add_child(instance=child_1)
        child_2 = SimplePage(title='child 2', slug='child-2', content='hello')
        self.child_page.add_child(instance=child_2)
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<input type="text" name="confirm_site_name"')
        self.assertTrue(SimplePage.objects.filter(id=self.child_page.id).exists())
        self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertFalse(SimplePage.objects.filter(id=self.child_page.id).exists())

    @override_settings(WAGTAILADMIN_UNSAFE_PAGE_DELETION_LIMIT=3)
    @override_settings(WAGTAIL_SITE_NAME='mysite')
    def test_confirm_delete_scenario_2(self):
        if False:
            for i in range(10):
                print('nop')
        child_1 = SimplePage(title='child 1', slug='child-1', content='hello')
        self.child_page.add_child(instance=child_1)
        child_2 = SimplePage(title='child 2', slug='child-2', content='hello')
        self.child_page.add_child(instance=child_2)
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This action will delete total <b>3</b> pages.')
        self.assertContains(response, 'Please type <b>mysite</b> to confirm.')
        self.assertContains(response, '<input type="text" name="confirm_site_name"')
        self.assertTrue(SimplePage.objects.filter(id=self.child_page.id).exists())

    @override_settings(WAGTAILADMIN_UNSAFE_PAGE_DELETION_LIMIT=3)
    @override_settings(WAGTAIL_SITE_NAME='mysite')
    def test_confirm_delete_scenario_3(self):
        if False:
            print('Hello World!')
        child_1 = SimplePage(title='child 1', slug='child-1', content='hello')
        self.child_page.add_child(instance=child_1)
        child_2 = SimplePage(title='child 2', slug='child-2', content='hello')
        self.child_page.add_child(instance=child_2)
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)), data={'confirm_site_name': 'random'})
        self.assertEqual(response.status_code, 200)
        messages = [m.message for m in response.context['messages']]
        self.assertEqual(len(messages), 1)
        self.assertContains(response, 'This action will delete total <b>3</b> pages.')
        self.assertContains(response, 'Please type <b>mysite</b> to confirm.')
        self.assertContains(response, '<input type="text" name="confirm_site_name"')
        self.assertTrue(SimplePage.objects.filter(id=self.child_page.id).exists())

    @override_settings(WAGTAILADMIN_UNSAFE_PAGE_DELETION_LIMIT=3)
    @override_settings(WAGTAIL_SITE_NAME='mysite')
    def test_confirm_delete_scenario_4(self):
        if False:
            for i in range(10):
                print('nop')
        child_1 = SimplePage(title='child 1', slug='child-1', content='hello')
        self.child_page.add_child(instance=child_1)
        child_2 = SimplePage(title='child 2', slug='child-2', content='hello')
        self.child_page.add_child(instance=child_2)
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)), data={'confirm_site_name': 'mysite'})
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertFalse(SimplePage.objects.filter(id=self.child_page.id).exists())
        self.assertFalse(SimplePage.objects.filter(id=child_1.id).exists())
        self.assertFalse(SimplePage.objects.filter(id=child_2.id).exists())

    def test_page_delete_specific_admin_title(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.child_page.get_admin_display_title())

    def test_page_delete_bad_permissions(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(SimplePage.objects.filter(id=self.child_page.id).exists())

    def test_page_delete_post(self):
        if False:
            i = 10
            return i + 15
        mock_handler = mock.MagicMock()
        page_unpublished.connect(mock_handler)
        try:
            response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')
            self.assertEqual(Page.objects.filter(path__startswith=self.root_page.path, slug='hello-world').count(), 0)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], self.child_page.specific_class)
            self.assertEqual(mock_call['instance'], self.child_page)
            self.assertIsInstance(mock_call['instance'], self.child_page.specific_class)
        finally:
            page_unpublished.disconnect(mock_handler)

    def test_page_delete_notlive_post(self):
        if False:
            while True:
                i = 10
        self.child_page.live = False
        self.child_page.save()
        mock_handler = mock.MagicMock()
        page_unpublished.connect(mock_handler)
        try:
            response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')
            self.assertEqual(Page.objects.filter(path__startswith=self.root_page.path, slug='hello-world').count(), 0)
            self.assertEqual(mock_handler.call_count, 0)
        finally:
            page_unpublished.disconnect(mock_handler)

    def test_subpage_deletion(self):
        if False:
            i = 10
            return i + 15
        unpublish_signals_received = []
        pre_delete_signals_received = []
        post_delete_signals_received = []

        def page_unpublished_handler(sender, instance, **kwargs):
            if False:
                return 10
            unpublish_signals_received.append((sender, instance.pk))

        def pre_delete_handler(sender, instance, **kwargs):
            if False:
                i = 10
                return i + 15
            pre_delete_signals_received.append((sender, instance.pk))

        def post_delete_handler(sender, instance, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            post_delete_signals_received.append((sender, instance.pk))
        page_unpublished.connect(page_unpublished_handler)
        pre_delete.connect(pre_delete_handler)
        post_delete.connect(post_delete_handler)
        try:
            response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_index.id,)))
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')
            self.assertFalse(StandardIndex.objects.filter(id=self.child_index.id).exists())
            self.assertFalse(Page.objects.filter(id=self.child_index.id).exists())
            self.assertFalse(StandardChild.objects.filter(id=self.grandchild_page.id).exists())
            self.assertFalse(Page.objects.filter(id=self.grandchild_page.id).exists())
            self.assertIn((StandardIndex, self.child_index.id), unpublish_signals_received)
            self.assertIn((StandardChild, self.grandchild_page.id), unpublish_signals_received)
            self.assertIn((StandardIndex, self.child_index.id), pre_delete_signals_received)
            self.assertIn((StandardChild, self.grandchild_page.id), pre_delete_signals_received)
            self.assertIn((StandardIndex, self.child_index.id), post_delete_signals_received)
            self.assertIn((StandardChild, self.grandchild_page.id), post_delete_signals_received)
        finally:
            page_unpublished.disconnect(page_unpublished_handler)
            pre_delete.disconnect(pre_delete_handler)
            post_delete.disconnect(post_delete_handler)

    def test_before_delete_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                return 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_page', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_delete_page_hook_post(self):
        if False:
            print('Hello World!')

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_page', hook_func):
            response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Page.objects.filter(id=self.child_page.id).exists())

    def test_after_delete_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                print('Hello World!')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('after_delete_page', hook_func):
            response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertFalse(Page.objects.filter(id=self.child_page.id).exists())