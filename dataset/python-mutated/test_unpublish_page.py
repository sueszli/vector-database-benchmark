from unittest import mock
from django.contrib.auth.models import Permission
from django.http import HttpRequest, HttpResponse
from django.test import TestCase
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.models import Page
from wagtail.signals import page_unpublished
from wagtail.test.testapp.models import SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestPageUnpublish(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()
        self.root_page = Page.objects.get(id=2)
        self.page = SimplePage(title='Hello world!', slug='hello-world', content='hello', live=True)
        self.root_page.add_child(instance=self.page)

    def test_unpublish_view(self):
        if False:
            while True:
                i = 10
        '\n        This tests that the unpublish view responds with an unpublish confirm page\n        '
        response = self.client.get(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/confirm_unpublish.html')

    def test_unpublish_view_invalid_page_id(self):
        if False:
            print('Hello World!')
        '\n        This tests that the unpublish view returns an error if the page id is invalid\n        '
        response = self.client.get(reverse('wagtailadmin_pages:unpublish', args=(12345,)))
        self.assertEqual(response.status_code, 404)

    def test_unpublish_view_bad_permissions(self):
        if False:
            while True:
                i = 10
        "\n        This tests that the unpublish view doesn't allow users without unpublish permissions\n        "
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)))
        self.assertEqual(response.status_code, 302)

    def test_unpublish_view_post(self):
        if False:
            i = 10
            return i + 15
        '\n        This posts to the unpublish view and checks that the page was unpublished\n        '
        mock_handler = mock.MagicMock()
        page_unpublished.connect(mock_handler)
        try:
            response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)))
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            self.assertFalse(SimplePage.objects.get(id=self.page.id).live)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], self.page.specific_class)
            self.assertEqual(mock_call['instance'], self.page)
            self.assertIsInstance(mock_call['instance'], self.page.specific_class)
        finally:
            page_unpublished.disconnect(mock_handler)

    def test_after_unpublish_page(self):
        if False:
            while True:
                i = 10

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('after_unpublish_page', hook_func):
            post_data = {}
            response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.page.refresh_from_db()
        self.assertEqual(self.page.status_string, _('draft'))

    def test_before_unpublish_page(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_unpublish_page', hook_func):
            post_data = {}
            response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.page.refresh_from_db()
        self.assertEqual(self.page.status_string, _('live'))

    def test_unpublish_descendants_view(self):
        if False:
            i = 10
            return i + 15
        "\n        This tests that the unpublish view responds with an unpublish confirm page that does not contain the form field 'include_descendants'\n        "
        response = self.client.get(reverse('wagtailadmin_pages:unpublish', args=(self.page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/confirm_unpublish.html')
        self.assertNotContains(response, 'name="include_descendants"')

class TestPageUnpublishIncludingDescendants(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        self.root_page = Page.objects.get(id=2)
        self.test_page = self.root_page.add_child(instance=SimplePage(title='Hello world!', slug='hello-world', content='hello', live=True, has_unpublished_changes=False))
        self.test_child_page = self.test_page.add_child(instance=SimplePage(title='Child page', slug='child-page', content='hello', live=True, has_unpublished_changes=True))
        self.test_another_child_page = self.test_page.add_child(instance=SimplePage(title='Another Child page', slug='another-child-page', content='hello', live=True, has_unpublished_changes=True))

    def test_unpublish_descendants_view(self):
        if False:
            return 10
        "\n        This tests that the unpublish view responds with an unpublish confirm page that contains the form field 'include_descendants'\n        "
        response = self.client.get(reverse('wagtailadmin_pages:unpublish', args=(self.test_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/confirm_unpublish.html')
        self.assertContains(response, 'name="include_descendants"')

    def test_unpublish_include_children_view_post(self):
        if False:
            print('Hello World!')
        '\n        This posts to the unpublish view and checks that the page and its descendants were unpublished\n        '
        response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.test_page.id,)), {'include_descendants': 'on'})
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertFalse(SimplePage.objects.get(id=self.test_page.id).live)
        self.assertFalse(SimplePage.objects.get(id=self.test_child_page.id).live)
        self.assertFalse(SimplePage.objects.get(id=self.test_another_child_page.id).live)

    def test_unpublish_not_include_children_view_post(self):
        if False:
            print('Hello World!')
        '\n        This posts to the unpublish view and checks that the page was unpublished but its descendants were not\n        '
        response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.test_page.id,)), {})
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertFalse(SimplePage.objects.get(id=self.test_page.id).live)
        self.assertTrue(SimplePage.objects.get(id=self.test_child_page.id).live)
        self.assertTrue(SimplePage.objects.get(id=self.test_another_child_page.id).live)