from unittest import mock
from django.contrib.auth.models import Permission
from django.http import HttpRequest, HttpResponse
from django.test import TestCase
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.views.pages.bulk_actions.page_bulk_action import PageBulkAction
from wagtail.models import Page
from wagtail.signals import page_unpublished
from wagtail.test.testapp.models import SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestBulkUnpublish(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=2)
        self.child_pages = [SimplePage(title=f'Hello world!-{i}', slug=f'hello-world-{i}', content=f'hello-{i}') for i in range(1, 5)]
        self.pages_to_be_unpublished = self.child_pages[:3]
        self.pages_not_to_be_unpublished = self.child_pages[3:]
        for child_page in self.child_pages:
            self.root_page.add_child(instance=child_page)
        self.url = reverse('wagtail_bulk_action', args=('wagtailcore', 'page', 'unpublish')) + '?'
        for child_page in self.pages_to_be_unpublished:
            self.url += f'&id={child_page.id}'
        self.redirect_url = reverse('wagtailadmin_explore', args=(self.root_page.id,))
        self.user = self.login()

    def test_unpublish_view(self):
        if False:
            i = 10
            return i + 15
        '\n        This tests that the unpublish view responds with an unpublish confirm page\n        '
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/bulk_actions/confirm_bulk_unpublish.html')

    def test_unpublish_view_invalid_page_id(self):
        if False:
            return 10
        '\n        This tests that the unpublish view returns an error if the page id is invalid\n        '
        response = self.client.get(reverse('wagtail_bulk_action', args=('wagtailcore', 'page', 'unpublish')))
        self.assertEqual(response.status_code, 404)

    def test_unpublish_view_bad_permissions(self):
        if False:
            while True:
                i = 10
        "\n        This tests that the unpublish view doesn't allow users without unpublish permissions\n        "
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertInHTML("<p>You don't have permission to unpublish these pages</p>", html)
        for child_page in self.pages_to_be_unpublished:
            self.assertInHTML(f'<li>{child_page.title}</li>', html)

    def test_unpublish_view_post(self):
        if False:
            i = 10
            return i + 15
        '\n        This posts to the unpublish view and checks that the page was unpublished\n        '
        mock_handler = mock.MagicMock()
        page_unpublished.connect(mock_handler)
        try:
            response = self.client.post(self.url)
            self.assertEqual(response.status_code, 302)
            for child_page in self.pages_to_be_unpublished:
                self.assertFalse(SimplePage.objects.get(id=child_page.id).live)
            for child_page in self.pages_not_to_be_unpublished:
                self.assertTrue(SimplePage.objects.get(id=child_page.id).live)
            self.assertEqual(mock_handler.call_count, len(self.pages_to_be_unpublished))
            for (i, child_page) in enumerate(self.pages_to_be_unpublished):
                mock_call = mock_handler.mock_calls[i][2]
                self.assertEqual(mock_call['sender'], child_page.specific_class)
                self.assertEqual(mock_call['instance'], child_page)
                self.assertIsInstance(mock_call['instance'], child_page.specific_class)
        finally:
            page_unpublished.disconnect(mock_handler)

    def test_after_unpublish_page(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, action_type, pages, action_class_instance):
            if False:
                return 10
            self.assertEqual(action_type, 'unpublish')
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(action_class_instance, PageBulkAction)
            for (i, page) in enumerate(pages):
                self.assertEqual(page.id, self.pages_to_be_unpublished[i].id)
            return HttpResponse('Overridden!')
        with self.register_hook('after_bulk_action', hook_func):
            response = self.client.post(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        for child_page in self.pages_to_be_unpublished:
            child_page.refresh_from_db()
            self.assertEqual(child_page.status_string, _('draft'))

    def test_before_unpublish_page(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, action_type, pages, action_class_instance):
            if False:
                while True:
                    i = 10
            self.assertEqual(action_type, 'unpublish')
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(action_class_instance, PageBulkAction)
            for (i, page) in enumerate(pages):
                self.assertEqual(page.id, self.pages_to_be_unpublished[i].id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_bulk_action', hook_func):
            response = self.client.post(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_unpublish_descendants_view(self):
        if False:
            print('Hello World!')
        "\n        This tests that the unpublish view responds with an unpublish confirm page that does not contain the form field 'include_descendants'\n        "
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/bulk_actions/confirm_bulk_unpublish.html')
        self.assertContains(response, 'name="include_descendants"', count=0)

class TestBulkUnpublishIncludingDescendants(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        self.child_pages = [SimplePage(title=f'Hello world!-{i}', slug=f'hello-world-{i}', content=f'hello-{i}') for i in range(1, 5)]
        self.pages_to_be_unpublished = self.child_pages[:3]
        self.pages_not_to_be_unpublished = self.child_pages[3:]
        for child_page in self.child_pages:
            self.root_page.add_child(instance=child_page)
        self.grandchildren_pages = {self.pages_to_be_unpublished[0]: [SimplePage(title='Hello world!-a', slug='hello-world-a', content='hello-a')], self.pages_to_be_unpublished[1]: [SimplePage(title='Hello world!-b', slug='hello-world-b', content='hello-b'), SimplePage(title='Hello world!-c', slug='hello-world-c', content='hello-c')]}
        for (child_page, grandchild_pages) in self.grandchildren_pages.items():
            for grandchild_page in grandchild_pages:
                child_page.add_child(instance=grandchild_page)
        self.url = reverse('wagtail_bulk_action', args=('wagtailcore', 'page', 'unpublish')) + '?'
        for child_page in self.pages_to_be_unpublished:
            self.url += f'&id={child_page.id}'
        self.redirect_url = reverse('wagtailadmin_explore', args=(self.root_page.id,))
        self.user = self.login()

    def test_unpublish_descendants_view(self):
        if False:
            i = 10
            return i + 15
        "\n        This tests that the unpublish view responds with an unpublish confirm page that contains the form field 'include_descendants'\n        "
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/bulk_actions/confirm_bulk_unpublish.html')
        self.assertContains(response, 'name="include_descendants"')

    def test_unpublish_include_children_view_post(self):
        if False:
            while True:
                i = 10
        '\n        This posts to the unpublish view and checks that the page and its descendants were unpublished\n        '
        response = self.client.post(self.url, {'include_descendants': 'on'})
        self.assertEqual(response.status_code, 302)
        for child_page in self.pages_to_be_unpublished:
            self.assertFalse(SimplePage.objects.get(id=child_page.id).live)
        for child_page in self.pages_not_to_be_unpublished:
            self.assertTrue(SimplePage.objects.get(id=child_page.id).live)
        for grandchild_pages in self.grandchildren_pages.values():
            for grandchild_page in grandchild_pages:
                self.assertFalse(SimplePage.objects.get(id=grandchild_page.id).live)

    def test_unpublish_not_include_children_view_post(self):
        if False:
            while True:
                i = 10
        '\n        This posts to the unpublish view and checks that the page was unpublished but its descendants were not\n        '
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 302)
        for child_page in self.pages_to_be_unpublished:
            self.assertFalse(SimplePage.objects.get(id=child_page.id).live)
        for grandchild_pages in self.grandchildren_pages.values():
            for grandchild_page in grandchild_pages:
                self.assertTrue(SimplePage.objects.get(id=grandchild_page.id).live)