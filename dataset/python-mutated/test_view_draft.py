from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.test import TestCase
from django.urls import reverse
from wagtail.models import Page
from wagtail.test.testapp.models import SimplePage, StreamPage
from wagtail.test.utils import WagtailTestUtils

class TestDraftAccess(WagtailTestUtils, TestCase):
    """Tests for the draft view access restrictions."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.child_page)
        self.stream_page = StreamPage(title='stream page', body=[('text', 'hello')])
        self.root_page.add_child(instance=self.stream_page)
        user = self.create_user(username='bob', password='password')
        user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))

    def test_draft_access_admin(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that admin can view draft.'
        self.user = self.login()
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)

    def test_page_without_preview_modes_is_unauthorised(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.stream_page.id,)))
        self.assertRedirects(response, '/admin/')

    def test_draft_access_unauthorised(self):
        if False:
            i = 10
            return i + 15
        "Test that user without edit/publish permission can't view draft."
        self.login(username='bob', password='password')
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 302)

    def test_draft_access_authorised(self):
        if False:
            print('Hello World!')
        'Test that user with edit permission can view draft.'
        user = get_user_model().objects.get(email='bob@example.com')
        user.groups.add(Group.objects.get(name='Moderators'))
        user.save()
        self.login(username='bob', password='password')
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)

    def test_middleware_response_is_returned(self):
        if False:
            i = 10
            return i + 15
        '\n        If middleware returns a response while serving a page preview, that response should be\n        returned back to the user\n        '
        self.login()
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.child_page.id,)), HTTP_USER_AGENT='EvilHacker')
        self.assertEqual(response.status_code, 403)

    def test_show_edit_link_in_userbar(self):
        if False:
            i = 10
            return i + 15
        self.login()
        response = self.client.get(reverse('wagtailadmin_pages:view_draft', args=(self.child_page.id,)))
        self.assertContains(response, 'Edit this page')
        self.assertContains(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))