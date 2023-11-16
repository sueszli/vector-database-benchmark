from django.contrib.auth.models import Permission
from django.test import TestCase
from django.urls import reverse
from wagtail.models import Page, PageLogEntry
from wagtail.test.testapp.models import SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestConvertAlias(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=2)
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.child_page)
        self.alias_page = self.child_page.create_alias(update_slug='alias-page')
        self.user = self.login()

    def test_convert_alias(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:convert_alias', args=[self.alias_page.id]))
        self.assertEqual(response.status_code, 200)

    def test_convert_alias_not_alias(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_pages:convert_alias', args=[self.child_page.id]))
        self.assertEqual(response.status_code, 404)

    def test_convert_alias_bad_permission(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:convert_alias', args=[self.alias_page.id]))
        self.assertRedirects(response, '/admin/')

    def test_post_convert_alias(self):
        if False:
            i = 10
            return i + 15
        response = self.client.post(reverse('wagtailadmin_pages:convert_alias', args=[self.alias_page.id]))
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.alias_page.id]))
        self.alias_page.refresh_from_db()
        self.assertIsNone(self.alias_page.alias_of)
        revision = self.alias_page.revisions.get()
        self.assertEqual(revision.user, self.user)
        self.assertEqual(self.alias_page.live_revision, revision)
        log = PageLogEntry.objects.get(action='wagtail.convert_alias')
        self.assertFalse(log.content_changed)
        self.assertEqual(log.data, {'page': {'id': self.alias_page.id, 'title': self.alias_page.get_admin_display_title()}})
        self.assertEqual(log.page, self.alias_page.page_ptr)
        self.assertEqual(log.revision, revision)
        self.assertEqual(log.user, self.user)