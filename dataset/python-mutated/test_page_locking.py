from django.contrib.auth.models import Group, Permission
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from django.utils.html import escape
from wagtail.models import Page
from wagtail.test.testapp.models import SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestLocking(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello', live=False)
        self.root_page.add_child(instance=self.child_page)

    def test_lock_post(self):
        if False:
            return 10
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_lock_get(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 405)
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_lock_post_already_locked(self):
        if False:
            while True:
                i = 10
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_lock_post_with_good_redirect(self):
        if False:
            return 10
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)), {'next': reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))})
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_lock_post_with_bad_redirect(self):
        if False:
            i = 10
            return i + 15
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)), {'next': 'http://www.google.co.uk'})
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_lock_post_bad_page(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(9999,)))
        self.assertEqual(response.status_code, 404)
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_lock_post_bad_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.post(reverse('wagtailadmin_pages:lock', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 302)
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_locked_pages_dashboard_panel(self):
        if False:
            i = 10
            return i + 15
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertContains(response, 'Your locked pages')
        self.assertContains(response, 'Unlock')

    def test_unlock_post(self):
        if False:
            return 10
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_unlock_get(self):
        if False:
            print('Hello World!')
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.get(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 405)
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_unlock_post_already_unlocked(self):
        if False:
            while True:
                i = 10
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_unlock_post_with_good_redirect(self):
        if False:
            while True:
                i = 10
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)), {'next': reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))}, follow=True)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertContains(response, escape("Page 'Hello world! (simple page)' is now unlocked."))
        self.assertNotContains(response, escape(("Page 'Hello world! (simple page)' is now unlocked.",)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_unlock_post_with_bad_redirect(self):
        if False:
            i = 10
            return i + 15
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)), {'next': 'http://www.google.co.uk'})
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)

    def test_unlock_post_bad_page(self):
        if False:
            for i in range(10):
                print('nop')
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(9999,)))
        self.assertEqual(response.status_code, 404)
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertEqual(page.locked_by, self.user)
        self.assertIsNotNone(page.locked_at)

    def test_unlock_post_bad_permissions(self):
        if False:
            i = 10
            return i + 15
        self.user.is_superuser = False
        self.user.groups.add(Group.objects.get(name='Editors'))
        self.user.save()
        self.child_page.locked = True
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 302)
        page = Page.objects.get(id=self.child_page.id)
        self.assertTrue(page.locked)
        self.assertIsNotNone(page.locked_at)

    def test_unlock_post_own_page_with_bad_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.groups.add(Group.objects.get(name='Editors'))
        self.user.save()
        self.child_page.locked = True
        self.child_page.locked_by = self.user
        self.child_page.locked_at = timezone.now()
        self.child_page.save()
        response = self.client.post(reverse('wagtailadmin_pages:unlock', args=(self.child_page.id,)), {'next': reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))})
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        page = Page.objects.get(id=self.child_page.id)
        self.assertFalse(page.locked)
        self.assertIsNone(page.locked_by)
        self.assertIsNone(page.locked_at)