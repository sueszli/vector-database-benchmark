from django.contrib.auth.models import Group, Permission
from django.http import HttpRequest, HttpResponse
from django.test import TestCase
from django.urls import reverse
from wagtail.models import GroupPagePermission, Page
from wagtail.test.testapp.models import EventPage, EventPageSpeaker, PageWithExcludedCopyField, SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestPageCopy(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=2)
        self.test_page = self.root_page.add_child(instance=SimplePage(title='Hello world!', slug='hello-world', content='hello', live=True, has_unpublished_changes=False))
        self.test_child_page = self.test_page.add_child(instance=SimplePage(title='Child page', slug='child-page', content='hello', live=True, has_unpublished_changes=True))
        self.test_unpublished_child_page = self.test_page.add_child(instance=SimplePage(title='Unpublished Child page', slug='unpublished-child-page', content='hello', live=False, has_unpublished_changes=True))
        self.user = self.login()

    def test_page_copy(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/copy.html')
        self.assertContains(response, 'New title')
        self.assertContains(response, 'New slug')
        self.assertContains(response, 'New parent page')
        self.assertContains(response, 'Copy subpages')
        self.assertContains(response, 'Publish copies')
        self.assertContains(response, 'Alias')

    def test_page_copy_bad_permissions(self):
        if False:
            return 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world', 'new_parent_page': str(self.test_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        publishers = Group.objects.create(name='Publishers')
        GroupPagePermission.objects.create(group=publishers, page=self.root_page, permission_type='publish')
        self.user.groups.add(publishers)
        self.user.save()
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world', 'new_parent_page': str(self.test_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        form = response.context['form']
        self.assertFalse(form.is_valid())
        self.assertIn('new_parent_page', form.errors)

    def test_page_copy_post(self):
        if False:
            i = 10
            return i + 15
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().filter(slug='hello-world-2').first()
        self.assertIsNotNone(page_copy)
        self.assertFalse(page_copy.live)
        self.assertTrue(page_copy.has_unpublished_changes)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 0)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_with_exclude_fields_in_copy(self):
        if False:
            print('Hello World!')
        original_page = self.test_page.add_child(instance=PageWithExcludedCopyField(title='Page with exclude_fields_in_copy', slug='page-with-exclude-fields-in-copy', content='Copy me', special_field="Don't copy me", live=True, has_unpublished_changes=False))
        post_data = {'new_title': f'{original_page.title} 2', 'new_slug': f'{original_page.slug}-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
        self.client.post(reverse('wagtailadmin_pages:copy', args=(original_page.id,)), post_data)
        page_copy = PageWithExcludedCopyField.objects.get(slug=post_data['new_slug'])
        self.assertEqual(page_copy.content, original_page.content)
        self.assertNotEqual(page_copy.special_field, original_page.special_field)
        self.assertEqual(page_copy.special_field, page_copy._meta.get_field('special_field').default)

    def test_page_copy_post_copy_subpages(self):
        if False:
            while True:
                i = 10
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': True, 'publish_copies': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().filter(slug='hello-world-2').first()
        self.assertIsNotNone(page_copy)
        self.assertFalse(page_copy.live)
        self.assertTrue(page_copy.has_unpublished_changes)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 2)
        child_copy = page_copy.get_children().filter(slug='child-page').first()
        self.assertIsNotNone(child_copy)
        self.assertFalse(child_copy.live)
        self.assertTrue(child_copy.has_unpublished_changes)
        unpublished_child_copy = page_copy.get_children().filter(slug='unpublished-child-page').first()
        self.assertIsNotNone(unpublished_child_copy)
        self.assertFalse(unpublished_child_copy.live)
        self.assertTrue(unpublished_child_copy.has_unpublished_changes)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_copy_post_copy_subpages_publish_copies(self):
        if False:
            for i in range(10):
                print('nop')
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': True, 'publish_copies': True, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().filter(slug='hello-world-2').first()
        self.assertIsNotNone(page_copy)
        self.assertTrue(page_copy.live)
        self.assertFalse(page_copy.has_unpublished_changes)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 2)
        child_copy = page_copy.get_children().filter(slug='child-page').first()
        self.assertIsNotNone(child_copy)
        self.assertTrue(child_copy.live)
        self.assertTrue(child_copy.has_unpublished_changes)
        unpublished_child_copy = page_copy.get_children().filter(slug='unpublished-child-page').first()
        self.assertIsNotNone(unpublished_child_copy)
        self.assertFalse(unpublished_child_copy.live)
        self.assertTrue(unpublished_child_copy.has_unpublished_changes)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_copy_post_new_parent(self):
        if False:
            print('Hello World!')
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.test_child_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.test_child_page.id,)))
        self.assertTrue(Page.objects.filter(slug='hello-world-2').first().get_parent(), msg=self.test_child_page)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_copy_post_existing_slug_within_same_parent_page(self):
        if False:
            return 10
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'new_slug', 'This slug is already in use within the context of its parent page "Welcome to your new Wagtail site!"')

    def test_page_copy_post_and_subpages_to_same_tree_branch(self):
        if False:
            i = 10
            return i + 15
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world', 'new_parent_page': str(self.test_child_page.id), 'copy_subpages': True, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'new_parent_page', 'You cannot copy a page into itself when copying subpages')

    def test_page_copy_post_existing_slug_to_another_parent_page(self):
        if False:
            return 10
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world', 'new_parent_page': str(self.test_child_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.test_child_page.id,)))

    def test_page_copy_post_invalid_slug(self):
        if False:
            i = 10
            return i + 15
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello world!', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'new_slug', 'Enter a valid “slug” consisting of Unicode letters, numbers, underscores, or hyphens.')

    def test_page_copy_post_valid_unicode_slug(self):
        if False:
            return 10
        post_data = {'new_title': 'Hello wɜːld', 'new_slug': 'hello-wɜːld', 'new_parent_page': str(self.test_page.id), 'copy_subpages': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.test_page.id,)))
        page_copy = self.test_page.get_children().filter(slug=post_data['new_slug']).first()
        self.assertIsNotNone(page_copy)
        self.assertEqual(page_copy.slug, post_data['new_slug'])

    def test_page_copy_no_publish_permission(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.groups.add(Group.objects.get(name='Editors'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/copy.html')
        self.assertNotContains(response, 'Publish copies')

    def test_page_copy_no_publish_permission_post_copy_subpages_publish_copies(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.groups.add(Group.objects.get(name='Editors'))
        self.user.save()
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': True, 'publish_copies': True, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().filter(slug='hello-world-2').first()
        self.assertIsNotNone(page_copy)
        self.assertFalse(page_copy.live)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 2)
        child_copy = page_copy.get_children().filter(slug='child-page').first()
        self.assertIsNotNone(child_copy)
        self.assertFalse(child_copy.live)
        unpublished_child_copy = page_copy.get_children().filter(slug='unpublished-child-page').first()
        self.assertIsNotNone(unpublished_child_copy)
        self.assertFalse(unpublished_child_copy.live)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_before_copy_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(page.specific, SimplePage)
            return HttpResponse('Overridden!')
        with self.register_hook('before_copy_page', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_copy_page_hook_post(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(page.specific, SimplePage)
            return HttpResponse('Overridden!')
        with self.register_hook('before_copy_page', hook_func):
            post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
            response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertFalse(Page.objects.filter(title='Hello world 2').exists())

    def test_after_copy_page_hook(self):
        if False:
            while True:
                i = 10

        def hook_func(request, page, new_page):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(page.specific, SimplePage)
            self.assertIsInstance(new_page.specific, SimplePage)
            return HttpResponse('Overridden!')
        with self.register_hook('after_copy_page', hook_func):
            post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
            response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Page.objects.filter(title='Hello world 2').exists())

    def test_page_copy_alias_post(self):
        if False:
            i = 10
            return i + 15
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': True}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().get(slug='hello-world-2')
        self.assertEqual(page_copy.alias_of, self.test_page.page_ptr)
        self.assertTrue(page_copy.live)
        self.assertFalse(page_copy.has_unpublished_changes)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 0)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_copy_alias_post_copy_subpages(self):
        if False:
            while True:
                i = 10
        post_data = {'new_title': 'Hello world 2', 'new_slug': 'hello-world-2', 'new_parent_page': str(self.root_page.id), 'copy_subpages': True, 'publish_copies': False, 'alias': True}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=(self.test_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_copy = self.root_page.get_children().get(slug='hello-world-2')
        self.assertEqual(page_copy.alias_of, self.test_page.page_ptr)
        self.assertTrue(page_copy.live)
        self.assertFalse(page_copy.has_unpublished_changes)
        self.assertEqual(page_copy.owner, self.user)
        self.assertEqual(page_copy.get_children().count(), 2)
        child_copy = page_copy.get_children().filter(slug='child-page').first()
        self.assertIsNotNone(child_copy)
        self.assertEqual(child_copy.alias_of, self.test_child_page.page_ptr)
        self.assertTrue(child_copy.live)
        self.assertFalse(child_copy.has_unpublished_changes)
        unpublished_child_copy = page_copy.get_children().filter(slug='unpublished-child-page').first()
        self.assertIsNotNone(unpublished_child_copy)
        self.assertEqual(unpublished_child_copy.alias_of, self.test_unpublished_child_page.page_ptr)
        self.assertFalse(unpublished_child_copy.live)
        self.assertTrue(unpublished_child_copy.has_unpublished_changes)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_page_copy_alias_post_without_source_publish_permission(self):
        if False:
            while True:
                i = 10
        self.destination_page = self.root_page.add_child(instance=SimplePage(title='Destination page', slug='destination-page', content='hello', live=True, has_unpublished_changes=False))
        self.user.is_superuser = False
        self.user.groups.add(Group.objects.get(name='Moderators'))
        self.user.save()
        GroupPagePermission.objects.filter(permission__codename='publish_page').update(page=self.destination_page)
        post_data = {'new_title': self.test_child_page.title, 'new_slug': self.test_child_page.slug, 'new_parent_page': str(self.destination_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
        response = self.client.post(reverse('wagtailadmin_pages:copy', args=[self.test_child_page.id]), post_data)
        self.assertEqual(response.status_code, 302)

    def test_copy_page_with_unique_uuids_in_orderables(self):
        if False:
            print('Hello World!')
        '\n        Test that a page with orderables can be copied and the translation\n        keys are updated.\n        '
        event_page = EventPage(title='Moon Landing', location='the moon', audience='public', cost='free on TV', date_from='1969-07-20')
        self.root_page.add_child(instance=event_page)
        event_page.speakers.add(EventPageSpeaker(first_name='Neil', last_name='Armstrong'))
        event_page.save_revision().publish()
        post_data = {'new_title': 'New Moon landing', 'new_slug': 'moon-landing-redux', 'new_parent_page': str(self.root_page.id), 'copy_subpages': False, 'publish_copies': False, 'alias': False}
        self.client.post(reverse('wagtailadmin_pages:copy', args=[event_page.id]), post_data)
        new_page = EventPage.objects.last()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[new_page.id]))
        new_page_on_edit_form = response.context['form'].instance
        new_page_on_edit_form.save_revision().publish()
        self.assertNotEqual(event_page.speakers.first().translation_key, new_page.speakers.first().translation_key)