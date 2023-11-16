import datetime
from unittest import mock
from django.contrib.auth.models import Group, Permission
from django.http import HttpRequest, HttpResponse
from django.test import TestCase
from django.test.utils import override_settings
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from wagtail.models import GroupPagePermission, Locale, Page, Revision
from wagtail.signals import page_published
from wagtail.test.testapp.models import BusinessChild, BusinessIndex, BusinessSubIndex, DefaultStreamPage, PersonPage, SimpleChildPage, SimplePage, SimpleParentPage, SingletonPage, SingletonPageViaMaxCount, StandardChild, StandardIndex
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.timestamps import submittable_timestamp

class TestPageCreation(WagtailTestUtils, TestCase):
    STATUS_TOGGLE_BADGE_REGEX = 'data-side-panel-toggle="status"[^<]+<svg[^<]+<use[^<]+</use[^<]+</svg[^<]+<div data-side-panel-toggle-counter[^>]+w-bg-critical-200[^>]+>\\s*%(num_errors)s\\s*</div>'

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_add_subpage(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Simple page')
        target_url = reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id))
        self.assertContains(response, 'href="%s"' % target_url)
        self.assertContains(response, 'A simple page description')
        self.assertNotContains(response, 'MTI base page')
        self.assertNotContains(response, 'Abstract page')
        self.assertNotContains(response, 'Business child')

    def test_add_subpage_with_subpage_types(self):
        if False:
            while True:
                i = 10
        business_index = BusinessIndex(title='Hello world!', slug='hello-world')
        self.root_page.add_child(instance=business_index)
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(business_index.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Business child')
        self.assertContains(response, 'A lazy business child page description')
        self.assertNotContains(response, 'Simple page')

    def test_no_subpage_type_available_to_create(self):
        if False:
            i = 10
            return i + 15
        simple_parent_page = SimpleParentPage(title='Hello World!', slug='hello-world')
        self.root_page.add_child(instance=simple_parent_page)
        simple_child_page = SimpleChildPage(title='Hello World!', slug='hello-world')
        simple_parent_page.add_child(instance=simple_child_page)
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(simple_parent_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Sorry, you cannot create a page at this location.')
        self.assertNotContains(response, "Choose which type of page you'd like to create.")

    def test_add_subpage_with_one_valid_subpage_type(self):
        if False:
            i = 10
            return i + 15
        business_index = BusinessIndex(title='Hello world!', slug='hello-world')
        self.root_page.add_child(instance=business_index)
        business_subindex = BusinessSubIndex(title='Hello world!', slug='hello-world')
        business_index.add_child(instance=business_subindex)
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(business_subindex.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_pages:add', args=('tests', 'businesschild', business_subindex.id)))

    def test_add_subpage_bad_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 302)

    def test_add_subpage_nonexistantparent(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(100000,)))
        self.assertEqual(response.status_code, 404)

    def test_add_subpage_with_next_param(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(self.root_page.id,)), {'next': '/admin/users/'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Simple page')
        target_url = reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id))
        self.assertContains(response, 'href="%s?next=/admin/users/"' % target_url)

    def test_create_simplepage(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/html; charset=utf-8')
        self.assertContains(response, '<a id="tab-label-content" href="#tab-content" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<a id="tab-label-promote" href="#tab-promote" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<button type="submit" name="action-panic" value="Panic!" class="button">Panic!</button>')
        self.assertContains(response, 'testapp/js/siren.js')
        self.assertContains(response, '<button type="submit" name="action-relax" value="Relax." class="button">Relax.</button>')
        self.assertContains(response, '<button type="submit" name="action-submit" value="Submit for moderation" class="button">')

    @override_settings(WAGTAIL_WORKFLOW_ENABLED=False)
    def test_workflow_buttons_not_shown_when_workflow_disabled(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'value="Submit for moderation"')

    def test_create_multipart(self):
        if False:
            while True:
                i = 10
        '\n        Test checks if \'enctype="multipart/form-data"\' is added and only to forms that require multipart encoding.\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'enctype="multipart/form-data"')
        self.assertTemplateUsed(response, 'wagtailadmin/pages/create.html')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'filepage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'enctype="multipart/form-data"')

    def test_create_page_without_promote_tab(self):
        if False:
            print('Hello World!')
        '\n        Test that the Promote tab is not rendered for page classes that define it as empty\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardindex', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<a id="tab-label-content" href="#tab-content" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertNotContains(response, 'tab-promote')

    def test_create_page_with_custom_tabs(self):
        if False:
            print('Hello World!')
        '\n        Test that custom edit handlers are rendered\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardchild', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<a id="tab-label-content" href="#tab-content" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<a id="tab-label-promote" href="#tab-promote" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<a id="tab-label-dinosaurs" href="#tab-dinosaurs" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')

    def test_create_page_with_non_model_field(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that additional fields defined on the form rather than the model are accepted and rendered\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'formclassadditionalfieldpage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/create.html')
        self.assertContains(response, 'Enter SMS authentication code')

    def test_create_simplepage_bad_permissions(self):
        if False:
            return 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertEqual(response.status_code, 302)

    def test_cannot_create_page_with_is_creatable_false(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'mtibasepage', self.root_page.id)))
        self.assertRedirects(response, '/admin/')

    def test_cannot_create_page_when_can_create_at_returns_false(self):
        if False:
            print('Hello World!')
        add_url = reverse('wagtailadmin_pages:add', args=[SingletonPage._meta.app_label, SingletonPage._meta.model_name, self.root_page.pk])
        self.assertTrue(SingletonPage.can_create_at(self.root_page))
        response = self.client.get(add_url)
        self.assertEqual(response.status_code, 200)
        self.root_page.add_child(instance=SingletonPage(title='singleton', slug='singleton'))
        self.assertFalse(SingletonPage.can_create_at(self.root_page))
        response = self.client.get(add_url)
        self.assertRedirects(response, '/admin/')

    def test_cannot_create_singleton_page_with_max_count(self):
        if False:
            print('Hello World!')
        add_url = reverse('wagtailadmin_pages:add', args=[SingletonPageViaMaxCount._meta.app_label, SingletonPageViaMaxCount._meta.model_name, self.root_page.pk])
        self.assertTrue(SingletonPageViaMaxCount.can_create_at(self.root_page))
        response = self.client.get(add_url)
        self.assertEqual(response.status_code, 200)
        self.root_page.add_child(instance=SingletonPageViaMaxCount(title='singleton', slug='singleton'))
        self.assertFalse(SingletonPageViaMaxCount.can_create_at(self.root_page))
        response = self.client.get(add_url)
        self.assertRedirects(response, '/admin/')

    def test_cannot_create_page_with_wrong_parent_page_types(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'businesschild', self.root_page.id)))
        self.assertRedirects(response, '/admin/')

    def test_cannot_create_page_with_wrong_subpage_types(self):
        if False:
            while True:
                i = 10
        business_index = BusinessIndex(title='Hello world!', slug='hello-world')
        self.root_page.add_child(instance=business_index)
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', business_index.id)))
        self.assertRedirects(response, '/admin/')

    def test_create_simplepage_post(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(page.id,)))
        self.assertEqual(page.title, post_data['title'])
        self.assertEqual(page.draft_title, post_data['title'])
        self.assertIsInstance(page, SimplePage)
        self.assertFalse(page.live)
        self.assertFalse(page.first_published_at)
        self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')

    def test_create_simplepage_scheduled(self):
        if False:
            print('Hello World!')
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 302)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertEqual(page.go_live_at.date(), go_live_at.date())
        self.assertEqual(page.expire_at.date(), expire_at.date())
        self.assertIs(page.expired, False)
        self.assertEqual(page.status_string, 'draft')
        self.assertFalse(Revision.page_revisions.filter(object_id=page.id).exclude(approved_go_live_at__isnull=True).exists())

    def test_create_simplepage_scheduled_go_live_before_expiry(self):
        if False:
            while True:
                i = 10
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=2)), 'expire_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=1))}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'go_live_at', 'Go live date/time must be before expiry date/time')
        self.assertFormError(response, 'form', 'expire_at', 'Go live date/time must be before expiry date/time')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})
        self.assertContains(response, 'alwaysDirty: true')

    def test_create_simplepage_scheduled_expire_in_the_past(self):
        if False:
            return 10
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'expire_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=-1))}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'expire_at', 'Expiry date/time must be in the future')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 1
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})
        self.assertContains(response, 'alwaysDirty: true')

    def test_create_simplepage_post_publish(self):
        if False:
            for i in range(10):
                print('nop')
        mock_handler = mock.MagicMock()
        page_published.connect(mock_handler)
        try:
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
            page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            self.assertEqual(page.title, post_data['title'])
            self.assertEqual(page.draft_title, post_data['title'])
            self.assertIsInstance(page, SimplePage)
            self.assertTrue(page.live)
            self.assertTrue(page.first_published_at)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], page.specific_class)
            self.assertEqual(mock_call['instance'], page)
            self.assertIsInstance(mock_call['instance'], page.specific_class)
            self.assertFalse(any(Page.find_problems()), msg='treebeard found consistency problems')
        finally:
            page_published.disconnect(mock_handler)

    def test_create_simplepage_post_publish_scheduled(self):
        if False:
            i = 10
            return i + 15
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 302)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertEqual(page.go_live_at.date(), go_live_at.date())
        self.assertEqual(page.expire_at.date(), expire_at.date())
        self.assertIs(page.expired, False)
        self.assertTrue(Revision.page_revisions.filter(object_id=page.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertFalse(page.live)
        self.assertFalse(page.first_published_at)
        self.assertEqual(page.status_string, 'scheduled')

    def test_create_simplepage_post_submit(self):
        if False:
            i = 10
            return i + 15
        self.create_superuser('moderator', 'moderator@email.com', 'password')
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(page.title, post_data['title'])
        self.assertIsInstance(page, SimplePage)
        self.assertFalse(page.live)
        self.assertFalse(page.first_published_at)
        self.assertEqual(page.current_workflow_state.status, page.current_workflow_state.STATUS_IN_PROGRESS)

    def test_create_simplepage_post_existing_slug(self):
        if False:
            while True:
                i = 10
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.child_page)
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'slug', "The slug 'hello-world' is already in use within the parent page")
        self.assertContains(response, 'alwaysDirty: true')

    def test_create_nonexistantparent(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', 100000)))
        self.assertEqual(response.status_code, 404)

    def test_create_nonpagetype(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('wagtailimages', 'image', self.root_page.id)))
        self.assertEqual(response.status_code, 404)

    def test_custom_validation(self):
        if False:
            while True:
                i = 10
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'validatedpage', self.root_page.id)), {'title': 'New page!', 'foo': 'not bar', 'slug': 'hello-world'})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'foo', 'Field foo must be bar')
        self.assertFalse(Page.objects.filter(path__startswith=self.root_page.path, slug='hello-world').exists())
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'validatedpage', self.root_page.id)), {'title': 'New page!', 'foo': 'superbar', 'slug': 'hello-world'})
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world')
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(page.id,)))

    def test_preview_on_create(self):
        if False:
            for i in range(10):
                print('nop')
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_add', args=('tests', 'simplepage', self.root_page.id))
        response = self.client.post(preview_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode(), {'is_valid': True, 'is_available': True})
        response = self.client.get(preview_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'tests/simple_page.html')
        self.assertContains(response, 'New page!')
        self.assertEqual(response.context['self'].depth, self.root_page.depth + 1)
        self.assertTrue(response.context['self'].path.startswith(self.root_page.path))
        self.assertEqual(response.context['self'].get_parent(), self.root_page)
        self.assertNotContains(response, 'Edit this page')

    def test_preview_with_custom_validation(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'New page!', 'foo': 'not bar', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_add', args=('tests', 'validatedpage', self.root_page.id))
        response = self.client.post(preview_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode(), {'is_valid': False, 'is_available': False})
        post_data = {'title': 'New page!', 'foo': 'superbar', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_add', args=('tests', 'validatedpage', self.root_page.id))
        response = self.client.post(preview_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode(), {'is_valid': True, 'is_available': True})
        response = self.client.get(preview_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'tests/validated_page.html')
        self.assertContains(response, 'foo = superbar')

    def test_whitespace_titles(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': ' ', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertFormError(response, 'form', 'title', 'This field is required.')

    def test_whitespace_titles_with_tab(self):
        if False:
            return 10
        post_data = {'title': '\t', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertFormError(response, 'form', 'title', 'This field is required.')

    def test_whitespace_titles_with_tab_in_seo_title(self):
        if False:
            while True:
                i = 10
        post_data = {'title': 'Hello', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit', 'seo_title': '\t'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 302)
        page = Page.objects.order_by('-id').first()
        self.assertEqual(page.seo_title, '')

    def test_whitespace_is_stripped_from_titles(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': '   Hello   ', 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit', 'seo_title': '   hello SEO   '}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 302)
        page = Page.objects.order_by('-id').first()
        self.assertEqual(page.title, 'Hello')
        self.assertEqual(page.draft_title, 'Hello')
        self.assertEqual(page.seo_title, 'hello SEO')

    def test_long_slug(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'Hello world', 'content': 'Some content', 'slug': 'hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world-hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'slug', 'Ensure this value has at most 255 characters (it has 287).')

    def test_title_field_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Should correctly add the sync field and placeholder attributes to the title field.\n        Note: Many test Page models use a FieldPanel for 'title', StandardChild does not\n        override content_panels (uses the default).\n        "
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardchild', self.root_page.id)))
        html = self.get_soup(response.content)
        actual_attrs = html.find('input', {'name': 'title'}).attrs
        expected_attrs = {'aria-describedby': 'panel-child-content-child-title-helptext', 'data-action': 'focus->w-sync#check blur->w-sync#apply change->w-sync#apply keyup->w-sync#apply', 'data-controller': 'w-sync', 'data-w-sync-target-value': '#id_slug', 'id': 'id_title', 'maxlength': '255', 'name': 'title', 'placeholder': 'Page title*', 'required': '', 'type': 'text'}
        self.assertEqual(actual_attrs, expected_attrs)

    def test_before_create_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, parent_page, page_class):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(parent_page.id, self.root_page.id)
            self.assertEqual(page_class, SimplePage)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_page', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_create_page_hook_post(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, parent_page, page_class):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(parent_page.id, self.root_page.id)
            self.assertEqual(page_class, SimplePage)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_page', hook_func):
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertFalse(Page.objects.filter(title='New page!').exists())

    def test_after_create_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(page, SimplePage)
            self.assertIsNone(page.first_published_at)
            self.assertIsNone(page.last_published_at)
            return HttpResponse('Overridden!')
        with self.register_hook('after_create_page', hook_func):
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Page.objects.filter(title='New page!').exists())

    def test_after_create_page_hook_with_page_publish(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, page):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(request, HttpRequest)
            self.assertIsInstance(page, SimplePage)
            self.assertIsNotNone(page.first_published_at)
            self.assertIsNotNone(page.last_published_at)
            return HttpResponse('Overridden!')
        with self.register_hook('after_create_page', hook_func):
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Page.objects.filter(title='New page!').exists())

    def test_after_publish_page(self):
        if False:
            return 10

        def hook_func(request, page):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.title, 'New page!')
            self.assertIsNotNone(page.first_published_at)
            self.assertIsNotNone(page.last_published_at)
            return HttpResponse('Overridden!')
        with self.register_hook('after_publish_page', hook_func):
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.root_page.refresh_from_db()
        self.assertEqual(self.root_page.get_children()[0].status_string, _('live'))

    def test_before_publish_page(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                print('Hello World!')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.title, 'New page!')
            self.assertIsNone(page.first_published_at)
            self.assertIsNone(page.last_published_at)
            return HttpResponse('Overridden!')
        with self.register_hook('before_publish_page', hook_func):
            post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.root_page.refresh_from_db()
        self.assertEqual(self.root_page.get_children()[0].status_string, _('live + draft'))

    def test_display_moderation_button_by_default(self):
        if False:
            while True:
                i = 10
        '\n        Tests that by default the "Submit for Moderation" button is shown in the action menu.\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertContains(response, '<button type="submit" name="action-submit" value="Submit for moderation" class="button"><svg class="icon icon-resubmit icon" aria-hidden="true"><use href="#icon-resubmit"></use></svg>Submit for moderation</button>')

    @override_settings(WAGTAIL_WORKFLOW_ENABLED=False)
    def test_hide_moderation_button(self):
        if False:
            while True:
                i = 10
        '\n        Tests that if WAGTAIL_WORKFLOW_ENABLED is set to False, the "Submit for Moderation" button is not shown.\n        '
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', self.root_page.id)))
        self.assertNotContains(response, '<button type="submit" name="action-submit" value="Submit for moderation" class="button">Submit for moderation</button>')

    def test_create_sets_locale_to_parent_locale(self):
        if False:
            return 10
        fr_locale = Locale.objects.create(language_code='fr')
        fr_homepage = self.root_page.add_child(instance=Page(title='Home', slug='home-fr', locale=fr_locale))
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', fr_homepage.id)))
        self.assertEqual(response.context['page'].locale, fr_locale)

class TestPermissionedFieldPanels(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        GroupPagePermission.objects.create(group=Group.objects.get(name='Site-wide editors'), page=self.root_page, permission_type='add')

    def test_create_page_with_permissioned_field_panel(self):
        if False:
            return 10
        '\n        Test that permission rules on field panels are honoured\n        '
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'secretpage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"boring_data"')
        self.assertNotContains(response, '"secret_data"')
        self.login(username='superuser', password='password')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'secretpage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"boring_data"')
        self.assertContains(response, '"secret_data"')

class TestSubpageBusinessRules(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        self.standard_index = StandardIndex()
        self.standard_index.title = 'Standard Index'
        self.standard_index.slug = 'standard-index'
        self.root_page.add_child(instance=self.standard_index)
        self.business_index = BusinessIndex()
        self.business_index.title = 'Business Index'
        self.business_index.slug = 'business-index'
        self.root_page.add_child(instance=self.business_index)
        self.business_child = BusinessChild()
        self.business_child.title = 'Business Child'
        self.business_child.slug = 'business-child'
        self.business_index.add_child(instance=self.business_child)
        self.business_subindex = BusinessSubIndex()
        self.business_subindex.title = 'Business Subindex'
        self.business_subindex.slug = 'business-subindex'
        self.business_index.add_child(instance=self.business_subindex)
        self.login()

    def test_standard_subpage(self):
        if False:
            while True:
                i = 10
        add_subpage_url = reverse('wagtailadmin_pages:add_subpage', args=(self.standard_index.id,))
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.standard_index.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, add_subpage_url)
        response = self.client.get(add_subpage_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, StandardChild.get_verbose_name())
        self.assertContains(response, BusinessIndex.get_verbose_name())
        self.assertNotContains(response, BusinessSubIndex.get_verbose_name())
        self.assertNotContains(response, BusinessChild.get_verbose_name())

    def test_business_subpage(self):
        if False:
            print('Hello World!')
        add_subpage_url = reverse('wagtailadmin_pages:add_subpage', args=(self.business_index.id,))
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.business_index.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, add_subpage_url)
        response = self.client.get(add_subpage_url)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, StandardIndex.get_verbose_name())
        self.assertNotContains(response, StandardChild.get_verbose_name())
        self.assertContains(response, BusinessSubIndex.get_verbose_name())
        self.assertContains(response, BusinessChild.get_verbose_name())

    def test_business_child_subpage(self):
        if False:
            while True:
                i = 10
        add_subpage_url = reverse('wagtailadmin_pages:add_subpage', args=(self.business_child.id,))
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.business_child.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, add_subpage_url)
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(self.business_child.id,)))
        self.assertEqual(response.status_code, 302)

    def test_cannot_add_invalid_subpage_type(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardchild', self.business_index.id)))
        self.assertRedirects(response, '/admin/')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardchild', self.business_child.id)))
        self.assertRedirects(response, '/admin/')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'businesschild', self.standard_index.id)))
        self.assertRedirects(response, '/admin/')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'businesschild', self.business_index.id)))
        self.assertEqual(response.status_code, 200)

    def test_not_prompted_for_page_type_when_only_one_choice(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:add_subpage', args=(self.business_subindex.id,)))
        self.assertRedirects(response, reverse('wagtailadmin_pages:add', args=('tests', 'businesschild', self.business_subindex.id)))

class TestInlinePanelMedia(WagtailTestUtils, TestCase):
    """
    Test that form media required by InlinePanels is correctly pulled in to the edit page
    """

    def test_inline_panel_media(self):
        if False:
            print('Hello World!')
        homepage = Page.objects.get(id=2)
        self.login()
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', homepage.id)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'wagtailadmin/js/draftail.js')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'sectionedrichtextpage', homepage.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'wagtailadmin/js/draftail.js')

class TestInlineStreamField(WagtailTestUtils, TestCase):
    """
    Test that streamfields inside an inline child work
    """

    def test_inline_streamfield(self):
        if False:
            while True:
                i = 10
        homepage = Page.objects.get(id=2)
        self.login()
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'inlinestreampage', homepage.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<div id="sections-__prefix__-body" data-block="')

class TestIssue2994(WagtailTestUtils, TestCase):
    """
    In contrast to most "standard" form fields, StreamField form widgets generally won't
    provide a postdata field with a name exactly matching the field name. To prevent Django
    from wrongly interpreting this as the field being omitted from the form,
    we need to provide a custom value_omitted_from_data method.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_page_edit_post_publish_url(self):
        if False:
            return 10
        post_data = {'title': 'Issue 2994 test', 'slug': 'issue-2994-test', 'body-count': '1', 'body-0-deleted': '', 'body-0-order': '0', 'body-0-type': 'text', 'body-0-value': 'hello world', 'action-publish': 'Publish'}
        self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'defaultstreampage', self.root_page.id)), post_data)
        new_page = DefaultStreamPage.objects.get(slug='issue-2994-test')
        self.assertEqual(1, len(new_page.body))
        self.assertEqual('hello world', new_page.body[0].value)

class TestInlinePanelWithTags(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_create(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'Mr Benn', 'slug': 'mr-benn', 'first_name': 'William', 'last_name': 'Benn', 'addresses-TOTAL_FORMS': 1, 'addresses-INITIAL_FORMS': 0, 'addresses-MIN_NUM_FORMS': 0, 'addresses-MAX_NUM_FORMS': 1000, 'addresses-0-address': '52 Festive Road, London', 'addresses-0-tags': 'shopkeeper, bowler-hat', 'action-publish': 'Publish', 'comments-TOTAL_FORMS': 0, 'comments-INITIAL_FORMS': 0, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'personpage', self.root_page.id)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        new_page = PersonPage.objects.get(slug='mr-benn')
        self.assertEqual(new_page.addresses.first().tags.count(), 2)

class TestInlinePanelNonFieldErrors(WagtailTestUtils, TestCase):
    """
    Test that non field errors will render for InlinePanels
    https://github.com/wagtail/wagtail/issues/3890
    """
    fixtures = ['demosite.json']

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_create(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': 'Issue 3890 test', 'slug': 'issue-3890-test', 'related_links-TOTAL_FORMS': 1, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 1000, 'related_links-0-id': 0, 'related_links-0-ORDER': 1, 'related_links-0-link_page': '', 'related_links-0-link_document': '', 'related_links-0-link_external': '', 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 1000, 'carousel_items-TOTAL_FORMS': 0, 'action-publish': 'Publish', 'comments-TOTAL_FORMS': 0, 'comments-INITIAL_FORMS': 0, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('demosite', 'homepage', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The page could not be created due to validation errors')
        self.assertContains(response, 'You must provide a related page, related document or an external URL', count=1)

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelector(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.events_page = Page.objects.get(url_path='/home/events/')
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.translated_events_page = self.events_page.copy_for_translation(self.fr_locale, copy_parents=True)
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.events_page.id]))
        self.assertContains(response, 'id="status-sidebar-english"')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.translated_events_page.id])
        self.assertContains(response, f'href="{add_translation_url}"')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.events_page.id]))
        self.assertNotContains(response, 'Page Locale:')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.translated_events_page.id])
        self.assertNotContains(response, f'href="{add_translation_url}"')

    def test_locale_selector_not_present_without_permission_to_add(self):
        if False:
            return 10
        group = Group.objects.get(name='Moderators')
        GroupPagePermission.objects.create(group=group, page=self.events_page, permission_type='add')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.groups.add(group)
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.events_page.id]))
        self.assertContains(response, 'id="status-sidebar-english"')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['tests', 'eventpage', self.translated_events_page.id])
        self.assertNotContains(response, f'href="{add_translation_url}"')

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelectorOnRootPage(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=1)
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]))
        self.assertContains(response, 'id="status-sidebar-english"')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=fr'
        self.assertContains(response, f'href="{add_translation_url}"')
        self_translation_url = reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=en'
        self.assertNotContains(response, f'href="{self_translation_url}"')

    def test_locale_selector_selected(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=fr')
        self.assertContains(response, 'id="status-sidebar-french"')
        self.assertContains(response, '<input type="hidden" name="locale" value="fr">')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=en'
        self.assertContains(response, f'href="{add_translation_url}"')
        self_translation_url = reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=fr'
        self.assertNotContains(response, f'href="{self_translation_url}"')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]))
        self.assertNotContains(response, 'Page Locale:')
        add_translation_url = reverse('wagtailadmin_pages:add', args=['demosite', 'homepage', self.root_page.id]) + '?locale=fr'
        self.assertNotContains(response, f'href="{add_translation_url}"')

class TestPageSubscriptionSettings(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_commment_notifications_switched_on_by_default(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_pages:add', args=['tests', 'simplepage', self.root_page.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<input type="checkbox" name="comment_notifications" id="id_comment_notifications" checked>')

    def test_post_with_comment_notifications_switched_on(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world', 'comment_notifications': 'on'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=['tests', 'simplepage', self.root_page.id]), post_data)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[page.id]))
        subscription = page.subscribers.get()
        self.assertEqual(subscription.user, self.user)
        self.assertTrue(subscription.comment_notifications)

    def test_post_with_comment_notifications_switched_off(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': 'New page!', 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=['tests', 'simplepage', self.root_page.id]), post_data)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[page.id]))
        subscription = page.subscribers.get()
        self.assertEqual(subscription.user, self.user)
        self.assertFalse(subscription.comment_notifications)