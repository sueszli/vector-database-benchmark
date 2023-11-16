import datetime
import json
import os
from unittest import mock
from django.conf import settings
from django.contrib.auth.models import Group, Permission
from django.core import mail
from django.core.files.base import ContentFile
from django.http import HttpRequest, HttpResponse
from django.test import TestCase, modify_settings, override_settings
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from wagtail.admin.admin_url_finder import AdminURLFinder
from wagtail.exceptions import PageClassNotFoundError
from wagtail.models import Comment, CommentReply, GroupPagePermission, Locale, Page, PageLogEntry, PageSubscription, Revision, Site
from wagtail.signals import page_published
from wagtail.test.testapp.models import EVENT_AUDIENCE_CHOICES, Advert, AdvertPlacement, EventCategory, EventPage, EventPageCarouselItem, FilePage, ManyToManyBlogPage, PageChooserModel, SimplePage, SingleEventPage, StandardIndex, StreamPage, TaggedPage
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.form_data import inline_formset, nested_form_data
from wagtail.test.utils.timestamps import submittable_timestamp
from wagtail.users.models import UserProfile
from wagtail.utils.timestamps import render_timestamp

class TestPageEdit(WagtailTestUtils, TestCase):
    STATUS_TOGGLE_BADGE_REGEX = 'data-side-panel-toggle="status"[^<]+<svg[^<]+<use[^<]+</use[^<]+</svg[^<]+<div data-side-panel-toggle-counter[^>]+w-bg-critical-200[^>]+>\\s*%(num_errors)s\\s*</div>'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=child_page)
        child_page.save_revision().publish()
        self.child_page = SimplePage.objects.get(id=child_page.id)
        fake_file = ContentFile('File for testing multipart')
        fake_file.name = 'test.txt'
        file_page = FilePage(title='File Page', slug='file-page', file_field=fake_file)
        self.root_page.add_child(instance=file_page)
        file_page.save_revision().publish()
        self.file_page = FilePage.objects.get(id=file_page.id)
        self.event_page = EventPage(title='Event page', slug='event-page', location='the moon', audience='public', cost='free', date_from='2001-01-01')
        self.root_page.add_child(instance=self.event_page)
        self.single_event_page = SingleEventPage(title='Mars landing', slug='mars-landing', location='mars', audience='public', cost='free', date_from='2001-01-01')
        self.root_page.add_child(instance=self.single_event_page)
        self.unpublished_page = SimplePage(title='Hello unpublished world!', slug='hello-unpublished-world', content='hello', live=False, has_unpublished_changes=True)
        self.root_page.add_child(instance=self.unpublished_page)
        self.user = self.login()

    def test_page_edit(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/html; charset=utf-8')
        self.assertContains(response, 'id="status-sidebar-live"')
        self.assertContains(response, 'Who this event is for')
        self.assertContains(response, '<label class="w-field__label" for="id_speakers-__prefix__-last_name" id="id_speakers-__prefix__-last_name-label">')
        self.assertContains(response, 'Add speakers')
        self.assertContains(response, 'Put the keynote speaker first')
        self.assertContains(response, 'For SEO nerds only')
        self.assertContains(response, '<button type="submit" name="action-panic" value="Panic!" class="button">Panic!</button>')
        self.assertContains(response, 'testapp/js/siren.js')
        self.assertContains(response, '<button type="submit" name="action-relax" value="Relax." class="button">Relax.</button>')
        self.assertContains(response, '<button type="submit" name="action-submit" value="Submit to Moderators approval" class="button">')
        self.assertContains(response, '<aside class="form-side form-side--initial" aria-label="Side panels" data-form-side>')
        self.assertNotContains(response, 'data-form-side-explorer')
        self.assertContains(response, 'Referenced 0 times')
        self.assertContains(response, reverse('wagtailadmin_pages:usage', args=(self.event_page.id,)))
        url_finder = AdminURLFinder(self.user)
        expected_url = '/admin/pages/%d/edit/' % self.event_page.id
        self.assertEqual(url_finder.get_edit_url(self.event_page), expected_url)

    def test_usage_count_information_shown(self):
        if False:
            for i in range(10):
                print('nop')
        PageChooserModel.objects.create(page=self.event_page)
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertContains(response, 'Referenced 1 time')
        self.assertContains(response, reverse('wagtailadmin_pages:usage', args=(self.event_page.id,)))

    @override_settings(WAGTAIL_WORKFLOW_ENABLED=False)
    def test_workflow_buttons_not_shown_when_workflow_disabled(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'value="Submit to Moderators approval"')

    def test_edit_draft_page_with_no_revisions(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.unpublished_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="status-sidebar-draft"')

    def test_edit_multipart(self):
        if False:
            return 10
        '\n        Test checks if \'enctype="multipart/form-data"\' is added and only to forms that require multipart encoding.\n        '
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'enctype="multipart/form-data"')
        self.assertTemplateUsed(response, 'wagtailadmin/pages/edit.html')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.file_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'enctype="multipart/form-data"')

    @mock.patch('wagtail.models.ContentType.model_class', return_value=None)
    def test_edit_when_specific_class_cannot_be_found(self, mocked_method):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(PageClassNotFoundError):
            self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))

    def test_upload_file_publish(self):
        if False:
            return 10
        '\n        Check that file uploads work when directly publishing\n        '
        file_upload = ContentFile(b'A new file', name='published-file.txt')
        post_data = {'title': 'New file', 'slug': 'new-file', 'file_field': file_upload, 'action-publish': 'Publish'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.file_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=[self.root_page.id]))
        file_page = FilePage.objects.get()
        self.assertEqual(file_page.file_field.name, file_upload.name)
        self.assertTrue(os.path.exists(file_page.file_field.path))
        self.assertEqual(file_page.file_field.read(), b'A new file')

    def test_upload_file_draft(self):
        if False:
            return 10
        '\n        Check that file uploads work when saving a draft\n        '
        file_upload = ContentFile(b'A new file', name='draft-file.txt')
        post_data = {'title': 'New file', 'slug': 'new-file', 'file_field': file_upload}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.file_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.file_page.id]))
        file_path = os.path.join(settings.MEDIA_ROOT, file_upload.name)
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, 'rb') as saved_file:
            self.assertEqual(saved_file.read(), b'A new file')
        FilePage.objects.get().get_latest_revision().publish()
        file_page = FilePage.objects.get()
        self.assertEqual(file_page.file_field.name, file_upload.name)
        self.assertTrue(os.path.exists(file_page.file_field.path))
        self.assertEqual(file_page.file_field.read(), b'A new file')

    def test_page_edit_bad_permissions(self):
        if False:
            return 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 302)
        url_finder = AdminURLFinder(self.user)
        self.assertIsNone(url_finder.get_edit_url(self.event_page))

    def test_page_edit_post(self):
        if False:
            print('Hello World!')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.has_unpublished_changes)
        self.assertEqual(child_page_new.title, self.child_page.title)
        self.assertEqual(child_page_new.content, self.child_page.content)
        self.assertEqual(child_page_new.slug, self.child_page.slug)
        self.assertEqual(child_page_new.draft_title, post_data['title'])

    def test_page_edit_post_when_locked(self):
        if False:
            print('Hello World!')
        self.child_page.locked = True
        self.child_page.save()
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertContains(response, 'The page could not be saved as it is locked')
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertFalse(child_page_new.has_unpublished_changes)

    def test_edit_post_scheduled(self):
        if False:
            return 10
        go_live_at = timezone.now() + datetime.timedelta(days=10)
        expire_at = timezone.now() + datetime.timedelta(days=20)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data, follow=True)
        self.assertRedirects(response, edit_url, 302, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertFalse(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__go_live_at__startswith=str(go_live_at.date())).exists())
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__expire_at__startswith=str(expire_at.date())).exists())
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'This publishing schedule will only take effect after you select the "Publish" option')

    def test_edit_post_scheduled_custom_timezone(self):
        if False:
            i = 10
            return i + 15
        UserProfile.objects.update_or_create(user=self.user, defaults={'current_time_zone': 'Asia/Jakarta'})
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': '2022-03-20 06:00'}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data, follow=True)
        html = response.content.decode()
        self.assertRedirects(response, edit_url, 302, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertFalse(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        if settings.USE_TZ:
            self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__go_live_at='2022-03-19T23:00:00Z').exists())
        else:
            self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__go_live_at='2022-03-20T06:00:00').exists())
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True, count=1)
        self.assertContains(response, '<span class="w-text-grey-600">Go-live:</span> March 20, 2022, 6 a.m.', html=True, count=1)
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<input type="text" name="go_live_at" value="2022-03-20 06:00">', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'This publishing schedule will only take effect after you select the "Publish" option')

    def test_schedule_panel_without_publish_permission(self):
        if False:
            return 10
        editor = self.create_user('editor', password='password')
        editor.groups.add(Group.objects.get(name='Editors'))
        self.login(username='editor')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Anyone with editing permissions can create schedules')

    def test_edit_scheduled_go_live_before_expiry(self):
        if False:
            while True:
                i = 10
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=2)), 'expire_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=1))}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'go_live_at', 'Go live date/time must be before expiry date/time')
        self.assertFormError(response, 'form', 'expire_at', 'Go live date/time must be before expiry date/time')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})
        self.assertContains(response, 'alwaysDirty: true')

    def test_edit_scheduled_expire_in_the_past(self):
        if False:
            while True:
                i = 10
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'expire_at': submittable_timestamp(timezone.now() + datetime.timedelta(days=-1))}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'expire_at', 'Expiry date/time must be in the future')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 1
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})
        self.assertContains(response, 'alwaysDirty: true')

    def test_edit_post_invalid_schedule_with_existing_draft_schedule(self):
        if False:
            for i in range(10):
                print('nop')
        self.child_page.go_live_at = timezone.now() + datetime.timedelta(days=1)
        self.child_page.expire_at = timezone.now() + datetime.timedelta(days=2)
        latest_revision = self.child_page.save_revision()
        go_live_at = timezone.now() + datetime.timedelta(days=10)
        expire_at = timezone.now() + datetime.timedelta(days=-20)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.latest_revision_id, latest_revision.pk)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertNotContains(response, '<span class="w-text-grey-600">Go-live:</span>', html=True)
        self.assertNotContains(response, '<span class="w-text-grey-600">Expiry:</span>', html=True)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_page_edit_post_publish(self):
        if False:
            i = 10
            return i + 15
        mock_handler = mock.MagicMock()
        page_published.connect(mock_handler)
        try:
            self.child_page.has_unpublished_changes = True
            self.child_page.save()
            first_published_at = SimplePage.objects.get(id=self.child_page.id).first_published_at
            post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world-new', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data, follow=True)
            self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
            child_page_new = SimplePage.objects.get(id=self.child_page.id)
            self.assertEqual(child_page_new.title, post_data['title'])
            self.assertEqual(child_page_new.draft_title, post_data['title'])
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], child_page_new.specific_class)
            self.assertEqual(mock_call['instance'], child_page_new)
            self.assertIsInstance(mock_call['instance'], child_page_new.specific_class)
            self.assertFalse(child_page_new.has_unpublished_changes)
            self.assertEqual(first_published_at, child_page_new.first_published_at)
            for message in response.context['messages']:
                self.assertIn('hello-world-new', message.message)
                break
        finally:
            page_published.disconnect(mock_handler)

    def test_first_published_at_editable(self):
        if False:
            print('Hello World!')
        'Test that we can update the first_published_at via the Page edit form,\n        for page models that expose it.'
        child_page = ManyToManyBlogPage(title='Hello world!', slug='hello-again-world', body='hello')
        self.root_page.add_child(instance=child_page)
        child_page.save_revision().publish()
        self.child_page = ManyToManyBlogPage.objects.get(id=child_page.id)
        initial_delta = self.child_page.first_published_at - timezone.now()
        first_published_at = timezone.now() - datetime.timedelta(days=2)
        post_data = {'title': "I've been edited!", 'body': 'Some content', 'slug': 'hello-again-world', 'action-publish': 'Publish', 'first_published_at': submittable_timestamp(first_published_at), 'comments-TOTAL_FORMS': 0, 'comments-INITIAL_FORMS': 0, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000}
        self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        child_page_new = ManyToManyBlogPage.objects.get(id=self.child_page.id)
        new_delta = child_page_new.first_published_at - timezone.now()
        self.assertNotEqual(new_delta.days, initial_delta.days)
        self.assertEqual(new_delta.days, -3)

    def test_edit_post_publish_scheduled_unpublished_page(self):
        if False:
            while True:
                i = 10
        self.child_page.live = False
        self.child_page.save()
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertFalse(child_page_new.live)
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(child_page_new.has_unpublished_changes, msg='A page scheduled for future publishing should have has_unpublished_changes=True')
        self.assertEqual(child_page_new.status_string, 'scheduled')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)

    def test_edit_post_publish_now_an_already_scheduled_unpublished_page(self):
        if False:
            while True:
                i = 10
        self.child_page.live = False
        self.child_page.save()
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertFalse(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'scheduled')
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        go_live_at = timezone.now()
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': go_live_at}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertFalse(child_page_new.live)
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)

    def test_edit_post_publish_scheduled_published_page(self):
        if False:
            i = 10
            return i + 15
        self.child_page.live = True
        self.child_page.save()
        live_revision = self.child_page.live_revision
        original_title = self.child_page.title
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live + scheduled')
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(child_page_new.has_unpublished_changes, msg='A page scheduled for future publishing should have has_unpublished_changes=True')
        self.assertNotEqual(child_page_new.get_latest_revision(), live_revision, 'A page scheduled for future publishing should have a new revision, that is not the live revision')
        self.assertEqual(child_page_new.title, original_title, msg='A live page with scheduled revisions should still have original content')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)

    def test_edit_post_publish_now_an_already_scheduled_published_page(self):
        if False:
            print('Hello World!')
        self.child_page.live = True
        self.child_page.save()
        original_title = self.child_page.title
        go_live_at = timezone.now() + datetime.timedelta(days=1)
        expire_at = timezone.now() + datetime.timedelta(days=2)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertEqual(child_page_new.title, original_title, 'A live page with scheduled revisions should still have original content')
        go_live_at = timezone.now()
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': go_live_at}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertEqual(child_page_new.title, 'Hello world!')

    def test_edit_post_save_schedule_before_a_scheduled_expire_page(self):
        if False:
            print('Hello World!')
        expire_at = timezone.now() + datetime.timedelta(days=20)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live')
        self.assertEqual(child_page_new.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = timezone.now() + datetime.timedelta(days=10)
        new_expire_at = timezone.now() + datetime.timedelta(days=15)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)}
        response = self.client.post(edit_url, post_data, follow=True)
        self.assertRedirects(response, edit_url, 302, 200)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertFalse(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__go_live_at__startswith=str(go_live_at.date())).exists())
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id, content__expire_at__startswith=str(expire_at.date())).exists())
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_schedule_before_a_scheduled_expire_page(self):
        if False:
            for i in range(10):
                print('nop')
        expire_at = timezone.now() + datetime.timedelta(days=20)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live')
        self.assertEqual(child_page_new.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = timezone.now() + datetime.timedelta(days=10)
        new_expire_at = timezone.now() + datetime.timedelta(days=15)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)}
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live + scheduled')
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertNotContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)

    def test_edit_post_publish_schedule_after_a_scheduled_expire_page(self):
        if False:
            print('Hello World!')
        expire_at = timezone.now() + datetime.timedelta(days=20)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)}
        edit_url = reverse('wagtailadmin_pages:edit', args=(self.child_page.id,))
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live')
        self.assertEqual(child_page_new.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = timezone.now() + datetime.timedelta(days=23)
        new_expire_at = timezone.now() + datetime.timedelta(days=25)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)}
        response = self.client.post(edit_url, post_data)
        self.assertEqual(response.status_code, 302)
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.live)
        self.assertEqual(child_page_new.status_string, 'live + scheduled')
        self.assertTrue(Revision.page_revisions.filter(object_id=child_page_new.id).exclude(approved_go_live_at__isnull=True).exists())
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)

    def test_page_edit_post_submit(self):
        if False:
            i = 10
            return i + 15
        self.create_superuser('moderator', 'moderator@email.com', 'password')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.has_unpublished_changes)
        self.assertEqual(child_page_new.current_workflow_state.status, child_page_new.current_workflow_state.STATUS_IN_PROGRESS)

    def test_page_edit_post_existing_slug(self):
        if False:
            i = 10
            return i + 15
        self.child_page = SimplePage(title='Hello world 2', slug='hello-world2', content='hello')
        self.root_page.add_child(instance=self.child_page)
        post_data = {'title': 'Hello world 2', 'slug': 'hello-world', 'action-submit': 'Submit'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'slug', "The slug 'hello-world' is already in use within the parent page")

    def test_preview_on_edit(self):
        if False:
            for i in range(10):
                print('nop')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_edit', args=(self.child_page.id,))
        response = self.client.post(preview_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode(), {'is_valid': True, 'is_available': True})
        response = self.client.get(preview_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'tests/simple_page.html')
        self.assertContains(response, 'I&#39;ve been edited!', html=True)
        self.assertNotContains(response, 'Edit this page')
        self.assertNotContains(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))

    def test_preview_on_edit_no_session_key(self):
        if False:
            print('Hello World!')
        preview_url = reverse('wagtailadmin_pages:preview_on_edit', args=(self.child_page.id,))
        response = self.client.get(preview_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<title>Preview not available - Wagtail</title>', html=True)
        self.assertContains(response, '<h1 class="preview-error__title">Preview not available</h1>', html=True)

    @override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})
    @modify_settings(MIDDLEWARE={'append': 'django.middleware.cache.FetchFromCacheMiddleware', 'prepend': 'django.middleware.cache.UpdateCacheMiddleware'})
    def test_preview_does_not_cache(self):
        if False:
            return 10
        '\n        Tests solution to issue #5975\n        '
        post_data = {'title': "I've been edited one time!", 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_edit', args=(self.child_page.id,))
        self.client.post(preview_url, post_data)
        response = self.client.get(preview_url)
        self.assertContains(response, 'I&#39;ve been edited one time!', html=True)
        post_data['title'] = "I've been edited two times!"
        self.client.post(preview_url, post_data)
        response = self.client.get(preview_url)
        self.assertContains(response, 'I&#39;ve been edited two times!', html=True)

    @modify_settings(ALLOWED_HOSTS={'append': 'childpage.example.com'})
    def test_preview_uses_correct_site(self):
        if False:
            print('Hello World!')
        Site.objects.create(hostname='childpage.example.com', root_page=self.child_page)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'action-submit': 'Submit'}
        preview_url = reverse('wagtailadmin_pages:preview_on_edit', args=(self.child_page.id,))
        response = self.client.post(preview_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode(), {'is_valid': True, 'is_available': True})
        response = self.client.get(preview_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'tests/simple_page.html')
        self.assertEqual(Site.find_for_request(response.context['request']).hostname, 'childpage.example.com')

    def test_editor_picks_up_direct_model_edits(self):
        if False:
            return 10
        self.child_page.title = 'This title only exists on the live database record'
        self.child_page.save()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This title only exists on the live database record')

    def test_editor_does_not_pick_up_direct_model_edits_when_draft_edits_exist(self):
        if False:
            return 10
        self.child_page.content = 'Some content with a draft edit'
        self.child_page.save_revision()
        self.child_page = SimplePage.objects.get(id=self.child_page.id)
        self.child_page.title = 'This title only exists on the live database record'
        self.child_page.save()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'This title only exists on the live database record')
        self.assertContains(response, 'Some content with a draft edit')

    def test_editor_page_shows_live_url_in_status_when_draft_edits_exist(self):
        if False:
            return 10
        self.child_page.content = 'Some content with a draft edit'
        self.child_page.slug = 'revised-slug-in-draft-only'
        self.child_page.save_revision()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        input_field_for_draft_slug = '<input type="text" name="slug" value="revised-slug-in-draft-only" data-controller="w-slug" data-action="blur-&gt;w-slug#slugify w-sync:check-&gt;w-slug#compare w-sync:apply-&gt;w-slug#urlify:prevent" data-w-slug-compare-as-param="urlify" data-w-slug-allow-unicode-value maxlength="255" aria-describedby="panel-child-promote-child-for_search_engines-child-slug-helptext" required id="id_slug">'
        input_field_for_live_slug = '<input type="text" name="slug" value="hello-world" maxlength="255" aria-describedby="panel-child-promote-child-for_search_engines-child-slug-helptext" required id="id_slug" />'
        self.assertNotContains(response, 'href="/revised-slug-in-draft-only/"', html=True)
        self.assertContains(response, input_field_for_draft_slug, html=True)
        self.assertNotContains(response, input_field_for_live_slug, html=True)

    def test_editor_page_shows_custom_live_url_in_status_when_draft_edits_exist(self):
        if False:
            return 10
        self.single_event_page.location = 'The other side of Mars'
        self.single_event_page.slug = 'revised-slug-in-draft-only'
        self.single_event_page.save_revision()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.single_event_page.id,)))
        input_field_for_draft_slug = '<input type="text" name="slug" value="revised-slug-in-draft-only" maxlength="255" aria-describedby="panel-child-promote-child-common_page_configuration-child-slug-helptext" required id="id_slug" />'
        input_field_for_live_slug = '<input type="text" name="slug" value="mars-landing" maxlength="255" aria-describedby="panel-child-promote-child-common_page_configuration-child-slug-helptext" required id="id_slug" />'
        self.assertNotContains(response, 'href="/revised-slug-in-draft-only/pointless-suffix/"', html=True)
        self.assertContains(response, input_field_for_draft_slug, html=True)
        self.assertNotContains(response, input_field_for_live_slug, html=True)

    def test_before_edit_page_hook(self):
        if False:
            print('Hello World!')

        def hook_func(request, page):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_page', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_edit_page_hook_post(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_page', hook_func):
            post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world-new', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertEqual(Page.objects.get(id=self.child_page.id).title, 'Hello world!')

    def test_after_edit_page_hook(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('after_edit_page', hook_func):
            post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world-new', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertEqual(Page.objects.get(id=self.child_page.id).title, "I've been edited!")

    def test_after_publish_page(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, page):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('after_publish_page', hook_func):
            post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world-new', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.child_page.refresh_from_db()
        self.assertEqual(self.child_page.status_string, _('live'))

    def test_before_publish_page(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, page):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(page.id, self.child_page.id)
            return HttpResponse('Overridden!')
        with self.register_hook('before_publish_page', hook_func):
            post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world-new', 'action-publish': 'Publish'}
            response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.child_page.refresh_from_db()
        self.assertEqual(self.child_page.status_string, _('live + draft'))

    def test_override_default_action_menu_item(self):
        if False:
            while True:
                i = 10

        def hook_func(menu_items, request, context):
            if False:
                i = 10
                return i + 15
            for (index, item) in enumerate(menu_items):
                if item.name == 'action-publish':
                    menu_items.pop(index)
                    menu_items.insert(0, item)
                    break
        with self.register_hook('construct_page_action_menu', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.single_event_page.id,)))
        publish_button = '\n            <button type="submit" name="action-publish" value="action-publish" class="button button-longrunning " data-controller="w-progress" data-action="w-progress#activate" data-w-progress-active-value="Publishing…">\n                <svg class="icon icon-upload button-longrunning__icon" aria-hidden="true"><use href="#icon-upload"></use></svg>\n\n                <svg class="icon icon-spinner icon" aria-hidden="true"><use href="#icon-spinner"></use></svg><em data-w-progress-target="label">Publish</em>\n            </button>\n        '
        save_button = '\n            <button type="submit" class="button action-save button-longrunning " data-controller="w-progress" data-action="w-progress#activate" data-w-progress-active-value="Saving…" >\n                <svg class="icon icon-draft button-longrunning__icon" aria-hidden="true"><use href="#icon-draft"></use></svg>\n\n                <svg class="icon icon-spinner icon" aria-hidden="true"><use href="#icon-spinner"></use></svg>\n                <em data-w-progress-target="label">Save draft</em>\n            </button>\n        '
        self.assertContains(response, '<li>%s</li>' % save_button, html=True)
        self.assertContains(response, publish_button, html=True)
        self.assertNotContains(response, '<li>%s</li>' % publish_button, html=True)

    def test_override_publish_action_menu_item_label(self):
        if False:
            return 10

        def hook_func(menu_items, request, context):
            if False:
                print('Hello World!')
            for item in menu_items:
                if item.name == 'action-publish':
                    item.label = 'Foobar'
                    break
        with self.register_hook('construct_page_action_menu', hook_func):
            response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.single_event_page.id,)))
        self.assertContains(response, 'Foobar')

    def test_edit_alias_page(self):
        if False:
            i = 10
            return i + 15
        alias_page = self.event_page.create_alias(update_slug='new-event-page')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[alias_page.id]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/html; charset=utf-8')
        self.assertContains(response, 'id="status-sidebar-live"')
        self.assertTemplateUsed(response, 'wagtailadmin/pages/edit_alias.html')
        original_page_edit_url = reverse('wagtailadmin_pages:edit', args=[self.event_page.id])
        self.assertContains(response, f'<a class="button button-secondary" href="{original_page_edit_url}">Edit original page</a>', html=True)

    def test_post_edit_alias_page(self):
        if False:
            for i in range(10):
                print('nop')
        alias_page = self.child_page.create_alias(update_slug='new-child-page')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[alias_page.id]), post_data)
        self.assertEqual(response.status_code, 405)

    def test_edit_after_change_language_code(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that changing LANGUAGE_CODE with no corresponding database change does not break editing\n        '
        self.child_page.title = 'Hello world updated'
        self.child_page.save_revision()
        Locale.objects.update(language_code='de')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))

    def test_edit_after_change_language_code_without_revisions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that changing LANGUAGE_CODE with no corresponding database change does not break editing\n        '
        Locale.objects.update(language_code='de')
        Revision.page_revisions.filter(object_id=self.child_page.id).delete()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))

    def test_page_edit_num_queries_as_superuser(self):
        if False:
            while True:
                i = 10
        self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        with self.assertNumQueries(35):
            self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))

    def test_page_edit_num_queries_as_editor(self):
        if False:
            print('Hello World!')
        editor = self.create_user('editor', password='password')
        editor.groups.add(Group.objects.get(name='Editors'))
        self.login(username='editor')
        self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        with self.assertNumQueries(39):
            self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))

class TestPageEditReordering(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=2)
        self.event_page = EventPage(title='Event page', slug='event-page', location='the moon', audience='public', cost='free', date_from='2001-01-01')
        self.event_page.carousel_items = [EventPageCarouselItem(caption='1234567', sort_order=1), EventPageCarouselItem(caption='7654321', sort_order=2), EventPageCarouselItem(caption='abcdefg', sort_order=3)]
        self.root_page.add_child(instance=self.event_page)
        self.user = self.login()

    def check_order(self, response, expected_order):
        if False:
            print('Hello World!')
        inline_panel = response.context['edit_handler'].children[0].children[9]
        order = [child.form.instance.caption for child in inline_panel.children]
        self.assertEqual(order, expected_order)

    def test_order(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.check_order(response, ['1234567', '7654321', 'abcdefg'])

    def test_reorder(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'Event page', 'slug': 'event-page', 'date_from': '01/01/2014', 'cost': '$10', 'audience': 'public', 'location': 'somewhere', 'related_links-INITIAL_FORMS': 0, 'related_links-MAX_NUM_FORMS': 1000, 'related_links-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MAX_NUM_FORMS': 1000, 'speakers-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 1000, 'head_counts-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 3, 'carousel_items-MAX_NUM_FORMS': 1000, 'carousel_items-TOTAL_FORMS': 3, 'carousel_items-0-id': self.event_page.carousel_items.all()[0].id, 'carousel_items-0-caption': self.event_page.carousel_items.all()[0].caption, 'carousel_items-0-ORDER': 2, 'carousel_items-1-id': self.event_page.carousel_items.all()[1].id, 'carousel_items-1-caption': self.event_page.carousel_items.all()[1].caption, 'carousel_items-1-ORDER': 3, 'carousel_items-2-id': self.event_page.carousel_items.all()[2].id, 'carousel_items-2-caption': self.event_page.carousel_items.all()[2].caption, 'carousel_items-2-ORDER': 1}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.check_order(response, ['abcdefg', '1234567', '7654321'])

    def test_reorder_with_validation_error(self):
        if False:
            return 10
        post_data = {'title': '', 'slug': 'event-page', 'date_from': '01/01/2014', 'cost': '$10', 'audience': 'public', 'location': 'somewhere', 'related_links-INITIAL_FORMS': 0, 'related_links-MAX_NUM_FORMS': 1000, 'related_links-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MAX_NUM_FORMS': 1000, 'speakers-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 1000, 'head_counts-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 3, 'carousel_items-MAX_NUM_FORMS': 1000, 'carousel_items-TOTAL_FORMS': 3, 'carousel_items-0-id': self.event_page.carousel_items.all()[0].id, 'carousel_items-0-caption': self.event_page.carousel_items.all()[0].caption, 'carousel_items-0-ORDER': 2, 'carousel_items-1-id': self.event_page.carousel_items.all()[1].id, 'carousel_items-1-caption': self.event_page.carousel_items.all()[1].caption, 'carousel_items-1-ORDER': 3, 'carousel_items-2-id': self.event_page.carousel_items.all()[2].id, 'carousel_items-2-caption': self.event_page.carousel_items.all()[2].caption, 'carousel_items-2-ORDER': 1}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.event_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.check_order(response, ['abcdefg', '1234567', '7654321'])

class TestIssue197(WagtailTestUtils, TestCase):

    def test_issue_197(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        self.tagged_page = self.root_page.add_child(instance=TaggedPage(title='Tagged page', slug='tagged-page', live=False))
        self.user = self.login()
        post_data = {'title': 'Tagged page', 'slug': 'tagged-page', 'tags': 'hello, world', 'action-publish': 'Publish'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.tagged_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = TaggedPage.objects.get(id=self.tagged_page.id)
        self.assertIn('hello', page.tags.slugs())
        self.assertIn('world', page.tags.slugs())

class TestChildRelationsOnSuperclass(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        self.test_advert = Advert.objects.get(id=1)
        self.index_page = StandardIndex(title='My lovely index', slug='my-lovely-index', advert_placements=[AdvertPlacement(advert=self.test_advert)])
        self.root_page.add_child(instance=self.index_page)
        self.login()

    def test_get_create_form(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'standardindex', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Adverts')
        self.assertContains(response, 'id_advert_placements-TOTAL_FORMS')

    def test_post_create_form(self):
        if False:
            return 10
        post_data = {'title': 'New index!', 'slug': 'new-index', 'advert_placements-TOTAL_FORMS': '1', 'advert_placements-INITIAL_FORMS': '0', 'advert_placements-MAX_NUM_FORMS': '1000', 'advert_placements-0-advert': '1', 'advert_placements-0-colour': 'yellow', 'advert_placements-0-id': ''}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'standardindex', self.root_page.id)), post_data)
        page = Page.objects.get(path__startswith=self.root_page.path, slug='new-index').specific
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(page.id,)))
        self.assertEqual(page.advert_placements.count(), 1)
        self.assertEqual(page.advert_placements.first().advert.text, 'test_advert')

    def test_post_create_form_with_validation_error_in_formset(self):
        if False:
            print('Hello World!')
        post_data = {'title': 'New index!', 'slug': 'new-index', 'advert_placements-TOTAL_FORMS': '1', 'advert_placements-INITIAL_FORMS': '0', 'advert_placements-MAX_NUM_FORMS': '1000', 'advert_placements-0-advert': '1', 'advert_placements-0-colour': '', 'advert_placements-0-id': ''}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'standardindex', self.root_page.id)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This field is required.')
        self.assertContains(response, 'alwaysDirty: true')

    def test_get_edit_form(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.index_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Adverts')
        self.assertContains(response, 'id_advert_placements-TOTAL_FORMS')
        self.assertContains(response, '<div class="chooser__title" data-chooser-title id="id_advert_placements-0-advert-title">test_advert</div>')
        self.assertContains(response, '<input type="hidden" name="advert_placements-0-advert" value="1" id="id_advert_placements-0-advert">', html=True)

    def test_post_edit_form(self):
        if False:
            while True:
                i = 10
        post_data = {'title': 'My lovely index', 'slug': 'my-lovely-index', 'advert_placements-TOTAL_FORMS': '2', 'advert_placements-INITIAL_FORMS': '1', 'advert_placements-MAX_NUM_FORMS': '1000', 'advert_placements-0-advert': '1', 'advert_placements-0-colour': 'yellow', 'advert_placements-0-id': self.index_page.advert_placements.first().id, 'advert_placements-1-advert': '1', 'advert_placements-1-colour': 'purple', 'advert_placements-1-id': '', 'action-publish': 'Publish'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.index_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page = Page.objects.get(id=self.index_page.id).specific
        self.assertEqual(page.advert_placements.count(), 2)
        self.assertEqual(page.advert_placements.all()[0].advert.text, 'test_advert')
        self.assertEqual(page.advert_placements.all()[1].advert.text, 'test_advert')

    def test_post_edit_form_with_validation_error_in_formset(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': 'My lovely index', 'slug': 'my-lovely-index', 'advert_placements-TOTAL_FORMS': '1', 'advert_placements-INITIAL_FORMS': '1', 'advert_placements-MAX_NUM_FORMS': '1000', 'advert_placements-0-advert': '1', 'advert_placements-0-colour': '', 'advert_placements-0-id': self.index_page.advert_placements.first().id, 'action-publish': 'Publish'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.index_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This field is required.')
        self.assertContains(response, 'alwaysDirty: true')

class TestIssue2492(WagtailTestUtils, TestCase):
    """
    The publication submission message generation was performed using
    the Page class, as opposed to the specific_class for that Page.
    This test ensures that the specific_class url method is called
    when the 'view live' message button is created.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        child_page = SingleEventPage(title='Test Event', slug='test-event', location='test location', cost='10', date_from=datetime.datetime.now(), audience=EVENT_AUDIENCE_CHOICES[0][0])
        self.root_page.add_child(instance=child_page)
        child_page.save_revision().publish()
        self.child_page = SingleEventPage.objects.get(id=child_page.id)
        self.user = self.login()

    def test_page_edit_post_publish_url(self):
        if False:
            for i in range(10):
                print('nop')
        post_data = {'action-publish': 'Publish', 'title': self.child_page.title, 'date_from': self.child_page.date_from, 'slug': self.child_page.slug, 'audience': self.child_page.audience, 'location': self.child_page.location, 'cost': self.child_page.cost, 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data, follow=True)
        new_url = SingleEventPage.objects.get(id=self.child_page.id).url
        for message in response.context['messages']:
            self.assertIn(f'"{new_url}"', message.message)
            break

class TestIssue3982(WagtailTestUtils, TestCase):
    """
    Pages that are not associated with a site, and thus do not have a live URL,
    should not display a "View live" link in the flash message after being
    edited.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login()

    def _create_page(self, parent):
        if False:
            print('Hello World!')
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'simplepage', parent.pk)), {'title': 'Hello, world!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'publish'}, follow=True)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(parent.pk,)))
        page = SimplePage.objects.get()
        self.assertTrue(page.live)
        return (response, page)

    def test_create_accessible(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a page under the site root, check the flash message has a valid\n        "View live" button.\n        '
        (response, page) = self._create_page(Page.objects.get(pk=2))
        self.assertIsNotNone(page.url)
        self.assertTrue(any(('View live' in message.message and page.url in message.message for message in response.context['messages'])))

    def test_create_inaccessible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a page outside of the site root, check the flash message does\n        not have a "View live" button.\n        '
        (response, page) = self._create_page(Page.objects.get(pk=1))
        self.assertIsNone(page.url)
        self.assertFalse(any(('View live' in message.message for message in response.context['messages'])))

    def _edit_page(self, parent):
        if False:
            while True:
                i = 10
        page = parent.add_child(instance=SimplePage(title='Hello, world!', content='Some content'))
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(page.pk,)), {'title': 'Hello, world!', 'content': 'Some content', 'slug': 'hello-world', 'action-publish': 'publish'}, follow=True)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(parent.pk,)))
        page = SimplePage.objects.get(pk=page.pk)
        self.assertTrue(page.live)
        return (response, page)

    def test_edit_accessible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Edit a page under the site root, check the flash message has a valid\n        "View live" button.\n        '
        (response, page) = self._edit_page(Page.objects.get(pk=2))
        self.assertIsNotNone(page.url)
        self.assertTrue(any(('View live' in message.message and page.url in message.message for message in response.context['messages'])))

    def test_edit_inaccessible(self):
        if False:
            while True:
                i = 10
        '\n        Edit a page outside of the site root, check the flash message does\n        not have a "View live" button.\n        '
        (response, page) = self._edit_page(Page.objects.get(pk=1))
        self.assertIsNone(page.url)
        self.assertFalse(any(('View live' in message.message for message in response.context['messages'])))

class TestParentalM2M(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.events_index = Page.objects.get(url_path='/home/events/')
        self.christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        self.user = self.login()
        self.holiday_category = EventCategory.objects.create(name='Holiday')
        self.men_with_beards_category = EventCategory.objects.create(name='Men with beards')

    def test_create_and_save(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': "Presidents' Day", 'date_from': '2017-02-20', 'slug': 'presidents-day', 'audience': 'public', 'location': 'America', 'cost': '$1', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0, 'categories': [self.holiday_category.id, self.men_with_beards_category.id]}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'eventpage', self.events_index.id)), post_data)
        created_page = EventPage.objects.get(url_path='/home/events/presidents-day/')
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(created_page.id,)))
        created_revision = created_page.get_latest_revision_as_object()
        self.assertIn(self.holiday_category, created_revision.categories.all())
        self.assertIn(self.men_with_beards_category, created_revision.categories.all())

    def test_create_and_publish(self):
        if False:
            print('Hello World!')
        post_data = {'action-publish': 'Publish', 'title': "Presidents' Day", 'date_from': '2017-02-20', 'slug': 'presidents-day', 'audience': 'public', 'location': 'America', 'cost': '$1', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0, 'categories': [self.holiday_category.id, self.men_with_beards_category.id]}
        response = self.client.post(reverse('wagtailadmin_pages:add', args=('tests', 'eventpage', self.events_index.id)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.events_index.id,)))
        created_page = EventPage.objects.get(url_path='/home/events/presidents-day/')
        self.assertIn(self.holiday_category, created_page.categories.all())
        self.assertIn(self.men_with_beards_category, created_page.categories.all())

    def test_edit_and_save(self):
        if False:
            i = 10
            return i + 15
        post_data = {'title': 'Christmas', 'date_from': '2017-12-25', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0, 'categories': [self.holiday_category.id, self.men_with_beards_category.id]}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)))
        updated_page = EventPage.objects.get(id=self.christmas_page.id)
        created_revision = updated_page.get_latest_revision_as_object()
        self.assertIn(self.holiday_category, created_revision.categories.all())
        self.assertIn(self.men_with_beards_category, created_revision.categories.all())
        self.assertEqual(0, updated_page.categories.count())

    def test_edit_and_publish(self):
        if False:
            return 10
        post_data = {'action-publish': 'Publish', 'title': 'Christmas', 'date_from': '2017-12-25', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0, 'categories': [self.holiday_category.id, self.men_with_beards_category.id]}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.events_index.id,)))
        updated_page = EventPage.objects.get(id=self.christmas_page.id)
        self.assertEqual(2, updated_page.categories.count())
        self.assertIn(self.holiday_category, updated_page.categories.all())
        self.assertIn(self.men_with_beards_category, updated_page.categories.all())

class TestValidationErrorMessages(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.events_index = Page.objects.get(url_path='/home/events/')
        self.christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        self.user = self.login()

    def test_field_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Field errors should be shown against the relevant fields, not in the header message'
        post_data = {'title': '', 'date_from': '2017-12-25', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The page could not be saved due to validation errors')
        self.assertContains(response, 'error-message', count=1)
        self.assertContains(response, 'This field is required', count=1)

    def test_non_field_error(self):
        if False:
            print('Hello World!')
        'Non-field errors should be shown in the header message'
        post_data = {'title': 'Christmas', 'date_from': '2017-12-25', 'date_to': '2017-12-24', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The page could not be saved due to validation errors')
        self.assertContains(response, '<li>The end date must be after the start date</li>', count=1)

    def test_field_and_non_field_error(self):
        if False:
            return 10
        '\n        If both field and non-field errors exist, all errors should be shown in the header message\n        with appropriate context to identify the field; and field errors should also be shown\n        against the relevant fields.\n        '
        post_data = {'title': '', 'date_from': '2017-12-25', 'date_to': '2017-12-24', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items-TOTAL_FORMS': 0, 'carousel_items-INITIAL_FORMS': 0, 'carousel_items-MIN_NUM_FORMS': 0, 'carousel_items-MAX_NUM_FORMS': 0, 'speakers-TOTAL_FORMS': 0, 'speakers-INITIAL_FORMS': 0, 'speakers-MIN_NUM_FORMS': 0, 'speakers-MAX_NUM_FORMS': 0, 'related_links-TOTAL_FORMS': 0, 'related_links-INITIAL_FORMS': 0, 'related_links-MIN_NUM_FORMS': 0, 'related_links-MAX_NUM_FORMS': 0, 'head_counts-TOTAL_FORMS': 0, 'head_counts-INITIAL_FORMS': 0, 'head_counts-MIN_NUM_FORMS': 0, 'head_counts-MAX_NUM_FORMS': 0}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The page could not be saved due to validation errors')
        self.assertContains(response, '<li>The end date must be after the start date</li>', count=1)
        self.assertContains(response, 'error-message', count=1)
        self.assertContains(response, '<li>Title: This field is required.</li>', count=1)

class TestNestedInlinePanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            print('Hello World!')
        self.events_index = Page.objects.get(url_path='/home/events/')
        self.christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        self.speaker = self.christmas_page.speakers.first()
        self.speaker.awards.create(name='Beard Of The Year', date_awarded=datetime.date(1997, 12, 25))
        self.speaker.save()
        self.user = self.login()

    def test_get_edit_form(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<input type="text" name="speakers-0-awards-0-name" value="Beard Of The Year" maxlength="255" id="id_speakers-0-awards-0-name">', count=1, html=True)
        self.assertContains(response, '<input type="hidden" name="speakers-0-awards-TOTAL_FORMS" value="1" id="id_speakers-0-awards-TOTAL_FORMS">', count=1, html=True)
        self.assertContains(response, '<input type="text" name="speakers-0-awards-1-name" value="" maxlength="255" id="id_speakers-0-awards-1-name">', count=0, html=True)
        self.assertContains(response, '<input type="text" name="speakers-0-awards-0-date_awarded" value="1997-12-25" autocomplete="off" id="id_speakers-0-awards-0-date_awarded">', count=1, html=True)

    def test_post_edit(self):
        if False:
            i = 10
            return i + 15
        post_data = nested_form_data({'title': 'Christmas', 'date_from': '2017-12-25', 'date_to': '2017-12-25', 'slug': 'christmas', 'audience': 'public', 'location': 'The North Pole', 'cost': 'Free', 'carousel_items': inline_formset([]), 'speakers': inline_formset([{'id': self.speaker.id, 'first_name': 'Jeff', 'last_name': 'Christmas', 'awards': inline_formset([{'id': self.speaker.awards.first().id, 'name': 'Beard Of The Century', 'date_awarded': '1997-12-25'}, {'name': 'Bobsleigh Olympic gold medallist', 'date_awarded': '2018-02-01'}], initial=1)}], initial=1), 'related_links': inline_formset([]), 'head_counts': inline_formset([]), 'action-publish': 'Publish'})
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.christmas_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_explore', args=(self.events_index.id,)))
        new_christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        self.assertEqual(new_christmas_page.speakers.first().first_name, 'Jeff')
        awards = new_christmas_page.speakers.first().awards.all()
        self.assertEqual(len(awards), 2)
        self.assertEqual(awards[0].name, 'Beard Of The Century')
        self.assertEqual(awards[1].name, 'Bobsleigh Olympic gold medallist')

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelector(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.translated_christmas_page = self.christmas_page.copy_for_translation(self.fr_locale, copy_parents=True)
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.christmas_page.id]))
        self.assertContains(response, 'id="status-sidebar-english"')
        edit_translation_url = reverse('wagtailadmin_pages:edit', args=[self.translated_christmas_page.id])
        self.assertContains(response, f'href="{edit_translation_url}"')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.christmas_page.id]))
        self.assertNotContains(response, 'Page Locale:')
        edit_translation_url = reverse('wagtailadmin_pages:edit', args=[self.translated_christmas_page.id])
        self.assertNotContains(response, f'href="{edit_translation_url}"')

    def test_locale_dropdown_not_present_without_permission_to_edit(self):
        if False:
            for i in range(10):
                print('nop')
        en_events_index = Page.objects.get(url_path='/home/events/')
        group = Group.objects.get(name='Moderators')
        GroupPagePermission.objects.create(group=group, page=en_events_index, permission_type='change')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.groups.add(group)
        self.user.save()
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.christmas_page.id]))
        self.assertContains(response, 'id="status-sidebar-english"')
        edit_translation_url = reverse('wagtailadmin_pages:edit', args=[self.translated_christmas_page.id])
        self.assertNotContains(response, f'href="{edit_translation_url}"')

class TestPageSubscriptionSettings(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.root_page = Page.objects.get(id=2)
        child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=child_page)
        child_page.save_revision().publish()
        self.child_page = SimplePage.objects.get(id=child_page.id)
        self.user = self.login()

    def test_commment_notifications_switched_off(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<input type="checkbox" name="comment_notifications" id="id_comment_notifications">')
        self.assertTrue(PageSubscription.objects.filter(page=self.child_page, user=self.user, comment_notifications=False).exists())

    def test_commment_notifications_switched_on(self):
        if False:
            i = 10
            return i + 15
        PageSubscription.objects.create(page=self.child_page, user=self.user, comment_notifications=True)
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<input type="checkbox" name="comment_notifications" id="id_comment_notifications" checked>')

    def test_post_with_comment_notifications_switched_on(self):
        if False:
            print('Hello World!')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comment_notifications': 'on'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        page = Page.objects.get(path__startswith=self.root_page.path, slug='hello-world').specific
        subscription = page.subscribers.get()
        self.assertEqual(subscription.user, self.user)
        self.assertTrue(subscription.comment_notifications)

    def test_post_with_comment_notifications_switched_off(self):
        if False:
            while True:
                i = 10
        subscription = PageSubscription.objects.create(page=self.child_page, user=self.user, comment_notifications=True)
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        subscription.refresh_from_db()
        self.assertFalse(subscription.comment_notifications)

    @override_settings(WAGTAILADMIN_COMMENTS_ENABLED=False)
    def test_comments_disabled(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.assertNotContains(response, 'data-side-panel-toggle="comments"')
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<input type="checkbox" name="comment_notifications" id="id_comment_notifications">')

    @override_settings(WAGTAILADMIN_COMMENTS_ENABLED=False)
    def test_post_comments_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comment_notifications': 'on'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.assertFalse(PageSubscription.objects.get().comment_notifications)

class TestCommenting(WagtailTestUtils, TestCase):
    """
    Tests both the comment notification and audit logging logic of the edit page view.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=child_page)
        child_page.save_revision().publish()
        self.child_page = SimplePage.objects.get(id=child_page.id)
        self.user = self.login()
        self.subscriber = self.create_user('subscriber')
        self.non_subscriber = self.create_user('non-subscriber')
        self.non_subscriber_2 = self.create_user('non-subscriber-2')
        self.never_emailed_user = self.create_user('never-emailed')
        PageSubscription.objects.create(page=self.child_page, user=self.user, comment_notifications=True)
        PageSubscription.objects.create(page=self.child_page, user=self.subscriber, comment_notifications=True)
        comment_on_other_page = Comment.objects.create(page=self.root_page, user=self.never_emailed_user, text='a comment')
        CommentReply.objects.create(user=self.never_emailed_user, comment=comment_on_other_page, text='a reply')

    def assertNeverEmailedWrongUser(self):
        if False:
            while True:
                i = 10
        self.assertNotIn(self.never_emailed_user.email, [to for email in mail.outbox for to in email.to])

    def test_new_comment(self):
        if False:
            return 10
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '0', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': '', 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        comment = self.child_page.wagtail_admin_comments.get()
        self.assertEqual(comment.text, 'A test comment')
        self.assertEqual(len(mail.outbox), 1)
        self.assertNeverEmailedWrongUser()
        self.assertEqual(mail.outbox[0].to, [self.subscriber.email])
        self.assertEqual(mail.outbox[0].subject, 'test@email.com has updated comments on "I\'ve been edited! (simple page)"')
        self.assertIn('New comments:\n - "A test comment"\n\n', mail.outbox[0].body)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.create')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)

    def test_edit_comment(self):
        if False:
            i = 10
            return i + 15
        comment = Comment.objects.create(page=self.child_page, user=self.user, text='A test comment', contentpath='title')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'Edited', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        comment.refresh_from_db()
        self.assertEqual(comment.text, 'Edited')
        self.assertEqual(len(mail.outbox), 0)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.edit')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)

    def test_edit_another_users_comment(self):
        if False:
            for i in range(10):
                print('nop')
        comment = Comment.objects.create(page=self.child_page, user=self.subscriber, text='A test comment', contentpath='title')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'Edited', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertEqual(response.context['form'].formsets['comments'].errors, [{'__all__': ["You cannot edit another user's comment."]}])
        self.child_page.refresh_from_db()
        comment.refresh_from_db()
        self.assertNotEqual(comment.text, 'Edited')
        self.assertFalse(PageLogEntry.objects.filter(action='wagtail.comments.edit').exists())

    def test_resolve_comment(self):
        if False:
            print('Hello World!')
        comment = Comment.objects.create(page=self.child_page, user=self.non_subscriber, text='A test comment', contentpath='title')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': 'on', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        comment.refresh_from_db()
        self.assertTrue(comment.resolved_at)
        self.assertEqual(comment.resolved_by, self.user)
        self.assertEqual(len(mail.outbox), 2)
        self.assertNeverEmailedWrongUser()
        self.assertEqual(mail.outbox[0].to, [self.non_subscriber.email])
        self.assertEqual(mail.outbox[0].subject, 'test@email.com has updated comments on "I\'ve been edited! (simple page)"')
        self.assertIn('Resolved comments:\n - "A test comment"\n\n', mail.outbox[0].body)
        self.assertEqual(mail.outbox[1].to, [self.subscriber.email])
        self.assertEqual(mail.outbox[1].subject, 'test@email.com has updated comments on "I\'ve been edited! (simple page)"')
        self.assertIn('Resolved comments:\n - "A test comment"\n\n', mail.outbox[1].body)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.resolve')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)

    def test_delete_comment(self):
        if False:
            for i in range(10):
                print('nop')
        comment = Comment.objects.create(page=self.child_page, user=self.user, text='A test comment', contentpath='title')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': 'on', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        self.assertFalse(self.child_page.wagtail_admin_comments.exists())
        self.assertEqual(len(mail.outbox), 1)
        self.assertNeverEmailedWrongUser()
        self.assertEqual(mail.outbox[0].to, [self.subscriber.email])
        self.assertEqual(mail.outbox[0].subject, 'test@email.com has updated comments on "I\'ve been edited! (simple page)"')
        self.assertIn('Deleted comments:\n - "A test comment"\n\n', mail.outbox[0].body)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.delete')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)

    def test_new_reply(self):
        if False:
            for i in range(10):
                print('nop')
        comment = Comment.objects.create(page=self.child_page, user=self.non_subscriber, text='A test comment', contentpath='title')
        reply = CommentReply.objects.create(comment=comment, user=self.non_subscriber_2, text='an old reply')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '2', 'comments-0-replies-INITIAL_FORMS': '1', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '', 'comments-0-replies-0-id': str(reply.id), 'comments-0-replies-0-text': 'an old reply', 'comments-0-replies-1-id': '', 'comments-0-replies-1-text': 'a new reply'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        comment.refresh_from_db()
        self.assertEqual(comment.replies.last().text, 'a new reply')
        self.assertEqual(len(mail.outbox), 3)
        self.assertNeverEmailedWrongUser()
        recipients = [mail.to for mail in mail.outbox]
        self.assertIn([self.non_subscriber_2.email], recipients)
        self.assertIn([self.non_subscriber.email], recipients)
        self.assertIn([self.subscriber.email], recipients)
        self.assertEqual(mail.outbox[2].subject, 'test@email.com has updated comments on "I\'ve been edited! (simple page)"')
        self.assertIn('  New replies to: "A test comment"\n   - "a new reply"', mail.outbox[2].body)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.create_reply')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)
        self.assertNotEqual(log_entry.data['reply']['id'], reply.id)
        self.assertEqual(log_entry.data['reply']['text'], 'a new reply')

    def test_edit_reply(self):
        if False:
            for i in range(10):
                print('nop')
        comment = Comment.objects.create(page=self.child_page, user=self.non_subscriber, text='A test comment', contentpath='title')
        reply = CommentReply.objects.create(comment=comment, user=self.user, text='an old reply')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '1', 'comments-0-replies-INITIAL_FORMS': '1', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '', 'comments-0-replies-0-id': str(reply.id), 'comments-0-replies-0-text': 'an edited reply'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        reply.refresh_from_db()
        self.assertEqual(reply.text, 'an edited reply')
        self.assertEqual(len(mail.outbox), 0)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.edit_reply')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)
        self.assertEqual(log_entry.data['reply']['id'], reply.id)
        self.assertEqual(log_entry.data['reply']['text'], 'an edited reply')

    def test_delete_reply(self):
        if False:
            return 10
        comment = Comment.objects.create(page=self.child_page, user=self.non_subscriber, text='A test comment', contentpath='title')
        reply = CommentReply.objects.create(comment=comment, user=self.user, text='an old reply')
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '1', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': str(comment.id), 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '1', 'comments-0-replies-INITIAL_FORMS': '1', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '', 'comments-0-replies-0-id': str(reply.id), 'comments-0-replies-0-text': 'an old reply', 'comments-0-replies-0-DELETE': 'on'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        self.child_page.refresh_from_db()
        self.assertFalse(comment.replies.exists())
        self.assertEqual(len(mail.outbox), 0)
        log_entry = PageLogEntry.objects.get(action='wagtail.comments.delete_reply')
        self.assertEqual(log_entry.page, self.child_page.page_ptr)
        self.assertEqual(log_entry.user, self.user)
        self.assertEqual(log_entry.revision, self.child_page.get_latest_revision())
        self.assertEqual(log_entry.data['comment']['id'], comment.id)
        self.assertEqual(log_entry.data['comment']['contentpath'], comment.contentpath)
        self.assertEqual(log_entry.data['comment']['text'], comment.text)
        self.assertEqual(log_entry.data['reply']['id'], reply.id)
        self.assertEqual(log_entry.data['reply']['text'], reply.text)

    def test_updated_comments_notifications_profile_setting(self):
        if False:
            for i in range(10):
                print('nop')
        profile = UserProfile.get_for_user(self.subscriber)
        profile.updated_comments_notifications = False
        profile.save()
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '0', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': '', 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        comment = self.child_page.wagtail_admin_comments.get()
        self.assertEqual(comment.text, 'A test comment')
        self.assertEqual(len(mail.outbox), 0)

    def test_updated_comments_notifications_active_users_only(self):
        if False:
            while True:
                i = 10
        self.subscriber.is_active = False
        self.subscriber.save()
        post_data = {'title': "I've been edited!", 'content': 'Some content', 'slug': 'hello-world', 'comments-TOTAL_FORMS': '1', 'comments-INITIAL_FORMS': '0', 'comments-MIN_NUM_FORMS': '0', 'comments-MAX_NUM_FORMS': '', 'comments-0-DELETE': '', 'comments-0-resolved': '', 'comments-0-id': '', 'comments-0-contentpath': 'title', 'comments-0-text': 'A test comment', 'comments-0-position': '', 'comments-0-replies-TOTAL_FORMS': '0', 'comments-0-replies-INITIAL_FORMS': '0', 'comments-0-replies-MIN_NUM_FORMS': '0', 'comments-0-replies-MAX_NUM_FORMS': '0'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        comment = self.child_page.wagtail_admin_comments.get()
        self.assertEqual(comment.text, 'A test comment')
        self.assertEqual(len(mail.outbox), 0)

class TestCommentOutput(WagtailTestUtils, TestCase):
    """
    Test that the correct set of comments is output on the edit page view
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        self.child_page = StreamPage(title='Hello world!', body=[{'id': '234', 'type': 'product', 'value': {'name': 'Cuddly toy', 'price': '$9.95'}}])
        self.root_page.add_child(instance=self.child_page)
        self.child_page.save_revision().publish()
        self.user = self.login()

    def test_only_comments_with_valid_paths_are_shown(self):
        if False:
            print('Hello World!')
        Comment.objects.create(user=self.user, page=self.child_page, text='A test comment', contentpath='title')
        Comment.objects.create(user=self.user, page=self.child_page, text="A comment on a field that doesn't exist", contentpath='sillytitle')
        Comment.objects.create(user=self.user, page=self.child_page, text='This is quite expensive', contentpath='body.234.price')
        Comment.objects.create(user=self.user, page=self.child_page, text="A comment on a block that doesn't exist", contentpath='body.234.colour')
        response = self.client.get(reverse('wagtailadmin_pages:edit', args=[self.child_page.id]))
        soup = self.get_soup(response.content)
        comments_data_json = soup.select_one('#comments-data').string
        comments_data = json.loads(comments_data_json)
        comment_text = [comment['text'] for comment in comments_data['comments']]
        comment_text.sort()
        self.assertEqual(comment_text, ['A test comment', 'This is quite expensive'])