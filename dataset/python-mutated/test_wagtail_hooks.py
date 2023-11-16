from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail import hooks
from wagtail.actions.create_alias import CreatePageAliasAction
from wagtail.actions.move_page import MovePageAction
from wagtail.admin import widgets as wagtailadmin_widgets
from wagtail.contrib.simple_translation.wagtail_hooks import page_listing_more_buttons, register_submit_translation_permission
from wagtail.models import Locale, Page
from wagtail.test.i18n.models import TestPage
from wagtail.test.utils import WagtailTestUtils

class Utils(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.en_locale = Locale.objects.first()
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.de_locale = Locale.objects.create(language_code='de')
        self.en_homepage = Page.objects.get(depth=2)
        self.fr_homepage = self.en_homepage.copy_for_translation(self.fr_locale)
        self.de_homepage = self.en_homepage.copy_for_translation(self.de_locale)
        self.en_blog_index = TestPage(title='Blog', slug='blog')
        self.en_homepage.add_child(instance=self.en_blog_index)
        self.en_blog_post = TestPage(title='Blog post', slug='blog-post')
        self.en_blog_index.add_child(instance=self.en_blog_post)

class TestWagtailHooksURLs(TestCase):

    def test_register_admin_urls_page(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(reverse('simple_translation:submit_page_translation', args=(1,)), '/admin/translation/submit/page/1/')

    def test_register_admin_urls_snippet(self):
        if False:
            i = 10
            return i + 15
        app_label = 'foo'
        model_name = 'bar'
        pk = 1
        self.assertEqual(reverse('simple_translation:submit_snippet_translation', args=(app_label, model_name, pk)), '/admin/translation/submit/snippet/foo/bar/1/')

class TestWagtailHooksPermission(Utils):

    def test_register_submit_translation_permission(self):
        if False:
            print('Hello World!')
        assert list(register_submit_translation_permission().values_list('id', flat=True)) == [Permission.objects.get(content_type__app_label='simple_translation', codename='submit_translation').id]

class TestWagtailHooksButtons(Utils):

    def test_page_listing_more_buttons(self):
        if False:
            print('Hello World!')
        root_page = self.en_blog_index.get_root()
        if get_user_model().USERNAME_FIELD == 'email':
            user = get_user_model().objects.create_user(email='jos@example.com')
        else:
            user = get_user_model().objects.create_user(username='jos')
        assert list(page_listing_more_buttons(root_page, user)) == []
        home_page = self.en_homepage
        assert list(page_listing_more_buttons(root_page, user)) == []
        perm = Permission.objects.get(codename='submit_translation')
        if get_user_model().USERNAME_FIELD == 'email':
            user = get_user_model().objects.create_user(email='henk@example.com')
        else:
            user = get_user_model().objects.create_user(username='henk')
        user.user_permissions.add(perm)
        group = Group.objects.get(name='Editors')
        user.groups.add(group)
        assert list(page_listing_more_buttons(home_page, user)) == []
        blog_page = self.en_blog_post
        assert isinstance(list(page_listing_more_buttons(blog_page, user))[0], wagtailadmin_widgets.Button)

class TestConstructSyncedPageTreeListHook(Utils):

    def unpublish_hook(self, pages, action):
        if False:
            return 10
        self.assertEqual(action, 'unpublish')
        self.assertIsInstance(pages, list)

    def missing_hook_action(self, pages, action):
        if False:
            i = 10
            return i + 15
        self.assertEqual(action, '')
        self.assertIsInstance(pages, list)

    def test_double_registered_hook(self):
        if False:
            return 10
        with hooks.register_temporarily('construct_translated_pages_to_cascade_actions', self.unpublish_hook):
            defined_hooks = hooks.get_hooks('construct_translated_pages_to_cascade_actions')
            self.assertEqual(len(defined_hooks), 2)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True)
    def test_page_tree_sync_on(self):
        if False:
            return 10
        with hooks.register_temporarily('construct_translated_pages_to_cascade_actions', self.unpublish_hook):
            for fn in hooks.get_hooks('construct_translated_pages_to_cascade_actions'):
                response = fn([self.en_homepage], 'unpublish')
                if response:
                    self.assertIsInstance(response, dict)
                    self.assertEqual(len(response.items()), 1)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=False)
    def test_page_tree_sync_off(self):
        if False:
            return 10
        with hooks.register_temporarily('construct_translated_pages_to_cascade_actions', self.unpublish_hook):
            for fn in hooks.get_hooks('construct_translated_pages_to_cascade_actions'):
                response = fn([self.en_homepage], 'unpublish')
                self.assertIsNone(response)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True)
    def test_missing_hook_action(self):
        if False:
            return 10
        with hooks.register_temporarily('construct_translated_pages_to_cascade_actions', self.missing_hook_action):
            for fn in hooks.get_hooks('construct_translated_pages_to_cascade_actions'):
                response = fn([self.en_homepage], '')
                if response is not None:
                    self.assertIsInstance(response, dict)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True, WAGTAIL_I18N_ENABLED=True)
    def test_other_l10n_pages_were_unpublished(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()
        self.fr_homepage.live = True
        self.fr_homepage.save()
        self.assertTrue(self.en_homepage.live)
        self.assertTrue(self.fr_homepage.live)
        response = self.client.post(reverse('wagtailadmin_pages:unpublish', args=(self.en_homepage.id,)), {'include_descendants': False}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.en_homepage.refresh_from_db()
        self.fr_homepage.refresh_from_db()
        self.assertFalse(self.en_homepage.live)
        self.assertFalse(self.fr_homepage.live)

class TestMovingTranslatedPages(Utils):

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True, WAGTAIL_I18N_ENABLED=True)
    def test_move_translated_pages(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.de_blog_index = self.en_blog_index.copy_for_translation(self.de_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        self.de_blog_post = self.en_blog_post.copy_for_translation(self.de_locale)
        self.assertEqual(self.en_blog_post.get_parent().id, self.en_blog_index.id)
        original_translated_parent_ids = [p.id for p in self.en_blog_index.get_translations()]
        self.assertIn(self.fr_blog_post.get_parent().id, original_translated_parent_ids)
        self.assertIn(self.de_blog_post.get_parent().id, original_translated_parent_ids)
        response = self.client.post(reverse('wagtailadmin_pages:move_confirm', args=(self.en_blog_post.id, self.en_homepage.id)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.fr_blog_post.refresh_from_db()
        self.de_blog_post.refresh_from_db()
        home_page_translation_ids = [p.id for p in self.en_homepage.get_translations()]
        self.assertIn(self.fr_blog_post.get_parent(update=True).id, home_page_translation_ids)
        self.assertIn(self.de_blog_post.get_parent(update=True).id, home_page_translation_ids)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=False)
    def test_unmovable_translation_pages(self):
        if False:
            while True:
                i = 10
        "\n        Test that moving a page with WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE\n        disabled doesn't apply to its translations.\n        "
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.de_blog_index = self.en_blog_index.copy_for_translation(self.de_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        self.de_blog_post = self.en_blog_post.copy_for_translation(self.de_locale)
        self.assertEqual(self.en_blog_post.get_parent().id, self.en_blog_index.id)
        original_translated_parent_ids = [p.id for p in self.en_blog_index.get_translations()]
        self.assertIn(self.fr_blog_post.get_parent().id, original_translated_parent_ids)
        self.assertIn(self.de_blog_post.get_parent().id, original_translated_parent_ids)
        response = self.client.post(reverse('wagtailadmin_pages:move_confirm', args=(self.en_blog_post.id, self.en_homepage.id)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.en_blog_post.refresh_from_db()
        self.fr_blog_post.refresh_from_db()
        self.de_blog_post.refresh_from_db()
        self.assertEqual(self.en_blog_post.get_parent(update=True).id, self.en_homepage.id)
        self.assertIn(self.fr_blog_post.get_parent(update=True).id, original_translated_parent_ids)
        self.assertIn(self.de_blog_post.get_parent(update=True).id, original_translated_parent_ids)

    @override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True, WAGTAIL_I18N_ENABLED=True)
    def test_translation_count_in_context(self):
        if False:
            print('Hello World!')
        'Test translation count is correct in the confirm_move.html template.'
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.de_blog_index = self.en_blog_index.copy_for_translation(self.de_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        self.de_blog_post = self.en_blog_post.copy_for_translation(self.de_locale, alias=True)
        response = self.client.get(reverse('wagtailadmin_pages:move_confirm', args=(self.en_blog_post.id, self.en_homepage.id)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['translations_to_move_count'], 1)
        self.assertIn('This will also move one translation of this page and its child pages', response.content.decode('utf-8'))

@override_settings(WAGTAILSIMPLETRANSLATION_SYNC_PAGE_TREE=True, WAGTAIL_I18N_ENABLED=True)
class TestDeletingTranslatedPages(Utils):

    def delete_hook(self, pages, action):
        if False:
            while True:
                i = 10
        self.assertEqual(action, 'delete')
        self.assertIsInstance(pages, list)

    def test_construct_translated_pages_to_cascade_actions_when_deleting(self):
        if False:
            while True:
                i = 10
        with hooks.register_temporarily('construct_translated_pages_to_cascade_actions', self.delete_hook):
            for fn in hooks.get_hooks('construct_translated_pages_to_cascade_actions'):
                response = fn([self.en_homepage], 'delete')
                if response is not None:
                    self.assertIsInstance(response, dict)
                    self.assertEqual(len(response.items()), 1)

    def test_delete_translated_pages(self):
        if False:
            i = 10
            return i + 15
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.en_blog_post.id,)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(Page.objects.filter(pk=self.fr_blog_post.id).first())

    def test_delete_confirmation_template(self):
        if False:
            while True:
                i = 10
        'Test the context info is correct in the confirm_delete.html template.'
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        new_page = CreatePageAliasAction(self.en_blog_post, recursive=False, parent=self.en_blog_index, update_slug='alias-page-slug', user=None)
        new_page.execute(skip_permission_checks=True)
        response = self.client.get(reverse('wagtailadmin_pages:delete', args=(self.en_blog_post.id,)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['translation_count'], 1)
        self.assertEqual(response.context['translation_descendant_count'], 0)
        self.assertIn('Deleting this page will also delete 1 translation of this page.', response.content.decode('utf-8'))

    def test_deleting_page_with_divergent_translation_tree(self):
        if False:
            while True:
                i = 10
        self.login()
        self.en_new_parent = TestPage(title='Test Parent', slug='test-parent')
        self.en_homepage.add_child(instance=self.en_new_parent)
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale)
        self.fr_new_parent = self.en_new_parent.copy_for_translation(self.fr_locale)
        action = MovePageAction(self.fr_blog_post, self.fr_new_parent, pos='last-child', user=None)
        action.execute(skip_permission_checks=True)
        self.fr_blog_post.refresh_from_db()
        self.en_blog_post.refresh_from_db()
        self.assertEqual(self.fr_blog_post.get_parent(update=True).id, self.fr_new_parent.id)
        self.assertEqual(self.en_blog_post.get_parent(update=True).id, self.en_blog_index.id)
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.en_blog_post.id,)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Page.objects.filter(pk=self.en_blog_post.id).exists())
        self.assertTrue(Page.objects.filter(pk=self.fr_blog_post.id).exists())
        self.fr_blog_post.refresh_from_db()
        self.assertEqual(self.fr_blog_post.get_parent(update=True).id, self.fr_new_parent.id)

    def test_alias_pages_when_deleting_source_page(self):
        if False:
            while True:
                i = 10
        '\n        When deleting a page that has an alias page in the same tree, the alias page\n        should continue to exist while the original page should be deleted\n        while using the `construct_translated_pages_to_cascade_actions` hook is active.\n        '
        self.login()
        self.assertEqual(self.en_blog_post.get_parent().id, self.en_blog_index.id)
        action = CreatePageAliasAction(self.en_blog_post, recursive=False, parent=self.en_blog_index, update_slug='sample-slug', user=None)
        new_page = action.execute(skip_permission_checks=True)
        self.assertEqual(new_page.get_parent().id, self.en_blog_index.id)
        self.assertEqual(new_page.alias_of_id, self.en_blog_post.id)
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.en_blog_post.id,)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Page.objects.filter(pk=self.en_blog_post.id).exists())
        self.assertTrue(Page.objects.filter(pk=new_page.id).exists())

    def test_translation_alias_pages_when_deleting_source_page(self):
        if False:
            i = 10
            return i + 15
        '\n        When deleting a page that has an alias page, the alias page\n        should be deleted while using the `construct_translated_pages_to_cascade_actions`\n        hook is active.\n        '
        self.login()
        self.fr_blog_index = self.en_blog_index.copy_for_translation(self.fr_locale)
        self.fr_blog_post = self.en_blog_post.copy_for_translation(self.fr_locale, alias=True)
        self.assertEqual(self.fr_blog_post.alias_of_id, self.en_blog_post.id)
        self.assertEqual(self.fr_blog_post.get_parent().id, self.fr_blog_index.id)
        translation_ids = [p.id for p in self.fr_blog_post.get_translations()]
        self.assertIn(self.fr_blog_post.alias_of_id, translation_ids)
        self.assertEqual(self.fr_blog_post.alias_of_id, self.en_blog_post.id)
        self.assertEqual(self.fr_blog_post.locale.language_code, 'fr')
        en_root = Page.objects.filter(depth__gt=1, locale=self.en_locale).first()
        fr_root = Page.objects.filter(depth__gt=1, locale=self.fr_locale).first()
        self.assertIn(self.en_blog_post, en_root.get_descendants().specific())
        self.assertIn(self.fr_blog_post, fr_root.get_descendants().specific())
        response = self.client.post(reverse('wagtailadmin_pages:delete', args=(self.en_blog_post.id,)), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Page.objects.filter(pk=self.en_blog_post.id).exists())
        self.assertFalse(Page.objects.filter(pk=self.fr_blog_post.id).exists())
        self.assertNotIn(self.en_blog_post, en_root.get_descendants().specific())
        self.assertNotIn(self.fr_blog_post, fr_root.get_descendants().specific())