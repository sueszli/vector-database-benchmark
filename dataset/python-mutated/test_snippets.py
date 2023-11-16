import datetime
import json
from io import StringIO
from unittest import mock
from django.conf import settings
from django.contrib.admin.utils import quote
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser, Permission
from django.contrib.contenttypes.models import ContentType
from django.core import checks, management
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.handlers.wsgi import WSGIRequest
from django.http import HttpRequest, HttpResponse
from django.test import RequestFactory, TestCase, TransactionTestCase
from django.test.utils import override_settings
from django.urls import reverse
from django.utils.timezone import make_aware, now
from freezegun import freeze_time
from taggit.models import Tag
from wagtail import hooks
from wagtail.admin.admin_url_finder import AdminURLFinder
from wagtail.admin.forms import WagtailAdminModelForm
from wagtail.admin.panels import FieldPanel, ObjectList, get_edit_handler
from wagtail.admin.widgets.button import ButtonWithDropdown
from wagtail.blocks.field_block import FieldBlockAdapter
from wagtail.models import Locale, ModelLogEntry, Revision
from wagtail.signals import published, unpublished
from wagtail.snippets.action_menu import ActionMenuItem, get_base_snippet_action_menu_items
from wagtail.snippets.blocks import SnippetChooserBlock
from wagtail.snippets.models import SNIPPET_MODELS, register_snippet
from wagtail.snippets.widgets import AdminSnippetChooser, SnippetChooserAdapter, SnippetListingButton
from wagtail.test.snippets.forms import FancySnippetForm
from wagtail.test.snippets.models import AlphaSnippet, FancySnippet, FileUploadSnippet, NonAutocompleteSearchableSnippet, RegisterDecorator, RegisterFunction, SearchableSnippet, StandardSnippet, StandardSnippetWithCustomPrimaryKey, TranslatableSnippet, ZuluSnippet
from wagtail.test.testapp.models import Advert, AdvertWithCustomPrimaryKey, AdvertWithCustomUUIDPrimaryKey, AdvertWithTabbedInterface, DraftStateCustomPrimaryKeyModel, DraftStateModel, FullFeaturedSnippet, MultiPreviewModesModel, RevisableChildModel, RevisableModel, SnippetChooserModel, SnippetChooserModelWithCustomPrimaryKey, VariousOnDeleteModel
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.template_tests import AdminTemplateTestUtils
from wagtail.test.utils.timestamps import submittable_timestamp
from wagtail.utils.deprecation import RemovedInWagtail70Warning
from wagtail.utils.timestamps import render_timestamp

class TestSnippetIndexView(AdminTemplateTestUtils, WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.user = self.login()

    def get(self, params={}):
        if False:
            print('Hello World!')
        return self.client.get(reverse('wagtailsnippets:index'), params)

    def test_get_with_limited_permissions(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.get()
        self.assertEqual(response.status_code, 302)

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/index.html')
        self.assertBreadcrumbsItemsRendered([{'url': '', 'label': 'Snippets'}], response.content)

    def test_displays_snippet(self):
        if False:
            while True:
                i = 10
        self.assertContains(self.get(), 'Adverts')

class TestSnippetListView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.login()
        user_model = get_user_model()
        self.user = user_model.objects.get()

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailsnippets_tests_advert:list'), params)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        self.assertEqual(response.context['header_icon'], 'snippet')

    def get_with_limited_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.get()
        self.assertEqual(response.status_code, 302)

    def test_ordering(self):
        if False:
            return 10
        '\n        Listing should be ordered descending by PK if no ordering has been set on the model\n        '
        for i in range(1, 11):
            Advert.objects.create(pk=i, text='advert %d' % i)
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_obj'][0].text, 'advert 10')

    def test_simple_pagination(self):
        if False:
            print('Hello World!')
        response = self.get({'p': 1})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        response = self.get({'p': 9999})
        self.assertEqual(response.status_code, 404)

    def test_displays_add_button(self):
        if False:
            i = 10
            return i + 15
        self.assertContains(self.get(), 'Add advert')

    def test_not_searchable(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.get().context['is_searchable'])

    def test_register_snippet_listing_buttons_hook(self):
        if False:
            return 10
        advert = Advert.objects.create(text='My Lovely advert')

        def snippet_listing_buttons(snippet, user, next_url=None):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(snippet, advert)
            self.assertEqual(user, self.user)
            self.assertEqual(next_url, reverse('wagtailsnippets_tests_advert:list'))
            yield SnippetListingButton('Another useless snippet listing button', '/custom-url', priority=10)
        with hooks.register_temporarily('register_snippet_listing_buttons', snippet_listing_buttons):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/buttons.html')
        soup = self.get_soup(response.content)
        actions = soup.select_one('tbody tr td ul.actions')
        top_level_custom_button = actions.select_one("li > a[href='/custom-url']")
        self.assertIsNone(top_level_custom_button)
        custom_button = actions.select_one("li [data-controller='w-dropdown'] a[href='/custom-url']")
        self.assertIsNotNone(custom_button)
        self.assertEqual(custom_button.text.strip(), 'Another useless snippet listing button')

    def test_register_snippet_listing_buttons_hook_with_dropdown(self):
        if False:
            print('Hello World!')
        advert = Advert.objects.create(text='My Lovely advert')

        def snippet_listing_buttons(snippet, user, next_url=None):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(snippet, advert)
            self.assertEqual(user, self.user)
            self.assertEqual(next_url, reverse('wagtailsnippets_tests_advert:list'))
            yield ButtonWithDropdown(label='Moar pls!', buttons=[SnippetListingButton('Alrighty', '/cheers', priority=10)])
        with hooks.register_temporarily('register_snippet_listing_buttons', snippet_listing_buttons):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/buttons.html')
        soup = self.get_soup(response.content)
        actions = soup.select_one('tbody tr td ul.actions')
        nested_dropdown = actions.select_one("li [data-controller='w-dropdown'] [data-controller='w-dropdown']")
        self.assertIsNone(nested_dropdown)
        dropdown_buttons = actions.select("li > [data-controller='w-dropdown']")
        self.assertEqual(len(dropdown_buttons), 2)
        custom_dropdown = None
        for button in dropdown_buttons:
            if 'Moar pls!' in button.text.strip():
                custom_dropdown = button
        self.assertIsNotNone(custom_dropdown)
        self.assertEqual(custom_dropdown.select_one('button').text.strip(), 'Moar pls!')
        custom_button = custom_dropdown.find('a', attrs={'href': '/cheers'})
        self.assertIsNotNone(custom_button)
        self.assertEqual(custom_button.text.strip(), 'Alrighty')

    def test_construct_snippet_listing_buttons_hook(self):
        if False:
            for i in range(10):
                print('nop')
        Advert.objects.create(text='My Lovely advert')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/buttons.html')
        soup = self.get_soup(response.content)
        dropdowns = soup.select("tbody tr td ul.actions > li > [data-controller='w-dropdown']")
        self.assertEqual(len(dropdowns), 1)
        more_dropdown = dropdowns[0]
        dummy_button = more_dropdown.find('a', attrs={'href': '/dummy-button'})
        self.assertIsNotNone(dummy_button)
        self.assertEqual(dummy_button.text.strip(), 'Dummy Button')

    def test_construct_snippet_listing_buttons_hook_contains_default_buttons(self):
        if False:
            print('Hello World!')
        advert = Advert.objects.create(text='My Lovely advert')
        delete_url = reverse('wagtailsnippets_tests_advert:delete', args=[quote(advert.pk)])

        def hide_delete_button_for_lovely_advert(buttons, snippet, user):
            if False:
                return 10
            self.assertEqual(len(buttons), 3)
            buttons[:] = [button for button in buttons if button.url != delete_url]
            self.assertEqual(len(buttons), 2)
        with hooks.register_temporarily('construct_snippet_listing_buttons', hide_delete_button_for_lovely_advert):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/buttons.html')
        self.assertNotContains(response, delete_url)

    def test_construct_snippet_listing_buttons_hook_deprecated_context(self):
        if False:
            i = 10
            return i + 15
        advert = Advert.objects.create(text='My Lovely advert')

        def register_snippet_listing_button_item(buttons, snippet, user, context):
            if False:
                while True:
                    i = 10
            self.assertEqual(snippet, advert)
            self.assertEqual(user, self.user)
            self.assertEqual(context, {})
        with hooks.register_temporarily('construct_snippet_listing_buttons', register_snippet_listing_button_item), self.assertWarnsMessage(RemovedInWagtail70Warning, 'construct_snippet_listing_buttons hook no longer accepts a context argument'):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/buttons.html')

    def test_use_latest_draft_as_title(self):
        if False:
            i = 10
            return i + 15
        snippet = DraftStateModel.objects.create(text='Draft-enabled Foo, Published')
        snippet.save_revision().publish()
        snippet.text = 'Draft-enabled Bar, In Draft'
        snippet.save_revision()
        response = self.client.get(reverse('wagtailsnippets_tests_draftstatemodel:list'))
        edit_url = reverse('wagtailsnippets_tests_draftstatemodel:edit', args=[quote(snippet.pk)])
        self.assertContains(response, f'<a href="{edit_url}">Draft-enabled Bar, In Draft</a>', html=True)

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelectorOnList(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_snippetstests_translatablesnippet:list'))
        switch_to_french_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:list') + '?locale=fr'
        self.assertContains(response, f'<a href="{switch_to_french_url}" data-locale-selector-link>')
        add_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=en'
        self.assertContains(response, f'<a href="{add_url}" class="button bicolor button--icon">')
        self.assertContains(response, f'No translatable snippets have been created. Why not <a href="{add_url}">add one</a>')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_snippetstests_translatablesnippet:list'))
        self.assertNotContains(response, 'data-locale-selector')
        add_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add')
        self.assertContains(response, f'<a href="{add_url}" class="button bicolor button--icon">')
        self.assertContains(response, f'No translatable snippets have been created. Why not <a href="{add_url}">add one</a>')

    def test_locale_selector_not_present_on_non_translatable_snippet(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:list'))
        self.assertNotContains(response, 'data-locale-selector')
        add_url = reverse('wagtailsnippets_tests_advert:add')
        self.assertContains(response, f'<a href="{add_url}" class="button bicolor button--icon">')
        self.assertContains(response, f'No adverts have been created. Why not <a href="{add_url}">add one</a>')

class TestModelOrdering(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        for i in range(1, 10):
            AdvertWithTabbedInterface.objects.create(text='advert %d' % i)
        AdvertWithTabbedInterface.objects.create(text='aaaadvert')
        self.login()

    def test_listing_respects_model_ordering(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advertwithtabbedinterface:list'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_obj'][0].text, 'aaaadvert')

    def test_chooser_respects_model_ordering(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailsnippetchoosers_tests_advertwithtabbedinterface:choose'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['results'][0].text, 'aaaadvert')

class TestListViewOrdering(WagtailTestUtils, TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        for i in range(1, 10):
            advert = Advert.objects.create(text=f"{i * 'a'}dvert {i}")
            draft = DraftStateModel.objects.create(text=f"{i * 'd'}raft {i}", live=False)
            if i % 2 == 0:
                ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(Advert), label='Test Advert', action='wagtail.create', timestamp=now(), object_id=advert.pk)
                draft.save_revision().publish()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()

    def test_listing_orderable_columns_with_no_mixin(self):
        if False:
            for i in range(10):
                print('nop')
        list_url = reverse('wagtailsnippets_tests_advert:list')
        response = self.client.get(list_url)
        sort_updated_url = list_url + '?ordering=_updated_at'
        sort_live_url = list_url + '?ordering=live'
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/tables/table.html')
        self.assertContains(response, f'<th><a href="{sort_updated_url}" title="Sort by &#x27;Updated&#x27; in ascending order." class="icon icon-arrow-down-after label">Updated</a></th>', html=True)
        self.assertNotContains(response, f'<th><a href="{sort_live_url}" title="Sort by &#x27;Status&#x27; in ascending order." class="icon icon-arrow-down-after label">Status</a></th>', html=True)

    def test_listing_orderable_columns_with_draft_state_mixin(self):
        if False:
            for i in range(10):
                print('nop')
        list_url = reverse('wagtailsnippets_tests_draftstatemodel:list')
        response = self.client.get(list_url)
        sort_updated_url = list_url + '?ordering=_updated_at'
        sort_live_url = list_url + '?ordering=live'
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/tables/table.html')
        self.assertContains(response, f'<th><a href="{sort_updated_url}" title="Sort by &#x27;Updated&#x27; in ascending order." class="icon icon-arrow-down-after label">Updated</a></th>', html=True)
        self.assertContains(response, f'<th><a href="{sort_live_url}" title="Sort by &#x27;Status&#x27; in ascending order." class="icon icon-arrow-down-after label">Status</a></th>', html=True)

    def test_order_by_updated_at_with_no_mixin(self):
        if False:
            print('Hello World!')
        list_url = reverse('wagtailsnippets_tests_advert:list')
        response = self.client.get(list_url + '?ordering=_updated_at')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context['page_obj'][0]._updated_at)
        self.assertEqual(response.context['page_obj'][-1].text, 'aaaaaaaadvert 8')
        self.assertIsNotNone(response.context['page_obj'][-1]._updated_at)
        self.assertContains(response, list_url + '?ordering=-_updated_at')
        response = self.client.get(list_url + '?ordering=-_updated_at')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_obj'][0].text, 'aaaaaaaadvert 8')
        self.assertIsNotNone(response.context['page_obj'][0]._updated_at)
        self.assertContains(response, list_url + '?ordering=_updated_at')

    def test_order_by_updated_at_with_draft_state_mixin(self):
        if False:
            i = 10
            return i + 15
        list_url = reverse('wagtailsnippets_tests_draftstatemodel:list')
        response = self.client.get(list_url + '?ordering=_updated_at')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context['page_obj'][0]._updated_at)
        self.assertEqual(response.context['page_obj'][-1].text, 'ddddddddraft 8')
        self.assertIsNotNone(response.context['page_obj'][-1]._updated_at)
        self.assertContains(response, list_url + '?ordering=-_updated_at')
        response = self.client.get(list_url + '?ordering=-_updated_at')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_obj'][0].text, 'ddddddddraft 8')
        self.assertIsNotNone(response.context['page_obj'][0]._updated_at)
        self.assertContains(response, list_url + '?ordering=_updated_at')

    def test_order_by_live(self):
        if False:
            return 10
        list_url = reverse('wagtailsnippets_tests_draftstatemodel:list')
        response = self.client.get(list_url + '?ordering=live')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['page_obj'][0].live)
        self.assertTrue(response.context['page_obj'][-1].live)
        self.assertContains(response, list_url + '?ordering=-live')
        response = self.client.get(list_url + '?ordering=-live')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['page_obj'][0].live)
        self.assertFalse(response.context['page_obj'][-1].live)
        self.assertContains(response, list_url + '?ordering=live')

class TestSnippetListViewWithSearchableSnippet(WagtailTestUtils, TransactionTestCase):

    def setUp(self):
        if False:
            return 10
        self.login()
        self.snippet_a = SearchableSnippet.objects.create(text='Hello')
        self.snippet_b = SearchableSnippet.objects.create(text='World')
        self.snippet_c = SearchableSnippet.objects.create(text='Hello World')

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailsnippets_snippetstests_searchablesnippet:list'), params)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        items = list(response.context['page_obj'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)
        self.assertNotContains(response, 'This field is required.')

    def test_empty_q(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        items = list(response.context['page_obj'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)
        self.assertNotContains(response, 'This field is required.')

    def test_is_searchable(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.get().context['is_searchable'])

    def test_search_hello(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': 'Hello'})
        items = list(response.context['page_obj'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertNotIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

    def test_search_world_autocomplete(self):
        if False:
            print('Hello World!')
        response = self.get({'q': 'wor'})
        items = list(response.context['page_obj'].object_list)
        self.assertNotIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

class TestSnippetListViewWithNonAutocompleteSearchableSnippet(WagtailTestUtils, TransactionTestCase):
    """
    Test that searchable snippets with no AutocompleteFields defined can still be searched using
    full words
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()
        self.snippet_a = NonAutocompleteSearchableSnippet.objects.create(text='Hello')
        self.snippet_b = NonAutocompleteSearchableSnippet.objects.create(text='World')
        self.snippet_c = NonAutocompleteSearchableSnippet.objects.create(text='Hello World')

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailsnippets_snippetstests_nonautocompletesearchablesnippet:list'), params)

    def test_search_hello(self):
        if False:
            return 10
        with self.assertWarnsRegex(RuntimeWarning, 'does not specify any AutocompleteFields'):
            response = self.get({'q': 'Hello'})
        items = list(response.context['page_obj'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertNotIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

class TestSnippetCreateView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()

    def get(self, params={}, model=Advert):
        if False:
            return 10
        return self.client.get(reverse(model.snippet_viewset.get_url_name('add')), params)

    def post(self, post_data={}, model=Advert):
        if False:
            i = 10
            return i + 15
        return self.client.post(reverse(model.snippet_viewset.get_url_name('add')), post_data)

    def test_get_with_limited_permissions(self):
        if False:
            return 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.get()
        self.assertEqual(response.status_code, 302)

    def test_simple(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/create.html')
        self.assertNotContains(response, 'role="tablist"', html=True)

    def test_snippet_with_tabbed_interface(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_tests_advertwithtabbedinterface:add'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/create.html')
        self.assertContains(response, 'role="tablist"')
        self.assertContains(response, '<a id="tab-label-advert" href="#tab-advert" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<a id="tab-label-other" href="#tab-other" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, 'Other panels help text')
        self.assertContains(response, 'Top-level help text')

    def test_create_with_limited_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.post(post_data={'text': 'test text', 'url': 'http://www.example.com/'})
        self.assertEqual(response.status_code, 302)

    def test_create_invalid(self):
        if False:
            return 10
        response = self.post(post_data={'foo': 'bar'})
        self.assertContains(response, 'The snippet could not be created due to errors.')
        self.assertContains(response, 'error-message', count=1)
        self.assertContains(response, 'This field is required', count=1)

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.post(post_data={'text': 'test_advert', 'url': 'http://www.example.com/'})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_advert:list'))
        snippets = Advert.objects.filter(text='test_advert')
        self.assertEqual(snippets.count(), 1)
        self.assertEqual(snippets.first().url, 'http://www.example.com/')

    def test_create_with_tags(self):
        if False:
            for i in range(10):
                print('nop')
        tags = ['hello', 'world']
        response = self.post(post_data={'text': 'test_advert', 'url': 'http://example.com/', 'tags': ', '.join(tags)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_advert:list'))
        snippet = Advert.objects.get(text='test_advert')
        expected_tags = list(Tag.objects.order_by('name').filter(name__in=tags))
        self.assertEqual(len(expected_tags), 2)
        self.assertEqual(list(snippet.tags.order_by('name')), expected_tags)

    def test_create_file_upload_multipart(self):
        if False:
            i = 10
            return i + 15
        response = self.get(model=FileUploadSnippet)
        self.assertContains(response, 'enctype="multipart/form-data"')
        response = self.post(model=FileUploadSnippet, post_data={'file': SimpleUploadedFile('test.txt', b'Uploaded file')})
        self.assertRedirects(response, reverse('wagtailsnippets_snippetstests_fileuploadsnippet:list'))
        snippet = FileUploadSnippet.objects.get()
        self.assertEqual(snippet.file.read(), b'Uploaded file')

    def test_create_with_revision(self):
        if False:
            i = 10
            return i + 15
        response = self.post(model=RevisableModel, post_data={'text': 'create_revisable'})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_revisablemodel:list'))
        snippets = RevisableModel.objects.filter(text='create_revisable')
        snippet = snippets.first()
        self.assertEqual(snippets.count(), 1)
        revisions = snippet.revisions
        revision = revisions.first()
        self.assertEqual(revisions.count(), 1)
        self.assertEqual(revision.content['text'], 'create_revisable')
        log_entries = ModelLogEntry.objects.for_instance(snippet).filter(action='wagtail.create')
        self.assertEqual(log_entries.count(), 1)
        self.assertEqual(log_entries.first().revision, revision)

    def test_before_create_snippet_hook_get(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, model):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(model, Advert)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_snippet', hook_func):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_create_snippet_hook_post(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_func(request, model):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(model, Advert)
            return HttpResponse('Overridden!')
        with self.register_hook('before_create_snippet', hook_func):
            post_data = {'text': 'Hook test', 'url': 'http://www.example.com/'}
            response = self.post(post_data=post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertFalse(Advert.objects.exists())

    def test_after_create_snippet_hook(self):
        if False:
            return 10

        def hook_func(request, instance):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(instance.text, 'Hook test')
            self.assertEqual(instance.url, 'http://www.example.com/')
            return HttpResponse('Overridden!')
        with self.register_hook('after_create_snippet', hook_func):
            post_data = {'text': 'Hook test', 'url': 'http://www.example.com/'}
            response = self.post(post_data=post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Advert.objects.exists())

    def test_register_snippet_action_menu_item(self):
        if False:
            print('Hello World!')

        class TestSnippetActionMenuItem(ActionMenuItem):
            label = 'Test'
            name = 'test'
            icon_name = 'check'
            classname = 'action-secondary'

            def is_shown(self, context):
                if False:
                    print('Hello World!')
                return True

        def hook_func(model):
            if False:
                print('Hello World!')
            return TestSnippetActionMenuItem(order=0)
        with self.register_hook('register_snippet_action_menu_item', hook_func):
            get_base_snippet_action_menu_items.cache_clear()
            response = self.get()
        get_base_snippet_action_menu_items.cache_clear()
        self.assertContains(response, '<button type="submit" name="test" value="Test" class="button action-secondary"><svg class="icon icon-check icon" aria-hidden="true"><use href="#icon-check"></use></svg>Test</button>', html=True)

    def test_register_snippet_action_menu_item_as_none(self):
        if False:
            while True:
                i = 10

        def hook_func(model):
            if False:
                while True:
                    i = 10
            return None
        with self.register_hook('register_snippet_action_menu_item', hook_func):
            get_base_snippet_action_menu_items.cache_clear()
            response = self.get()
        get_base_snippet_action_menu_items.cache_clear()
        self.assertEqual(response.status_code, 200)

    def test_construct_snippet_action_menu(self):
        if False:
            return 10

        class TestSnippetActionMenuItem(ActionMenuItem):
            label = 'Test'
            name = 'test'
            icon_name = 'check'
            classname = 'action-secondary'

            def is_shown(self, context):
                if False:
                    i = 10
                    return i + 15
                return True

        def hook_func(menu_items, request, context):
            if False:
                return 10
            self.assertIsInstance(menu_items, list)
            self.assertIsInstance(request, WSGIRequest)
            self.assertEqual(context['view'], 'create')
            self.assertEqual(context['model'], Advert)
            menu_items[:] = [TestSnippetActionMenuItem(order=0)]
        with self.register_hook('construct_snippet_action_menu', hook_func):
            response = self.get()
        self.assertContains(response, '<button type="submit" name="test" value="Test" class="button action-secondary"><svg class="icon icon-check icon" aria-hidden="true"><use href="#icon-check"></use></svg>Test</button>', html=True)
        self.assertNotContains(response, "<em>'Save'</em>")

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelectorOnCreate(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_snippetstests_translatablesnippet:add'))
        self.assertContains(response, 'Switch locales')
        switch_to_french_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=fr'
        self.assertContains(response, f'<a href="{switch_to_french_url}" lang="fr">')

    def test_locale_selector_with_existing_locale(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=fr')
        self.assertContains(response, 'Switch locales')
        switch_to_english_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=en'
        self.assertContains(response, f'<a href="{switch_to_english_url}" lang="en">')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailsnippets_snippetstests_translatablesnippet:add'))
        self.assertNotContains(response, 'Switch locales')
        switch_to_french_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=fr'
        self.assertNotContains(response, f'<a href="{switch_to_french_url}" lang="fr">')

    def test_locale_selector_not_present_on_non_translatable_snippet(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:add'))
        self.assertNotContains(response, 'Switch locales')
        switch_to_french_url = reverse('wagtailsnippets_snippetstests_translatablesnippet:add') + '?locale=fr'
        self.assertNotContains(response, f'<a href="{switch_to_french_url}" lang="fr">')

class TestCreateDraftStateSnippet(WagtailTestUtils, TestCase):
    STATUS_TOGGLE_BADGE_REGEX = 'data-side-panel-toggle="status"[^<]+<svg[^<]+<use[^<]+</use[^<]+</svg[^<]+<div data-side-panel-toggle-counter[^>]+w-bg-critical-200[^>]+>\\s*%(num_errors)s\\s*</div>'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailsnippets_tests_draftstatemodel:add'))

    def post(self, post_data={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.post(reverse('wagtailsnippets_tests_draftstatemodel:add'), post_data)

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/create.html')
        self.assertContains(response, 'Save draft')
        self.assertContains(response, 'Publish')
        self.assertContains(response, '<button\n    type="submit"\n    name="action-publish"\n    value="action-publish"\n    class="button action-save button-longrunning"\n    data-controller="w-progress"\n    data-action="w-progress#activate"\n')
        self.assertContains(response, '<div class="form-side__panel" data-side-panel="status" hidden>')
        self.assertContains(response, 'No publishing schedule set')
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Set schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'Choose when this draft state model should go live and/or expire')
        unpublish_url = '/admin/snippets/tests/draftstatemodel/unpublish/'
        self.assertNotContains(response, unpublish_url)
        self.assertNotContains(response, 'Unpublish')

    def test_save_draft(self):
        if False:
            i = 10
            return i + 15
        response = self.post(post_data={'text': 'Draft-enabled Foo'})
        snippet = DraftStateModel.objects.get(text='Draft-enabled Foo')
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:edit', args=[snippet.pk]))
        self.assertEqual(snippet.text, 'Draft-enabled Foo')
        self.assertFalse(snippet.live)
        self.assertTrue(snippet.has_unpublished_changes)
        self.assertIsNone(snippet.first_published_at)
        self.assertIsNone(snippet.last_published_at)
        self.assertIsNone(snippet.live_revision)
        self.assertIsNotNone(snippet.latest_revision)
        self.assertEqual(snippet.latest_revision.content['text'], 'Draft-enabled Foo')

    def test_publish(self):
        if False:
            return 10
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            timestamp = now()
            with freeze_time(timestamp):
                response = self.post(post_data={'text': 'Draft-enabled Foo, Published', 'action-publish': 'action-publish'})
            snippet = DraftStateModel.objects.get(text='Draft-enabled Foo, Published')
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:list'))
            self.assertEqual(snippet.text, 'Draft-enabled Foo, Published')
            self.assertTrue(snippet.live)
            self.assertFalse(snippet.has_unpublished_changes)
            self.assertEqual(snippet.first_published_at, timestamp)
            self.assertEqual(snippet.last_published_at, timestamp)
            self.assertIsNotNone(snippet.live_revision)
            self.assertEqual(snippet.live_revision, snippet.latest_revision)
            self.assertEqual(snippet.live_revision.content['text'], 'Draft-enabled Foo, Published')
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateModel)
            self.assertEqual(mock_call['instance'], snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateModel)
        finally:
            published.disconnect(mock_handler)

    def test_publish_bad_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        add_permission = Permission.objects.get(content_type__app_label='tests', codename='add_draftstatemodel')
        edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatemodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(add_permission, edit_permission, admin_permission)
        self.user.save()
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            response = self.post(post_data={'text': 'Draft-enabled Foo', 'action-publish': 'action-publish'})
            snippet = DraftStateModel.objects.get(text='Draft-enabled Foo')
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:edit', args=[snippet.pk]))
            self.assertEqual(snippet.text, 'Draft-enabled Foo')
            self.assertFalse(snippet.live)
            self.assertTrue(snippet.has_unpublished_changes)
            self.assertIsNotNone(snippet.latest_revision)
            self.assertIsNone(snippet.live_revision)
            self.assertEqual(snippet.latest_revision.content['text'], 'Draft-enabled Foo')
            self.assertEqual(mock_handler.call_count, 0)
        finally:
            published.disconnect(mock_handler)

    def test_publish_with_publish_permission(self):
        if False:
            i = 10
            return i + 15
        self.user.is_superuser = False
        add_permission = Permission.objects.get(content_type__app_label='tests', codename='add_draftstatemodel')
        publish_permission = Permission.objects.get(content_type__app_label='tests', codename='publish_draftstatemodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(add_permission, publish_permission, admin_permission)
        self.user.save()
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            timestamp = now()
            with freeze_time(timestamp):
                response = self.post(post_data={'text': 'Draft-enabled Foo, Published', 'action-publish': 'action-publish'})
            snippet = DraftStateModel.objects.get(text='Draft-enabled Foo, Published')
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:list'))
            self.assertEqual(snippet.text, 'Draft-enabled Foo, Published')
            self.assertTrue(snippet.live)
            self.assertFalse(snippet.has_unpublished_changes)
            self.assertEqual(snippet.first_published_at, timestamp)
            self.assertEqual(snippet.last_published_at, timestamp)
            self.assertIsNotNone(snippet.live_revision)
            self.assertEqual(snippet.live_revision, snippet.latest_revision)
            self.assertEqual(snippet.live_revision.content['text'], 'Draft-enabled Foo, Published')
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateModel)
            self.assertEqual(mock_call['instance'], snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateModel)
        finally:
            published.disconnect(mock_handler)

    def test_create_scheduled(self):
        if False:
            while True:
                i = 10
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': 'Some content', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        snippet = DraftStateModel.objects.get(text='Some content')
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:edit', args=[snippet.pk]))
        self.assertEqual(snippet.go_live_at.date(), go_live_at.date())
        self.assertEqual(snippet.expire_at.date(), expire_at.date())
        self.assertIs(snippet.expired, False)
        self.assertEqual(snippet.status_string, 'draft')
        self.assertFalse(Revision.objects.for_instance(snippet).exclude(approved_go_live_at__isnull=True).exists())

    def test_create_scheduled_go_live_before_expiry(self):
        if False:
            while True:
                i = 10
        response = self.post(post_data={'text': 'Some content', 'go_live_at': submittable_timestamp(now() + datetime.timedelta(days=2)), 'expire_at': submittable_timestamp(now() + datetime.timedelta(days=1))})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'go_live_at', 'Go live date/time must be before expiry date/time')
        self.assertFormError(response, 'form', 'expire_at', 'Go live date/time must be before expiry date/time')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_create_scheduled_expire_in_the_past(self):
        if False:
            print('Hello World!')
        response = self.post(post_data={'text': 'Some content', 'expire_at': submittable_timestamp(now() + datetime.timedelta(days=-1))})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'expire_at', 'Expiry date/time must be in the future')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 1
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_create_post_publish_scheduled(self):
        if False:
            for i in range(10):
                print('nop')
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatemodel:list'))
        snippet = DraftStateModel.objects.get(text='Some content')
        self.assertEqual(snippet.go_live_at.date(), go_live_at.date())
        self.assertEqual(snippet.expire_at.date(), expire_at.date())
        self.assertIs(snippet.expired, False)
        self.assertTrue(Revision.objects.for_instance(snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertFalse(snippet.live)
        self.assertFalse(snippet.first_published_at)
        self.assertEqual(snippet.status_string, 'scheduled')

class BaseTestSnippetEditView(WagtailTestUtils, TestCase):

    def get_edit_url(self):
        if False:
            for i in range(10):
                print('nop')
        snippet = self.test_snippet
        args = [quote(snippet.pk)]
        return reverse(snippet.snippet_viewset.get_url_name('edit'), args=args)

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(self.get_edit_url(), params)

    def post(self, post_data={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.post(self.get_edit_url(), post_data)

    def setUp(self):
        if False:
            return 10
        self.user = self.login()

class TestSnippetEditView(BaseTestSnippetEditView):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.test_snippet = Advert.objects.get(pk=1)
        ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(Advert), label='Test Advert', action='wagtail.create', timestamp=now() - datetime.timedelta(weeks=3), user=self.user, object_id='1')

    def test_get_with_limited_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.get()
        self.assertEqual(response.status_code, 302)

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertNotContains(response, 'role="tablist"')
        self.assertNotContains(response, 'No publishing schedule set')
        history_url = reverse('wagtailsnippets_tests_advert:history', args=[quote(self.test_snippet.pk)])
        self.assertContains(response, history_url, count=2)
        usage_url = reverse('wagtailsnippets_tests_advert:usage', args=[quote(self.test_snippet.pk)])
        self.assertContains(response, usage_url)
        self.assertContains(response, '3\xa0weeks ago')
        self.assertTagInHTML(f'<a href="{history_url}" aria-describedby="status-sidebar-live">View history</a>', html, allow_extra_attrs=True)
        url_finder = AdminURLFinder(self.user)
        expected_url = '/admin/snippets/tests/advert/edit/%d/' % self.test_snippet.pk
        self.assertEqual(url_finder.get_edit_url(self.test_snippet), expected_url)

    def test_non_existant_model(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(f'/admin/snippets/tests/foo/edit/{quote(self.test_snippet.pk)}/')
        self.assertEqual(response.status_code, 404)

    def test_nonexistant_id(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:edit', args=[999999]))
        self.assertEqual(response.status_code, 404)

    def test_edit_with_limited_permissions(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.post(post_data={'text': 'test text', 'url': 'http://www.example.com/'})
        self.assertEqual(response.status_code, 302)
        url_finder = AdminURLFinder(self.user)
        self.assertIsNone(url_finder.get_edit_url(self.test_snippet))

    def test_edit_invalid(self):
        if False:
            while True:
                i = 10
        response = self.post(post_data={'foo': 'bar'})
        self.assertContains(response, 'The snippet could not be saved due to errors.')
        self.assertContains(response, 'error-message', count=1)
        self.assertContains(response, 'This field is required', count=1)

    def test_edit(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.post(post_data={'text': 'edited_test_advert', 'url': 'http://www.example.com/edited'})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_advert:list'))
        snippets = Advert.objects.filter(text='edited_test_advert')
        self.assertEqual(snippets.count(), 1)
        self.assertEqual(snippets.first().url, 'http://www.example.com/edited')

    def test_edit_with_tags(self):
        if False:
            i = 10
            return i + 15
        tags = ['hello', 'world']
        response = self.post(post_data={'text': 'edited_test_advert', 'url': 'http://www.example.com/edited', 'tags': ', '.join(tags)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_advert:list'))
        snippet = Advert.objects.get(text='edited_test_advert')
        expected_tags = list(Tag.objects.order_by('name').filter(name__in=tags))
        self.assertEqual(len(expected_tags), 2)
        self.assertEqual(list(snippet.tags.order_by('name')), expected_tags)

    def test_before_edit_snippet_hook_get(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, instance):
            if False:
                return 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(instance.text, 'test_advert')
            self.assertEqual(instance.url, 'http://www.example.com')
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_snippet', hook_func):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_edit_snippet_hook_post(self):
        if False:
            while True:
                i = 10

        def hook_func(request, instance):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(instance.text, 'test_advert')
            self.assertEqual(instance.url, 'http://www.example.com')
            return HttpResponse('Overridden!')
        with self.register_hook('before_edit_snippet', hook_func):
            response = self.post(post_data={'text': 'Edited and runs hook', 'url': 'http://www.example.com/hook-enabled-edited'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertEqual(Advert.objects.get().text, 'test_advert')

    def test_after_edit_snippet_hook(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, instance):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(instance.text, 'Edited and runs hook')
            self.assertEqual(instance.url, 'http://www.example.com/hook-enabled-edited')
            return HttpResponse('Overridden!')
        with self.register_hook('after_edit_snippet', hook_func):
            response = self.post(post_data={'text': 'Edited and runs hook', 'url': 'http://www.example.com/hook-enabled-edited'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertEqual(Advert.objects.get().text, 'Edited and runs hook')

    def test_register_snippet_action_menu_item(self):
        if False:
            return 10

        class TestSnippetActionMenuItem(ActionMenuItem):
            label = 'Test'
            name = 'test'
            icon_name = 'check'
            classname = 'action-secondary'

            def is_shown(self, context):
                if False:
                    while True:
                        i = 10
                return True

        def hook_func(model):
            if False:
                i = 10
                return i + 15
            return TestSnippetActionMenuItem(order=0)
        with self.register_hook('register_snippet_action_menu_item', hook_func):
            get_base_snippet_action_menu_items.cache_clear()
            response = self.get()
        get_base_snippet_action_menu_items.cache_clear()
        self.assertContains(response, '<button type="submit" name="test" value="Test" class="button action-secondary"><svg class="icon icon-check icon" aria-hidden="true"><use href="#icon-check"></use></svg>Test</button>', html=True)

    def test_construct_snippet_action_menu(self):
        if False:
            while True:
                i = 10

        def hook_func(menu_items, request, context):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(menu_items, list)
            self.assertIsInstance(request, WSGIRequest)
            self.assertEqual(context['view'], 'edit')
            self.assertEqual(context['instance'], self.test_snippet)
            self.assertEqual(context['model'], Advert)
            del menu_items[0]
        with self.register_hook('construct_snippet_action_menu', hook_func):
            response = self.get()
        self.assertNotContains(response, '<em>Save</em>')

class TestEditTabbedSnippet(BaseTestSnippetEditView):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.test_snippet = AdvertWithTabbedInterface.objects.create(text='test_advert', url='http://www.example.com', something_else='Model with tabbed interface')

    def test_snippet_with_tabbed_interface(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, 'role="tablist"')
        self.assertContains(response, '<a id="tab-label-advert" href="#tab-advert" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')
        self.assertContains(response, '<a id="tab-label-other" href="#tab-other" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">')

class TestEditFileUploadSnippet(BaseTestSnippetEditView):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.test_snippet = FileUploadSnippet.objects.create(file=ContentFile(b'Simple text document', 'test.txt'))

    def test_edit_file_upload_multipart(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertContains(response, 'enctype="multipart/form-data"')
        response = self.post(post_data={'file': SimpleUploadedFile('replacement.txt', b'Replacement document')})
        self.assertRedirects(response, reverse('wagtailsnippets_snippetstests_fileuploadsnippet:list'))
        snippet = FileUploadSnippet.objects.get()
        self.assertEqual(snippet.file.read(), b'Replacement document')

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelectorOnEdit(BaseTestSnippetEditView):
    fixtures = ['test.json']
    LOCALE_SELECTOR_LABEL = 'Switch locales'
    LOCALE_INDICATOR_HTML = '<h3 id="status-sidebar-english"'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.test_snippet = TranslatableSnippet.objects.create(text='This is a test')
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.test_snippet_fr = self.test_snippet.copy_for_translation(self.fr_locale)
        self.test_snippet_fr.save()

    def test_locale_selector(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertContains(response, self.LOCALE_SELECTOR_LABEL)
        self.assertContains(response, self.LOCALE_INDICATOR_HTML)

    def test_locale_selector_without_translation(self):
        if False:
            print('Hello World!')
        self.test_snippet_fr.delete()
        response = self.get()
        self.assertNotContains(response, self.LOCALE_SELECTOR_LABEL)
        self.assertContains(response, self.LOCALE_INDICATOR_HTML)
        self.assertContains(response, 'No other translations')

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertNotContains(response, self.LOCALE_SELECTOR_LABEL)
        self.assertNotContains(response, self.LOCALE_INDICATOR_HTML)

    def test_locale_selector_not_present_on_non_translatable_snippet(self):
        if False:
            return 10
        self.test_snippet = Advert.objects.get(pk=1)
        response = self.get()
        self.assertNotContains(response, self.LOCALE_SELECTOR_LABEL)
        self.assertNotContains(response, self.LOCALE_INDICATOR_HTML)

class TestEditRevisionSnippet(BaseTestSnippetEditView):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.test_snippet = RevisableModel.objects.create(text='foo')

    def test_edit_snippet_with_revision(self):
        if False:
            return 10
        response = self.post(post_data={'text': 'bar'})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_revisablemodel:list'))
        snippets = RevisableModel.objects.filter(text='bar')
        self.assertEqual(snippets.count(), 1)
        revisions = self.test_snippet.revisions
        revision = revisions.first()
        self.assertEqual(revisions.count(), 1)
        self.assertEqual(revision.content['text'], 'bar')
        log_entries = ModelLogEntry.objects.for_instance(self.test_snippet).filter(action='wagtail.edit')
        self.assertEqual(log_entries.count(), 1)
        self.assertEqual(log_entries.first().revision, revision)

class TestEditDraftStateSnippet(BaseTestSnippetEditView):
    STATUS_TOGGLE_BADGE_REGEX = 'data-side-panel-toggle="status"[^<]+<svg[^<]+<use[^<]+</use[^<]+</svg[^<]+<div data-side-panel-toggle-counter[^>]+w-bg-critical-200[^>]+>\\s*%(num_errors)s\\s*</div>'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.test_snippet = DraftStateCustomPrimaryKeyModel.objects.create(custom_id='custom/1', text='Draft-enabled Foo', live=False)

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, 'Save draft')
        self.assertContains(response, 'Publish')
        self.assertContains(response, '<button\n    type="submit"\n    name="action-publish"\n    value="action-publish"\n    class="button action-save button-longrunning"\n    data-controller="w-progress"\n    data-action="w-progress#activate"\n')
        self.assertContains(response, 'No publishing schedule set')
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Set schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'Choose when this draft state custom primary key model should go live and/or expire')
        unpublish_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(self.test_snippet.pk),))
        self.assertNotContains(response, f'<a class="button action-secondary" href="{unpublish_url}">')
        self.assertNotContains(response, 'Unpublish')

    def test_save_draft(self):
        if False:
            while True:
                i = 10
        response = self.post(post_data={'text': 'Draft-enabled Bar'})
        self.test_snippet.refresh_from_db()
        revisions = Revision.objects.for_instance(self.test_snippet)
        latest_revision = self.test_snippet.latest_revision
        self.assertRedirects(response, self.get_edit_url())
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Foo')
        self.assertFalse(self.test_snippet.live)
        self.assertTrue(self.test_snippet.has_unpublished_changes)
        self.assertIsNone(self.test_snippet.first_published_at)
        self.assertIsNone(self.test_snippet.last_published_at)
        self.assertIsNone(self.test_snippet.live_revision)
        self.assertEqual(revisions.count(), 1)
        self.assertEqual(latest_revision, revisions.first())
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar')

    def test_publish(self):
        if False:
            print('Hello World!')
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            timestamp = now()
            with freeze_time(timestamp):
                response = self.post(post_data={'text': 'Draft-enabled Bar, Published', 'action-publish': 'action-publish'})
            self.test_snippet.refresh_from_db()
            revisions = Revision.objects.for_instance(self.test_snippet)
            latest_revision = self.test_snippet.latest_revision
            log_entries = ModelLogEntry.objects.filter(content_type=ContentType.objects.get_for_model(DraftStateCustomPrimaryKeyModel), action='wagtail.publish', object_id=self.test_snippet.pk)
            log_entry = log_entries.first()
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
            self.assertEqual(self.test_snippet.text, 'Draft-enabled Bar, Published')
            self.assertTrue(self.test_snippet.live)
            self.assertFalse(self.test_snippet.has_unpublished_changes)
            self.assertEqual(self.test_snippet.first_published_at, timestamp)
            self.assertEqual(self.test_snippet.last_published_at, timestamp)
            self.assertEqual(self.test_snippet.live_revision, latest_revision)
            self.assertEqual(revisions.count(), 1)
            self.assertEqual(latest_revision, revisions.first())
            self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar, Published')
            self.assertEqual(log_entries.count(), 1)
            self.assertEqual(log_entry.timestamp, timestamp)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateCustomPrimaryKeyModel)
            self.assertEqual(mock_call['instance'], self.test_snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateCustomPrimaryKeyModel)
        finally:
            published.disconnect(mock_handler)

    def test_publish_bad_permissions(self):
        if False:
            return 10
        self.user.is_superuser = False
        edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatecustomprimarykeymodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(edit_permission, admin_permission)
        self.user.save()
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            response = self.post(post_data={'text': 'Edited draft Foo', 'action-publish': 'action-publish'})
            self.test_snippet.refresh_from_db()
            self.assertRedirects(response, self.get_edit_url())
            self.assertEqual(self.test_snippet.text, 'Draft-enabled Foo')
            self.assertFalse(self.test_snippet.live)
            self.assertTrue(self.test_snippet.has_unpublished_changes)
            self.assertIsNotNone(self.test_snippet.latest_revision)
            self.assertIsNone(self.test_snippet.live_revision)
            self.assertEqual(self.test_snippet.latest_revision.content['text'], 'Edited draft Foo')
            self.assertEqual(mock_handler.call_count, 0)
        finally:
            published.disconnect(mock_handler)

    def test_publish_with_publish_permission(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatecustomprimarykeymodel')
        publish_permission = Permission.objects.get(content_type__app_label='tests', codename='publish_draftstatecustomprimarykeymodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(edit_permission, publish_permission, admin_permission)
        self.user.save()
        mock_handler = mock.MagicMock()
        published.connect(mock_handler)
        try:
            timestamp = now()
            with freeze_time(timestamp):
                response = self.post(post_data={'text': 'Draft-enabled Bar, Published', 'action-publish': 'action-publish'})
            self.test_snippet.refresh_from_db()
            revisions = Revision.objects.for_instance(self.test_snippet)
            latest_revision = self.test_snippet.latest_revision
            log_entries = ModelLogEntry.objects.filter(content_type=ContentType.objects.get_for_model(DraftStateCustomPrimaryKeyModel), action='wagtail.publish', object_id=self.test_snippet.pk)
            log_entry = log_entries.first()
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
            self.assertEqual(self.test_snippet.text, 'Draft-enabled Bar, Published')
            self.assertTrue(self.test_snippet.live)
            self.assertFalse(self.test_snippet.has_unpublished_changes)
            self.assertEqual(self.test_snippet.first_published_at, timestamp)
            self.assertEqual(self.test_snippet.last_published_at, timestamp)
            self.assertEqual(self.test_snippet.live_revision, latest_revision)
            self.assertEqual(revisions.count(), 1)
            self.assertEqual(latest_revision, revisions.first())
            self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar, Published')
            self.assertEqual(log_entries.count(), 1)
            self.assertEqual(log_entry.timestamp, timestamp)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateCustomPrimaryKeyModel)
            self.assertEqual(mock_call['instance'], self.test_snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateCustomPrimaryKeyModel)
        finally:
            published.disconnect(mock_handler)

    def test_save_draft_then_publish(self):
        if False:
            return 10
        save_timestamp = now()
        with freeze_time(save_timestamp):
            self.test_snippet.text = 'Draft-enabled Bar, In Draft'
            self.test_snippet.save_revision()
        publish_timestamp = now()
        with freeze_time(publish_timestamp):
            response = self.post(post_data={'text': 'Draft-enabled Bar, Now Published', 'action-publish': 'action-publish'})
        self.test_snippet.refresh_from_db()
        revisions = Revision.objects.for_instance(self.test_snippet).order_by('pk')
        latest_revision = self.test_snippet.latest_revision
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Bar, Now Published')
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(self.test_snippet.has_unpublished_changes)
        self.assertEqual(self.test_snippet.first_published_at, publish_timestamp)
        self.assertEqual(self.test_snippet.last_published_at, publish_timestamp)
        self.assertEqual(self.test_snippet.live_revision, latest_revision)
        self.assertEqual(revisions.count(), 2)
        self.assertEqual(latest_revision, revisions.last())
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar, Now Published')

    def test_publish_then_save_draft(self):
        if False:
            for i in range(10):
                print('nop')
        publish_timestamp = now()
        with freeze_time(publish_timestamp):
            self.test_snippet.text = 'Draft-enabled Bar, Published'
            self.test_snippet.save_revision().publish()
        save_timestamp = now()
        with freeze_time(save_timestamp):
            response = self.post(post_data={'text': 'Draft-enabled Bar, Published and In Draft'})
        self.test_snippet.refresh_from_db()
        revisions = Revision.objects.for_instance(self.test_snippet).order_by('pk')
        latest_revision = self.test_snippet.latest_revision
        self.assertRedirects(response, self.get_edit_url())
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Bar, Published')
        self.assertTrue(self.test_snippet.live)
        self.assertTrue(self.test_snippet.has_unpublished_changes)
        self.assertEqual(self.test_snippet.first_published_at, publish_timestamp)
        self.assertEqual(self.test_snippet.last_published_at, publish_timestamp)
        self.assertEqual(self.test_snippet.live_revision, revisions.first())
        self.assertEqual(revisions.count(), 2)
        self.assertEqual(latest_revision, revisions.last())
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar, Published and In Draft')

    def test_publish_twice(self):
        if False:
            for i in range(10):
                print('nop')
        first_timestamp = now()
        with freeze_time(first_timestamp):
            self.test_snippet.text = 'Draft-enabled Bar, Published Once'
            self.test_snippet.save_revision().publish()
        second_timestamp = now() + datetime.timedelta(days=1)
        with freeze_time(second_timestamp):
            response = self.post(post_data={'text': 'Draft-enabled Bar, Published Twice', 'action-publish': 'action-publish'})
        self.test_snippet.refresh_from_db()
        revisions = Revision.objects.for_instance(self.test_snippet).order_by('pk')
        latest_revision = self.test_snippet.latest_revision
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Bar, Published Twice')
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(self.test_snippet.has_unpublished_changes)
        self.assertEqual(self.test_snippet.first_published_at, first_timestamp)
        self.assertEqual(self.test_snippet.last_published_at, second_timestamp)
        self.assertEqual(self.test_snippet.live_revision, revisions.last())
        self.assertEqual(revisions.count(), 2)
        self.assertEqual(latest_revision, revisions.last())
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Bar, Published Twice')

    def test_get_after_save_draft(self):
        if False:
            return 10
        self.post(post_data={'text': 'Draft-enabled Bar'})
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertNotContains(response, '<h3 id="status-sidebar-live" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Live</h3>', html=True)
        self.assertContains(response, '<h3 id="status-sidebar-draft" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Draft</h3>', html=True)
        unpublish_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(self.test_snippet.pk),))
        self.assertNotContains(response, f'<a class="button action-secondary" href="{unpublish_url}">')
        self.assertNotContains(response, 'Unpublish')

    def test_get_after_publish(self):
        if False:
            i = 10
            return i + 15
        self.post(post_data={'text': 'Draft-enabled Bar, Published', 'action-publish': 'action-publish'})
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, '<h3 id="status-sidebar-live" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Live</h3>', html=True)
        self.assertNotContains(response, '<h3 id="status-sidebar-draft" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Draft</h3>', html=True)
        unpublish_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(self.test_snippet.pk),))
        self.assertContains(response, f'<a class="button action-secondary" href="{unpublish_url}">')
        self.assertContains(response, 'Unpublish')

    def test_get_after_publish_and_save_draft(self):
        if False:
            i = 10
            return i + 15
        self.post(post_data={'text': 'Draft-enabled Bar, Published', 'action-publish': 'action-publish'})
        self.post(post_data={'text': 'Draft-enabled Bar, In Draft'})
        response = self.get()
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, '<h3 id="status-sidebar-live" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Live</h3>', html=True)
        self.assertContains(response, '<h3 id="status-sidebar-draft" class="w-label-1 !w-mt-0 w-mb-1"><span class="w-sr-only">Status: </span>Draft</h3>', html=True)
        unpublish_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(self.test_snippet.pk),))
        self.assertContains(response, f'<a class="button action-secondary" href="{unpublish_url}">')
        self.assertContains(response, 'Unpublish')
        self.assertContains(response, '<h1 class="w-header__title" id="header-title"><svg class="icon icon-snippet w-header__glyph" aria-hidden="true"><use href="#icon-snippet"></use></svg>Draft-enabled Bar, In Draft</h1>', html=True)
        self.assertTagInHTML('<textarea name="text">Draft-enabled Bar, In Draft</textarea>', html, allow_extra_attrs=True)

    def test_edit_post_scheduled(self):
        if False:
            while True:
                i = 10
        self.test_snippet.save_revision().publish()
        go_live_at = now() + datetime.timedelta(days=10)
        expire_at = now() + datetime.timedelta(days=20)
        response = self.post(post_data={'text': 'Some content', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:edit', args=[quote(self.test_snippet.pk)]))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).filter(content__go_live_at__startswith=str(go_live_at.date())).exists())
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).filter(content__expire_at__startswith=str(expire_at.date())).exists())
        response = self.get()
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_scheduled_go_live_before_expiry(self):
        if False:
            while True:
                i = 10
        response = self.post(post_data={'text': 'Some content', 'go_live_at': submittable_timestamp(now() + datetime.timedelta(days=2)), 'expire_at': submittable_timestamp(now() + datetime.timedelta(days=1))})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'go_live_at', 'Go live date/time must be before expiry date/time')
        self.assertFormError(response, 'form', 'expire_at', 'Go live date/time must be before expiry date/time')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_edit_scheduled_expire_in_the_past(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.post(post_data={'text': 'Some content', 'expire_at': submittable_timestamp(now() + datetime.timedelta(days=-1))})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'expire_at', 'Expiry date/time must be in the future')
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 1
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_edit_post_invalid_schedule_with_existing_draft_schedule(self):
        if False:
            print('Hello World!')
        self.test_snippet.go_live_at = now() + datetime.timedelta(days=1)
        self.test_snippet.expire_at = now() + datetime.timedelta(days=2)
        latest_revision = self.test_snippet.save_revision()
        go_live_at = now() + datetime.timedelta(days=10)
        expire_at = now() + datetime.timedelta(days=-20)
        response = self.post(post_data={'text': 'Some edited content', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertEqual(response.status_code, 200)
        self.test_snippet.refresh_from_db()
        self.assertFalse(self.test_snippet.live)
        self.assertEqual(self.test_snippet.latest_revision_id, latest_revision.pk)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertNotContains(response, '<span class="w-text-grey-600">Go-live:</span>', html=True)
        self.assertNotContains(response, '<span class="w-text-grey-600">Expiry:</span>', html=True)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Invalid schedule</div>', html=True)
        num_errors = 2
        self.assertRegex(response.content.decode(), self.STATUS_TOGGLE_BADGE_REGEX % {'num_errors': num_errors})

    def test_first_published_at_editable(self):
        if False:
            while True:
                i = 10
        'Test that we can update the first_published_at via the edit form,\n        for models that expose it.'
        self.test_snippet.save_revision().publish()
        self.test_snippet.refresh_from_db()
        initial_delta = self.test_snippet.first_published_at - now()
        first_published_at = now() - datetime.timedelta(days=2)
        self.post(post_data={'text': "I've been edited!", 'action-publish': 'action-publish', 'first_published_at': submittable_timestamp(first_published_at)})
        self.test_snippet.refresh_from_db()
        new_delta = self.test_snippet.first_published_at - now()
        self.assertNotEqual(new_delta.days, initial_delta.days)
        self.assertEqual(new_delta.days, -3)

    def test_edit_post_publish_scheduled_unpublished(self):
        if False:
            for i in range(10):
                print('nop')
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertFalse(self.test_snippet.live)
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(self.test_snippet.has_unpublished_changes, msg='An object scheduled for future publishing should have has_unpublished_changes=True')
        self.assertEqual(self.test_snippet.status_string, 'scheduled')
        response = self.get()
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_now_an_already_scheduled_unpublished(self):
        if False:
            print('Hello World!')
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertFalse(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'scheduled')
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': ''})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        response = self.get()
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_scheduled_published(self):
        if False:
            print('Hello World!')
        self.test_snippet.save_revision().publish()
        self.test_snippet.refresh_from_db()
        live_revision = self.test_snippet.live_revision
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': "I've been edited!", 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet = DraftStateCustomPrimaryKeyModel.objects.get(pk=self.test_snippet.pk)
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live + scheduled')
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(self.test_snippet.has_unpublished_changes, msg='An object scheduled for future publishing should have has_unpublished_changes=True')
        self.assertNotEqual(self.test_snippet.get_latest_revision(), live_revision, 'An object scheduled for future publishing should have a new revision, that is not the live revision')
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Foo', 'A live object with a scheduled revision should still have the original content')
        response = self.get()
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_now_an_already_scheduled_published(self):
        if False:
            i = 10
            return i + 15
        self.test_snippet.save_revision().publish()
        go_live_at = now() + datetime.timedelta(days=1)
        expire_at = now() + datetime.timedelta(days=2)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertEqual(self.test_snippet.text, 'Draft-enabled Foo', 'A live object with scheduled revisions should still have original content')
        response = self.post(post_data={'text': "I've been updated!", 'action-publish': 'Publish', 'go_live_at': ''})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertEqual(self.test_snippet.text, "I've been updated!")

    def test_edit_post_save_schedule_before_a_scheduled_expire(self):
        if False:
            i = 10
            return i + 15
        expire_at = now() + datetime.timedelta(days=20)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live')
        self.assertEqual(self.test_snippet.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = now() + datetime.timedelta(days=10)
        new_expire_at = now() + datetime.timedelta(days=15)
        response = self.post(post_data={'text': 'Some content', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:edit', args=[quote(self.test_snippet.pk)]))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).filter(content__go_live_at__startswith=str(go_live_at.date())).exists())
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).filter(content__expire_at__startswith=str(expire_at.date())).exists())
        response = self.get()
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        self.assertContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_schedule_before_a_scheduled_expire(self):
        if False:
            while True:
                i = 10
        expire_at = now() + datetime.timedelta(days=20)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live')
        self.assertEqual(self.test_snippet.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = now() + datetime.timedelta(days=10)
        new_expire_at = now() + datetime.timedelta(days=15)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet = DraftStateCustomPrimaryKeyModel.objects.get(pk=self.test_snippet.pk)
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live + scheduled')
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        response = self.get()
        self.assertNotContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

    def test_edit_post_publish_schedule_after_a_scheduled_expire(self):
        if False:
            i = 10
            return i + 15
        expire_at = now() + datetime.timedelta(days=20)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'expire_at': submittable_timestamp(expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet.refresh_from_db()
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live')
        self.assertEqual(self.test_snippet.expire_at, expire_at.replace(second=0, microsecond=0))
        go_live_at = now() + datetime.timedelta(days=23)
        new_expire_at = now() + datetime.timedelta(days=25)
        response = self.post(post_data={'text': 'Some content', 'action-publish': 'Publish', 'go_live_at': submittable_timestamp(go_live_at), 'expire_at': submittable_timestamp(new_expire_at)})
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
        self.test_snippet = DraftStateCustomPrimaryKeyModel.objects.get(pk=self.test_snippet.pk)
        self.assertTrue(self.test_snippet.live)
        self.assertEqual(self.test_snippet.status_string, 'live + scheduled')
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())
        response = self.get()
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(expire_at)}', html=True, count=1)
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(go_live_at)}', html=True, count=1)
        self.assertContains(response, f'<span class="w-text-grey-600">Expiry:</span> {render_timestamp(new_expire_at)}', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<template data-controller="w-teleport" data-w-teleport-target-value="[data-edit-form]">', html, count=1, allow_extra_attrs=True)
        self.assertTagInHTML('<div id="schedule-publishing-dialog" class="w-dialog publishing" data-controller="w-dialog">', html, count=1, allow_extra_attrs=True)

class TestScheduledForPublishLock(BaseTestSnippetEditView):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.test_snippet = DraftStateModel.objects.create(text='Draft-enabled Foo', live=False)
        self.go_live_at = now() + datetime.timedelta(days=1)
        self.test_snippet.text = "I've been edited!"
        self.test_snippet.go_live_at = self.go_live_at
        self.latest_revision = self.test_snippet.save_revision()
        self.latest_revision.publish()
        self.test_snippet.refresh_from_db()

    def test_edit_get_scheduled_for_publishing_with_publish_permission(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatemodel')
        publish_permission = Permission.objects.get(content_type__app_label='tests', codename='publish_draftstatemodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(edit_permission, publish_permission, admin_permission)
        self.user.save()
        response = self.get()
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(self.go_live_at)}', html=True, count=1)
        self.assertContains(response, "Draft state model 'I&#x27;ve been edited!' is locked and has been scheduled to go live at", count=1)
        self.assertContains(response, 'Locked by schedule')
        self.assertContains(response, '<div class="w-help-text">Currently locked and will go live on the scheduled date</div>', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)
        unschedule_url = reverse('wagtailsnippets_tests_draftstatemodel:revisions_unschedule', args=[self.test_snippet.pk, self.latest_revision.pk])
        self.assertTagInHTML(f'<button data-action="w-action#post" data-controller="w-action" data-w-action-url-value="{unschedule_url}">Cancel scheduled publish</button>', html, count=1, allow_extra_attrs=True)

    def test_edit_get_scheduled_for_publishing_without_publish_permission(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatemodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        self.user.user_permissions.add(edit_permission, admin_permission)
        self.user.save()
        response = self.get()
        self.assertNotContains(response, '<div class="w-label-3 w-text-primary">Once published:</div>', html=True)
        self.assertContains(response, f'<span class="w-text-grey-600">Go-live:</span> {render_timestamp(self.go_live_at)}', html=True, count=1)
        self.assertContains(response, "Draft state model 'I&#x27;ve been edited!' is locked and has been scheduled to go live at", count=1)
        self.assertContains(response, 'Locked by schedule')
        self.assertContains(response, '<div class="w-help-text">Currently locked and will go live on the scheduled date</div>', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)
        unschedule_url = reverse('wagtailsnippets_tests_draftstatemodel:revisions_unschedule', args=[self.test_snippet.pk, self.latest_revision.pk])
        self.assertTagInHTML(f'<button data-action="w-action#post" data-controller="w-action" data-w-action-url-value="{unschedule_url}">Cancel scheduled publish</button>', html, count=0, allow_extra_attrs=True)

    def test_edit_post_scheduled_for_publishing(self):
        if False:
            while True:
                i = 10
        response = self.post(post_data={'text': "I'm edited while it's locked for scheduled publishing!", 'go_live_at': submittable_timestamp(self.go_live_at)})
        self.test_snippet.refresh_from_db()
        self.assertEqual(self.test_snippet.latest_revision, self.latest_revision)
        self.assertEqual(self.test_snippet.latest_revision.content['text'], "I've been edited!")
        self.assertContains(response, 'The draft state model could not be saved as it is locked', count=1)
        self.assertNotContains(response, "Draft state model 'I&#x27;ve been edited!' is locked and has been scheduled to go live at")
        self.assertContains(response, 'Locked by schedule')
        self.assertContains(response, '<div class="w-help-text">Currently locked and will go live on the scheduled date</div>', html=True, count=1)
        html = response.content.decode()
        self.assertTagInHTML('<button type="button" data-a11y-dialog-show="schedule-publishing-dialog">Edit schedule</button>', html, count=0, allow_extra_attrs=True)
        unschedule_url = reverse('wagtailsnippets_tests_draftstatemodel:revisions_unschedule', args=[self.test_snippet.pk, self.latest_revision.pk])
        self.assertTagInHTML(f'<button data-action="w-action#post" data-controller="w-action" data-w-action-url-value="{unschedule_url}">Cancel scheduled publish</button>', html, count=0, allow_extra_attrs=True)

class TestSnippetUnschedule(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.login()
        self.test_snippet = DraftStateCustomPrimaryKeyModel.objects.create(custom_id='custom/1', text='Draft-enabled Foo', live=False)
        self.go_live_at = now() + datetime.timedelta(days=1)
        self.test_snippet.text = "I've been edited!"
        self.test_snippet.go_live_at = self.go_live_at
        self.latest_revision = self.test_snippet.save_revision()
        self.latest_revision.publish()
        self.test_snippet.refresh_from_db()
        self.unschedule_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:revisions_unschedule', args=[quote(self.test_snippet.pk), self.latest_revision.pk])

    def set_permissions(self, set_publish_permission):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        permissions = [Permission.objects.get(content_type__app_label='tests', codename='change_draftstatecustomprimarykeymodel'), Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')]
        if set_publish_permission:
            permissions.append(Permission.objects.get(content_type__app_label='tests', codename='publish_draftstatecustomprimarykeymodel'))
        self.user.user_permissions.add(*permissions)
        self.user.save()

    def test_get_unschedule_view_with_publish_permissions(self):
        if False:
            while True:
                i = 10
        self.set_permissions(True)
        response = self.client.get(self.unschedule_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/revisions/confirm_unschedule.html')

    def test_get_unschedule_view_bad_permissions(self):
        if False:
            return 10
        self.set_permissions(False)
        response = self.client.get(self.unschedule_url)
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_post_unschedule_view_with_publish_permissions(self):
        if False:
            return 10
        self.set_permissions(True)
        response = self.client.post(self.unschedule_url)
        self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:history', args=[quote(self.test_snippet.pk)]))
        self.test_snippet.refresh_from_db()
        self.latest_revision.refresh_from_db()
        self.assertIsNone(self.latest_revision.approved_go_live_at)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())

    def test_post_unschedule_view_bad_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_permissions(False)
        response = self.client.post(self.unschedule_url)
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.test_snippet.refresh_from_db()
        self.latest_revision.refresh_from_db()
        self.assertIsNotNone(self.latest_revision.approved_go_live_at)
        self.assertTrue(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())

    def test_post_unschedule_view_with_next_url(self):
        if False:
            while True:
                i = 10
        self.set_permissions(True)
        edit_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:edit', args=[quote(self.test_snippet.pk)])
        response = self.client.post(self.unschedule_url + f'?next={edit_url}')
        self.assertRedirects(response, edit_url)
        self.test_snippet.refresh_from_db()
        self.latest_revision.refresh_from_db()
        self.assertIsNone(self.latest_revision.approved_go_live_at)
        self.assertFalse(Revision.objects.for_instance(self.test_snippet).exclude(approved_go_live_at__isnull=True).exists())

class TestSnippetUnpublish(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()
        self.snippet = DraftStateCustomPrimaryKeyModel.objects.create(custom_id='custom/1', text='to be unpublished')
        self.unpublish_url = reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(self.snippet.pk),))

    def test_unpublish_view(self):
        if False:
            return 10
        '\n        This tests that the unpublish view responds with an unpublish confirm page\n        '
        response = self.client.get(self.unpublish_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_unpublish.html')

    def test_unpublish_view_invalid_pk(self):
        if False:
            i = 10
            return i + 15
        '\n        This tests that the unpublish view returns an error if the object pk is invalid\n        '
        response = self.client.get(reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:unpublish', args=(quote(12345),)))
        self.assertEqual(response.status_code, 404)

    def test_unpublish_view_get_bad_permissions(self):
        if False:
            print('Hello World!')
        "\n        This tests that the unpublish view doesn't allow users without unpublish permissions\n        "
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(self.unpublish_url)
        self.assertEqual(response.status_code, 302)

    def test_unpublish_view_post_bad_permissions(self):
        if False:
            while True:
                i = 10
        "\n        This tests that the unpublish view doesn't allow users without unpublish permissions\n        "
        mock_handler = mock.MagicMock()
        unpublished.connect(mock_handler)
        try:
            self.user.is_superuser = False
            self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
            self.user.save()
            response = self.client.post(self.unpublish_url)
            self.assertRedirects(response, reverse('wagtailadmin_home'))
            self.assertTrue(DraftStateCustomPrimaryKeyModel.objects.get(pk=self.snippet.pk).live)
            self.assertEqual(mock_handler.call_count, 0)
        finally:
            unpublished.disconnect(mock_handler)

    def test_unpublish_view_post_with_publish_permission(self):
        if False:
            print('Hello World!')
        '\n        This posts to the unpublish view and checks that the object was unpublished,\n        using a specific publish permission instead of relying on the superuser flag\n        '
        mock_handler = mock.MagicMock()
        unpublished.connect(mock_handler)
        try:
            self.user.is_superuser = False
            edit_permission = Permission.objects.get(content_type__app_label='tests', codename='change_draftstatecustomprimarykeymodel')
            publish_permission = Permission.objects.get(content_type__app_label='tests', codename='publish_draftstatecustomprimarykeymodel')
            admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
            self.user.user_permissions.add(edit_permission, publish_permission, admin_permission)
            self.user.save()
            response = self.client.post(self.unpublish_url)
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
            self.assertFalse(DraftStateCustomPrimaryKeyModel.objects.get(pk=self.snippet.pk).live)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateCustomPrimaryKeyModel)
            self.assertEqual(mock_call['instance'], self.snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateCustomPrimaryKeyModel)
        finally:
            unpublished.disconnect(mock_handler)

    def test_unpublish_view_post(self):
        if False:
            while True:
                i = 10
        '\n        This posts to the unpublish view and checks that the object was unpublished\n        '
        mock_handler = mock.MagicMock()
        unpublished.connect(mock_handler)
        try:
            response = self.client.post(self.unpublish_url)
            self.assertRedirects(response, reverse('wagtailsnippets_tests_draftstatecustomprimarykeymodel:list'))
            self.assertFalse(DraftStateCustomPrimaryKeyModel.objects.get(pk=self.snippet.pk).live)
            self.assertEqual(mock_handler.call_count, 1)
            mock_call = mock_handler.mock_calls[0][2]
            self.assertEqual(mock_call['sender'], DraftStateCustomPrimaryKeyModel)
            self.assertEqual(mock_call['instance'], self.snippet)
            self.assertIsInstance(mock_call['instance'], DraftStateCustomPrimaryKeyModel)
        finally:
            unpublished.disconnect(mock_handler)

    def test_after_unpublish_hook(self):
        if False:
            while True:
                i = 10

        def hook_func(request, snippet):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(snippet.pk, self.snippet.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('after_unpublish', hook_func):
            post_data = {}
            response = self.client.post(self.unpublish_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.snippet.refresh_from_db()
        self.assertEqual(self.snippet.status_string, 'draft')

    def test_before_unpublish(self):
        if False:
            i = 10
            return i + 15

        def hook_func(request, snippet):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertEqual(snippet.pk, self.snippet.pk)
            return HttpResponse('Overridden!')
        with self.register_hook('before_unpublish', hook_func):
            post_data = {}
            response = self.client.post(self.unpublish_url, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.snippet.refresh_from_db()
        self.assertEqual(self.snippet.status_string, 'live')

class TestSnippetDelete(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_snippet = Advert.objects.get(pk=1)
        self.user = self.login()

    def test_delete_get_with_limited_permissions(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertEqual(response.status_code, 302)

    def test_delete_get(self):
        if False:
            print('Hello World!')
        delete_url = reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)])
        response = self.client.get(delete_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Yes, delete')
        self.assertContains(response, delete_url)

    @override_settings(WAGTAIL_I18N_ENABLED=True)
    def test_delete_get_with_i18n_enabled(self):
        if False:
            while True:
                i = 10
        delete_url = reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)])
        response = self.client.get(delete_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Yes, delete')
        self.assertContains(response, delete_url)

    def test_delete_get_with_protected_reference(self):
        if False:
            while True:
                i = 10
        VariousOnDeleteModel.objects.create(text='Undeletable', on_delete_protect=self.test_snippet)
        delete_url = reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)])
        response = self.client.get(delete_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This advert is referenced 1 time.')
        self.assertContains(response, 'One or more references to this advert prevent it from being deleted.')
        self.assertContains(response, reverse('wagtailsnippets_tests_advert:usage', args=[quote(self.test_snippet.pk)]) + '?describe_on_delete=1')
        self.assertNotContains(response, 'Yes, delete')
        self.assertNotContains(response, delete_url)

    def test_delete_post_with_limited_permissions(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.post(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertEqual(response.status_code, 302)

    def test_delete_post(self):
        if False:
            i = 10
            return i + 15
        response = self.client.post(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertRedirects(response, reverse('wagtailsnippets_tests_advert:list'))
        self.assertEqual(Advert.objects.filter(text='test_advert').count(), 0)

    def test_delete_post_with_protected_reference(self):
        if False:
            print('Hello World!')
        VariousOnDeleteModel.objects.create(text='Undeletable', on_delete_protect=self.test_snippet)
        delete_url = reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)])
        response = self.client.post(delete_url)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('wagtailadmin_home'))
        self.assertTrue(Advert.objects.filter(pk=self.test_snippet.pk).exists())

    def test_usage_link(self):
        if False:
            while True:
                i = 10
        output = StringIO()
        management.call_command('rebuild_references_index', stdout=output)
        response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_delete.html')
        self.assertContains(response, 'This advert is referenced 2 times')
        self.assertContains(response, reverse('wagtailsnippets_tests_advert:usage', args=[quote(self.test_snippet.pk)]) + '?describe_on_delete=1')

    def test_before_delete_snippet_hook_get(self):
        if False:
            while True:
                i = 10
        advert = Advert.objects.create(url='http://www.example.com/', text='Test hook')

        def hook_func(request, instances):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertQuerysetEqual(instances, ['<Advert: Test hook>'], transform=repr)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_snippet', hook_func):
            response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(advert.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')

    def test_before_delete_snippet_hook_post(self):
        if False:
            while True:
                i = 10
        advert = Advert.objects.create(url='http://www.example.com/', text='Test hook')

        def hook_func(request, instances):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(request, HttpRequest)
            self.assertQuerysetEqual(instances, ['<Advert: Test hook>'], transform=repr)
            return HttpResponse('Overridden!')
        with self.register_hook('before_delete_snippet', hook_func):
            response = self.client.post(reverse('wagtailsnippets_tests_advert:delete', args=[quote(advert.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertTrue(Advert.objects.filter(pk=advert.pk).exists())

    def test_after_delete_snippet_hook(self):
        if False:
            i = 10
            return i + 15
        advert = Advert.objects.create(url='http://www.example.com/', text='Test hook')

        def hook_func(request, instances):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(request, HttpRequest)
            self.assertQuerysetEqual(instances, ['<Advert: Test hook>'], transform=repr)
            return HttpResponse('Overridden!')
        with self.register_hook('after_delete_snippet', hook_func):
            response = self.client.post(reverse('wagtailsnippets_tests_advert:delete', args=[quote(advert.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Overridden!')
        self.assertFalse(Advert.objects.filter(pk=advert.pk).exists())

class TestSnippetChooserPanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        model = SnippetChooserModel
        self.advert_text = 'Test advert text'
        test_snippet = model.objects.create(advert=Advert.objects.create(text=self.advert_text))
        self.edit_handler = get_edit_handler(model)
        self.form_class = self.edit_handler.get_form_class()
        form = self.form_class(instance=test_snippet)
        edit_handler = self.edit_handler.get_bound_panel(instance=test_snippet, form=form, request=self.request)
        self.snippet_chooser_panel = [panel for panel in edit_handler.children if getattr(panel, 'field_name', None) == 'advert'][0]

    def test_render_html(self):
        if False:
            for i in range(10):
                print('nop')
        field_html = self.snippet_chooser_panel.render_html()
        self.assertIn(self.advert_text, field_html)
        self.assertIn('Choose advert', field_html)
        self.assertIn('Choose another advert', field_html)
        self.assertIn('icon icon-snippet icon', field_html)

    def test_render_as_empty_field(self):
        if False:
            for i in range(10):
                print('nop')
        test_snippet = SnippetChooserModel()
        form = self.form_class(instance=test_snippet)
        edit_handler = self.edit_handler.get_bound_panel(instance=test_snippet, form=form, request=self.request)
        snippet_chooser_panel = [panel for panel in edit_handler.children if getattr(panel, 'field_name', None) == 'advert'][0]
        field_html = snippet_chooser_panel.render_html()
        self.assertIn('Choose advert', field_html)
        self.assertIn('Choose another advert', field_html)

    def test_render_js(self):
        if False:
            return 10
        self.assertIn('new SnippetChooser("id_advert", {"modalUrl": "/admin/snippets/choose/tests/advert/"});', self.snippet_chooser_panel.render_html())

    def test_target_model_autodetected(self):
        if False:
            for i in range(10):
                print('nop')
        edit_handler = ObjectList([FieldPanel('advert')]).bind_to_model(SnippetChooserModel)
        form_class = edit_handler.get_form_class()
        form = form_class()
        widget = form.fields['advert'].widget
        self.assertIsInstance(widget, AdminSnippetChooser)
        self.assertEqual(widget.model, Advert)

class TestSnippetRegistering(TestCase):

    def test_register_function(self):
        if False:
            i = 10
            return i + 15
        self.assertIn(RegisterFunction, SNIPPET_MODELS)

    def test_register_decorator(self):
        if False:
            print('Hello World!')
        self.assertIsNotNone(RegisterDecorator)
        self.assertIn(RegisterDecorator, SNIPPET_MODELS)

class TestSnippetOrdering(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        register_snippet(ZuluSnippet)
        register_snippet(AlphaSnippet)

    def test_snippets_ordering(self):
        if False:
            while True:
                i = 10
        self.assertLess(SNIPPET_MODELS.index(AlphaSnippet), SNIPPET_MODELS.index(ZuluSnippet))

class TestSnippetHistory(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def get(self, snippet, params={}):
        if False:
            i = 10
            return i + 15
        return self.client.get(self.get_url(snippet, 'history'), params)

    def get_url(self, snippet, url_name, args=None):
        if False:
            return 10
        if args is None:
            args = [quote(snippet.pk)]
        return reverse(snippet.snippet_viewset.get_url_name(url_name), args=args)

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        self.non_revisable_snippet = Advert.objects.get(pk=1)
        ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(Advert), label='Test Advert', action='wagtail.create', timestamp=make_aware(datetime.datetime(2021, 9, 30, 10, 1, 0)), object_id='1')
        ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(Advert), label='Test Advert Updated', action='wagtail.edit', timestamp=make_aware(datetime.datetime(2022, 5, 10, 12, 34, 0)), object_id='1')
        self.revisable_snippet = FullFeaturedSnippet.objects.create(text='Foo')
        self.initial_revision = self.revisable_snippet.save_revision(user=self.user)
        ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(FullFeaturedSnippet), label='Foo', action='wagtail.create', timestamp=make_aware(datetime.datetime(2022, 5, 10, 20, 22, 0)), object_id=self.revisable_snippet.pk, revision=self.initial_revision, content_changed=True)
        self.revisable_snippet.text = 'Bar'
        self.edit_revision = self.revisable_snippet.save_revision(user=self.user, log_action=True)

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get(self.non_revisable_snippet)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<td class="title">Created</td>', html=True)
        self.assertContains(response, 'data-w-tooltip-content-value="Sept. 30, 2021, 10:01 a.m."')

    def test_filters(self):
        if False:
            for i in range(10):
                print('nop')
        snippets = [self.non_revisable_snippet, self.revisable_snippet]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                response = self.get(snippet, {'action': 'wagtail.edit'})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'Edited', count=1)
                self.assertNotContains(response, 'Created')

    def test_should_not_show_actions_on_non_revisable_snippet(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(self.non_revisable_snippet)
        edit_url = self.get_url(self.non_revisable_snippet, 'edit')
        self.assertNotContains(response, f'<a href="{edit_url}" class="button button-small button-secondary">Edit</a>')

    def test_should_show_actions_on_revisable_snippet(self):
        if False:
            i = 10
            return i + 15
        response = self.get(self.revisable_snippet)
        edit_url = self.get_url(self.revisable_snippet, 'edit')
        revert_url = self.get_url(self.revisable_snippet, 'revisions_revert', args=[self.revisable_snippet.pk, self.initial_revision.pk])
        self.assertNotContains(response, '<span class="w-status w-status--primary">Live version</span>')
        self.assertNotContains(response, '<span class="w-status w-status--primary">Current draft</span>')
        self.assertContains(response, f'<a href="{edit_url}" class="button button-small button-secondary">Edit</a>', count=1)
        self.assertContains(response, f'<a href="{revert_url}" class="button button-small button-secondary">Review this version</a>', count=1)

    def test_with_live_and_draft_status(self):
        if False:
            for i in range(10):
                print('nop')
        snippet = DraftStateModel.objects.create(text='Draft-enabled Foo, Published')
        snippet.save_revision().publish()
        snippet.refresh_from_db()
        snippet.text = 'Draft-enabled Bar, In Draft'
        snippet.save_revision(log_action=True)
        response = self.get(snippet)
        self.assertContains(response, '<span class="w-status w-status--primary">Live version</span>', count=1, html=True)
        self.assertContains(response, '<span class="w-status w-status--primary">Current draft</span>', count=1, html=True)
        self.assertContains(response, '<span class="w-header__subtitle">Draft-enabled Bar, In Draft</span>')

    @override_settings(WAGTAIL_I18N_ENABLED=True)
    def test_get_with_i18n_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(self.non_revisable_snippet)
        self.assertEqual(response.status_code, 200)
        response = self.get(self.revisable_snippet)
        self.assertEqual(response.status_code, 200)

class TestSnippetRevisions(WagtailTestUtils, TestCase):

    @property
    def revert_url(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_url('revisions_revert', args=[quote(self.snippet.pk), self.initial_revision.pk])

    def get(self):
        if False:
            while True:
                i = 10
        return self.client.get(self.revert_url)

    def post(self, post_data={}):
        if False:
            i = 10
            return i + 15
        return self.client.post(self.revert_url, post_data)

    def get_url(self, url_name, args=None):
        if False:
            return 10
        view_name = self.snippet.snippet_viewset.get_url_name(url_name)
        if args is None:
            args = [quote(self.snippet.pk)]
        return reverse(view_name, args=args)

    def setUp(self):
        if False:
            return 10
        self.user = self.login()
        with freeze_time('2022-05-10 11:00:00'):
            self.snippet = RevisableModel.objects.create(text='The original text')
            self.initial_revision = self.snippet.save_revision(user=self.user)
            ModelLogEntry.objects.create(content_type=ContentType.objects.get_for_model(RevisableModel), label='The original text', action='wagtail.create', timestamp=now(), object_id=self.snippet.pk, revision=self.initial_revision, content_changed=True)
        self.snippet.text = 'The edited text'
        self.snippet.save()
        self.edit_revision = self.snippet.save_revision(user=self.user, log_action=True)

    def test_get_revert_revision(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        if settings.USE_TZ:
            expected_date_string = 'May 10, 2022, 8 p.m.'
        else:
            expected_date_string = 'May 10, 2022, 11 a.m.'
        self.assertContains(response, f'You are viewing a previous version of this Revisable model from <b>{expected_date_string}</b> by', count=1)
        self.assertContains(response, 'The original text', count=1)
        form_tag = f'<form action="{self.revert_url}" method="POST">'
        html = response.content.decode()
        self.assertTagInHTML(form_tag, html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'Replace current revision', count=1)

    def test_get_revert_revision_with_non_revisable_snippet(self):
        if False:
            for i in range(10):
                print('nop')
        snippet = Advert.objects.create(text='foo')
        response = self.client.get(f'/admin/snippets/tests/advert/history/{snippet.pk}/revisions/1/revert/')
        self.assertEqual(response.status_code, 404)

    def test_get_with_limited_permissions(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.get()
        self.assertEqual(response.status_code, 302)

    def test_get_with_draft_state_snippet(self):
        if False:
            print('Hello World!')
        self.snippet = DraftStateModel.objects.create(text='Draft-enabled Foo')
        self.initial_revision = self.snippet.save_revision()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, 'Replace current draft')
        self.assertContains(response, 'Publish this version')
        self.assertContains(response, '<button\n    type="submit"\n    name="action-publish"\n    value="action-publish"\n    class="button action-save button-longrunning warning"\n    data-controller="w-progress"\n    data-action="w-progress#activate"\n')
        unpublish_url = reverse('wagtailsnippets_tests_draftstatemodel:unpublish', args=(quote(self.snippet.pk),))
        self.assertNotContains(response, f'<a class="button action-secondary" href="{unpublish_url}">')
        self.assertNotContains(response, 'Unpublish')

    def test_get_with_previewable_snippet(self):
        if False:
            for i in range(10):
                print('nop')
        self.snippet = MultiPreviewModesModel.objects.create(text='Preview-enabled foo')
        self.initial_revision = self.snippet.save_revision()
        self.snippet.text = 'Preview-enabled bar'
        self.snippet.save_revision()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertContains(response, 'You are viewing a previous version of this', count=1)
        self.assertContains(response, 'Preview-enabled foo')
        form_tag = f'<form action="{self.revert_url}" method="POST">'
        html = response.content.decode()
        self.assertTagInHTML(form_tag, html, count=1, allow_extra_attrs=True)
        self.assertContains(response, 'Replace current revision', count=1)
        preview_url = self.get_url('preview_on_edit')
        self.assertContains(response, 'data-side-panel-toggle="preview"')
        self.assertContains(response, 'data-side-panel="preview"')
        self.assertContains(response, f'data-action="{preview_url}"')

    def test_replace_revision(self):
        if False:
            i = 10
            return i + 15
        get_response = self.get()
        text_from_revision = get_response.context['form'].initial['text']
        post_response = self.post(post_data={'text': text_from_revision + ' reverted', 'revision': self.initial_revision.pk})
        self.assertRedirects(post_response, self.get_url('list', args=[]))
        self.snippet.refresh_from_db()
        latest_revision = self.snippet.get_latest_revision()
        log_entry = ModelLogEntry.objects.filter(revision=latest_revision).first()
        self.assertEqual(self.snippet.text, 'The original text reverted')
        self.assertEqual(self.snippet.revisions.count(), 3)
        self.assertEqual(latest_revision.content['text'], 'The original text reverted')
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.action, 'wagtail.revert')

    def test_replace_with_limited_permissions(self):
        if False:
            i = 10
            return i + 15
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.post(post_data={'text': 'test text', 'revision': self.initial_revision.pk})
        self.assertEqual(response.status_code, 302)
        self.snippet.refresh_from_db()
        self.assertNotEqual(self.snippet.text, 'test text')
        self.assertEqual(self.snippet.revisions.count(), 2)

    def test_replace_draft(self):
        if False:
            i = 10
            return i + 15
        self.snippet = DraftStateModel.objects.create(text='Draft-enabled Foo', live=False)
        self.initial_revision = self.snippet.save_revision()
        self.snippet.text = 'Draft-enabled Foo edited'
        self.edit_revision = self.snippet.save_revision()
        get_response = self.get()
        text_from_revision = get_response.context['form'].initial['text']
        post_response = self.post(post_data={'text': text_from_revision + ' reverted', 'revision': self.initial_revision.pk})
        self.assertRedirects(post_response, self.get_url('edit'))
        self.snippet.refresh_from_db()
        latest_revision = self.snippet.get_latest_revision()
        log_entry = ModelLogEntry.objects.filter(revision=latest_revision).first()
        publish_log_entries = ModelLogEntry.objects.filter(content_type=ContentType.objects.get_for_model(DraftStateModel), action='wagtail.publish', object_id=self.snippet.pk)
        self.assertEqual(self.snippet.text, 'Draft-enabled Foo')
        self.assertEqual(self.snippet.revisions.count(), 3)
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Foo reverted')
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.action, 'wagtail.revert')
        self.assertEqual(publish_log_entries.count(), 0)
        self.assertFalse(self.snippet.live)
        self.assertTrue(self.snippet.has_unpublished_changes)
        self.assertIsNone(self.snippet.first_published_at)
        self.assertIsNone(self.snippet.last_published_at)
        self.assertIsNone(self.snippet.live_revision)

    def test_replace_publish(self):
        if False:
            while True:
                i = 10
        self.snippet = DraftStateModel.objects.create(text='Draft-enabled Foo')
        self.initial_revision = self.snippet.save_revision()
        self.snippet.text = 'Draft-enabled Foo edited'
        self.edit_revision = self.snippet.save_revision()
        get_response = self.get()
        text_from_revision = get_response.context['form'].initial['text']
        timestamp = now()
        with freeze_time(timestamp):
            post_response = self.post(post_data={'text': text_from_revision + ' reverted', 'revision': self.initial_revision.pk, 'action-publish': 'action-publish'})
        self.assertRedirects(post_response, self.get_url('list', args=[]))
        self.snippet.refresh_from_db()
        latest_revision = self.snippet.get_latest_revision()
        log_entry = ModelLogEntry.objects.filter(revision=latest_revision).first()
        revert_log_entries = ModelLogEntry.objects.filter(content_type=ContentType.objects.get_for_model(DraftStateModel), action='wagtail.revert', object_id=self.snippet.pk)
        self.assertEqual(self.snippet.text, 'Draft-enabled Foo reverted')
        self.assertEqual(self.snippet.revisions.count(), 3)
        self.assertEqual(latest_revision.content['text'], 'Draft-enabled Foo reverted')
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.action, 'wagtail.publish')
        self.assertEqual(revert_log_entries.count(), 1)
        self.assertTrue(self.snippet.live)
        self.assertFalse(self.snippet.has_unpublished_changes)
        self.assertEqual(self.snippet.first_published_at, timestamp)
        self.assertEqual(self.snippet.last_published_at, timestamp)
        self.assertEqual(self.snippet.live_revision, self.snippet.latest_revision)

class TestCompareRevisions(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.snippet = RevisableModel.objects.create(text='Initial revision')
        self.initial_revision = self.snippet.save_revision()
        self.initial_revision.created_at = make_aware(datetime.datetime(2022, 5, 10))
        self.initial_revision.save()
        self.snippet.text = 'First edit'
        self.edit_revision = self.snippet.save_revision()
        self.edit_revision.created_at = make_aware(datetime.datetime(2022, 5, 11))
        self.edit_revision.save()
        self.snippet.text = 'Final revision'
        self.final_revision = self.snippet.save_revision()
        self.final_revision.created_at = make_aware(datetime.datetime(2022, 5, 12))
        self.final_revision.save()
        self.login()

    def get(self, revision_a_id, revision_b_id):
        if False:
            for i in range(10):
                print('nop')
        compare_url = reverse('wagtailsnippets_tests_revisablemodel:revisions_compare', args=(quote(self.snippet.pk), revision_a_id, revision_b_id))
        return self.client.get(compare_url)

    def test_compare_revisions(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(self.initial_revision.pk, self.edit_revision.pk)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="deletion">Initial revision</span><span class="addition">First edit</span>', html=True)

    def test_compare_revisions_earliest(self):
        if False:
            while True:
                i = 10
        response = self.get('earliest', self.edit_revision.pk)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="deletion">Initial revision</span><span class="addition">First edit</span>', html=True)

    def test_compare_revisions_latest(self):
        if False:
            while True:
                i = 10
        response = self.get(self.edit_revision.id, 'latest')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="deletion">First edit</span><span class="addition">Final revision</span>', html=True)

    def test_compare_revisions_live(self):
        if False:
            print('Hello World!')
        self.snippet.text = 'Live edited'
        self.snippet.save(update_fields=['text'])
        response = self.get(self.final_revision.id, 'live')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="deletion">Final revision</span><span class="addition">Live edited</span>', html=True)

class TestCompareRevisionsWithPerUserPanels(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.snippet = RevisableChildModel.objects.create(text='Foo bar', secret_text='Secret text')
        self.old_revision = self.snippet.save_revision()
        self.snippet.text = 'Foo baz'
        self.snippet.secret_text = 'Secret unseen note'
        self.new_revision = self.snippet.save_revision()
        self.compare_url = reverse('wagtailsnippets_tests_revisablechildmodel:revisions_compare', args=(quote(self.snippet.pk), self.old_revision.pk, self.new_revision.pk))

    def test_comparison_as_superuser(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()
        response = self.client.get(self.compare_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Foo <span class="deletion">bar</span><span class="addition">baz</span>', html=True)
        self.assertContains(response, 'Secret <span class="deletion">text</span><span class="addition">unseen note</span>', html=True)

    def test_comparison_as_ordinary_user(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user(username='editor', password='password')
        add_permission = Permission.objects.get(content_type__app_label='tests', codename='change_revisablechildmodel')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        user.user_permissions.add(add_permission, admin_permission)
        self.login(username='editor', password='password')
        response = self.client.get(self.compare_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Foo <span class="deletion">bar</span><span class="addition">baz</span>', html=True)
        self.assertNotContains(response, 'unseen note')

class TestSnippetChoose(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.login()
        self.url_args = ['tests', 'advert']

    def get(self, params=None):
        if False:
            return 10
        (app_label, model_name) = self.url_args
        return self.client.get(reverse(f'wagtailsnippetchoosers_{app_label}_{model_name}:choose'), params or {})

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertNotIn('<select data-chooser-modal-search-filter name="lang">', response.json()['html'])

    def test_no_results(self):
        if False:
            return 10
        Advert.objects.all().delete()
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        response_html = response.json()['html']
        self.assertIn('href="/admin/snippets/tests/advert/add/"', response_html)

    def test_ordering(self):
        if False:
            print('Hello World!')
        '\n        Listing should be ordered by PK if no ordering has been set on the model\n        '
        Advert.objects.all().delete()
        for i in range(10, 0, -1):
            Advert.objects.create(pk=i, text='advert %d' % i)
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['results'][0].text, 'advert 1')

    def test_simple_pagination(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'p': 1})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        response = self.get({'p': 9999})
        self.assertEqual(response.status_code, 404)

    def test_not_searchable(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.get().context['filter_form'].fields.get('q'))

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_filter_requires_i18n_enabled(self):
        if False:
            print('Hello World!')
        self.url_args = ['snippetstests', 'translatablesnippet']
        fr_locale = Locale.objects.create(language_code='fr')
        TranslatableSnippet.objects.create(text='English snippet')
        TranslatableSnippet.objects.create(text='French snippet', locale=fr_locale)
        response = self.get()
        response_html = response.json()['html']
        self.assertNotIn('data-chooser-modal-search-filter', response_html)
        self.assertNotIn('name="locale"', response_html)

    @override_settings(WAGTAIL_I18N_ENABLED=True)
    def test_filter_by_locale(self):
        if False:
            for i in range(10):
                print('nop')
        self.url_args = ['snippetstests', 'translatablesnippet']
        fr_locale = Locale.objects.create(language_code='fr')
        TranslatableSnippet.objects.create(text='English snippet')
        TranslatableSnippet.objects.create(text='French snippet', locale=fr_locale)
        response = self.get()
        response_html = response.json()['html']
        self.assertIn('data-chooser-modal-search-filter', response_html)
        self.assertIn('name="locale"', response_html)
        self.assertEqual(len(response.context['results']), 2)
        self.assertEqual(response.context['results'][0].text, 'English snippet')
        self.assertEqual(response.context['results'][1].text, 'French snippet')
        response = self.get({'locale': 'en'})
        self.assertEqual(len(response.context['results']), 1)
        self.assertEqual(response.context['results'][0].text, 'English snippet')

class TestSnippetChooseResults(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()

    def get(self, params=None):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailsnippetchoosers_tests_advert:choose_results'), params or {})

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailsnippets/chooser/results.html')

    def test_no_results(self):
        if False:
            for i in range(10):
                print('nop')
        Advert.objects.all().delete()
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailsnippets/chooser/results.html')
        self.assertContains(response, 'href="/admin/snippets/tests/advert/add/"')

class TestSnippetChooseStatus(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        cls.draft = DraftStateModel.objects.create(text='foo', live=False)
        cls.live = DraftStateModel.objects.create(text='bar', live=True)
        cls.live_draft = DraftStateModel.objects.create(text='baz', live=True)
        cls.live_draft.save_revision()

    def get(self, view_name, params=None):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse(f'wagtailsnippetchoosers_tests_draftstatemodel:{view_name}'), params or {})

    def test_choose_view_shows_status_column(self):
        if False:
            print('Hello World!')
        response = self.get('choose')
        html = response.json()['html']
        self.assertTagInHTML('<th>Status</th>', html)
        self.assertTagInHTML('<span class="w-status">draft</span>', html)
        self.assertTagInHTML('<span class="w-status w-status--primary">live</span>', html)
        self.assertTagInHTML('<span class="w-status w-status--primary">live + draft</span>', html)

    def test_choose_results_view_shows_status_column(self):
        if False:
            i = 10
            return i + 15
        response = self.get('choose_results')
        self.assertContains(response, '<th>Status</th>', html=True)
        self.assertContains(response, '<span class="w-status">draft</span>', html=True)
        self.assertContains(response, '<span class="w-status w-status--primary">live</span>', html=True)
        self.assertContains(response, '<span class="w-status w-status--primary">live + draft</span>', html=True)

class TestSnippetChooseWithSearchableSnippet(WagtailTestUtils, TransactionTestCase):

    def setUp(self):
        if False:
            return 10
        self.login()
        self.snippet_a = SearchableSnippet.objects.create(text='Hello')
        self.snippet_b = SearchableSnippet.objects.create(text='World')
        self.snippet_c = SearchableSnippet.objects.create(text='Hello World')

    def get(self, params=None):
        if False:
            return 10
        return self.client.get(reverse('wagtailsnippetchoosers_snippetstests_searchablesnippet:choose'), params or {})

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        items = list(response.context['results'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

    def test_is_searchable(self):
        if False:
            return 10
        self.assertTrue(self.get().context['filter_form'].fields.get('q'))

    def test_search_hello(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': 'Hello'})
        items = list(response.context['results'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertNotIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

    def test_search_world(self):
        if False:
            print('Hello World!')
        response = self.get({'q': 'World'})
        items = list(response.context['results'].object_list)
        self.assertNotIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

    def test_partial_match(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': 'hello wo'})
        items = list(response.context['results'].object_list)
        self.assertNotIn(self.snippet_a, items)
        self.assertNotIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

class TestSnippetChooseWithNonAutocompleteSearchableSnippet(WagtailTestUtils, TransactionTestCase):
    """
    Test that searchable snippets with no AutocompleteFields defined can still be searched using
    full words
    """

    def setUp(self):
        if False:
            return 10
        self.login()
        self.snippet_a = NonAutocompleteSearchableSnippet.objects.create(text='Hello')
        self.snippet_b = NonAutocompleteSearchableSnippet.objects.create(text='World')
        self.snippet_c = NonAutocompleteSearchableSnippet.objects.create(text='Hello World')

    def get(self, params=None):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailsnippetchoosers_snippetstests_nonautocompletesearchablesnippet:choose'), params or {})

    def test_search_hello(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertWarnsRegex(RuntimeWarning, 'does not specify any AutocompleteFields'):
            response = self.get({'q': 'Hello'})
        items = list(response.context['results'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertNotIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

class TestSnippetChosen(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.login()

    def get(self, pk, params=None):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailsnippetchoosers_tests_advert:chosen', args=(pk,)), params or {})

    def test_choose_a_page(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(pk=Advert.objects.all()[0].pk)
        response_json = json.loads(response.content.decode())
        self.assertEqual(response_json['step'], 'chosen')

    def test_choose_a_non_existing_page(self):
        if False:
            return 10
        response = self.get(999999)
        self.assertEqual(response.status_code, 404)

class TestAddOnlyPermissions(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_snippet = Advert.objects.get(pk=1)
        user = self.create_user(username='addonly', email='addonly@example.com', password='password')
        add_permission = Permission.objects.get(content_type__app_label='tests', codename='add_advert')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        user.user_permissions.add(add_permission, admin_permission)
        self.login(username='addonly', password='password')

    def test_get_index(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_tests_advert:list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        self.assertContains(response, 'Add advert')

    def test_get_add(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:add'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/create.html')
        self.assertEqual(response.context['header_icon'], 'snippet')

    def test_get_edit(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:edit', args=[quote(self.test_snippet.pk)]))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_get_delete(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

class TestEditOnlyPermissions(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_snippet = Advert.objects.get(pk=1)
        user = self.create_user(username='changeonly', email='changeonly@example.com', password='password')
        change_permission = Permission.objects.get(content_type__app_label='tests', codename='change_advert')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        user.user_permissions.add(change_permission, admin_permission)
        self.login(username='changeonly', password='password')

    def test_get_index(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        self.assertNotContains(response, 'Add advert')

    def test_get_add(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_tests_advert:add'))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_get_edit(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:edit', args=[quote(self.test_snippet.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')
        self.assertEqual(response.context['header_icon'], 'snippet')

    def test_get_delete(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

class TestDeleteOnlyPermissions(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_snippet = Advert.objects.get(pk=1)
        user = self.create_user(username='deleteonly', password='password')
        change_permission = Permission.objects.get(content_type__app_label='tests', codename='delete_advert')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        user.user_permissions.add(change_permission, admin_permission)
        self.login(username='deleteonly', password='password')

    def test_get_index(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailsnippets_tests_advert:list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        self.assertNotContains(response, 'Add advert')

    def test_get_add(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_tests_advert:add'))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_get_edit(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:edit', args=[quote(self.test_snippet.pk)]))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_get_delete(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_tests_advert:delete', args=[quote(self.test_snippet.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_delete.html')
        self.assertEqual(response.context['header_icon'], 'snippet')

class TestSnippetEditHandlers(WagtailTestUtils, TestCase):

    def test_standard_edit_handler(self):
        if False:
            print('Hello World!')
        edit_handler = get_edit_handler(StandardSnippet)
        form_class = edit_handler.get_form_class()
        self.assertTrue(issubclass(form_class, WagtailAdminModelForm))
        self.assertFalse(issubclass(form_class, FancySnippetForm))

    def test_fancy_edit_handler(self):
        if False:
            print('Hello World!')
        edit_handler = get_edit_handler(FancySnippet)
        form_class = edit_handler.get_form_class()
        self.assertTrue(issubclass(form_class, WagtailAdminModelForm))
        self.assertTrue(issubclass(form_class, FancySnippetForm))

class TestInlinePanelMedia(WagtailTestUtils, TestCase):
    """
    Test that form media required by InlinePanels is correctly pulled in to the edit page
    """

    def test_inline_panel_media(self):
        if False:
            return 10
        self.login()
        response = self.client.get(reverse('wagtailsnippets_snippetstests_multisectionrichtextsnippet:add'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'wagtailadmin/js/draftail.js')

class TestSnippetChooserBlock(TestCase):
    fixtures = ['test.json']

    def test_serialize(self):
        if False:
            return 10
        'The value of a SnippetChooserBlock (a snippet instance) should serialize to an ID'
        block = SnippetChooserBlock(Advert)
        test_advert = Advert.objects.get(text='test_advert')
        self.assertEqual(block.get_prep_value(test_advert), test_advert.id)
        self.assertIsNone(block.get_prep_value(None))

    def test_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        'The serialized value of a SnippetChooserBlock (an ID) should deserialize to a snippet instance'
        block = SnippetChooserBlock(Advert)
        test_advert = Advert.objects.get(text='test_advert')
        self.assertEqual(block.to_python(test_advert.id), test_advert)
        self.assertIsNone(block.to_python(None))

    def test_reference_model_by_string(self):
        if False:
            while True:
                i = 10
        block = SnippetChooserBlock('tests.Advert')
        test_advert = Advert.objects.get(text='test_advert')
        self.assertEqual(block.to_python(test_advert.id), test_advert)

    def test_adapt(self):
        if False:
            while True:
                i = 10
        block = SnippetChooserBlock(Advert, help_text='pick an advert, any advert')
        block.set_name('test_snippetchooserblock')
        js_args = FieldBlockAdapter().js_args(block)
        self.assertEqual(js_args[0], 'test_snippetchooserblock')
        self.assertIsInstance(js_args[1], AdminSnippetChooser)
        self.assertEqual(js_args[1].model, Advert)
        self.assertEqual(js_args[2], {'label': 'Test snippetchooserblock', 'required': True, 'icon': 'snippet', 'helpText': 'pick an advert, any advert', 'classname': 'w-field w-field--model_choice_field w-field--admin_snippet_chooser', 'showAddCommentButton': True, 'strings': {'ADD_COMMENT': 'Add Comment'}})

    def test_form_response(self):
        if False:
            for i in range(10):
                print('nop')
        block = SnippetChooserBlock(Advert)
        test_advert = Advert.objects.get(text='test_advert')
        value = block.value_from_datadict({'advert': str(test_advert.id)}, {}, 'advert')
        self.assertEqual(value, test_advert)
        empty_value = block.value_from_datadict({'advert': ''}, {}, 'advert')
        self.assertIsNone(empty_value)

    def test_clean(self):
        if False:
            while True:
                i = 10
        required_block = SnippetChooserBlock(Advert)
        nonrequired_block = SnippetChooserBlock(Advert, required=False)
        test_advert = Advert.objects.get(text='test_advert')
        self.assertEqual(required_block.clean(test_advert), test_advert)
        with self.assertRaises(ValidationError):
            required_block.clean(None)
        self.assertEqual(nonrequired_block.clean(test_advert), test_advert)
        self.assertIsNone(nonrequired_block.clean(None))

    def test_deconstruct(self):
        if False:
            return 10
        block = SnippetChooserBlock(Advert, required=False)
        (path, args, kwargs) = block.deconstruct()
        self.assertEqual(path, 'wagtail.snippets.blocks.SnippetChooserBlock')
        self.assertEqual(args, (Advert,))
        self.assertEqual(kwargs, {'required': False})

    def test_extract_references(self):
        if False:
            i = 10
            return i + 15
        block = SnippetChooserBlock(Advert)
        test_advert = Advert.objects.get(text='test_advert')
        self.assertListEqual(list(block.extract_references(test_advert)), [(Advert, str(test_advert.id), '', '')])
        self.assertListEqual(list(block.extract_references(None)), [])

class TestAdminSnippetChooserWidget(WagtailTestUtils, TestCase):

    def test_adapt(self):
        if False:
            i = 10
            return i + 15
        widget = AdminSnippetChooser(Advert)
        js_args = SnippetChooserAdapter().js_args(widget)
        self.assertEqual(len(js_args), 3)
        self.assertInHTML('<input type="hidden" name="__NAME__" id="__ID__">', js_args[0])
        self.assertIn('Choose advert', js_args[0])
        self.assertEqual(js_args[1], '__ID__')

class TestSnippetListViewWithCustomPrimaryKey(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()
        self.snippet_a = StandardSnippetWithCustomPrimaryKey.objects.create(snippet_id='snippet/01', text='Hello')
        self.snippet_b = StandardSnippetWithCustomPrimaryKey.objects.create(snippet_id='snippet/02', text='Hello')
        self.snippet_c = StandardSnippetWithCustomPrimaryKey.objects.create(snippet_id='snippet/03', text='Hello')

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:list'), params)

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/index.html')
        items = list(response.context['page_obj'].object_list)
        self.assertIn(self.snippet_a, items)
        self.assertIn(self.snippet_b, items)
        self.assertIn(self.snippet_c, items)

class TestSnippetViewWithCustomPrimaryKey(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.login()
        self.snippet_a = StandardSnippetWithCustomPrimaryKey.objects.create(snippet_id='snippet/01', text='Hello')

    def get(self, snippet, params={}):
        if False:
            return 10
        args = [quote(snippet.pk)]
        return self.client.get(reverse(snippet.snippet_viewset.get_url_name('edit'), args=args), params)

    def post(self, snippet, post_data={}):
        if False:
            i = 10
            return i + 15
        args = [quote(snippet.pk)]
        return self.client.post(reverse(snippet.snippet_viewset.get_url_name('edit'), args=args), post_data)

    def create(self, snippet, post_data={}, model=Advert):
        if False:
            i = 10
            return i + 15
        return self.client.post(reverse(snippet.snippet_viewset.get_url_name('add')), post_data)

    def test_show_edit_view(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(self.snippet_a)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailsnippets/snippets/edit.html')

    def test_edit_invalid(self):
        if False:
            i = 10
            return i + 15
        response = self.post(self.snippet_a, post_data={'foo': 'bar'})
        self.assertContains(response, 'The snippet could not be saved due to errors.')
        self.assertContains(response, 'This field is required.')

    def test_edit(self):
        if False:
            while True:
                i = 10
        response = self.post(self.snippet_a, post_data={'text': 'Edited snippet', 'snippet_id': 'snippet_id_edited'})
        self.assertRedirects(response, reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:list'))
        snippets = StandardSnippetWithCustomPrimaryKey.objects.all()
        self.assertEqual(snippets.count(), 2)
        self.assertEqual(snippets.last().snippet_id, 'snippet_id_edited')

    def test_create(self):
        if False:
            i = 10
            return i + 15
        response = self.create(self.snippet_a, post_data={'text': 'test snippet', 'snippet_id': 'snippet/02'})
        self.assertRedirects(response, reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:list'))
        snippets = StandardSnippetWithCustomPrimaryKey.objects.all()
        self.assertEqual(snippets.count(), 2)
        self.assertEqual(snippets.last().text, 'test snippet')

    def test_get_delete(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:delete', args=[quote(self.snippet_a.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_delete.html')

    def test_usage_link(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:delete', args=[quote(self.snippet_a.pk)]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_delete.html')
        self.assertContains(response, 'This standard snippet with custom primary key is referenced 0 times')
        self.assertContains(response, reverse('wagtailsnippets_snippetstests_standardsnippetwithcustomprimarykey:usage', args=[quote(self.snippet_a.pk)]) + '?describe_on_delete=1')

    def test_redirect_to_edit(self):
        if False:
            return 10
        with self.assertWarnsRegex(RemovedInWagtail70Warning, '`/<pk>/` edit view URL pattern has been deprecated in favour of /edit/<pk>/.'):
            response = self.client.get('/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/snippet_2F01/')
        self.assertRedirects(response, '/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/edit/snippet_2F01/', status_code=301)

    def test_redirect_to_delete(self):
        if False:
            print('Hello World!')
        with self.assertWarnsRegex(RemovedInWagtail70Warning, '`/<pk>/delete/` delete view URL pattern has been deprecated in favour of /delete/<pk>/.'):
            response = self.client.get('/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/snippet_2F01/delete/')
        self.assertRedirects(response, '/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/delete/snippet_2F01/', status_code=301)

    def test_redirect_to_usage(self):
        if False:
            print('Hello World!')
        with self.assertWarnsRegex(RemovedInWagtail70Warning, '`/<pk>/usage/` usage view URL pattern has been deprecated in favour of /usage/<pk>/.'):
            response = self.client.get('/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/snippet_2F01/usage/')
        self.assertRedirects(response, '/admin/snippets/snippetstests/standardsnippetwithcustomprimarykey/usage/snippet_2F01/', status_code=301)

class TestSnippetChooserBlockWithCustomPrimaryKey(TestCase):
    fixtures = ['test.json']

    def test_serialize(self):
        if False:
            print('Hello World!')
        'The value of a SnippetChooserBlock (a snippet instance) should serialize to an ID'
        block = SnippetChooserBlock(AdvertWithCustomPrimaryKey)
        test_advert = AdvertWithCustomPrimaryKey.objects.get(pk='advert/01')
        self.assertEqual(block.get_prep_value(test_advert), test_advert.pk)
        self.assertIsNone(block.get_prep_value(None))

    def test_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        'The serialized value of a SnippetChooserBlock (an ID) should deserialize to a snippet instance'
        block = SnippetChooserBlock(AdvertWithCustomPrimaryKey)
        test_advert = AdvertWithCustomPrimaryKey.objects.get(pk='advert/01')
        self.assertEqual(block.to_python(test_advert.pk), test_advert)
        self.assertIsNone(block.to_python(None))

    def test_adapt(self):
        if False:
            return 10
        block = SnippetChooserBlock(AdvertWithCustomPrimaryKey, help_text='pick an advert, any advert')
        block.set_name('test_snippetchooserblock')
        js_args = FieldBlockAdapter().js_args(block)
        self.assertEqual(js_args[0], 'test_snippetchooserblock')
        self.assertIsInstance(js_args[1], AdminSnippetChooser)
        self.assertEqual(js_args[1].model, AdvertWithCustomPrimaryKey)
        self.assertEqual(js_args[2], {'label': 'Test snippetchooserblock', 'required': True, 'icon': 'snippet', 'helpText': 'pick an advert, any advert', 'classname': 'w-field w-field--model_choice_field w-field--admin_snippet_chooser', 'showAddCommentButton': True, 'strings': {'ADD_COMMENT': 'Add Comment'}})

    def test_form_response(self):
        if False:
            return 10
        block = SnippetChooserBlock(AdvertWithCustomPrimaryKey)
        test_advert = AdvertWithCustomPrimaryKey.objects.get(pk='advert/01')
        value = block.value_from_datadict({'advertwithcustomprimarykey': str(test_advert.pk)}, {}, 'advertwithcustomprimarykey')
        self.assertEqual(value, test_advert)
        empty_value = block.value_from_datadict({'advertwithcustomprimarykey': ''}, {}, 'advertwithcustomprimarykey')
        self.assertIsNone(empty_value)

    def test_clean(self):
        if False:
            for i in range(10):
                print('nop')
        required_block = SnippetChooserBlock(AdvertWithCustomPrimaryKey)
        nonrequired_block = SnippetChooserBlock(AdvertWithCustomPrimaryKey, required=False)
        test_advert = AdvertWithCustomPrimaryKey.objects.get(pk='advert/01')
        self.assertEqual(required_block.clean(test_advert), test_advert)
        with self.assertRaises(ValidationError):
            required_block.clean(None)
        self.assertEqual(nonrequired_block.clean(test_advert), test_advert)
        self.assertIsNone(nonrequired_block.clean(None))

class TestSnippetChooserPanelWithCustomPrimaryKey(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        model = SnippetChooserModelWithCustomPrimaryKey
        self.advert_text = 'Test advert text'
        test_snippet = model.objects.create(advertwithcustomprimarykey=AdvertWithCustomPrimaryKey.objects.create(advert_id='advert/02', text=self.advert_text))
        self.edit_handler = get_edit_handler(model)
        self.form_class = self.edit_handler.get_form_class()
        form = self.form_class(instance=test_snippet)
        edit_handler = self.edit_handler.get_bound_panel(instance=test_snippet, form=form, request=self.request)
        self.snippet_chooser_panel = [panel for panel in edit_handler.children if getattr(panel, 'field_name', None) == 'advertwithcustomprimarykey'][0]

    def test_render_html(self):
        if False:
            i = 10
            return i + 15
        field_html = self.snippet_chooser_panel.render_html()
        self.assertIn(self.advert_text, field_html)
        self.assertIn('Choose advert with custom primary key', field_html)
        self.assertIn('Choose another advert with custom primary key', field_html)

    def test_render_as_empty_field(self):
        if False:
            for i in range(10):
                print('nop')
        test_snippet = SnippetChooserModelWithCustomPrimaryKey()
        form = self.form_class(instance=test_snippet)
        edit_handler = self.edit_handler.get_bound_panel(instance=test_snippet, form=form, request=self.request)
        snippet_chooser_panel = [panel for panel in edit_handler.children if getattr(panel, 'field_name', None) == 'advertwithcustomprimarykey'][0]
        field_html = snippet_chooser_panel.render_html()
        self.assertIn('Choose advert with custom primary key', field_html)
        self.assertIn('Choose another advert with custom primary key', field_html)

    def test_render_js(self):
        if False:
            print('Hello World!')
        self.assertIn('new SnippetChooser("id_advertwithcustomprimarykey", {"modalUrl": "/admin/snippets/choose/tests/advertwithcustomprimarykey/"});', self.snippet_chooser_panel.render_html())

    def test_target_model_autodetected(self):
        if False:
            print('Hello World!')
        edit_handler = ObjectList([FieldPanel('advertwithcustomprimarykey')]).bind_to_model(SnippetChooserModelWithCustomPrimaryKey)
        form_class = edit_handler.get_form_class()
        form = form_class()
        widget = form.fields['advertwithcustomprimarykey'].widget
        self.assertIsInstance(widget, AdminSnippetChooser)
        self.assertEqual(widget.model, AdvertWithCustomPrimaryKey)

class TestSnippetChooseWithCustomPrimaryKey(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.login()

    def get(self, params=None):
        if False:
            return 10
        return self.client.get(reverse('wagtailsnippetchoosers_tests_advertwithcustomprimarykey:choose'), params or {})

    def test_simple(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertEqual(response.context['header_icon'], 'snippet')
        self.assertEqual(response.context['icon'], 'snippet')

    def test_ordering(self):
        if False:
            while True:
                i = 10
        '\n        Listing should be ordered by PK if no ordering has been set on the model\n        '
        AdvertWithCustomPrimaryKey.objects.all().delete()
        for i in range(10, 0, -1):
            AdvertWithCustomPrimaryKey.objects.create(pk=i, text='advert %d' % i)
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['results'][0].text, 'advert 1')

class TestSnippetChosenWithCustomPrimaryKey(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.login()

    def get(self, pk, params=None):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailsnippetchoosers_tests_advertwithcustomprimarykey:chosen', args=(quote(pk),)), params or {})

    def test_choose_a_page(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get(pk=AdvertWithCustomPrimaryKey.objects.all()[0].pk)
        response_json = json.loads(response.content.decode())
        self.assertEqual(response_json['step'], 'chosen')

class TestSnippetChosenWithCustomUUIDPrimaryKey(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.login()

    def get(self, pk, params=None):
        if False:
            print('Hello World!')
        return self.client.get(reverse('wagtailsnippetchoosers_tests_advertwithcustomuuidprimarykey:chosen', args=(quote(pk),)), params or {})

    def test_choose_a_page(self):
        if False:
            return 10
        response = self.get(pk=AdvertWithCustomUUIDPrimaryKey.objects.all()[0].pk)
        response_json = json.loads(response.content.decode())
        self.assertEqual(response_json['step'], 'chosen')

class TestPanelConfigurationChecks(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.warning_id = 'wagtailadmin.W002'

        def get_checks_result():
            if False:
                for i in range(10):
                    print('nop')
            checks_result = checks.run_checks(tags=['panels'])
            return [warning for warning in checks_result if warning.id == self.warning_id]
        self.get_checks_result = get_checks_result

    def test_model_with_single_tabbed_panel_only(self):
        if False:
            print('Hello World!')
        StandardSnippet.content_panels = [FieldPanel('text')]
        warning = checks.Warning('StandardSnippet.content_panels will have no effect on snippets editing', hint='Ensure that StandardSnippet uses `panels` instead of `content_panels`or set up an `edit_handler` if you want a tabbed editing interface.\nThere are no default tabs on non-Page models so there will be no Content tab for the content_panels to render in.', obj=StandardSnippet, id='wagtailadmin.W002')
        checks_results = self.get_checks_result()
        self.assertEqual([warning], checks_results)
        delattr(StandardSnippet, 'content_panels')