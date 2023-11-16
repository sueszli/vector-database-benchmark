from datetime import date, datetime, timezone
from functools import wraps
from typing import Any, List, Mapping, Optional
from unittest import mock
from django import forms
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser, Permission
from django.core import checks
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from django.utils.html import escape, json_script
from freezegun import freeze_time
from wagtail.admin.forms import WagtailAdminModelForm, WagtailAdminPageForm
from wagtail.admin.panels import CommentPanel, FieldPanel, FieldRowPanel, HelpPanel, InlinePanel, MultiFieldPanel, MultipleChooserPanel, ObjectList, PageChooserPanel, Panel, PublishingPanel, TabbedInterface, TitleFieldPanel, extract_panel_definitions_from_model_class, get_form_for_model
from wagtail.admin.rich_text import DraftailRichTextArea
from wagtail.admin.widgets import AdminAutoHeightTextInput, AdminDateInput, AdminPageChooser
from wagtail.contrib.forms.models import FormSubmission
from wagtail.contrib.forms.panels import FormSubmissionsPanel
from wagtail.coreutils import get_dummy_request
from wagtail.images import get_image_model
from wagtail.models import Comment, CommentReply, Page, Site
from wagtail.test.testapp.forms import ValidatedPageForm
from wagtail.test.testapp.models import Advert, EventPage, EventPageChooserModel, EventPageSpeaker, FormPageWithRedirect, GalleryPage, PageChooserModel, RestaurantPage, RestaurantTag, SimplePage, ValidatedPage
from wagtail.test.utils import WagtailTestUtils

class TestGetFormForModel(TestCase):

    def test_get_form_without_model(self):
        if False:
            print('Hello World!')
        edit_handler = ObjectList()
        with self.assertRaisesMessage(AttributeError, 'ObjectList is not bound to a model yet. Use `.bind_to_model(model)` before using this method.'):
            edit_handler.get_form_class()

    def test_get_form_for_model_without_explicit_fields(self):
        if False:
            for i in range(10):
                print('nop')
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm)
        self.assertTrue(issubclass(EventPageForm, WagtailAdminPageForm))
        form = EventPageForm()
        self.assertNotIn('title', form.fields)
        self.assertNotIn('path', form.fields)

    def test_get_form_for_model_without_formsets(self):
        if False:
            while True:
                i = 10
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'date_from', 'date_to'])
        form = EventPageForm()
        self.assertTrue(issubclass(EventPageForm, WagtailAdminModelForm))
        self.assertEqual(type(form.fields['title']), forms.CharField)
        self.assertEqual(type(form.fields['date_from']), forms.DateField)
        self.assertEqual(type(form.fields['date_from'].widget), AdminDateInput)
        self.assertNotIn('path', form.fields)

    def test_get_form_for_model_with_formsets(self):
        if False:
            i = 10
            return i + 15
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'date_from', 'date_to'], formsets=['speakers', 'related_links'])
        form = EventPageForm()
        self.assertIn('speakers', form.formsets)
        self.assertIn('related_links', form.formsets)

    def test_direct_form_field_overrides(self):
        if False:
            i = 10
            return i + 15
        SimplePageForm = get_form_for_model(SimplePage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'content'])
        self.assertTrue(issubclass(SimplePageForm, WagtailAdminPageForm))
        simple_form = SimplePageForm()
        self.assertEqual(type(simple_form.fields['content'].widget), AdminAutoHeightTextInput)
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'body'])
        event_form = EventPageForm()
        self.assertEqual(type(event_form.fields['body'].widget), DraftailRichTextArea)

    def test_get_form_for_model_with_specific_fields(self):
        if False:
            return 10
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['date_from'], formsets=['speakers'])
        form = EventPageForm()
        self.assertEqual(type(form.fields['date_from']), forms.DateField)
        self.assertEqual(type(form.fields['date_from'].widget), AdminDateInput)
        self.assertNotIn('title', form.fields)
        self.assertIn('speakers', form.formsets)
        self.assertNotIn('related_links', form.formsets)

    def test_get_form_for_model_without_explicit_formsets(self):
        if False:
            i = 10
            return i + 15
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['date_from'])
        form = EventPageForm()
        self.assertNotIn('speakers', form.formsets)
        self.assertNotIn('related_links', form.formsets)

    def test_get_form_for_model_with_excluded_fields(self):
        if False:
            i = 10
            return i + 15
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, exclude=['title'], exclude_formsets=['related_links'])
        form = EventPageForm()
        self.assertEqual(type(form.fields['date_from']), forms.DateField)
        self.assertEqual(type(form.fields['date_from'].widget), AdminDateInput)
        self.assertNotIn('title', form.fields)
        self.assertIn('speakers', form.formsets)
        self.assertNotIn('related_links', form.formsets)

    def test_get_form_for_model_with_widget_overides_by_class(self):
        if False:
            while True:
                i = 10
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['date_to', 'date_from'], widgets={'date_from': forms.PasswordInput})
        form = EventPageForm()
        self.assertEqual(type(form.fields['date_from']), forms.DateField)
        self.assertEqual(type(form.fields['date_from'].widget), forms.PasswordInput)

    def test_get_form_for_model_with_widget_overides_by_instance(self):
        if False:
            for i in range(10):
                print('nop')
        EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['date_to', 'date_from'], widgets={'date_from': forms.PasswordInput()})
        form = EventPageForm()
        self.assertEqual(type(form.fields['date_from']), forms.DateField)
        self.assertEqual(type(form.fields['date_from'].widget), forms.PasswordInput)

    def test_tag_widget_is_passed_tag_model(self):
        if False:
            for i in range(10):
                print('nop')
        RestaurantPageForm = get_form_for_model(RestaurantPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'tags'])
        form_html = RestaurantPageForm().as_p()
        self.assertIn('data-w-tag-url-value="/admin/tag-autocomplete/tests/restauranttag/"', form_html)
        self.assertIn(escape('"autocompleteOnly": true'), form_html)
        RestaurantTag.objects.create(name='Italian', slug='italian')
        RestaurantTag.objects.create(name='Indian', slug='indian')
        form = RestaurantPageForm({'title': 'Buonasera', 'slug': 'buonasera', 'tags': 'Italian, delicious'})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data['tags'], ['Italian'])

def clear_edit_handler(page_cls):
    if False:
        i = 10
        return i + 15

    def decorator(fn):
        if False:
            return 10

        @wraps(fn)
        def decorated(*args, **kwargs):
            if False:
                return 10
            page_cls.get_edit_handler.cache_clear()
            try:
                fn(*args, **kwargs)
            finally:
                page_cls.get_edit_handler.cache_clear()
        return decorated
    return decorator

class TestPageEditHandlers(TestCase):

    @clear_edit_handler(EventPage)
    def test_get_edit_handler(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Forms for pages should have a base class of WagtailAdminPageForm.\n        '
        edit_handler = EventPage.get_edit_handler()
        EventPageForm = edit_handler.get_form_class()
        self.assertTrue(issubclass(EventPageForm, WagtailAdminPageForm))

    @clear_edit_handler(ValidatedPage)
    def test_get_form_for_page_with_custom_base(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        ValidatedPage sets a custom base_form_class. This should be used as the\n        base class when constructing a form for ValidatedPages\n        '
        edit_handler = ValidatedPage.get_edit_handler()
        GeneratedValidatedPageForm = edit_handler.get_form_class()
        self.assertTrue(issubclass(GeneratedValidatedPageForm, ValidatedPageForm))

    @clear_edit_handler(ValidatedPage)
    def test_check_invalid_base_form_class(self):
        if False:
            return 10

        class BadFormClass:
            pass
        invalid_base_form = checks.Error('ValidatedPage.base_form_class does not extend WagtailAdminPageForm', hint='Ensure that wagtail.admin.tests.test_edit_handlers.BadFormClass extends WagtailAdminPageForm', obj=ValidatedPage, id='wagtailadmin.E001')
        invalid_edit_handler = checks.Error('ValidatedPage.get_edit_handler().get_form_class() does not extend WagtailAdminPageForm', hint='Ensure that the panel definition for ValidatedPage creates a subclass of WagtailAdminPageForm', obj=ValidatedPage, id='wagtailadmin.E002')
        with mock.patch.object(ValidatedPage, 'base_form_class', new=BadFormClass):
            errors = checks.run_checks()
            errors = [e for e in errors if e.level >= checks.ERROR]
            errors.sort(key=lambda e: e.id)
            self.assertEqual(errors, [invalid_base_form, invalid_edit_handler])

    @clear_edit_handler(ValidatedPage)
    def test_custom_edit_handler_form_class(self):
        if False:
            i = 10
            return i + 15
        '\n        Set a custom edit handler on a Page class, but dont customise\n        ValidatedPage.base_form_class, or provide a custom form class for the\n        edit handler. Check the generated form class is of the correct type.\n        '
        ValidatedPage.edit_handler = TabbedInterface()
        with mock.patch.object(ValidatedPage, 'edit_handler', new=TabbedInterface(), create=True):
            form_class = ValidatedPage.get_edit_handler().get_form_class()
            self.assertTrue(issubclass(form_class, WagtailAdminPageForm))
            errors = ValidatedPage.check()
            self.assertEqual(errors, [])

    @clear_edit_handler(ValidatedPage)
    def test_repr(self):
        if False:
            while True:
                i = 10
        edit_handler = ValidatedPage.get_edit_handler()
        handler_repr = repr(edit_handler)
        self.assertIn("model=<class 'wagtail.test.testapp.models.ValidatedPage'>", handler_repr)
        bound_handler = edit_handler.get_bound_panel(instance=None, request=None, form=None)
        bound_handler_repr = repr(bound_handler)
        self.assertIn("model=<class 'wagtail.test.testapp.models.ValidatedPage'>", bound_handler_repr)
        self.assertIn('instance=None', bound_handler_repr)
        self.assertIn('request=None', bound_handler_repr)
        self.assertIn('form=None', bound_handler_repr)

class TestExtractPanelDefinitionsFromModelClass(TestCase):

    def test_can_extract_panel_property(self):
        if False:
            i = 10
            return i + 15
        result = extract_panel_definitions_from_model_class(EventPageSpeaker)
        self.assertEqual(len(result), 5)
        self.assertTrue(any((isinstance(panel, MultiFieldPanel) for panel in result)))

    def test_exclude(self):
        if False:
            i = 10
            return i + 15
        panels = extract_panel_definitions_from_model_class(Site, exclude=['hostname'])
        for panel in panels:
            self.assertNotEqual(panel.field_name, 'hostname')

    def test_can_build_panel_list(self):
        if False:
            while True:
                i = 10
        panels = extract_panel_definitions_from_model_class(EventPage)
        self.assertTrue(any((isinstance(panel, FieldPanel) and panel.field_name == 'date_from' for panel in panels)))

class TestPanelAttributes(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.request = RequestFactory().get('/')
        user = self.create_superuser(username='admin')
        self.request.user = user
        self.user = self.login()
        self.event_page_tabbed_interface = TabbedInterface([ObjectList([HelpPanel('Double-check event details before submit.', attrs={'data-panel-type': 'help'}), FieldPanel('title', widget=forms.Textarea), FieldRowPanel([FieldPanel('date_from'), FieldPanel('date_to', attrs={'data-panel-type': 'field'})], attrs={'data-panel-type': 'field-row'})], heading='Event details', classname='shiny', attrs={'data-panel-type': 'object-list'}), ObjectList([InlinePanel('speakers', label='Speakers', attrs={'data-panel-type': 'inline'})], heading='Speakers'), ObjectList([MultiFieldPanel([HelpPanel('Double-check cost details before submit.', attrs={'data-panel-type': 'help-cost'}), FieldPanel('cost'), FieldRowPanel([FieldPanel('cost'), FieldPanel('cost', attrs={'data-panel-type': 'nested-object_list-multi_field-field_row-field'})], attrs={'data-panel-type': 'nested-object_list-multi_field-field_row'})], attrs={'data-panel-type': 'multi-field'})], heading='Secret')], attrs={'data-panel-type': 'tabs'}).bind_to_model(EventPage)

    def test_render(self):
        if False:
            while True:
                i = 10
        EventPageForm = self.event_page_tabbed_interface.get_form_class()
        event = EventPage(title='Abergavenny sheepdog trials')
        form = EventPageForm(instance=event)
        tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
        result = tabbed_interface.render_html()
        self.assertEqual(result.count('data-panel-type="tabs"'), 1)
        self.assertEqual(result.count('data-panel-type="multi-field"'), 1)
        self.assertEqual(result.count('data-panel-type="nested-object_list-multi_field-field_row"'), 1)
        self.assertEqual(result.count('data-panel-type="nested-object_list-multi_field-field_row-field"'), 1)
        self.assertEqual(result.count('data-panel-type="help-cost"'), 1)
        self.assertEqual(result.count('data-panel-type="inline"'), 1)
        self.assertEqual(result.count('data-panel-type="object-list"'), 1)
        self.assertEqual(result.count('data-panel-type="field-row"'), 1)
        self.assertEqual(result.count('data-panel-type="field"'), 1)
        self.assertEqual(result.count('data-panel-type="help"'), 1)

class TestTabbedInterface(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.request = RequestFactory().get('/')
        user = self.create_superuser(username='admin')
        self.request.user = user
        self.user = self.login()
        self.other_user = self.create_user(username='admin2', email='test2@email.com')
        p = Permission.objects.get(codename='custom_see_panel_setting')
        self.other_user.user_permissions.add(p)
        self.event_page_tabbed_interface = TabbedInterface([ObjectList([FieldPanel('title', widget=forms.Textarea), FieldPanel('date_from'), FieldPanel('date_to')], heading='Event details', classname='shiny'), ObjectList([InlinePanel('speakers', label='Speakers')], heading='Speakers'), ObjectList([FieldPanel('cost', permission='superuser')], heading='Secret'), ObjectList([FieldPanel('cost')], permission='tests.custom_see_panel_setting', heading='Custom Setting'), ObjectList([FieldPanel('cost')], permission='tests.other_custom_see_panel_setting', heading='Other Custom Setting')], attrs={'data-controller': 'my-tabbed-interface'}).bind_to_model(EventPage)

    def test_get_form_class(self):
        if False:
            for i in range(10):
                print('nop')
        EventPageForm = self.event_page_tabbed_interface.get_form_class()
        form = EventPageForm()
        self.assertIn('speakers', form.formsets)
        self.assertEqual(type(form.fields['title'].widget), forms.Textarea)

    def test_render(self):
        if False:
            print('Hello World!')
        EventPageForm = self.event_page_tabbed_interface.get_form_class()
        event = EventPage(title='Abergavenny sheepdog trials')
        form = EventPageForm(instance=event)
        tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
        result = tabbed_interface.render_html()
        self.assertIn('<a id="tab-label-event_details" href="#tab-event_details" class="w-tabs__tab shiny" role="tab" aria-selected="false" tabindex="-1">', result)
        self.assertIn('<a id="tab-label-speakers" href="#tab-speakers" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">', result)
        self.assertIn('aria-labelledby="tab-label-event_details"', result)
        self.assertIn('aria-labelledby="tab-label-speakers"', result)
        self.assertIn('Abergavenny sheepdog trials</textarea>', result)
        self.assertIn('data-controller="my-tabbed-interface"', result)
        self.assertNotIn('signup_link', result)

    def test_required_fields(self):
        if False:
            return 10
        result = set(self.event_page_tabbed_interface.get_form_options()['fields'])
        self.assertEqual(result, {'title', 'date_from', 'date_to', 'cost'})

    def test_render_form_content(self):
        if False:
            while True:
                i = 10
        EventPageForm = self.event_page_tabbed_interface.get_form_class()
        event = EventPage(title='Abergavenny sheepdog trials')
        form = EventPageForm(instance=event)
        tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
        result = tabbed_interface.render_form_content()
        self.assertIn('Abergavenny sheepdog trials</textarea>', result)
        self.assertNotIn('signup_link', result)

    def test_tabs_permissions(self):
        if False:
            print('Hello World!')
        '\n        test that three tabs show when the current user has permission to see all three\n        test that two tabs show when the current user does not have permission to see all three\n        '
        EventPageForm = self.event_page_tabbed_interface.get_form_class()
        event = EventPage(title='Abergavenny sheepdog trials')
        form = EventPageForm(instance=event)
        with self.subTest('Super user test'):
            tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
            result = tabbed_interface.render_html()
            self.assertIn('<a id="tab-label-event_details" href="#tab-event_details" class="w-tabs__tab shiny" role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertIn('<a id="tab-label-speakers" href="#tab-speakers" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertIn('<a id="tab-label-secret" href="#tab-secret" ', result)
            self.assertIn('<a id="tab-label-custom_setting" href="#tab-custom_setting" ', result)
            self.assertIn('<a id="tab-label-other_custom_setting" href="#tab-other_custom_setting" ', result)
        with self.subTest('Not superuser permissions'):
            '\n            The super user panel should not show, nor should the panel they dont have\n            permission for.\n            '
            self.request.user = self.other_user
            tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
            result = tabbed_interface.render_html()
            self.assertIn('<a id="tab-label-event_details" href="#tab-event_details" class="w-tabs__tab shiny" role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertIn('<a id="tab-label-speakers" href="#tab-speakers" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertNotIn('<a id="tab-label-secret" href="#tab-secret" ', result)
            self.assertIn('<a id="tab-label-custom_setting" href="#tab-custom_setting" ', result)
            self.assertNotIn('<a id="tab-label-other_custom_setting" href="#tab-other-custom_setting" ', result)
        with self.subTest('Non superuser'):
            user = AnonymousUser()
            self.request.user = user
            tabbed_interface = self.event_page_tabbed_interface.get_bound_panel(instance=event, form=form, request=self.request)
            result = tabbed_interface.render_html()
            self.assertIn('<a id="tab-label-event_details" href="#tab-event_details" class="w-tabs__tab shiny" role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertIn('<a id="tab-label-speakers" href="#tab-speakers" class="w-tabs__tab " role="tab" aria-selected="false" tabindex="-1">', result)
            self.assertNotIn('<a id="tab-label-secret" href="#tab-secret" ', result)
            self.assertNotIn('<a id="tab-label-custom_setting" href="#tab-custom_setting" ', result)
            self.assertNotIn('<a id="tab-label-other_custom_setting" href="#tab-other-custom_setting" ', result)

class TestObjectList(TestCase):

    def setUp(self):
        if False:
            return 10
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        self.event_page_object_list = ObjectList([FieldPanel('title', widget=forms.Textarea), FieldPanel('date_from'), FieldPanel('date_to'), InlinePanel('speakers', label='Speakers')], heading='Event details', classname='shiny', attrs={'data-controller': 'my-object-list'}).bind_to_model(EventPage)

    def test_get_form_class(self):
        if False:
            print('Hello World!')
        EventPageForm = self.event_page_object_list.get_form_class()
        form = EventPageForm()
        self.assertIn('speakers', form.formsets)
        self.assertEqual(type(form.fields['title'].widget), forms.Textarea)

    def test_render(self):
        if False:
            i = 10
            return i + 15
        EventPageForm = self.event_page_object_list.get_form_class()
        event = EventPage(title='Abergavenny sheepdog trials')
        form = EventPageForm(instance=event)
        object_list = self.event_page_object_list.get_bound_panel(instance=event, form=form, request=self.request)
        result = object_list.render_html()
        self.assertIn('<div class="w-panel__header">', result)
        self.assertIn('data-controller="my-object-list"', result)
        self.assertIn('<label for="id_date_from" id="id_date_from-label">', result)
        self.assertInHTML('<div class="help">Not required if event is on a single day</div>', result)
        self.assertIn('Abergavenny sheepdog trials</textarea>', result)
        self.assertNotIn('signup_link', result)

class TestFormatValueForDisplay(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.panel = Panel()
        self.event = EventPage(title='Abergavenny sheepdog trials', date_from=date(2014, 7, 20), date_to=date(2014, 7, 21), audience='public')

    def test_charfield_return_value(self):
        if False:
            while True:
                i = 10
        result = self.panel.format_value_for_display(self.event.title)
        self.assertIs(result, self.event.title)

    def test_datefield_return_value(self):
        if False:
            i = 10
            return i + 15
        result = self.panel.format_value_for_display(self.event.date_from)
        self.assertIs(result, self.event.date_from)

    def test_queryset_return_value(self):
        if False:
            return 10
        result = self.panel.format_value_for_display(Page.objects.all())
        self.assertEqual(result, 'Root, Welcome to your new Wagtail site!')

class TestFieldPanel(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        self.event = EventPage(title='Abergavenny sheepdog trials', date_from=date(2014, 7, 20), date_to=date(2014, 7, 21), audience='public')
        self.end_date_panel = FieldPanel('date_to', classname='full-width').bind_to_model(EventPage)
        self.read_only_end_date_panel = FieldPanel('date_to', read_only=True).bind_to_model(EventPage)
        self.read_only_audience_panel = FieldPanel('audience', read_only=True).bind_to_model(EventPage)
        self.read_only_image_panel = FieldPanel('feed_image', read_only=True).bind_to_model(EventPage)
        self.pontypridd_event_data = {'title': 'Pontypridd sheepdog trials', 'date_from': '2014-06-01', 'date_to': '2014-06-02'}

    def _get_form(self, data: Optional[Mapping[str, Any]]=None, fields: Optional[List[str]]=None) -> WagtailAdminPageForm:
        if False:
            return 10
        cls = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=fields if fields is not None else ['title', 'slug', 'date_to'], formsets=[])
        return cls(data=data, instance=self.event)

    def _get_bound_panel(self, panel: FieldPanel, form: WagtailAdminPageForm=None) -> FieldPanel.BoundPanel:
        if False:
            for i in range(10):
                print('nop')
        if not panel.model:
            panel = panel.bind_to_model(EventPage)
        return panel.get_bound_panel(form=form or self._get_form(), request=self.request, instance=self.event)

    def test_non_model_field(self):
        if False:
            for i in range(10):
                print('nop')
        field_panel = FieldPanel('barbecue').bind_to_model(Page)
        with self.assertRaises(FieldDoesNotExist):
            field_panel.db_field

    def test_get_form_options_includes_non_read_only_fields(self):
        if False:
            for i in range(10):
                print('nop')
        panel = self.end_date_panel
        result = panel.get_form_options()
        self.assertIn('fields', result)
        self.assertEqual(result['fields'], ['date_to'])

    def test_get_form_options_does_not_include_read_only_fields(self):
        if False:
            i = 10
            return i + 15
        panel = self.read_only_end_date_panel
        result = panel.get_form_options()
        self.assertNotIn('fields', result)

    def test_boundpanel_is_shown(self):
        if False:
            for i in range(10):
                print('nop')
        form = self._get_form(fields=['body', 'title'])
        for (field_name, make_read_only, expected_value) in (('title', True, True), ('body', True, True)):
            panel = FieldPanel(field_name, read_only=make_read_only)
            bound_panel = self._get_bound_panel(panel, form=form)
            with self.subTest(f'{field_name}, read_only={make_read_only}'):
                self.assertIs(bound_panel.is_shown(), expected_value)

    def test_override_heading(self):
        if False:
            print('Hello World!')
        bound_panel = self._get_bound_panel(self.end_date_panel)
        self.assertEqual(bound_panel.heading, bound_panel.bound_field.label)
        bound_panel = self._get_bound_panel(FieldPanel('date_to', classname='full-width', heading='New heading'))
        self.assertEqual(bound_panel.heading, 'New heading')
        self.assertEqual(bound_panel.bound_field.label, 'New heading')

    def test_render_html(self):
        if False:
            print('Hello World!')
        for (data, expected_input_value) in ((None, str(self.event.date_to)), (self.pontypridd_event_data, self.pontypridd_event_data['date_to'])):
            form = self._get_form(data=data, fields=['title', 'slug', 'date_to'])
            form.is_valid()
            bound_panel = self._get_bound_panel(self.end_date_panel, form=form)
            result = bound_panel.render_html()
            with self.subTest(f'form data = {data}'):
                self.assertIn('<input', result)
                self.assertIn(f'value="{expected_input_value}"', result)
                self.assertIn('data-field-wrapper', result)
                self.assertIn('Not required if event is on a single day', result)
                self.assertNotIn('error-message', result)

    def test_render_html_when_read_only(self):
        if False:
            return 10
        expected_value_output = self.event.date_to.strftime('%B %-d, %Y')
        for (panel, data) in ((self.read_only_end_date_panel, None), (self.read_only_end_date_panel, self.pontypridd_event_data)):
            form = self._get_form(data=data, fields=['title', 'slug'])
            form.is_valid()
            bound_panel = self._get_bound_panel(panel, form=form)
            with self.subTest(f'form data = {data}'):
                result = bound_panel.render_html()
                self.assertNotIn('<input', result)
                self.assertIn(expected_value_output, result)
                self.assertIn('Not required if event is on a single day', result)

    def test_format_value_for_display_with_choicefield(self):
        if False:
            return 10
        result = self.read_only_audience_panel.format_value_for_display(self.event.audience)
        self.assertEqual(result, 'Public')

    def test_format_value_for_display_with_modelchoicefield(self):
        if False:
            print('Hello World!')
        "\n        `ForeignKey.formfield()` returns a `ModelChoiceField`, which returns a\n        `ModelChoiceIterator` instance when it's `choices` property is\n        accessed. This test is to show that `format_value_for_display()` avoids\n        evaluating `ModelChoiceIterator` instances, and the database query\n        that would trigger.\n        "
        image = get_image_model()(title='Title')
        with self.assertNumQueries(0):
            self.assertEqual(self.read_only_image_panel.format_value_for_display(image), image)

    def test_required_fields(self):
        if False:
            while True:
                i = 10
        result = self.end_date_panel.get_form_options()['fields']
        self.assertEqual(result, ['date_to'])

    def test_error_message_is_rendered(self):
        if False:
            return 10
        form = self._get_form(data={'title': 'Pontypridd sheepdog trials', 'date_from': '2014-07-20', 'date_to': '2014-07-33'})
        form.is_valid()
        bound_panel = self._get_bound_panel(self.end_date_panel, form)
        result = bound_panel.render_html()
        self.assertIn('Enter a valid date.', result)

    def test_repr(self):
        if False:
            return 10
        bound_panel = self._get_bound_panel(self.end_date_panel)
        field_panel_repr = repr(bound_panel)
        self.assertIn("model=<class 'wagtail.test.testapp.models.EventPage'>", field_panel_repr)
        self.assertIn('instance=Abergavenny sheepdog trials', field_panel_repr)
        self.assertIn("request=<WSGIRequest: GET '/'>", field_panel_repr)
        self.assertIn('form=EventPageForm', field_panel_repr)

class TestFieldRowPanel(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        self.EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'date_from', 'date_to'], formsets=[])
        self.event = EventPage(title='Abergavenny sheepdog trials', date_from=date(2014, 7, 20), date_to=date(2014, 7, 21))
        self.dates_panel = FieldRowPanel([FieldPanel('date_from', classname='col4', heading='Start'), FieldPanel('date_to', classname='coltwo')], help_text='Confirmed event dates only').bind_to_model(EventPage)

    def test_render_html(self):
        if False:
            print('Hello World!')
        form = self.EventPageForm({'title': 'Pontypridd sheepdog trials', 'date_from': '2014-07-20', 'date_to': '2014-07-22'}, instance=self.event)
        form.is_valid()
        field_panel = self.dates_panel.get_bound_panel(instance=self.event, form=form, request=self.request)
        result = field_panel.render_html()
        self.assertIn('<label class="w-field__label" for="id_date_to" id="id_date_to-label">', result)
        self.assertIn('Not required if event is on a single day', result)
        self.assertIn('Confirmed event dates only', result)
        self.assertIn('value="2014-07-22"', result)
        self.assertNotIn('error-message', result)

    def test_error_message_is_rendered(self):
        if False:
            for i in range(10):
                print('nop')
        form = self.EventPageForm({'title': 'Pontypridd sheepdog trials', 'date_from': '2014-07-20', 'date_to': '2014-07-33'}, instance=self.event)
        form.is_valid()
        field_panel = self.dates_panel.get_bound_panel(instance=self.event, form=form, request=self.request)
        result = field_panel.render_html()
        self.assertIn('Enter a valid date.', result)

class TestFieldRowPanelWithChooser(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        self.EventPageForm = get_form_for_model(EventPage, form_class=WagtailAdminPageForm, fields=['title', 'slug', 'date_from', 'date_to'], formsets=[])
        self.event = EventPage(title='Abergavenny sheepdog trials', date_from=date(2014, 7, 19), date_to=date(2014, 7, 21))
        self.dates_panel = FieldRowPanel([FieldPanel('date_from'), FieldPanel('feed_image')]).bind_to_model(EventPage)

    def test_render_html(self):
        if False:
            return 10
        form = self.EventPageForm({'title': 'Pontypridd sheepdog trials', 'date_from': '2014-07-20', 'date_to': '2014-07-22'}, instance=self.event)
        form.is_valid()
        field_panel = self.dates_panel.get_bound_panel(instance=self.event, form=form, request=self.request)
        result = field_panel.render_html()
        self.assertIn('value="2014-07-20"', result)
        self.assertNotIn('error-message', result)

class TestPageChooserPanel(TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        model = PageChooserModel
        self.edit_handler = ObjectList([PageChooserPanel('page')]).bind_to_model(PageChooserModel)
        self.my_page_chooser_panel = self.edit_handler.children[0]
        self.PageChooserForm = self.edit_handler.get_form_class()
        self.christmas_page = Page.objects.get(slug='christmas')
        self.events_index_page = Page.objects.get(slug='events')
        self.test_instance = model.objects.create(page=self.christmas_page)
        self.form = self.PageChooserForm(instance=self.test_instance)
        self.page_chooser_panel = self.my_page_chooser_panel.get_bound_panel(instance=self.test_instance, form=self.form, request=self.request)

    def test_page_chooser_uses_correct_widget(self):
        if False:
            return 10
        self.assertEqual(type(self.form.fields['page'].widget), AdminPageChooser)

    def test_render_js_init(self):
        if False:
            while True:
                i = 10
        result = self.page_chooser_panel.render_html()
        expected_js = 'new PageChooser("{id}", {{"modelNames": ["{model}"], "canChooseRoot": false, "userPerms": null, "modalUrl": "/admin/choose-page/", "parentId": {parent}}});'.format(id='id_page', model='wagtailcore.page', parent=self.events_index_page.id)
        self.assertIn(expected_js, result)

    def test_render_js_init_with_can_choose_root_true(self):
        if False:
            i = 10
            return i + 15
        my_page_object_list = ObjectList([PageChooserPanel('page', can_choose_root=True)]).bind_to_model(PageChooserModel)
        my_page_chooser_panel = my_page_object_list.children[0]
        PageChooserForm = my_page_object_list.get_form_class()
        form = PageChooserForm(instance=self.test_instance)
        page_chooser_panel = my_page_chooser_panel.get_bound_panel(instance=self.test_instance, form=form, request=self.request)
        result = page_chooser_panel.render_html()
        expected_js = 'new PageChooser("{id}", {{"modelNames": ["{model}"], "canChooseRoot": true, "userPerms": null, "modalUrl": "/admin/choose-page/", "parentId": {parent}}});'.format(id='id_page', model='wagtailcore.page', parent=self.events_index_page.id)
        self.assertIn(expected_js, result)

    def test_render_html(self):
        if False:
            print('Hello World!')
        result = self.page_chooser_panel.render_html()
        self.assertIn('<div class="help">help text</div>', result)
        self.assertIn('<div class="chooser__title" data-chooser-title id="id_page-title">Christmas</div>', result)
        self.assertIn('<a data-chooser-edit-link href="/admin/pages/%d/edit/" aria-describedby="id_page-title"' % self.christmas_page.id, result)

    def test_render_as_empty_field(self):
        if False:
            while True:
                i = 10
        test_instance = PageChooserModel()
        form = self.PageChooserForm(instance=test_instance)
        page_chooser_panel = self.my_page_chooser_panel.get_bound_panel(instance=test_instance, form=form, request=self.request)
        result = page_chooser_panel.render_html()
        self.assertIn('<div class="help">help text</div>', result)
        self.assertIn('<div class="chooser__title" data-chooser-title id="id_page-title"></div>', result)
        self.assertIn('Choose a page', result)

    def test_render_error(self):
        if False:
            i = 10
            return i + 15
        form = self.PageChooserForm({'page': ''}, instance=self.test_instance)
        self.assertFalse(form.is_valid())
        page_chooser_panel = self.my_page_chooser_panel.get_bound_panel(instance=self.test_instance, form=form, request=self.request)
        self.assertIn('error-message', page_chooser_panel.render_html())

    def test_override_page_type(self):
        if False:
            while True:
                i = 10
        my_page_object_list = ObjectList([PageChooserPanel('page', 'tests.EventPage')]).bind_to_model(EventPageChooserModel)
        my_page_chooser_panel = my_page_object_list.children[0]
        PageChooserForm = my_page_object_list.get_form_class()
        form = PageChooserForm(instance=self.test_instance)
        page_chooser_panel = my_page_chooser_panel.get_bound_panel(instance=self.test_instance, form=form, request=self.request)
        result = page_chooser_panel.render_html()
        expected_js = 'new PageChooser("{id}", {{"modelNames": ["{model}"], "canChooseRoot": false, "userPerms": null, "modalUrl": "/admin/choose-page/", "parentId": {parent}}});'.format(id='id_page', model='tests.eventpage', parent=self.events_index_page.id)
        self.assertIn(expected_js, result)

    def test_autodetect_page_type(self):
        if False:
            for i in range(10):
                print('nop')
        my_page_object_list = ObjectList([PageChooserPanel('page')]).bind_to_model(EventPageChooserModel)
        my_page_chooser_panel = my_page_object_list.children[0]
        PageChooserForm = my_page_object_list.get_form_class()
        form = PageChooserForm(instance=self.test_instance)
        page_chooser_panel = my_page_chooser_panel.get_bound_panel(instance=self.test_instance, form=form, request=self.request)
        result = page_chooser_panel.render_html()
        expected_js = 'new PageChooser("{id}", {{"modelNames": ["{model}"], "canChooseRoot": false, "userPerms": null, "modalUrl": "/admin/choose-page/", "parentId": {parent}}});'.format(id='id_page', model='tests.eventpage', parent=self.events_index_page.id)
        self.assertIn(expected_js, result)

    def test_target_models(self):
        if False:
            while True:
                i = 10
        panel = PageChooserPanel('page', 'wagtailcore.site').bind_to_model(PageChooserModel)
        widget = panel.get_form_options()['widgets']['page']
        self.assertEqual(widget.target_models, [Site])

    def test_target_models_malformed_type(self):
        if False:
            return 10
        panel = PageChooserPanel('page', 'snowman').bind_to_model(PageChooserModel)
        self.assertRaises(ImproperlyConfigured, panel.get_form_options)

    def test_target_models_nonexistent_type(self):
        if False:
            return 10
        panel = PageChooserPanel('page', 'snowman.lorry').bind_to_model(PageChooserModel)
        self.assertRaises(ImproperlyConfigured, panel.get_form_options)

class TestInlinePanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user

    def test_render(self):
        if False:
            print('Hello World!')
        "\n        Check that the inline panel renders the panels set on the model\n        when no 'panels' parameter is passed in the InlinePanel definition\n        "
        speaker_object_list = ObjectList([InlinePanel('speakers', label='Speakers', classname='classname-for-speakers', attrs={'data-controller': 'test'})]).bind_to_model(EventPage)
        EventPageForm = speaker_object_list.get_form_class()
        self.assertEqual(['speakers'], list(EventPageForm.formsets.keys()))
        event_page = EventPage.objects.get(slug='christmas')
        form = EventPageForm(instance=event_page)
        panel = speaker_object_list.get_bound_panel(instance=event_page, form=form, request=self.request)
        result = panel.render_html()
        self.assertIn('<label class="w-field__label" for="id_speakers-0-first_name" id="id_speakers-0-first_name-label">', result)
        self.assertIn('value="Father"', result)
        self.assertIn('<label class="w-field__label" for="id_speakers-0-last_name" id="id_speakers-0-last_name-label">', result)
        self.assertIn('<label class="w-field__label" for="id_speakers-0-image" id="id_speakers-0-image-label">', result)
        self.assertIn('Choose an image', result)
        self.assertTagInHTML('<input id="id_speakers-0-id" name="speakers-0-id" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-0-DELETE" name="speakers-0-DELETE" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-0-ORDER" name="speakers-0-ORDER" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-TOTAL_FORMS" name="speakers-TOTAL_FORMS" type="hidden">', result, allow_extra_attrs=True)
        self.assertIn('var panel = new InlinePanel({', result)
        self.assertIn('data-contentpath-disabled', result)
        self.assertIn('data-controller="test"', result)

    def test_render_with_panel_overrides(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that inline panel renders the panels listed in the InlinePanel definition\n        where one is specified\n        '
        speaker_object_list = ObjectList([InlinePanel('speakers', label='Speakers', panels=[FieldPanel('first_name', widget=forms.Textarea), FieldPanel('image')])]).bind_to_model(EventPage)
        speaker_inline_panel = speaker_object_list.children[0]
        EventPageForm = speaker_object_list.get_form_class()
        self.assertEqual(['speakers'], list(EventPageForm.formsets.keys()))
        event_page = EventPage.objects.get(slug='christmas')
        form = EventPageForm(instance=event_page)
        panel = speaker_inline_panel.get_bound_panel(instance=event_page, form=form, request=self.request)
        result = panel.render_html()
        self.assertIn('<label class="w-field__label" for="id_speakers-0-first_name" id="id_speakers-0-first_name-label">', result)
        self.assertIn('Father</textarea>', result)
        self.assertNotIn('<label class="w-field__label" for="id_speakers-0-last_name" id="id_speakers-0-last_name-label">', result)
        self.assertTagInHTML('<input id="id_speakers-0-last_name">', result, count=0, allow_extra_attrs=True)
        self.assertIn('<label class="w-field__label" for="id_speakers-0-image" id="id_speakers-0-image-label">', result)
        self.assertIn('Choose an image', result)
        self.assertTagInHTML('<input id="id_speakers-0-id" name="speakers-0-id" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-0-DELETE" name="speakers-0-DELETE" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-0-ORDER" name="speakers-0-ORDER" type="hidden">', result, allow_extra_attrs=True)
        self.assertTagInHTML('<input id="id_speakers-TOTAL_FORMS" name="speakers-TOTAL_FORMS" type="hidden">', result, allow_extra_attrs=True)
        self.assertIn('var panel = new InlinePanel({', panel.render_html())

    @override_settings(USE_L10N=True, USE_THOUSAND_SEPARATOR=True)
    def test_no_thousand_separators_in_js(self):
        if False:
            return 10
        '\n        Test that the USE_THOUSAND_SEPARATOR setting does not screw up the rendering of numbers\n        (specifically maxForms=1000) in the JS initializer:\n        https://github.com/wagtail/wagtail/pull/2699\n        https://github.com/wagtail/wagtail/issues/3227\n        '
        speaker_object_list = ObjectList([InlinePanel('speakers', label='Speakers', panels=[FieldPanel('first_name', widget=forms.Textarea), FieldPanel('image')])]).bind_to_model(EventPage)
        speaker_inline_panel = speaker_object_list.children[0]
        EventPageForm = speaker_object_list.get_form_class()
        event_page = EventPage.objects.get(slug='christmas')
        form = EventPageForm(instance=event_page)
        panel = speaker_inline_panel.get_bound_panel(instance=event_page, form=form, request=self.request)
        self.assertIn('maxForms: 1000', panel.render_html())

    def test_invalid_inlinepanel_declaration(self):
        if False:
            print('Hello World!')
        with self.ignore_deprecation_warnings():
            self.assertRaises(TypeError, lambda : InlinePanel(label='Speakers'))
            self.assertRaises(TypeError, lambda : InlinePanel(EventPage, 'speakers', label='Speakers', bacon='chunky'))

class TestInlinePanelGetComparison(TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            print('Hello World!')
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user

    def test_get_comparison(self):
        if False:
            return 10
        page = Page.objects.get(id=4).specific
        comparison = page.get_edit_handler().get_bound_panel(instance=page, request=self.request).get_comparison()
        comparison = [comp(page, page) for comp in comparison]
        field_labels = [comp.field_label() for comp in comparison]
        self.assertIn('Speakers', field_labels)

class TestInlinePanelRelatedModelPanelConfigChecks(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.original_panels = EventPageSpeaker.panels
        delattr(EventPageSpeaker, 'panels')

        def get_checks_result():
            if False:
                while True:
                    i = 10
            checks_result = checks.run_checks(tags=['panels'])
            return [warning for warning in checks_result if warning.obj == EventPageSpeaker]
        self.warning_id = 'wagtailadmin.W002'
        self.get_checks_result = get_checks_result

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        EventPageSpeaker.panels = self.original_panels

    def test_page_with_inline_model_with_tabbed_panel_only(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that checks will warn against setting single tabbed panel on InlinePanel model'
        EventPageSpeaker.settings_panels = [FieldPanel('first_name'), FieldPanel('last_name')]
        warning = checks.Warning('EventPageSpeaker.settings_panels will have no effect on InlinePanel model editing', hint='Ensure that EventPageSpeaker uses `panels` instead of `settings_panels`.\nThere are no tabs on non-Page model editing within InlinePanels.', obj=EventPageSpeaker, id=self.warning_id)
        checks_results = self.get_checks_result()
        self.assertIn(warning, checks_results)
        delattr(EventPageSpeaker, 'settings_panels')

    def test_page_with_inline_model_with_two_tabbed_panels(self):
        if False:
            return 10
        'Test that checks will warn against multiple tabbed panels on InlinePanel models'
        EventPageSpeaker.content_panels = [FieldPanel('first_name')]
        EventPageSpeaker.promote_panels = [FieldPanel('last_name')]
        warning_1 = checks.Warning('EventPageSpeaker.content_panels will have no effect on InlinePanel model editing', hint='Ensure that EventPageSpeaker uses `panels` instead of `content_panels`.\nThere are no tabs on non-Page model editing within InlinePanels.', obj=EventPageSpeaker, id=self.warning_id)
        warning_2 = checks.Warning('EventPageSpeaker.promote_panels will have no effect on InlinePanel model editing', hint='Ensure that EventPageSpeaker uses `panels` instead of `promote_panels`.\nThere are no tabs on non-Page model editing within InlinePanels.', obj=EventPageSpeaker, id=self.warning_id)
        checks_results = self.get_checks_result()
        self.assertIn(warning_1, checks_results)
        self.assertIn(warning_2, checks_results)
        delattr(EventPageSpeaker, 'content_panels')
        delattr(EventPageSpeaker, 'promote_panels')

    def test_page_with_inline_model_with_edit_handler(self):
        if False:
            i = 10
            return i + 15
        'Checks should NOT warn if InlinePanel models use tabbed panels AND edit_handler'
        EventPageSpeaker.content_panels = [FieldPanel('first_name')]
        EventPageSpeaker.edit_handler = TabbedInterface([ObjectList([FieldPanel('last_name')], heading='test')])
        self.assertEqual(self.get_checks_result(), [])
        delattr(EventPageSpeaker, 'edit_handler')
        delattr(EventPageSpeaker, 'content_panels')

class TestCommentPanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.commenting_user = get_user_model().objects.get(pk=7)
        self.other_user = get_user_model().objects.get(pk=6)
        self.request = RequestFactory().get('/')
        self.request.user = self.commenting_user
        unbound_object_list = ObjectList([CommentPanel()])
        self.object_list = unbound_object_list.bind_to_model(EventPage)
        self.tabbed_interface = TabbedInterface([unbound_object_list]).bind_to_model(EventPage)
        self.EventPageForm = self.object_list.get_form_class()
        self.event_page = EventPage.objects.get(slug='christmas')
        self.comment = Comment.objects.create(page=self.event_page, text='test', user=self.other_user, contentpath='location')
        self.reply_1 = CommentReply.objects.create(comment=self.comment, text='reply_1', user=self.other_user)
        self.reply_2 = CommentReply.objects.create(comment=self.comment, text='reply_2', user=self.commenting_user)

    def test_comments_toggle_enabled(self):
        if False:
            return 10
        '\n        Test that the comments toggle is enabled for a TabbedInterface containing CommentPanel, and disabled otherwise\n        '
        form_class = self.tabbed_interface.get_form_class()
        form = form_class()
        self.assertTrue(form.show_comments_toggle)
        tabbed_interface_without_content_panel = TabbedInterface([ObjectList(self.event_page.content_panels)]).bind_to_model(EventPage)
        form_class = tabbed_interface_without_content_panel.get_form_class()
        form = form_class()
        self.assertFalse(form.show_comments_toggle)

    @override_settings(WAGTAILADMIN_COMMENTS_ENABLED=False)
    def test_comments_disabled_setting(self):
        if False:
            return 10
        '\n        Test that the comment panel is missing if WAGTAILADMIN_COMMENTS_ENABLED=False\n        '
        self.assertFalse(any((isinstance(panel, CommentPanel) for panel in Page.settings_panels)))
        form_class = Page.get_edit_handler().get_form_class()
        form = form_class()
        self.assertFalse(form.show_comments_toggle)

    def test_comments_enabled_setting(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the comment panel is present by default\n        '
        self.assertTrue(any((isinstance(panel, CommentPanel) for panel in Page.settings_panels)))
        form_class = Page.get_edit_handler().get_form_class()
        form = form_class()
        self.assertTrue(form.show_comments_toggle)

    def test_context(self):
        if False:
            return 10
        '\n        Test that the context contains the data about existing comments necessary to initialize the commenting app\n        '
        form = self.EventPageForm(instance=self.event_page)
        panel = self.object_list.get_bound_panel(request=self.request, instance=self.event_page, form=form).children[0]
        data = panel.get_context_data()['comments_data']
        self.assertEqual(data['user'], self.commenting_user.pk)
        self.assertEqual(len(data['comments']), 1)
        self.assertEqual(data['comments'][0]['user'], self.comment.user.pk)
        self.assertEqual(len(data['comments'][0]['replies']), 2)
        self.assertEqual(data['comments'][0]['replies'][0]['user'], self.reply_1.user.pk)
        self.assertEqual(data['comments'][0]['replies'][1]['user'], self.reply_2.user.pk)
        self.assertIn(str(self.commenting_user.pk), data['authors'])
        self.assertIn(str(self.other_user.pk), data['authors'])
        try:
            json_script(data, 'comments-data')
        except TypeError:
            self.fail('Failed to serialize comments data. This is likely due to a custom user model using an unsupported field.')

    def test_form(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the form has the comments/replies formsets, and that the\n        user has been set on each CommentForm/CommentReplyForm instance\n        '
        form = self.EventPageForm(instance=self.event_page, for_user=self.commenting_user)
        self.assertIn('comments', form.formsets)
        comments_formset = form.formsets['comments']
        self.assertEqual(len(comments_formset.forms), 1)
        self.assertEqual(comments_formset.forms[0].for_user, self.commenting_user)
        replies_formset = comments_formset.forms[0].formsets['replies']
        self.assertEqual(len(replies_formset.forms), 2)
        self.assertEqual(replies_formset.forms[0].for_user, self.commenting_user)

    def test_comment_form_validation(self):
        if False:
            i = 10
            return i + 15
        form = self.EventPageForm({'comments-TOTAL_FORMS': 2, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': 'edited text', 'comments-0-contentpath': self.comment.contentpath, 'comments-0-replies-TOTAL_FORMS': 0, 'comments-0-replies-INITIAL_FORMS': 0, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000, 'comments-1-id': '', 'comments-1-text': 'new comment', 'comments-1-contentpath': 'new.path', 'comments-1-replies-TOTAL_FORMS': 0, 'comments-1-replies-INITIAL_FORMS': 0, 'comments-1-replies-MIN_NUM_FORMS': 0, 'comments-1-replies-MAX_NUM_FORMS': 1000}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        self.assertFalse(comment_form.is_valid())
        comment_form = form.formsets['comments'].forms[1]
        self.assertTrue(comment_form.is_valid())
        self.assertEqual(comment_form.instance.user, self.commenting_user)
        form = self.EventPageForm({'comments-TOTAL_FORMS': 1, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': self.comment.text, 'comments-0-contentpath': self.comment.contentpath, 'comments-0-DELETE': 1, 'comments-0-replies-TOTAL_FORMS': 0, 'comments-0-replies-INITIAL_FORMS': 0, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        self.assertFalse(comment_form.is_valid())

    def test_users_can_edit_comment_positions(self):
        if False:
            print('Hello World!')
        form = self.EventPageForm({'comments-TOTAL_FORMS': 1, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': self.comment.text, 'comments-0-contentpath': self.comment.contentpath, 'comments-0-position': 'a_new_position', 'comments-0-DELETE': 0, 'comments-0-replies-TOTAL_FORMS': 0, 'comments-0-replies-INITIAL_FORMS': 0, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        self.assertTrue(comment_form.is_valid())

    @freeze_time('2017-01-01 12:00:00')
    def test_comment_resolve(self):
        if False:
            print('Hello World!')
        form = self.EventPageForm({'comments-TOTAL_FORMS': 1, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': self.comment.text, 'comments-0-contentpath': self.comment.contentpath, 'comments-0-resolved': 1, 'comments-0-replies-TOTAL_FORMS': 0, 'comments-0-replies-INITIAL_FORMS': 0, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        self.assertTrue(comment_form.is_valid())
        comment_form.save()
        resolved_comment = Comment.objects.get(pk=self.comment.pk)
        self.assertEqual(resolved_comment.resolved_by, self.commenting_user)
        if settings.USE_TZ:
            self.assertEqual(resolved_comment.resolved_at, datetime(2017, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        else:
            self.assertEqual(resolved_comment.resolved_at, datetime(2017, 1, 1, 12, 0, 0))

    def test_comment_reply_form_validation(self):
        if False:
            return 10
        form = self.EventPageForm({'comments-TOTAL_FORMS': 1, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': self.comment.text, 'comments-0-contentpath': self.comment.contentpath, 'comments-0-replies-TOTAL_FORMS': 3, 'comments-0-replies-INITIAL_FORMS': 2, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000, 'comments-0-replies-0-id': self.reply_1.pk, 'comments-0-replies-0-text': 'edited_text', 'comments-0-replies-1-id': self.reply_2.pk, 'comments-0-replies-1-text': 'Edited text 2', 'comments-0-replies-2-id': '', 'comments-0-replies-2-text': 'New reply'}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        reply_forms = comment_form.formsets['replies'].forms
        self.assertFalse(reply_forms[0].is_valid())
        self.assertTrue(reply_forms[1].is_valid())
        self.assertTrue(reply_forms[2].is_valid())
        self.assertEqual(reply_forms[2].instance.user, self.commenting_user)
        form = self.EventPageForm({'comments-TOTAL_FORMS': 1, 'comments-INITIAL_FORMS': 1, 'comments-MIN_NUM_FORMS': 0, 'comments-MAX_NUM_FORMS': 1000, 'comments-0-id': self.comment.pk, 'comments-0-text': self.comment.text, 'comments-0-contentpath': self.comment.contentpath, 'comments-0-replies-TOTAL_FORMS': 2, 'comments-0-replies-INITIAL_FORMS': 2, 'comments-0-replies-MIN_NUM_FORMS': 0, 'comments-0-replies-MAX_NUM_FORMS': 1000, 'comments-0-replies-0-id': self.reply_1.pk, 'comments-0-replies-0-text': self.reply_1.text, 'comments-0-replies-0-DELETE': 1, 'comments-0-replies-1-id': self.reply_2.pk, 'comments-0-replies-1-text': 'Edited text 2', 'comments-0-replies-1-DELETE': 1}, instance=self.event_page, for_user=self.commenting_user)
        comment_form = form.formsets['comments'].forms[0]
        reply_forms = comment_form.formsets['replies'].forms
        self.assertFalse(reply_forms[0].is_valid())
        self.assertTrue(reply_forms[1].is_valid())

class TestPublishingPanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()
        unbound_object_list = ObjectList([PublishingPanel()])
        self.object_list = unbound_object_list.bind_to_model(EventPage)
        self.tabbed_interface = TabbedInterface([unbound_object_list]).bind_to_model(EventPage)
        self.EventPageForm = self.object_list.get_form_class()
        self.event_page = EventPage.objects.get(slug='christmas')

    def test_schedule_publishing_toggle_toggle_shown(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the schedule publishing toggle is shown for a TabbedInterface containing PublishingPanel, and disabled otherwise\n        '
        form_class = self.tabbed_interface.get_form_class()
        form = form_class()
        self.assertTrue(form.show_schedule_publishing_toggle)
        tabbed_interface_without_publishing_panel = TabbedInterface([ObjectList(self.event_page.content_panels)]).bind_to_model(EventPage)
        form_class = tabbed_interface_without_publishing_panel.get_form_class()
        form = form_class()
        self.assertFalse(form.show_schedule_publishing_toggle)

    def test_publishing_panel_shown_by_default(self):
        if False:
            while True:
                i = 10
        '\n        Test that the publishing panel is present by default\n        '
        self.assertTrue(any((isinstance(panel, PublishingPanel) for panel in Page.settings_panels)))
        form_class = Page.get_edit_handler().get_form_class()
        form = form_class()
        self.assertTrue(form.show_schedule_publishing_toggle)
        expire_at_input = form.fields['expire_at'].widget
        data_controller = expire_at_input.attrs.get('data-controller', None)
        data_action = expire_at_input.attrs.get('data-action', None)
        data_w_dialog_target = expire_at_input.attrs.get('data-w-dialog-target', None)
        self.assertEqual(data_controller, 'w-action')
        self.assertEqual(data_action, 'w-dialog:hidden->w-action#reset')
        self.assertEqual(data_w_dialog_target, 'notify')

    def test_form(self):
        if False:
            return 10
        '\n        Check that the form has the scheduled publishing fields\n        '
        form = self.EventPageForm(instance=self.event_page, for_user=self.user)
        self.assertIn('go_live_at', form.base_fields)
        self.assertIn('expire_at', form.base_fields)

class TestMultipleChooserPanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.root_page = Page.objects.get(id=2)
        self.user = self.login()

    def test_can_render_panel(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_pages:add', args=('tests', 'gallerypage', self.root_page.id)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="gallery_images-TOTAL_FORMS"')
        self.assertContains(response, 'chooserFieldName: "image"')

class TestMultipleChooserPanelGetComparison(TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = RequestFactory().get('/')
        user = AnonymousUser()
        self.request.user = user
        self.page = GalleryPage(title='Test page')
        parent_page = Page.objects.get(id=2)
        parent_page.add_child(instance=self.page)

    def test_get_comparison(self):
        if False:
            i = 10
            return i + 15
        comparison = self.page.get_edit_handler().get_bound_panel(instance=self.page, request=self.request).get_comparison()
        comparison = [comp(self.page, self.page) for comp in comparison]
        field_labels = [comp.field_label() for comp in comparison]
        self.assertIn('Gallery images', field_labels)

class TestPanelIcons(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()
        self.request = get_dummy_request()
        self.request.user = self.user

    def test_default_fieldpanel_icon(self):
        if False:
            print('Hello World!')
        cases = [(FieldPanel('signup_link'), 'link-external', 'link-external', 1), (FieldPanel('audience'), None, 'placeholder', 1), (FieldPanel('body'), 'pilcrow', 'pilcrow', 1), (FieldPanel('feed_image'), 'image', 'image', 2)]
        edit_handler = ObjectList([panel for (panel, *_) in cases])
        edit_handler = edit_handler.bind_to_model(EventPage)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class())
        html = bound_edit_handler.render_form_content()
        for (i, (_, expected_icon, rendered_default, default_count)) in enumerate(cases):
            bound_panel = bound_edit_handler.children[i]
            panel = bound_panel.panel
            field_type = type(panel.db_field).__name__
            with self.subTest(field_type=field_type, field_name=panel.field_name):
                self.assertEqual(bound_panel.icon, expected_icon)
                self.assertEqual(html.count(f'#icon-{rendered_default}'), default_count)

    def test_override_fieldpanel_icon(self):
        if False:
            print('Hello World!')
        cases = [(FieldPanel('signup_link', icon='cog'), 'cog', 'link-external', 0), (FieldPanel('audience', icon='check'), 'check', 'placeholder', 0), (FieldPanel('body', icon='cut'), 'cut', 'pilcrow', 0), (FieldPanel('feed_image', icon='snippet'), 'snippet', 'image', 1)]
        edit_handler = ObjectList([panel for (panel, *_) in cases])
        edit_handler = edit_handler.bind_to_model(EventPage)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class())
        html = bound_edit_handler.render_form_content()
        for (i, (_, expected_icon, rendered_default, default_count)) in enumerate(cases):
            bound_panel = bound_edit_handler.children[i]
            panel = bound_panel.panel
            field_type = type(panel.db_field).__name__
            with self.subTest(field_type=field_type, field_name=panel.field_name):
                self.assertEqual(bound_panel.icon, expected_icon)
                self.assertIn(f'#icon-{expected_icon}', html)
                self.assertEqual(html.count(f'#icon-{rendered_default}'), default_count)

    def test_override_panelgroup_icon(self):
        if False:
            return 10
        cases = [(MultiFieldPanel((FieldPanel('date_from'), FieldPanel('date_to')), heading='Dateys', icon='calendar-alt'), 'calendar-alt'), (FieldRowPanel((FieldPanel('time_from'), FieldPanel('time_to')), heading='Timeys', icon='history'), 'history')]
        edit_handler = ObjectList([panel for (panel, *_) in cases])
        edit_handler = edit_handler.bind_to_model(EventPage)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class())
        html = bound_edit_handler.render_form_content()
        for (i, (panel, expected_icon)) in enumerate(cases):
            bound_panel = bound_edit_handler.children[i]
            with self.subTest(panel_type=type(panel)):
                self.assertEqual(bound_panel.icon, expected_icon)
                self.assertIn(f'#icon-{expected_icon}', html)

    def test_override_inlinepanel_icon(self):
        if False:
            for i in range(10):
                print('nop')
        cases = [(InlinePanel('carousel_items', label='Carousey', icon='cogs'), 'cogs'), (MultipleChooserPanel('related_links', label='Linky', chooser_field_name='link_page', icon='pick'), 'pick')]
        edit_handler = ObjectList([panel for (panel, *_) in cases])
        edit_handler = edit_handler.bind_to_model(EventPage)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class())
        html = bound_edit_handler.render_form_content()
        for (i, (panel, expected_icon)) in enumerate(cases):
            bound_panel = bound_edit_handler.children[i]
            with self.subTest(panel_type=type(panel)):
                self.assertEqual(bound_panel.icon, expected_icon)
                self.assertIn(f'#icon-{expected_icon}', html)

    def test_override_misc_panel_icon(self):
        if False:
            return 10
        root_page = Page.objects.get(id=2)
        form_page = FormPageWithRedirect(title='Contact us', slug='contact-us', to_address='to@email.com', from_address='from@email.com', subject='The subject')
        form_page = root_page.add_child(instance=form_page)
        FormSubmission.objects.create(form_data={}, page=form_page)
        cases = [(PageChooserPanel('thank_you_redirect_page', icon='reset'), 'reset'), (FormSubmissionsPanel(icon='thumbtack'), 'thumbtack')]
        edit_handler = ObjectList([panel for (panel, *_) in cases])
        edit_handler = edit_handler.bind_to_model(FormPageWithRedirect)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class(), instance=form_page)
        html = bound_edit_handler.render_form_content()
        for (i, (panel, expected_icon)) in enumerate(cases):
            bound_panel = bound_edit_handler.children[i]
            with self.subTest(panel_type=type(panel)):
                self.assertEqual(bound_panel.icon, expected_icon)
                self.assertIn(f'#icon-{expected_icon}', html)

class TestTitleFieldPanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.login()
        self.request = get_dummy_request()
        self.request.user = self.user

    def get_edit_handler_html(self, edit_handler, model=EventPage, instance=None):
        if False:
            i = 10
            return i + 15
        edit_handler = edit_handler.bind_to_model(model)
        form_class = edit_handler.get_form_class()
        bound_edit_handler = edit_handler.get_bound_panel(request=self.request, form=form_class(), instance=instance)
        html = bound_edit_handler.render_form_content()
        return self.get_soup(html)

    @clear_edit_handler(Page)
    def test_default_page_content_panels_uses_title_field(self):
        if False:
            for i in range(10):
                print('nop')
        edit_handler = Page.get_edit_handler()
        first_inner_panel_child = edit_handler.children[0].children[0]
        self.assertTrue(isinstance(first_inner_panel_child, TitleFieldPanel))

    def test_default_title_field_panel(self):
        if False:
            print('Hello World!')
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title'), FieldPanel('slug')]))
        self.assertIsNotNone(html.find(attrs={'class': 'w-panel title'}))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['placeholder'], 'Page title*')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_slug')
        self.assertEqual(attrs['data-action'], 'focus->w-sync#check blur->w-sync#apply change->w-sync#apply keyup->w-sync#apply')

    def test_not_using_apply_actions_if_live(self):
        if False:
            while True:
                i = 10
        '\n        If the Page (or any model) has `live = True`, do not apply the actions by default.\n        Allow this to be overridden though.\n        '
        event_live = EventPage.objects.get(slug='christmas')
        self.assertEqual(event_live.live, True)
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title'), FieldPanel('slug')]), instance=event_live)
        self.assertIsNone(html.find('input').attrs.get('data-action'))
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', apply_if_live=True), FieldPanel('slug')]), instance=event_live)
        self.assertIsNotNone(html.find('input').attrs.get('data-action'))

    def test_using_apply_actions_if_non_page_model(self):
        if False:
            for i in range(10):
                print('nop')
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('text', targets=['url']), FieldPanel('url')]), model=Advert)
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_url')
        self.assertIsNotNone(attrs['data-action'])

    def test_using_apply_actions_if_non_page_model_with_live_property(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check for instance being live should be agnostic to how that is implemented.\n        '
        advert_live = Advert(text='Free sheepdog', url='https://example.com', id=5000)
        advert_live.live = True
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('text', targets=['url']), FieldPanel('url')]), model=Advert, instance=advert_live)
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_url')
        self.assertIsNone(attrs.get('data-action'))
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('text', targets=['url'], apply_if_live=True), FieldPanel('url')]), model=Advert, instance=advert_live)
        attrs = html.find('input').attrs
        self.assertIsNotNone(attrs.get('data-action'))

    def test_targets_override_with_empty(self):
        if False:
            for i in range(10):
                print('nop')
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', targets=[]), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-w-sync-target-value'], '')

    def test_targets_override_with_non_slug_field(self):
        if False:
            return 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('location', targets=['title']), FieldPanel('title')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_title')

    def test_targets_override_with_multiple_fields(self):
        if False:
            while True:
                i = 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', targets=['cost', 'location']), FieldPanel('cost'), FieldPanel('location')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_cost, #id_location')

    def test_classname_override(self):
        if False:
            while True:
                i = 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', classname='super-title'), FieldPanel('slug')]))
        self.assertIsNone(html.find(attrs={'class': 'w-panel title'}))
        self.assertIsNotNone(html.find(attrs={'class': 'w-panel super-title'}))

    def test_merging_data_attrs(self):
        if False:
            i = 10
            return i + 15
        widget = forms.TextInput(attrs={'data-controller': 'w-clean', 'data-action': 'w-clean#clean blur->w-clean#clean', 'data-w-clean-filters-value': 'trim upper', 'data-w-sync-target-value': '.will-be-ignored'})
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', widget=widget), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['data-controller'], 'w-clean w-sync')
        self.assertEqual(attrs['data-action'], ' '.join(['w-clean#clean blur->w-clean#clean', 'focus->w-sync#check blur->w-sync#apply change->w-sync#apply keyup->w-sync#apply']))
        self.assertEqual(attrs['data-w-sync-target-value'], '#id_slug')
        self.assertEqual(attrs['data-w-clean-filters-value'], 'trim upper')

    def test_placeholder_override_false(self):
        if False:
            print('Hello World!')
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', placeholder=False), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertNotIn('placeholder', attrs)

    def test_placeholder_override_none(self):
        if False:
            i = 10
            return i + 15
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', placeholder=None), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertNotIn('placeholder', attrs)

    def test_placeholder_override_empty_string(self):
        if False:
            while True:
                i = 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', placeholder=''), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertNotIn('placeholder', attrs)

    def test_placeholder_override_via_widget(self):
        if False:
            for i in range(10):
                print('nop')
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', widget=forms.TextInput(attrs={'placeholder': 'My custom placeholder'})), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['placeholder'], 'My custom placeholder')

    def test_placeholder_override_via_widget_over_kwarg(self):
        if False:
            while True:
                i = 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', placeholder='PANEL placeholder', widget=forms.TextInput(attrs={'placeholder': 'WIDGET placeholder'})), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['placeholder'], 'WIDGET placeholder')

    def test_placeholder_override_via_widget_over_false_kwarg(self):
        if False:
            return 10
        html = self.get_edit_handler_html(ObjectList([TitleFieldPanel('title', placeholder=False, widget=forms.TextInput(attrs={'placeholder': 'WIDGET placeholder'})), FieldPanel('slug')]))
        attrs = html.find('input').attrs
        self.assertEqual(attrs['name'], 'title')
        self.assertEqual(attrs['data-controller'], 'w-sync')
        self.assertEqual(attrs['placeholder'], 'WIDGET placeholder')