import json
from django.contrib.auth.models import AnonymousUser, Permission
from django.template import Context, Template
from django.test import TestCase
from django.urls import reverse
from wagtail import hooks
from wagtail.admin.userbar import AccessibilityItem
from wagtail.coreutils import get_dummy_request
from wagtail.models import PAGE_TEMPLATE_VAR, Page, Site
from wagtail.test.testapp.models import BusinessChild, BusinessIndex, SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestUserbarTag(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.create_superuser(username='test', email='test@email.com', password='password')
        self.homepage = Page.objects.get(id=2)

    def dummy_request(self, user=None, *, is_preview=False, in_preview_panel=False, revision_id=None, is_editing=False):
        if False:
            for i in range(10):
                print('nop')
        request = get_dummy_request()
        request.user = user or AnonymousUser()
        request.is_preview = is_preview
        request.is_editing = is_editing
        request.in_preview_panel = in_preview_panel
        if revision_id:
            request.revision_id = revision_id
        return request

    def test_userbar_tag(self):
        if False:
            for i in range(10):
                print('nop')
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        context = Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request(self.user)})
        with self.assertNumQueries(5):
            content = template.render(context)
        self.assertIn('<!-- Wagtail user bar embed code -->', content)

    def test_userbar_does_not_break_without_request(self):
        if False:
            for i in range(10):
                print('nop')
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}boom')
        content = template.render(Context({}))
        self.assertEqual('boom', content)

    def test_userbar_tag_self(self):
        if False:
            while True:
                i = 10
        '\n        Ensure the userbar renders with `self` instead of `PAGE_TEMPLATE_VAR`\n        '
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({'self': self.homepage, 'request': self.dummy_request(self.user)}))
        self.assertIn('<!-- Wagtail user bar embed code -->', content)

    def test_userbar_tag_anonymous_user(self):
        if False:
            print('Hello World!')
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request()}))
        self.assertEqual(content, '')

    def test_userbar_tag_no_page(self):
        if False:
            i = 10
            return i + 15
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({'request': self.dummy_request(self.user)}))
        self.assertIn('<!-- Wagtail user bar embed code -->', content)

    def test_edit_link(self):
        if False:
            for i in range(10):
                print('nop')
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request(self.user, is_preview=False)}))
        self.assertIn('<!-- Wagtail user bar embed code -->', content)
        self.assertIn('Edit this page', content)

    def test_userbar_edit_menu_in_previews(self):
        if False:
            for i in range(10):
                print('nop')
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request(self.user, is_preview=True)}))
        self.assertIn('<!-- Wagtail user bar embed code -->', content)
        self.assertIn('Edit this page', content)
        self.assertIn(reverse('wagtailadmin_pages:edit', args=(self.homepage.id,)), content)

    def test_userbar_edit_menu_not_in_preview(self):
        if False:
            while True:
                i = 10
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request(self.user, is_preview=True, is_editing=True)}))
        self.assertIn('<!-- Wagtail user bar embed code -->', content)
        self.assertNotIn('Edit this page', content)
        self.assertNotIn(reverse('wagtailadmin_pages:edit', args=(self.homepage.id,)), content)

    def test_userbar_not_in_preview_panel(self):
        if False:
            return 10
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({PAGE_TEMPLATE_VAR: self.homepage, 'request': self.dummy_request(self.user, is_preview=True, in_preview_panel=True)}))
        self.assertEqual(content, '')

class TestAccessibilityCheckerConfig(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        self.request = get_dummy_request()
        self.request.user = self.user

    def get_script(self):
        if False:
            i = 10
            return i + 15
        template = Template('{% load wagtailuserbar %}{% wagtailuserbar %}')
        content = template.render(Context({'request': self.request}))
        soup = self.get_soup(content)
        return soup.find('script', id='accessibility-axe-configuration')

    def get_config(self):
        if False:
            return 10
        return json.loads(self.get_script().string)

    def get_hook(self, item_class):
        if False:
            return 10

        def customise_accessibility_checker(request, items):
            if False:
                i = 10
                return i + 15
            items[:] = [item_class() if isinstance(item, AccessibilityItem) else item for item in items]
        return customise_accessibility_checker

    def test_config_json(self):
        if False:
            while True:
                i = 10
        script = self.get_script()
        self.assertIsNotNone(script)
        self.assertEqual(script.attrs['type'], 'application/json')
        config_string = script.string.strip()
        self.assertGreater(len(config_string), 0)
        config = json.loads(config_string)
        self.assertIsInstance(config, dict)
        self.assertGreater(len(config.keys()), 0)

    def test_messages(self):
        if False:
            i = 10
            return i + 15
        config = self.get_config()
        self.assertIsInstance(config.get('messages'), dict)
        self.assertEqual(config['messages']['empty-heading'], 'Empty heading found. Use meaningful text for screen reader users.')

    def test_custom_message(self):
        if False:
            while True:
                i = 10

        class CustomMessageAccessibilityItem(AccessibilityItem):
            axe_messages = {'empty-heading': 'Headings should not be empty!'}

            def get_axe_messages(self, request):
                if False:
                    print('Hello World!')
                return {**super().get_axe_messages(request), 'color-contrast-enhanced': 'Increase colour contrast!'}
        with hooks.register_temporarily('construct_wagtail_userbar', self.get_hook(CustomMessageAccessibilityItem)):
            config = self.get_config()
            self.assertEqual(config['messages'], {'empty-heading': 'Headings should not be empty!', 'color-contrast-enhanced': 'Increase colour contrast!'})

    def test_unset_run_only(self):
        if False:
            print('Hello World!')

        class UnsetRunOnlyAccessibilityItem(AccessibilityItem):
            axe_run_only = None
            axe_rules = {'focus-order-semantics': {'enabled': True}}
        with hooks.register_temporarily('construct_wagtail_userbar', self.get_hook(UnsetRunOnlyAccessibilityItem)):
            config = self.get_config()
            self.assertEqual(config['options'], {'rules': {'focus-order-semantics': {'enabled': True}}})

    def test_custom_context(self):
        if False:
            return 10

        class CustomContextAccessibilityItem(AccessibilityItem):
            axe_include = ['article', 'section']
            axe_exclude = ['.sr-only']

            def get_axe_exclude(self, request):
                if False:
                    return 10
                return [*super().get_axe_exclude(request), '[data-please-ignore]']
        with hooks.register_temporarily('construct_wagtail_userbar', self.get_hook(CustomContextAccessibilityItem)):
            config = self.get_config()
            self.assertEqual(config['context'], {'include': ['article', 'section'], 'exclude': ['.sr-only', {'fromShadowDOM': ['wagtail-userbar']}, '[data-please-ignore]']})

    def test_custom_run_only_and_rules_per_request(self):
        if False:
            i = 10
            return i + 15

        class CustomRunOnlyAccessibilityItem(AccessibilityItem):
            axe_run_only = ['wcag2a', 'wcag2aa', 'wcag2aaa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice']
            axe_rules = {'color-contrast-enhanced': {'enabled': False}}

            def get_axe_rules(self, request):
                if False:
                    i = 10
                    return i + 15
                if request.user.is_superuser:
                    return {}
                return super().get_axe_rules(request)
        with hooks.register_temporarily('construct_wagtail_userbar', self.get_hook(CustomRunOnlyAccessibilityItem)):
            config = self.get_config()
            self.assertEqual(config['options'], {'runOnly': CustomRunOnlyAccessibilityItem.axe_run_only, 'rules': {}})
            self.user.is_superuser = False
            self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
            self.user.save()
            config = self.get_config()
            self.assertEqual(config['options'], {'runOnly': CustomRunOnlyAccessibilityItem.axe_run_only, 'rules': CustomRunOnlyAccessibilityItem.axe_rules})

class TestUserbarInPageServe(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()
        self.request = get_dummy_request(site=Site.objects.first())
        self.request.user = self.user
        self.homepage = Page.objects.get(id=2).specific
        self.page = SimplePage(title='Rendang', content='Enak', live=True)
        self.homepage.add_child(instance=self.page)

    def test_userbar_rendered(self):
        if False:
            return 10
        response = self.page.serve(self.request)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<template id="wagtail-userbar-template">')

    def test_userbar_anonymous_user_cannot_see(self):
        if False:
            while True:
                i = 10
        self.request.user = AnonymousUser()
        response = self.page.serve(self.request)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<template id="wagtail-userbar-template">')

class TestUserbarAddLink(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.login()
        self.request = get_dummy_request(site=Site.objects.first())
        self.request.user = self.user
        self.homepage = Page.objects.get(url_path='/home/')
        self.event_index = Page.objects.get(url_path='/home/events/').specific
        self.business_index = BusinessIndex(title='Business', live=True)
        self.homepage.add_child(instance=self.business_index)
        self.business_child = BusinessChild(title='Business Child', live=True)
        self.business_index.add_child(instance=self.business_child)

    def test_page_allowing_subpages(self):
        if False:
            i = 10
            return i + 15
        response = self.event_index.serve(self.request)
        response.render()
        self.assertEqual(response.status_code, 200)
        expected_url = reverse('wagtailadmin_pages:add_subpage', args=(self.event_index.id,))
        needle = f'\n            <a href="{expected_url}" target="_parent" role="menuitem">\n                <svg class="icon icon-plus w-action-icon" aria-hidden="true">\n                    <use href="#icon-plus"></use>\n                </svg>\n                Add a child page\n            </a>\n            '
        self.assertTagInHTML(needle, response.content.decode())

    def test_page_disallowing_subpages(self):
        if False:
            print('Hello World!')
        response = self.business_child.serve(self.request)
        response.render()
        self.assertEqual(response.status_code, 200)
        expected_url = reverse('wagtailadmin_pages:add_subpage', args=(self.business_index.id,))
        soup = self.get_soup(response.content)
        link = soup.find('a', attrs={'href': expected_url})
        self.assertIsNone(link)