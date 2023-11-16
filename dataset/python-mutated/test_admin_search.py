"""
Tests for the search box in the admin side menu, and the custom search hooks.
"""
from django.contrib.auth.models import Permission
from django.template import Context, Template
from django.test import RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse
from wagtail.admin.auth import user_has_any_page_permission
from wagtail.admin.search import SearchArea
from wagtail.test.utils import WagtailTestUtils

class BaseSearchAreaTestCase(WagtailTestUtils, TestCase):
    rf = RequestFactory()

    def search_other(self, current_url='/admin/', data=None):
        if False:
            while True:
                i = 10
        request = self.rf.get(current_url, data=data)
        request.user = self.user
        template = Template('{% load wagtailadmin_tags %}{% search_other %}')
        return template.render(Context({'request': request}))

class TestSearchAreas(BaseSearchAreaTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.user = self.login()

    def test_other_searches(self):
        if False:
            while True:
                i = 10
        search_url = reverse('wagtailadmin_pages:search')
        query = 'Hello'
        base_css = 'search--custom-class'
        icon = '<svg class="icon icon-custom filter-options__icon" aria-hidden="true"><use href="#icon-custom"></use></svg>'
        test_string = '<a href="/customsearch/?q=%s" class="%s" is-custom="true">%sMy Search</a>'
        response = self.client.get(search_url, {'q': query})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/search.html')
        self.assertTemplateUsed(response, 'wagtailadmin/shared/search_area.html')
        self.assertTemplateUsed(response, 'wagtailadmin/shared/search_other.html')
        self.assertContains(response, test_string % (query, base_css, icon), html=True)
        response = self.client.get(search_url, {'q': query, 'hide-option': 'true'})
        self.assertNotContains(response, test_string % (query, base_css, icon), status_code=200, html=True)
        response = self.client.get(search_url, {'q': query, 'active-option': 'true'})
        self.assertContains(response, test_string % (query, base_css + ' nolink', icon), status_code=200, html=True)

    def test_search_other(self):
        if False:
            for i in range(10):
                print('nop')
        rendered = self.search_other()
        self.assertIn(reverse('wagtailadmin_pages:search'), rendered)
        self.assertIn('/customsearch/', rendered)
        self.assertIn('Pages', rendered)
        self.assertIn('My Search', rendered)

class TestSearchAreaNoPagePermissions(BaseSearchAreaTestCase):
    """
    Test the admin search when the user does not have permission to manage
    pages. The search bar should show the first available search area instead.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        self.assertFalse(user_has_any_page_permission(self.user))

    def create_test_user(self):
        if False:
            while True:
                i = 10
        user = super().create_test_user()
        user.is_superuser = False
        user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        user.save()
        return user

    def test_dashboard(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the menu search area on the dashboard is not searching\n        pages, as they are not allowed.\n        '
        response = self.client.get('/admin/')
        self.assertNotContains(response, reverse('wagtailadmin_pages:search'))
        self.assertContains(response, '{"_type": "wagtail.sidebar.SearchModule", "_args": ["/customsearch/"]}')

    def test_search_other(self):
        if False:
            while True:
                i = 10
        'The pages search link should be hidden, custom search should be visible.'
        rendered = self.search_other()
        self.assertNotIn(reverse('wagtailadmin_pages:search'), rendered)
        self.assertIn('/customsearch/', rendered)
        self.assertNotIn('Pages', rendered)
        self.assertIn('My Search', rendered)

class SearchAreaComparisonTestCase(SimpleTestCase):
    """Tests the comparison functions."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.search_area1 = SearchArea('Label 1', '/url1', order=100)
        self.search_area2 = SearchArea('Label 2', '/url2', order=200)
        self.search_area3 = SearchArea('Label 1', '/url3', order=300)
        self.search_area4 = SearchArea('Label 1', '/url1', order=100)

    def test_eq(self):
        if False:
            return 10
        self.assertTrue(self.search_area1 == self.search_area4)
        self.assertFalse(self.search_area1 == self.search_area2)
        self.assertFalse(self.search_area1 == 'Something')

    def test_lt(self):
        if False:
            return 10
        self.assertTrue(self.search_area1 < self.search_area2)
        self.assertTrue(self.search_area1 < self.search_area3)
        self.assertFalse(self.search_area2 < self.search_area1)
        with self.assertRaises(TypeError):
            self.search_area1 < 'Something'

    def test_le(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.search_area1 <= self.search_area2)
        self.assertTrue(self.search_area1 <= self.search_area3)
        self.assertTrue(self.search_area1 <= self.search_area1)
        self.assertTrue(self.search_area1 <= self.search_area4)
        self.assertFalse(self.search_area2 <= self.search_area1)
        with self.assertRaises(TypeError):
            self.search_area1 <= 'Something'

    def test_gt(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.search_area2 > self.search_area1)
        self.assertTrue(self.search_area3 > self.search_area1)
        self.assertFalse(self.search_area1 > self.search_area2)
        with self.assertRaises(TypeError):
            self.search_area1 > 'Something'

    def test_ge(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.search_area2 >= self.search_area1)
        self.assertTrue(self.search_area3 >= self.search_area1)
        self.assertTrue(self.search_area1 >= self.search_area1)
        self.assertTrue(self.search_area1 >= self.search_area4)
        self.assertFalse(self.search_area1 >= self.search_area2)
        with self.assertRaises(TypeError):
            self.search_area1 >= 'Something'