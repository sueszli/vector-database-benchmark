import pickle
from django.test import TestCase, override_settings
from wagtail.models import Site
from wagtail.test.testapp.models import ImportantPagesSiteSetting, TestSiteSetting
from .base import SiteSettingsTestMixin

@override_settings(ALLOWED_HOSTS=['localhost', 'other'])
class SettingModelTestCase(SiteSettingsTestMixin, TestCase):

    def test_for_site_returns_expected_settings(self):
        if False:
            print('Hello World!')
        for (site, expected_settings) in ((self.default_site, self.default_settings), (self.other_site, self.other_settings)):
            with self.subTest(site=site):
                self.assertEqual(TestSiteSetting.for_site(site), expected_settings)

    def test_for_request_returns_expected_settings(self):
        if False:
            while True:
                i = 10
        default_site_request = self.get_request()
        other_site_request = self.get_request(site=self.other_site)
        for (request, expected_settings) in ((default_site_request, self.default_settings), (other_site_request, self.other_settings)):
            with self.subTest(request=request):
                self.assertEqual(TestSiteSetting.for_request(request), expected_settings)

    def test_for_request_result_caching(self):
        if False:
            while True:
                i = 10
        for (i, request) in enumerate([self.get_request(), self.get_request()], 1):
            with self.subTest(attempt=i):
                Site.find_for_request(request)
                with self.assertNumQueries(1):
                    for i in range(4):
                        TestSiteSetting.for_request(request)

    def test_pickle_after_lookup_via_for_request(self):
        if False:
            while True:
                i = 10
        request = self.get_request()
        settings = TestSiteSetting.for_request(request)
        pickled = pickle.dumps(settings)
        unpickled = pickle.loads(pickled)
        self.assertEqual(unpickled.title, 'Site title')

    def _create_importantpagessitesetting_object(self):
        if False:
            for i in range(10):
                print('nop')
        site = self.default_site
        return ImportantPagesSiteSetting.objects.create(site=site, sign_up_page=site.root_page, general_terms_page=site.root_page, privacy_policy_page=self.other_site.root_page)

    def test_importantpages_object_is_pickleable(self):
        if False:
            while True:
                i = 10
        obj = self._create_importantpagessitesetting_object()
        signup_page_url = obj.page_url.sign_up_page
        try:
            pickled = pickle.dumps(obj, -1)
        except Exception as e:
            raise AssertionError(f'An error occured when attempting to pickle {obj!r}: {e}')
        try:
            unpickled = pickle.loads(pickled)
        except Exception as e:
            raise AssertionError(f'An error occured when attempting to unpickle {obj!r}: {e}')
        self.assertEqual(unpickled.page_url.sign_up_page, signup_page_url)

    def test_select_related(self, expected_queries=4):
        if False:
            print('Hello World!')
        'The `select_related` attribute on setting models is `None` by default, so fetching foreign keys values requires additional queries'
        request = self.get_request()
        self._create_importantpagessitesetting_object()
        Site.find_for_request(request)
        with self.assertNumQueries(expected_queries):
            settings = ImportantPagesSiteSetting.for_request(request)
            settings.sign_up_page
            settings.general_terms_page
            settings.privacy_policy_page

    def test_select_related_use_reduces_total_queries(self):
        if False:
            return 10
        'But, `select_related` can be used to reduce the number of queries needed to fetch foreign keys'
        try:
            ImportantPagesSiteSetting.select_related = ['sign_up_page', 'general_terms_page', 'privacy_policy_page']
            self.test_select_related(expected_queries=1)
        finally:
            ImportantPagesSiteSetting.select_related = None

    def test_get_page_url_when_settings_fetched_via_for_request(self):
        if False:
            print('Hello World!')
        'Using ImportantPagesSiteSetting.for_request() makes the setting\n        object request-aware, improving efficiency and allowing\n        site-relative URLs to be returned'
        self._create_importantpagessitesetting_object()
        request = self.get_request()
        settings = ImportantPagesSiteSetting.for_request(request)
        self.default_site.root_page._get_site_root_paths(request)
        for (page_fk_field, expected_result) in (('sign_up_page', '/'), ('general_terms_page', '/'), ('privacy_policy_page', 'http://other/')):
            with self.subTest(page_fk_field=page_fk_field):
                with self.assertNumQueries(1):
                    self.assertEqual(settings.get_page_url(page_fk_field), expected_result)
                    self.assertEqual(settings.get_page_url(page_fk_field), expected_result)
                    self.assertEqual(getattr(settings.page_url, page_fk_field), expected_result)

    def test_get_page_url_when_for_settings_fetched_via_for_site(self):
        if False:
            for i in range(10):
                print('nop')
        'ImportantPagesSiteSetting.for_site() cannot make the settings object\n        request-aware, so things are a little less efficient, and the\n        URLs returned will not be site-relative'
        self._create_importantpagessitesetting_object()
        settings = ImportantPagesSiteSetting.for_site(self.default_site)
        self.default_site.root_page._get_site_root_paths()
        for (page_fk_field, expected_result) in (('sign_up_page', 'http://localhost/'), ('general_terms_page', 'http://localhost/'), ('privacy_policy_page', 'http://other/')):
            with self.subTest(page_fk_field=page_fk_field):
                with self.assertNumQueries(2):
                    self.assertEqual(settings.get_page_url(page_fk_field), expected_result)
                    self.assertEqual(settings.get_page_url(page_fk_field), expected_result)
                    self.assertEqual(getattr(settings.page_url, page_fk_field), expected_result)

    def test_get_page_url_raises_attributeerror_if_attribute_name_invalid(self):
        if False:
            while True:
                i = 10
        settings = self._create_importantpagessitesetting_object()
        with self.assertRaises(AttributeError):
            settings.get_page_url('not_an_attribute')
        with self.assertRaises(AttributeError):
            settings.page_url.not_an_attribute

    def test_get_page_url_returns_empty_string_if_attribute_value_not_a_page(self):
        if False:
            return 10
        settings = self._create_importantpagessitesetting_object()
        for value in (None, self.default_site):
            with self.subTest(attribute_value=value):
                settings.test_attribute = value
                self.assertEqual(settings.get_page_url('test_attribute'), '')
                self.assertEqual(settings.page_url.test_attribute, '')