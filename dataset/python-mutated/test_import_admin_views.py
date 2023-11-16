import os
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail.contrib.redirects.models import Redirect
from wagtail.models import Site
from wagtail.test.utils import WagtailTestUtils
TEST_ROOT = os.path.abspath(os.path.dirname(__file__))

@override_settings(ALLOWED_HOSTS=['testserver', 'localhost', 'test.example.com', 'other.example.com'])
class TestImportAdminViews(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.login()

    def get(self, params={}):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailredirects:start_import'), params)

    def post(self, post_data={}, follow=False):
        if False:
            while True:
                i = 10
        return self.client.post(reverse('wagtailredirects:start_import'), post_data, follow=follow)

    def post_import(self, post_data={}, follow=False):
        if False:
            return 10
        return self.client.post(reverse('wagtailredirects:process_import'), post_data, follow=follow)

    def test_request_start_with_get_returns_initial_form(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.templates[0].name, 'wagtailredirects/choose_import_file.html')

    def test_empty_import_file_returns_error(self):
        if False:
            print('Hello World!')
        response = self.post({'import_file': '', 'input_format': '0'})
        self.assertIn('import_file', response.context['form'].errors)

    def test_non_valid_format_returns_error(self):
        if False:
            return 10
        f = f'{TEST_ROOT}/files/example.yaml'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            response = self.post({'import_file': upload_file}, follow=True)
            self.assertContains(response, 'File format of type &quot;yaml&quot; is not supported')

    def test_valid_csv_triggers_confirm_view(self):
        if False:
            while True:
                i = 10
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            response = self.post({'import_file': upload_file})
            self.assertEqual(response.templates[0].name, 'wagtailredirects/confirm_import.html')
            self.assertEqual(len(response.context['dataset']), 3)

    def test_import_step(self):
        if False:
            return 10
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': True})
            self.assertTemplateUsed(import_response, 'wagtailredirects/import_summary.html')
            self.assertEqual(Redirect.objects.all().count(), 2)

    def test_import_step_with_offset_columns(self):
        if False:
            print('Hello World!')
        f = f'{TEST_ROOT}/files/example_offset_columns.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 16, 'to_index': 17, 'permanent': True})
            self.assertTemplateUsed(import_response, 'wagtailredirects/import_summary.html')
            self.assertEqual(Redirect.objects.all().count(), 2)

    def test_permanent_setting(self):
        if False:
            for i in range(10):
                print('nop')
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': False})
            self.assertTemplateUsed(import_response, 'wagtailredirects/import_summary.html')
            self.assertFalse(Redirect.objects.first().is_permanent)

    def test_site_setting(self):
        if False:
            return 10
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        default_site = Site.objects.first()
        new_site = Site.objects.create(hostname='hello.dev', root_page=default_site.root_page)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': False, 'site': new_site.pk})
            self.assertTemplateUsed(import_response, 'wagtailredirects/import_summary.html')
            self.assertEqual(Redirect.objects.count(), 2)
            self.assertEqual(Redirect.objects.first().site, new_site)

    def test_import_xlsx(self):
        if False:
            print('Hello World!')
        f = f'{TEST_ROOT}/files/example.xlsx'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            self.assertTemplateUsed(response, 'wagtailredirects/confirm_import.html')
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': True}, follow=True)
            self.assertTemplateUsed(import_response, 'wagtailredirects/index.html')
            self.assertEqual(Redirect.objects.all().count(), 3)

    def test_unicode_error_when_importing(self):
        if False:
            i = 10
            return i + 15
        f = f'{TEST_ROOT}/files/example_faulty.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file}, follow=True)
            self.assertIn(b'Imported file has a wrong encoding:', response.content)

    def test_not_valid_method_for_import_file(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailredirects:process_import'))
        self.assertEqual(response.status_code, 405)

    def test_error_in_data_renders_confirm_view_on_import(self):
        if False:
            while True:
                i = 10
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            response = self.post({'import_file': upload_file})
            self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': True, 'site': 99})
            self.assertTemplateUsed(response, 'wagtailredirects/confirm_import.html')

    def test_import_tsv(self):
        if False:
            return 10
        f = f'{TEST_ROOT}/files/example.tsv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            self.assertTemplateUsed(response, 'wagtailredirects/confirm_import.html')
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': True})
            self.assertTemplateUsed(import_response, 'wagtailredirects/import_summary.html')
            self.assertEqual(Redirect.objects.all().count(), 2)

    @override_settings(WAGTAIL_REDIRECTS_FILE_STORAGE='cache')
    def test_import_xlsx_with_cache_store_engine(self):
        if False:
            while True:
                i = 10
        f = f'{TEST_ROOT}/files/example.xlsx'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            self.assertTemplateUsed(response, 'wagtailredirects/confirm_import.html')
            import_response = self.post_import({**response.context['form'].initial, 'from_index': 0, 'to_index': 1, 'permanent': True}, follow=True)
            self.assertTemplateUsed(import_response, 'wagtailredirects/index.html')
            self.assertEqual(Redirect.objects.all().count(), 3)

    @override_settings(WAGTAIL_REDIRECTS_FILE_STORAGE='cache')
    def test_process_validation_works_when_using_plaintext_files_and_cache(self):
        if False:
            return 10
        f = f'{TEST_ROOT}/files/example.csv'
        (_, filename) = os.path.split(f)
        with open(f, 'rb') as infile:
            upload_file = SimpleUploadedFile(filename, infile.read())
            self.assertEqual(Redirect.objects.all().count(), 0)
            response = self.post({'import_file': upload_file})
            self.assertTemplateUsed(response, 'wagtailredirects/confirm_import.html')
            import_response = self.post_import({**response.context['form'].initial, 'permanent': True})
            self.assertTemplateUsed(import_response, 'wagtailredirects/confirm_import.html')