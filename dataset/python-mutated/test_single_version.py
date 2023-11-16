import django_dynamic_fixture as fixture
from django.test import TestCase
from django.test.utils import override_settings
from readthedocs.projects.models import Project

@override_settings(PUBLIC_DOMAIN='public.readthedocs.org', SERVE_PUBLIC_DOCS=True)
class RedirectSingleVersionTests(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.pip = fixture.get(Project, slug='pip', single_version=True, main_language_project=None)

    def test_docs_url_generation(self):
        if False:
            return 10
        self.assertEqual(self.pip.get_docs_url(), 'http://pip.public.readthedocs.org/')
        self.pip.single_version = False
        self.assertEqual(self.pip.get_docs_url(), 'http://pip.public.readthedocs.org/en/latest/')