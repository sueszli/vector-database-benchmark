from contextlib import contextmanager
from django.core import checks
from django.test import TestCase
from wagtail.models import Page
from wagtail.search import index
from wagtail.test.search import models
from wagtail.test.testapp.models import TaggedChildPage, TaggedGrandchildPage, TaggedPage

@contextmanager
def patch_search_fields(model, new_search_fields):
    if False:
        for i in range(10):
            print('nop')
    "\n    A context manager to allow testing of different search_fields configurations\n    without permanently changing the models' search_fields.\n    "
    old_search_fields = model.search_fields
    model.search_fields = new_search_fields
    yield
    model.search_fields = old_search_fields

class TestContentTypeNames(TestCase):

    def test_base_content_type_name(self):
        if False:
            print('Hello World!')
        name = models.Novel.indexed_get_toplevel_content_type()
        self.assertEqual(name, 'searchtests_book')

    def test_qualified_content_type_name(self):
        if False:
            return 10
        name = models.Novel.indexed_get_content_type()
        self.assertEqual(name, 'searchtests_book_searchtests_novel')

class TestSearchFields(TestCase):

    def make_dummy_type(self, search_fields):
        if False:
            return 10
        return type('DummyType', (index.Indexed,), {'search_fields': search_fields})

    def get_checks_result(warning_id=None):
        if False:
            for i in range(10):
                print('nop')
        "Run Django checks on any with the 'search' tag used when registering the check"
        checks_result = checks.run_checks()
        if warning_id:
            return [warning for warning in checks_result if warning.id == warning_id]
        return checks_result

    def test_basic(self):
        if False:
            while True:
                i = 10
        cls = self.make_dummy_type([index.SearchField('test', boost=100), index.FilterField('filter_test')])
        self.assertEqual(len(cls.get_search_fields()), 2)
        self.assertEqual(len(cls.get_searchable_search_fields()), 1)
        self.assertEqual(len(cls.get_filterable_search_fields()), 1)

    def test_overriding(self):
        if False:
            print('Hello World!')
        cls = self.make_dummy_type([index.SearchField('test', boost=100), index.SearchField('test')])
        self.assertEqual(len(cls.get_search_fields()), 1)
        self.assertEqual(len(cls.get_searchable_search_fields()), 1)
        self.assertEqual(len(cls.get_filterable_search_fields()), 0)
        field = cls.get_search_fields()[0]
        self.assertIsInstance(field, index.SearchField)
        self.assertIsNone(field.boost)

    def test_different_field_types_dont_override(self):
        if False:
            print('Hello World!')
        cls = self.make_dummy_type([index.SearchField('test', boost=100), index.FilterField('test')])
        self.assertEqual(len(cls.get_search_fields()), 2)
        self.assertEqual(len(cls.get_searchable_search_fields()), 1)
        self.assertEqual(len(cls.get_filterable_search_fields()), 1)

    def test_checking_search_fields(self):
        if False:
            i = 10
            return i + 15
        with patch_search_fields(models.Book, models.Book.search_fields + [index.SearchField('foo')]):
            expected_errors = [checks.Warning("Book.search_fields contains non-existent field 'foo'", obj=models.Book, id='wagtailsearch.W004')]
            errors = models.Book.check()
            self.assertEqual(errors, expected_errors)

    def test_checking_core_page_fields_are_indexed(self):
        if False:
            while True:
                i = 10
        'Run checks to ensure that when core page fields are missing we get a warning'
        errors = [error for error in checks.run_checks() if error.id == 'wagtailsearch.W001']
        self.assertEqual([TaggedPage, TaggedChildPage, TaggedGrandchildPage], [error.obj for error in errors])
        for error in errors:
            self.assertEqual(error.msg, 'Core Page fields missing in `search_fields`')
            self.assertIn('Page model search fields `search_fields = Page.search_fields + [...]`', error.hint)
        with patch_search_fields(TaggedPage, Page.search_fields + TaggedPage.search_fields):
            errors = [error for error in checks.run_checks() if error.id == 'wagtailsearch.W001']
            self.assertEqual([], errors)
        with patch_search_fields(TaggedPage, []):
            errors = [error for error in checks.run_checks() if error.id == 'wagtailsearch.W001']
            self.assertEqual([], errors)