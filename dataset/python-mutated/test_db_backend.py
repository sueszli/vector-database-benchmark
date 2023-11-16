import unittest
from django.test import TestCase
from django.test.utils import override_settings
from .test_backends import BackendTests

@override_settings(WAGTAILSEARCH_BACKENDS={'default': {'BACKEND': 'wagtail.search.backends.database.fallback'}})
class TestDBBackend(BackendTests, TestCase):
    backend_path = 'wagtail.search.backends.database.fallback'

    @unittest.expectedFailure
    def test_ranking(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_ranking()

    @unittest.expectedFailure
    def test_annotate_score(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_annotate_score()

    @unittest.expectedFailure
    def test_annotate_score_with_slice(self):
        if False:
            while True:
                i = 10
        super().test_annotate_score_with_slice()

    @unittest.expectedFailure
    def test_search_boosting_on_related_fields(self):
        if False:
            print('Hello World!')
        super().test_search_boosting_on_related_fields()

    @unittest.expectedFailure
    def test_search_child_class_field_from_parent(self):
        if False:
            return 10
        super().test_search_child_class_field_from_parent()

    @unittest.expectedFailure
    def test_search_on_related_fields(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_search_on_related_fields()

    @unittest.expectedFailure
    def test_search_callable_field(self):
        if False:
            return 10
        super().test_search_callable_field()

    @unittest.expectedFailure
    def test_incomplete_plain_text(self):
        if False:
            i = 10
            return i + 15
        super().test_incomplete_plain_text()

    @unittest.expectedFailure
    def test_boost(self):
        if False:
            i = 10
            return i + 15
        super().test_boost()