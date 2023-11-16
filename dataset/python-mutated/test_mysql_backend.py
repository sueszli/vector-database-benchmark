import unittest
from unittest import skip
from django.db import connection
from django.test.testcases import TransactionTestCase
from django.test.utils import override_settings
from wagtail.search.query import Not, PlainText
from wagtail.search.tests.test_backends import BackendTests
from wagtail.test.search import models

@unittest.skipUnless(connection.vendor == 'mysql', 'The current database is not MySQL')
@override_settings(WAGTAILSEARCH_BACKENDS={'default': {'BACKEND': 'wagtail.search.backends.database.mysql.mysql'}})
class TestMySQLSearchBackend(BackendTests, TransactionTestCase):
    backend_path = 'wagtail.search.backends.database.mysql.mysql'

    def test_not(self):
        if False:
            for i in range(10):
                print('nop')
        all_other_titles = {'A Clash of Kings', 'A Game of Thrones', 'A Storm of Swords', 'Foundation', 'Learning Python', 'The Hobbit', 'The Two Towers', 'The Fellowship of the Ring', 'The Return of the King', 'The Rust Programming Language', 'Two Scoops of Django 1.11', 'Programming Rust'}
        results = self.backend.search(Not(PlainText('javascript')), models.Book.objects.all())
        self.assertSetEqual({r.title for r in results}, all_other_titles)
        results = self.backend.search(~PlainText('javascript'), models.Book.objects.all())
        self.assertSetEqual({r.title for r in results}, all_other_titles)
        results = self.backend.search(~PlainText('javascript the'), models.Book.objects.all())
        self.assertSetEqual({r.title for r in results}, all_other_titles | {'JavaScript: The Definitive Guide', 'JavaScript: The good parts'})
        results = self.backend.search(~PlainText('javascript parts'), models.Book.objects.all())
        self.assertSetEqual({r.title for r in results}, all_other_titles | {'JavaScript: The Definitive Guide'})

    @skip("The MySQL backend doesn't support choosing individual fields for the search, only (body, title) or (autocomplete) fields may be searched.")
    def test_search_on_individual_field(self):
        if False:
            print('Hello World!')
        return super().test_search_on_individual_field()

    @skip("The MySQL backend doesn't support boosting.")
    def test_search_boosting_on_related_fields(self):
        if False:
            i = 10
            return i + 15
        return super().test_search_boosting_on_related_fields()

    @skip("The MySQL backend doesn't support boosting.")
    def test_boost(self):
        if False:
            while True:
                i = 10
        return super().test_boost()

    @skip("The MySQL backend doesn't score annotations.")
    def test_annotate_score(self):
        if False:
            i = 10
            return i + 15
        return super().test_annotate_score()

    @skip("The MySQL backend doesn't score annotations.")
    def test_annotate_score_with_slice(self):
        if False:
            while True:
                i = 10
        return super().test_annotate_score_with_slice()

    @skip("The MySQL backend doesn't guarantee correct ranking of results.")
    def test_ranking(self):
        if False:
            return 10
        return super().test_ranking()