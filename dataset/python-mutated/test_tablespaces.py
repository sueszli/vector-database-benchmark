from django.apps import apps
from django.conf import settings
from django.db import connection
from django.test import TransactionTestCase, skipIfDBFeature, skipUnlessDBFeature
from .models.tablespaces import Article, ArticleRef, Authors, Reviewers, Scientist, ScientistRef

def sql_for_table(model):
    if False:
        return 10
    with connection.schema_editor(collect_sql=True) as editor:
        editor.create_model(model)
    return editor.collected_sql[0]

def sql_for_index(model):
    if False:
        while True:
            i = 10
    return '\n'.join((str(sql) for sql in connection.schema_editor()._model_indexes_sql(model)))

class TablespacesTests(TransactionTestCase):
    available_apps = ['model_options']

    def setUp(self):
        if False:
            print('Hello World!')
        self._old_models = apps.app_configs['model_options'].models.copy()
        for model in (Article, Authors, Reviewers, Scientist):
            model._meta.managed = True

    def tearDown(self):
        if False:
            print('Hello World!')
        for model in (Article, Authors, Reviewers, Scientist):
            model._meta.managed = False
        apps.app_configs['model_options'].models = self._old_models
        apps.all_models['model_options'] = self._old_models
        apps.clear_cache()

    def assertNumContains(self, haystack, needle, count):
        if False:
            for i in range(10):
                print('nop')
        real_count = haystack.count(needle)
        self.assertEqual(real_count, count, "Found %d instances of '%s', expected %d" % (real_count, needle, count))

    @skipUnlessDBFeature('supports_tablespaces')
    def test_tablespace_for_model(self):
        if False:
            i = 10
            return i + 15
        sql = sql_for_table(Scientist).lower()
        if settings.DEFAULT_INDEX_TABLESPACE:
            self.assertNumContains(sql, 'tbl_tbsp', 1)
            self.assertNumContains(sql, settings.DEFAULT_INDEX_TABLESPACE, 1)
        else:
            self.assertNumContains(sql, 'tbl_tbsp', 2)

    @skipIfDBFeature('supports_tablespaces')
    def test_tablespace_ignored_for_model(self):
        if False:
            while True:
                i = 10
        self.assertEqual(sql_for_table(Scientist), sql_for_table(ScientistRef))

    @skipUnlessDBFeature('supports_tablespaces')
    def test_tablespace_for_indexed_field(self):
        if False:
            while True:
                i = 10
        sql = sql_for_table(Article).lower()
        if settings.DEFAULT_INDEX_TABLESPACE:
            self.assertNumContains(sql, 'tbl_tbsp', 1)
            self.assertNumContains(sql, settings.DEFAULT_INDEX_TABLESPACE, 2)
        else:
            self.assertNumContains(sql, 'tbl_tbsp', 3)
        self.assertNumContains(sql, 'idx_tbsp', 1)

    @skipIfDBFeature('supports_tablespaces')
    def test_tablespace_ignored_for_indexed_field(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(sql_for_table(Article), sql_for_table(ArticleRef))

    @skipUnlessDBFeature('supports_tablespaces')
    def test_tablespace_for_many_to_many_field(self):
        if False:
            for i in range(10):
                print('nop')
        sql = sql_for_table(Authors).lower()
        if settings.DEFAULT_INDEX_TABLESPACE:
            self.assertNumContains(sql, 'tbl_tbsp', 1)
            self.assertNumContains(sql, settings.DEFAULT_INDEX_TABLESPACE, 1)
        else:
            self.assertNumContains(sql, 'tbl_tbsp', 2)
        self.assertNumContains(sql, 'idx_tbsp', 0)
        sql = sql_for_index(Authors).lower()
        if settings.DEFAULT_INDEX_TABLESPACE:
            self.assertNumContains(sql, settings.DEFAULT_INDEX_TABLESPACE, 2)
        else:
            self.assertNumContains(sql, 'tbl_tbsp', 2)
        self.assertNumContains(sql, 'idx_tbsp', 0)
        sql = sql_for_table(Reviewers).lower()
        if settings.DEFAULT_INDEX_TABLESPACE:
            self.assertNumContains(sql, 'tbl_tbsp', 1)
            self.assertNumContains(sql, settings.DEFAULT_INDEX_TABLESPACE, 1)
        else:
            self.assertNumContains(sql, 'tbl_tbsp', 2)
        self.assertNumContains(sql, 'idx_tbsp', 0)
        sql = sql_for_index(Reviewers).lower()
        self.assertNumContains(sql, 'tbl_tbsp', 0)
        self.assertNumContains(sql, 'idx_tbsp', 2)