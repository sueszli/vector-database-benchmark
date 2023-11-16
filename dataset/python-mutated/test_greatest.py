from datetime import datetime, timedelta
from decimal import Decimal
from unittest import skipUnless
from django.db import connection
from django.db.models.expressions import RawSQL
from django.db.models.functions import Coalesce, Greatest
from django.test import TestCase, skipIfDBFeature, skipUnlessDBFeature
from django.utils import timezone
from ..models import Article, Author, DecimalModel, Fan

class GreatestTests(TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        now = timezone.now()
        before = now - timedelta(hours=1)
        Article.objects.create(title='Testing with Django', written=before, published=now)
        articles = Article.objects.annotate(last_updated=Greatest('written', 'published'))
        self.assertEqual(articles.first().last_updated, now)

    @skipUnlessDBFeature('greatest_least_ignores_nulls')
    def test_ignores_null(self):
        if False:
            i = 10
            return i + 15
        now = timezone.now()
        Article.objects.create(title='Testing with Django', written=now)
        articles = Article.objects.annotate(last_updated=Greatest('written', 'published'))
        self.assertEqual(articles.first().last_updated, now)

    @skipIfDBFeature('greatest_least_ignores_nulls')
    def test_propagates_null(self):
        if False:
            print('Hello World!')
        Article.objects.create(title='Testing with Django', written=timezone.now())
        articles = Article.objects.annotate(last_updated=Greatest('written', 'published'))
        self.assertIsNone(articles.first().last_updated)

    def test_coalesce_workaround(self):
        if False:
            return 10
        past = datetime(1900, 1, 1)
        now = timezone.now()
        Article.objects.create(title='Testing with Django', written=now)
        articles = Article.objects.annotate(last_updated=Greatest(Coalesce('written', past), Coalesce('published', past)))
        self.assertEqual(articles.first().last_updated, now)

    @skipUnless(connection.vendor == 'mysql', 'MySQL-specific workaround')
    def test_coalesce_workaround_mysql(self):
        if False:
            print('Hello World!')
        past = datetime(1900, 1, 1)
        now = timezone.now()
        Article.objects.create(title='Testing with Django', written=now)
        past_sql = RawSQL('cast(%s as datetime)', (past,))
        articles = Article.objects.annotate(last_updated=Greatest(Coalesce('written', past_sql), Coalesce('published', past_sql)))
        self.assertEqual(articles.first().last_updated, now)

    def test_all_null(self):
        if False:
            i = 10
            return i + 15
        Article.objects.create(title='Testing with Django', written=timezone.now())
        articles = Article.objects.annotate(last_updated=Greatest('published', 'updated'))
        self.assertIsNone(articles.first().last_updated)

    def test_one_expressions(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesMessage(ValueError, 'Greatest must take at least two expressions'):
            Greatest('written')

    def test_related_field(self):
        if False:
            while True:
                i = 10
        author = Author.objects.create(name='John Smith', age=45)
        Fan.objects.create(name='Margaret', age=50, author=author)
        authors = Author.objects.annotate(highest_age=Greatest('age', 'fans__age'))
        self.assertEqual(authors.first().highest_age, 50)

    def test_update(self):
        if False:
            return 10
        author = Author.objects.create(name='James Smith', goes_by='Jim')
        Author.objects.update(alias=Greatest('name', 'goes_by'))
        author.refresh_from_db()
        self.assertEqual(author.alias, 'Jim')

    def test_decimal_filter(self):
        if False:
            print('Hello World!')
        obj = DecimalModel.objects.create(n1=Decimal('1.1'), n2=Decimal('1.2'))
        self.assertCountEqual(DecimalModel.objects.annotate(greatest=Greatest('n1', 'n2')).filter(greatest=Decimal('1.2')), [obj])