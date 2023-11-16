from django.db.models import CharField, Value
from django.db.models.functions import Left, Ord
from django.test import TestCase
from django.test.utils import register_lookup
from ..models import Author

class OrdTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        cls.john = Author.objects.create(name='John Smith', alias='smithj')
        cls.elena = Author.objects.create(name='Élena Jordan', alias='elena')
        cls.rhonda = Author.objects.create(name='Rhonda')

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        authors = Author.objects.annotate(name_part=Ord('name'))
        self.assertCountEqual(authors.filter(name_part__gt=Ord(Value('John'))), [self.elena, self.rhonda])
        self.assertCountEqual(authors.exclude(name_part__gt=Ord(Value('John'))), [self.john])

    def test_transform(self):
        if False:
            print('Hello World!')
        with register_lookup(CharField, Ord):
            authors = Author.objects.annotate(first_initial=Left('name', 1))
            self.assertCountEqual(authors.filter(first_initial__ord=ord('J')), [self.john])
            self.assertCountEqual(authors.exclude(first_initial__ord=ord('J')), [self.elena, self.rhonda])