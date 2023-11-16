from django.db import connection
from django.db.models import IntegerField, Value
from django.db.models.functions import Length, Lower, Right
from django.test import TestCase
from ..models import Author

class RightTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        Author.objects.create(name='John Smith', alias='smithj')
        Author.objects.create(name='Rhonda')

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        authors = Author.objects.annotate(name_part=Right('name', 5))
        self.assertQuerySetEqual(authors.order_by('name'), ['Smith', 'honda'], lambda a: a.name_part)
        Author.objects.filter(alias__isnull=True).update(alias=Lower(Right('name', 2)))
        self.assertQuerySetEqual(authors.order_by('name'), ['smithj', 'da'], lambda a: a.alias)

    def test_invalid_length(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesMessage(ValueError, "'length' must be greater than 0"):
            Author.objects.annotate(raises=Right('name', 0))

    def test_zero_length(self):
        if False:
            print('Hello World!')
        Author.objects.create(name='Tom', alias='tom')
        authors = Author.objects.annotate(name_part=Right('name', Length('name') - Length('alias')))
        self.assertQuerySetEqual(authors.order_by('name'), ['mith', '' if connection.features.interprets_empty_strings_as_nulls else None, ''], lambda a: a.name_part)

    def test_expressions(self):
        if False:
            i = 10
            return i + 15
        authors = Author.objects.annotate(name_part=Right('name', Value(3, output_field=IntegerField())))
        self.assertQuerySetEqual(authors.order_by('name'), ['ith', 'nda'], lambda a: a.name_part)