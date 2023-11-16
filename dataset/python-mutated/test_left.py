from django.db.models import IntegerField, Value
from django.db.models.functions import Left, Lower
from django.test import TestCase
from ..models import Author

class LeftTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        Author.objects.create(name='John Smith', alias='smithj')
        Author.objects.create(name='Rhonda')

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        authors = Author.objects.annotate(name_part=Left('name', 5))
        self.assertQuerySetEqual(authors.order_by('name'), ['John ', 'Rhond'], lambda a: a.name_part)
        Author.objects.filter(alias__isnull=True).update(alias=Lower(Left('name', 2)))
        self.assertQuerySetEqual(authors.order_by('name'), ['smithj', 'rh'], lambda a: a.alias)

    def test_invalid_length(self):
        if False:
            print('Hello World!')
        with self.assertRaisesMessage(ValueError, "'length' must be greater than 0"):
            Author.objects.annotate(raises=Left('name', 0))

    def test_expressions(self):
        if False:
            return 10
        authors = Author.objects.annotate(name_part=Left('name', Value(3, output_field=IntegerField())))
        self.assertQuerySetEqual(authors.order_by('name'), ['Joh', 'Rho'], lambda a: a.name_part)