from django.db.models import CharField
from django.db.models.functions import Upper
from django.test import TestCase
from django.test.utils import register_lookup
from ..models import Author

class UpperTests(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        Author.objects.create(name='John Smith', alias='smithj')
        Author.objects.create(name='Rhonda')
        authors = Author.objects.annotate(upper_name=Upper('name'))
        self.assertQuerySetEqual(authors.order_by('name'), ['JOHN SMITH', 'RHONDA'], lambda a: a.upper_name)
        Author.objects.update(name=Upper('name'))
        self.assertQuerySetEqual(authors.order_by('name'), [('JOHN SMITH', 'JOHN SMITH'), ('RHONDA', 'RHONDA')], lambda a: (a.upper_name, a.name))

    def test_transform(self):
        if False:
            return 10
        with register_lookup(CharField, Upper):
            Author.objects.create(name='John Smith', alias='smithj')
            Author.objects.create(name='Rhonda')
            authors = Author.objects.filter(name__upper__exact='JOHN SMITH')
            self.assertQuerySetEqual(authors.order_by('name'), ['John Smith'], lambda a: a.name)