from django.db.models import Value as V
from django.db.models.functions import Lower, StrIndex, Substr, Upper
from django.test import TestCase
from ..models import Author

class SubstrTests(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        Author.objects.create(name='John Smith', alias='smithj')
        Author.objects.create(name='Rhonda')
        authors = Author.objects.annotate(name_part=Substr('name', 5, 3))
        self.assertQuerySetEqual(authors.order_by('name'), [' Sm', 'da'], lambda a: a.name_part)
        authors = Author.objects.annotate(name_part=Substr('name', 2))
        self.assertQuerySetEqual(authors.order_by('name'), ['ohn Smith', 'honda'], lambda a: a.name_part)
        Author.objects.filter(alias__isnull=True).update(alias=Lower(Substr('name', 1, 5)))
        self.assertQuerySetEqual(authors.order_by('name'), ['smithj', 'rhond'], lambda a: a.alias)

    def test_start(self):
        if False:
            for i in range(10):
                print('nop')
        Author.objects.create(name='John Smith', alias='smithj')
        a = Author.objects.annotate(name_part_1=Substr('name', 1), name_part_2=Substr('name', 2)).get(alias='smithj')
        self.assertEqual(a.name_part_1[1:], a.name_part_2)

    def test_pos_gt_zero(self):
        if False:
            print('Hello World!')
        with self.assertRaisesMessage(ValueError, "'pos' must be greater than 0"):
            Author.objects.annotate(raises=Substr('name', 0))

    def test_expressions(self):
        if False:
            print('Hello World!')
        Author.objects.create(name='John Smith', alias='smithj')
        Author.objects.create(name='Rhonda')
        substr = Substr(Upper('name'), StrIndex('name', V('h')), 5)
        authors = Author.objects.annotate(name_part=substr)
        self.assertQuerySetEqual(authors.order_by('name'), ['HN SM', 'HONDA'], lambda a: a.name_part)