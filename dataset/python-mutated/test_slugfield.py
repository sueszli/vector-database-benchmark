from django.test import TestCase
from .models import BigS, UnicodeSlugField

class SlugFieldTests(TestCase):

    def test_slugfield_max_length(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        SlugField honors max_length.\n        '
        bs = BigS.objects.create(s='slug' * 50)
        bs = BigS.objects.get(pk=bs.pk)
        self.assertEqual(bs.s, 'slug' * 50)

    def test_slugfield_unicode_max_length(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        SlugField with allow_unicode=True honors max_length.\n        '
        bs = UnicodeSlugField.objects.create(s='你好你好' * 50)
        bs = UnicodeSlugField.objects.get(pk=bs.pk)
        self.assertEqual(bs.s, '你好你好' * 50)