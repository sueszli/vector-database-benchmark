from pelican.tests.support import unittest
from pelican.urlwrappers import Author, Category, Tag, URLWrapper

class TestURLWrapper(unittest.TestCase):

    def test_ordering(self):
        if False:
            i = 10
            return i + 15
        wrapper_a = URLWrapper(name='first', settings={})
        wrapper_b = URLWrapper(name='last', settings={})
        self.assertFalse(wrapper_a > wrapper_b)
        self.assertFalse(wrapper_a >= wrapper_b)
        self.assertFalse(wrapper_a == wrapper_b)
        self.assertTrue(wrapper_a != wrapper_b)
        self.assertTrue(wrapper_a <= wrapper_b)
        self.assertTrue(wrapper_a < wrapper_b)
        wrapper_b.name = 'first'
        self.assertFalse(wrapper_a > wrapper_b)
        self.assertTrue(wrapper_a >= wrapper_b)
        self.assertTrue(wrapper_a == wrapper_b)
        self.assertFalse(wrapper_a != wrapper_b)
        self.assertTrue(wrapper_a <= wrapper_b)
        self.assertFalse(wrapper_a < wrapper_b)
        wrapper_a.name = 'last'
        self.assertTrue(wrapper_a > wrapper_b)
        self.assertTrue(wrapper_a >= wrapper_b)
        self.assertFalse(wrapper_a == wrapper_b)
        self.assertTrue(wrapper_a != wrapper_b)
        self.assertFalse(wrapper_a <= wrapper_b)
        self.assertFalse(wrapper_a < wrapper_b)

    def test_equality(self):
        if False:
            while True:
                i = 10
        tag = Tag('test', settings={})
        cat = Category('test', settings={})
        author = Author('test', settings={})
        self.assertNotEqual(tag, cat)
        self.assertNotEqual(tag, author)
        self.assertEqual(tag, 'test')
        self.assertNotEqual(tag, b'test')
        tag_equal = Tag('Test', settings={})
        self.assertEqual(tag, tag_equal)
        author_equal = Author('Test', settings={})
        self.assertEqual(author, author_equal)
        cat_ascii = Category('指導書', settings={})
        self.assertEqual(cat_ascii, 'zhi dao shu')

    def test_slugify_with_substitutions_and_dots(self):
        if False:
            for i in range(10):
                print('nop')
        tag = Tag('Tag Dot', settings={'TAG_REGEX_SUBSTITUTIONS': [('Tag Dot', 'tag.dot')]})
        cat = Category('Category Dot', settings={'CATEGORY_REGEX_SUBSTITUTIONS': [('Category Dot', 'cat.dot')]})
        self.assertEqual(tag.slug, 'tag.dot')
        self.assertEqual(cat.slug, 'cat.dot')

    def test_author_slug_substitutions(self):
        if False:
            while True:
                i = 10
        settings = {'AUTHOR_REGEX_SUBSTITUTIONS': [('Alexander Todorov', 'atodorov'), ('Krasimir Tsonev', 'krasimir'), ('[^\\w\\s-]', ''), ('(?u)\\A\\s*', ''), ('(?u)\\s*\\Z', ''), ('[-\\s]+', '-')]}
        author1 = Author('Mr. Senko', settings=settings)
        author2 = Author('Alexander Todorov', settings=settings)
        author3 = Author('Krasimir Tsonev', settings=settings)
        self.assertEqual(author1.slug, 'mr-senko')
        self.assertEqual(author2.slug, 'atodorov')
        self.assertEqual(author3.slug, 'krasimir')