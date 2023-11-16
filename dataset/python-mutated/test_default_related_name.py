from django.core.exceptions import FieldError
from django.test import TestCase
from .models.default_related_name import Author, Book, Editor

class DefaultRelatedNameTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        cls.author = Author.objects.create(first_name='Dave', last_name='Loper')
        cls.editor = Editor.objects.create(name='Test Editions', bestselling_author=cls.author)
        cls.book = Book.objects.create(title='Test Book', editor=cls.editor)
        cls.book.authors.add(cls.author)

    def test_no_default_related_name(self):
        if False:
            return 10
        self.assertEqual(list(self.author.editor_set.all()), [self.editor])

    def test_default_related_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(list(self.author.books.all()), [self.book])

    def test_default_related_name_in_queryset_lookup(self):
        if False:
            while True:
                i = 10
        self.assertEqual(Author.objects.get(books=self.book), self.author)

    def test_model_name_not_available_in_queryset_lookup(self):
        if False:
            i = 10
            return i + 15
        msg = "Cannot resolve keyword 'book' into field."
        with self.assertRaisesMessage(FieldError, msg):
            Author.objects.get(book=self.book)

    def test_related_name_overrides_default_related_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(list(self.editor.edited_books.all()), [self.book])

    def test_inheritance(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(list(self.book.model_options_bookstores.all()), [])

    def test_inheritance_with_overridden_default_related_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(list(self.book.editor_stores.all()), [])