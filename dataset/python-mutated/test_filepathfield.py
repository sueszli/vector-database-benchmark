import os
from django.db.models import FilePathField
from django.test import SimpleTestCase

class FilePathFieldTests(SimpleTestCase):

    def test_path(self):
        if False:
            while True:
                i = 10
        path = os.path.dirname(__file__)
        field = FilePathField(path=path)
        self.assertEqual(field.path, path)
        self.assertEqual(field.formfield().path, path)

    def test_callable_path(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.dirname(__file__)

        def generate_path():
            if False:
                for i in range(10):
                    print('nop')
            return path
        field = FilePathField(path=generate_path)
        self.assertEqual(field.path(), path)
        self.assertEqual(field.formfield().path, path)