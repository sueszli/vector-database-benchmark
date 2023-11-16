import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from django.core.exceptions import SuspiciousFileOperation
from django.core.files import File, temp
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.db import IntegrityError, models
from django.test import TestCase, override_settings
from django.test.utils import isolate_apps
from .models import Document

class FileFieldTests(TestCase):

    def test_clearable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        FileField.save_form_data() will clear its instance attribute value if\n        passed False.\n        '
        d = Document(myfile='something.txt')
        self.assertEqual(d.myfile, 'something.txt')
        field = d._meta.get_field('myfile')
        field.save_form_data(d, False)
        self.assertEqual(d.myfile, '')

    def test_unchanged(self):
        if False:
            return 10
        '\n        FileField.save_form_data() considers None to mean "no change" rather\n        than "clear".\n        '
        d = Document(myfile='something.txt')
        self.assertEqual(d.myfile, 'something.txt')
        field = d._meta.get_field('myfile')
        field.save_form_data(d, None)
        self.assertEqual(d.myfile, 'something.txt')

    def test_changed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        FileField.save_form_data(), if passed a truthy value, updates its\n        instance attribute.\n        '
        d = Document(myfile='something.txt')
        self.assertEqual(d.myfile, 'something.txt')
        field = d._meta.get_field('myfile')
        field.save_form_data(d, 'else.txt')
        self.assertEqual(d.myfile, 'else.txt')

    def test_delete_when_file_unset(self):
        if False:
            while True:
                i = 10
        '\n        Calling delete on an unset FileField should not call the file deletion\n        process, but fail silently (#20660).\n        '
        d = Document()
        d.myfile.delete()

    def test_refresh_from_db(self):
        if False:
            print('Hello World!')
        d = Document.objects.create(myfile='something.txt')
        d.refresh_from_db()
        self.assertIs(d.myfile.instance, d)

    @unittest.skipIf(sys.platform == 'win32', 'Crashes with OSError on Windows.')
    def test_save_without_name(self):
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            document = Document.objects.create(myfile='something.txt')
            document.myfile = File(tmp)
            msg = f"Detected path traversal attempt in '{tmp.name}'"
            with self.assertRaisesMessage(SuspiciousFileOperation, msg):
                document.save()

    def test_defer(self):
        if False:
            print('Hello World!')
        Document.objects.create(myfile='something.txt')
        self.assertEqual(Document.objects.defer('myfile')[0].myfile, 'something.txt')

    def test_unique_when_same_filename(self):
        if False:
            i = 10
            return i + 15
        "\n        A FileField with unique=True shouldn't allow two instances with the\n        same name to be saved.\n        "
        Document.objects.create(myfile='something.txt')
        with self.assertRaises(IntegrityError):
            Document.objects.create(myfile='something.txt')

    @unittest.skipIf(sys.platform == 'win32', "Windows doesn't support moving open files.")
    @override_settings(MEDIA_ROOT=temp.gettempdir())
    def test_move_temporary_file(self):
        if False:
            print('Hello World!')
        '\n        The temporary uploaded file is moved rather than copied to the\n        destination.\n        '
        with TemporaryUploadedFile('something.txt', 'text/plain', 0, 'UTF-8') as tmp_file:
            tmp_file_path = tmp_file.temporary_file_path()
            Document.objects.create(myfile=tmp_file)
            self.assertFalse(os.path.exists(tmp_file_path), 'Temporary file still exists')

    def test_open_returns_self(self):
        if False:
            while True:
                i = 10
        '\n        FieldField.open() returns self so it can be used as a context manager.\n        '
        d = Document.objects.create(myfile='something.txt')
        d.myfile.file = ContentFile(b'', name='bla')
        self.assertEqual(d.myfile, d.myfile.open())

    def test_media_root_pathlib(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            with override_settings(MEDIA_ROOT=Path(tmp_dir)):
                with TemporaryUploadedFile('foo.txt', 'text/plain', 1, 'utf-8') as tmp_file:
                    document = Document.objects.create(myfile=tmp_file)
                    self.assertIs(document.myfile.storage.exists(os.path.join('unused', 'foo.txt')), True)

    def test_pickle(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_dir:
            with override_settings(MEDIA_ROOT=Path(tmp_dir)):
                with open(__file__, 'rb') as fp:
                    file1 = File(fp, name='test_file.py')
                    document = Document(myfile='test_file.py')
                    document.myfile.save('test_file.py', file1)
                    try:
                        dump = pickle.dumps(document)
                        loaded_document = pickle.loads(dump)
                        self.assertEqual(document.myfile, loaded_document.myfile)
                        self.assertEqual(document.myfile.url, loaded_document.myfile.url)
                        self.assertEqual(document.myfile.storage, loaded_document.myfile.storage)
                        self.assertEqual(document.myfile.instance, loaded_document.myfile.instance)
                        self.assertEqual(document.myfile.field, loaded_document.myfile.field)
                        myfile_dump = pickle.dumps(document.myfile)
                        loaded_myfile = pickle.loads(myfile_dump)
                        self.assertEqual(document.myfile, loaded_myfile)
                        self.assertEqual(document.myfile.url, loaded_myfile.url)
                        self.assertEqual(document.myfile.storage, loaded_myfile.storage)
                        self.assertEqual(document.myfile.instance, loaded_myfile.instance)
                        self.assertEqual(document.myfile.field, loaded_myfile.field)
                    finally:
                        document.myfile.delete()

    @isolate_apps('model_fields')
    def test_abstract_filefield_model(self):
        if False:
            print('Hello World!')
        '\n        FileField.model returns the concrete model for fields defined in an\n        abstract model.\n        '

        class AbstractMyDocument(models.Model):
            myfile = models.FileField(upload_to='unused')

            class Meta:
                abstract = True

        class MyDocument(AbstractMyDocument):
            pass
        document = MyDocument(myfile='test_file.py')
        self.assertEqual(document.myfile.field.model, MyDocument)