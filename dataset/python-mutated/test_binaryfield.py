from django.core.exceptions import ValidationError
from django.db import models
from django.test import TestCase
from .models import DataModel

class BinaryFieldTests(TestCase):
    binary_data = b'\x00F\xfe'

    def test_set_and_retrieve(self):
        if False:
            i = 10
            return i + 15
        data_set = (self.binary_data, bytearray(self.binary_data), memoryview(self.binary_data))
        for bdata in data_set:
            with self.subTest(data=repr(bdata)):
                dm = DataModel(data=bdata)
                dm.save()
                dm = DataModel.objects.get(pk=dm.pk)
                self.assertEqual(bytes(dm.data), bytes(bdata))
                dm.save()
                dm = DataModel.objects.get(pk=dm.pk)
                self.assertEqual(bytes(dm.data), bytes(bdata))
                self.assertEqual(bytes(dm.short_data), b'\x08')

    def test_max_length(self):
        if False:
            while True:
                i = 10
        dm = DataModel(short_data=self.binary_data * 4)
        with self.assertRaises(ValidationError):
            dm.full_clean()

    def test_editable(self):
        if False:
            print('Hello World!')
        field = models.BinaryField()
        self.assertIs(field.editable, False)
        field = models.BinaryField(editable=True)
        self.assertIs(field.editable, True)
        field = models.BinaryField(editable=False)
        self.assertIs(field.editable, False)

    def test_filter(self):
        if False:
            i = 10
            return i + 15
        dm = DataModel.objects.create(data=self.binary_data)
        DataModel.objects.create(data=b'\xef\xbb\xbf')
        self.assertSequenceEqual(DataModel.objects.filter(data=self.binary_data), [dm])

    def test_filter_bytearray(self):
        if False:
            print('Hello World!')
        dm = DataModel.objects.create(data=self.binary_data)
        DataModel.objects.create(data=b'\xef\xbb\xbf')
        self.assertSequenceEqual(DataModel.objects.filter(data=bytearray(self.binary_data)), [dm])

    def test_filter_memoryview(self):
        if False:
            return 10
        dm = DataModel.objects.create(data=self.binary_data)
        DataModel.objects.create(data=b'\xef\xbb\xbf')
        self.assertSequenceEqual(DataModel.objects.filter(data=memoryview(self.binary_data)), [dm])