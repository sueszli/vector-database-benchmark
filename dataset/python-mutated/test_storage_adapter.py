from unittest import TestCase
from chatterbot.storage import StorageAdapter

class StorageAdapterTestCase(TestCase):
    """
    This test case is for the StorageAdapter base class.
    Although this class is not intended for direct use,
    this test case ensures that exceptions requiring
    basic functionality are triggered when needed.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.adapter = StorageAdapter()

    def test_count(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.count()

    def test_filter(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.filter()

    def test_remove(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.remove('')

    def test_create(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.create()

    def test_create_many(self):
        if False:
            return 10
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.create_many([])

    def test_update(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.update('')

    def test_get_random(self):
        if False:
            return 10
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.get_random()

    def test_drop(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(StorageAdapter.AdapterMethodNotImplementedError):
            self.adapter.drop()

    def test_get_model_invalid(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(AttributeError):
            self.adapter.get_model('invalid')

    def test_get_object_invalid(self):
        if False:
            return 10
        with self.assertRaises(AttributeError):
            self.adapter.get_object('invalid')