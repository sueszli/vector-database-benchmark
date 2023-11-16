from django.test import TestCase
from wagtail.models import Collection

class TestCollectionTreeOperations(TestCase):

    def setUp(self):
        if False:
            return 10
        self.root_collection = Collection.get_first_root_node()
        self.holiday_photos_collection = self.root_collection.add_child(name='Holiday photos')
        self.evil_plans_collection = self.root_collection.add_child(name='Evil plans')
        self.holiday_photos_collection.refresh_from_db()

    def test_alphabetic_sorting(self):
        if False:
            print('Hello World!')
        old_evil_path = self.evil_plans_collection.path
        old_holiday_path = self.holiday_photos_collection.path
        alpha_collection = self.root_collection.add_child(name='Alpha')
        self.assertEqual(old_evil_path, self.evil_plans_collection.path)
        self.assertEqual(old_holiday_path, self.holiday_photos_collection.path)
        self.evil_plans_collection.refresh_from_db()
        self.holiday_photos_collection.refresh_from_db()
        self.assertNotEqual(old_evil_path, self.evil_plans_collection.path)
        self.assertNotEqual(old_holiday_path, self.holiday_photos_collection.path)
        self.assertLess(alpha_collection.path, self.evil_plans_collection.path)
        self.assertLess(alpha_collection.path, self.holiday_photos_collection.path)

    def test_get_ancestors(self):
        if False:
            while True:
                i = 10
        self.assertEqual(list(self.holiday_photos_collection.get_ancestors().order_by('path')), [self.root_collection])
        self.assertEqual(list(self.holiday_photos_collection.get_ancestors(inclusive=True).order_by('path')), [self.root_collection, self.holiday_photos_collection])

    def test_get_descendants(self):
        if False:
            print('Hello World!')
        self.assertEqual(list(self.root_collection.get_descendants().order_by('path')), [self.evil_plans_collection, self.holiday_photos_collection])
        self.assertEqual(list(self.root_collection.get_descendants(inclusive=True).order_by('path')), [self.root_collection, self.evil_plans_collection, self.holiday_photos_collection])

    def test_get_siblings(self):
        if False:
            print('Hello World!')
        self.assertEqual(list(self.holiday_photos_collection.get_siblings().order_by('path')), [self.evil_plans_collection, self.holiday_photos_collection])
        self.assertEqual(list(self.holiday_photos_collection.get_siblings(inclusive=False).order_by('path')), [self.evil_plans_collection])

    def test_get_next_siblings(self):
        if False:
            while True:
                i = 10
        self.assertEqual(list(self.evil_plans_collection.get_next_siblings().order_by('path')), [self.holiday_photos_collection])
        self.assertEqual(list(self.holiday_photos_collection.get_next_siblings(inclusive=True).order_by('path')), [self.holiday_photos_collection])
        self.assertEqual(list(self.holiday_photos_collection.get_next_siblings().order_by('path')), [])

    def test_get_prev_siblings(self):
        if False:
            while True:
                i = 10
        self.assertEqual(list(self.holiday_photos_collection.get_prev_siblings().order_by('path')), [self.evil_plans_collection])
        self.assertEqual(list(self.evil_plans_collection.get_prev_siblings().order_by('path')), [])
        self.assertEqual(list(self.evil_plans_collection.get_prev_siblings(inclusive=True).order_by('path')), [self.evil_plans_collection])