import unittest
from patterns.creational.lazy_evaluation import Person

class TestDynamicExpanding(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.John = Person('John', 'Coder')

    def test_innate_properties(self):
        if False:
            i = 10
            return i + 15
        self.assertDictEqual({'name': 'John', 'occupation': 'Coder', 'call_count2': 0}, self.John.__dict__)

    def test_relatives_not_in_properties(self):
        if False:
            while True:
                i = 10
        self.assertNotIn('relatives', self.John.__dict__)

    def test_extended_properties(self):
        if False:
            for i in range(10):
                print('nop')
        print(f"John's relatives: {self.John.relatives}")
        self.assertDictEqual({'name': 'John', 'occupation': 'Coder', 'relatives': 'Many relatives.', 'call_count2': 0}, self.John.__dict__)

    def test_relatives_after_access(self):
        if False:
            return 10
        print(f"John's relatives: {self.John.relatives}")
        self.assertIn('relatives', self.John.__dict__)

    def test_parents(self):
        if False:
            i = 10
            return i + 15
        for _ in range(2):
            self.assertEqual(self.John.parents, 'Father and mother')
        self.assertEqual(self.John.call_count2, 1)