"""Tests for abc_hierarchy.py."""
from pytype.pytd import abc_hierarchy
import unittest

class TestAbcHierarchy(unittest.TestCase):
    """Test abc_hierarchy.py."""

    def test_get_superclasses(self):
        if False:
            for i in range(10):
                print('nop')
        superclasses = abc_hierarchy.GetSuperClasses()
        self.assertDictEqual(superclasses, abc_hierarchy.SUPERCLASSES)
        self.assertIsNot(superclasses, abc_hierarchy.SUPERCLASSES)

    def test_get_subclasses(self):
        if False:
            i = 10
            return i + 15
        subclasses = abc_hierarchy.GetSubClasses()
        self.assertSetEqual(set(subclasses['Sized']), {'Set', 'Mapping', 'MappingView', 'Sequence'})
if __name__ == '__main__':
    unittest.main()