"""Tests for pytd_visitors."""
from pytype.pytd import base_visitor
import unittest

class TestAncestorMap(unittest.TestCase):

    def test_get_ancestor_map(self):
        if False:
            print('Hello World!')
        ancestors = base_visitor._GetAncestorMap()
        self.assertEqual({'TypeDeclUnit'}, ancestors['TypeDeclUnit'])
        named_type = ancestors['NamedType']
        self.assertIn('TypeDeclUnit', named_type)
        self.assertIn('Parameter', named_type)
        self.assertIn('GenericType', named_type)
        self.assertIn('NamedType', named_type)
        self.assertNotIn('ClassType', named_type)
        self.assertNotIn('NothingType', named_type)
        self.assertNotIn('AnythingType', named_type)
if __name__ == '__main__':
    unittest.main()