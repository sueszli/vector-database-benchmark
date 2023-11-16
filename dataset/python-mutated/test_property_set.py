"""Transpiler PropertySet testing"""
import unittest
from qiskit.transpiler import PropertySet
from qiskit.test import QiskitTestCase

class TestPropertySet(QiskitTestCase):
    """Tests for PropertySet methods."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.pset = PropertySet()

    def test_get_non_existent(self):
        if False:
            i = 10
            return i + 15
        'Getting non-existent property should return None.'
        self.assertIsNone(self.pset['does_not_exists'])

    def test_get_set_and_retrive(self):
        if False:
            i = 10
            return i + 15
        'Setting and retrieving.'
        self.pset['property'] = 'value'
        self.assertEqual(self.pset['property'], 'value')

    def test_str(self):
        if False:
            return 10
        'Test __str__ method.'
        self.pset['property'] = 'value'
        self.assertEqual(str(self.pset), "{'property': 'value'}")

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        'Test __repr__ method.'
        self.pset['property'] = 'value'
        self.assertEqual(str(repr(self.pset)), "{'property': 'value'}")
if __name__ == '__main__':
    unittest.main()