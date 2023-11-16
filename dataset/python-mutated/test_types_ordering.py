from qiskit.circuit.classical import types
from qiskit.test import QiskitTestCase

class TestTypesOrdering(QiskitTestCase):

    def test_order(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(types.order(types.Uint(8), types.Uint(16)), types.Ordering.LESS)
        self.assertIs(types.order(types.Uint(16), types.Uint(8)), types.Ordering.GREATER)
        self.assertIs(types.order(types.Uint(8), types.Uint(8)), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Bool(), types.Bool()), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Bool(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Bool()), types.Ordering.NONE)

    def test_is_subtype(self):
        if False:
            print('Hello World!')
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(16)))
        self.assertFalse(types.is_subtype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Uint(8), strict=True))
        self.assertTrue(types.is_subtype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Bool(), strict=True))
        self.assertFalse(types.is_subtype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Bool()))

    def test_is_supertype(self):
        if False:
            print('Hello World!')
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(16)))
        self.assertTrue(types.is_supertype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_supertype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(8), strict=True))
        self.assertTrue(types.is_supertype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Bool(), strict=True))
        self.assertFalse(types.is_supertype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Bool()))

    def test_greater(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(types.greater(types.Uint(16), types.Uint(8)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(16)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(8)), types.Uint(8))
        self.assertEqual(types.greater(types.Bool(), types.Bool()), types.Bool())
        with self.assertRaisesRegex(TypeError, 'no ordering'):
            types.greater(types.Bool(), types.Uint(8))