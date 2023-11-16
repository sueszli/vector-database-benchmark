import unittest
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle

class TestPropertySave(Dy2StTestBase):
    """test jit property save"""

    def setUp(self):
        if False:
            print('Hello World!')
        a = paddle.framework.core.Property()
        a.set_float('a', 1.0)
        a.set_floats('b', [1.02, 2.3, 4.23])
        b = paddle.framework.core.Property()
        b.parse_from_string(a.serialize_to_string())
        self.a = a
        self.b = b

    def test_property_save(self):
        if False:
            return 10
        self.assertEqual(self.a.get_float('a'), self.b.get_float('a'))
        self.assertEqual(self.a.get_float(0), 1.0)

    def test_size(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.b.size(), 2)
        self.assertEqual(self.a.size(), 2)

    def test_load_float(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self.a.get_float(1)

    def test_set(self):
        if False:
            for i in range(10):
                print('nop')
        'test property set.'
        try:
            a = paddle.framework.core.Property()
            a.set_float('float', 10.0)
            a.set_floats('floats', [5.0, 4.0, 3.0])
            a.set_int('int', 5)
            a.set_ints('ints', [1, 2, 3])
            a.set_string('str', 'hello')
            a.set_strings('strs', ['1', '2', '3'])
        except Exception as e:
            self.assertEqual(False, True)
if __name__ == '__main__':
    unittest.main()