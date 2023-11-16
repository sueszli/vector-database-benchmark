import unittest
from troposphere import Name
from troposphere.iottwinmaker import DataType, DataValue

class TestPlacementTemplate(unittest.TestCase):

    def test_datatype_nestedtype(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            DataType(NestedType='foo')
        DataType(NestedType=DataType())
        DataType(NestedType=Name('foo'))

    def test_datavalue_listvalue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            DataValue(ListValue='foo')
        with self.assertRaises(TypeError):
            DataValue(ListValue=['foo'])
        DataValue(ListValue=[DataValue(), Name('foo')])