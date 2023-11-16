import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_api
from paddle.jit.dy2static.utils import ast_to_source_code
from paddle.jit.dy2static.variable_trans_func import create_fill_constant_node

class TestVariableTransFunc(Dy2StTestBase):

    @test_legacy_and_pir_api
    def test_create_fill_constant_node(self):
        if False:
            return 10
        node = create_fill_constant_node('a', 1.0)
        source = "a = paddle.full(shape=[1], dtype='float64', fill_value=1.0, name='a')"
        self.assertEqual(ast_to_source_code(node).replace('\n', '').replace(' ', ''), source.replace(' ', ''))
        node = create_fill_constant_node('b', True)
        source = "b = paddle.full(shape=[1], dtype='bool', fill_value=True, name='b')"
        self.assertEqual(ast_to_source_code(node).replace('\n', '').replace(' ', ''), source.replace(' ', ''))
        node = create_fill_constant_node('c', 4293)
        source = "c = paddle.full(shape=[1], dtype='int64', fill_value=4293, name='c')"
        self.assertEqual(ast_to_source_code(node).replace('\n', '').replace(' ', ''), source.replace(' ', ''))
        self.assertIsNone(create_fill_constant_node('e', None))
        self.assertIsNone(create_fill_constant_node('e', []))
if __name__ == '__main__':
    unittest.main()