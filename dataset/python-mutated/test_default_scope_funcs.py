import unittest
from paddle.base.default_scope_funcs import enter_local_scope, find_var, get_cur_scope, leave_local_scope, scoped_function, var

class TestDefaultScopeFuncs(unittest.TestCase):

    def test_cur_scope(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNotNone(get_cur_scope())

    def test_none_variable(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(find_var('test'))

    def test_create_var_get_var(self):
        if False:
            for i in range(10):
                print('nop')
        var_a = var('var_a')
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        enter_local_scope()
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        leave_local_scope()

    def test_var_get_int(self):
        if False:
            print('Hello World!')

        def __new_scope__():
            if False:
                return 10
            i = var('var_i')
            self.assertFalse(i.is_int())
            i.set_int(10)
            self.assertTrue(i.is_int())
            self.assertEqual(10, i.get_int())
        for _ in range(10):
            scoped_function(__new_scope__)
if __name__ == '__main__':
    unittest.main()