import unittest
from paddle import base

class TestGetAndSetFlags(unittest.TestCase):

    def test_api(self):
        if False:
            for i in range(10):
                print('nop')
        flags = {'FLAGS_eager_delete_tensor_gb': 1.0, 'FLAGS_check_nan_inf': True}
        base.set_flags(flags)
        flags_list = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
        flag = 'FLAGS_eager_delete_tensor_gb'
        res_list = base.get_flags(flags_list)
        res = base.get_flags(flag)
        self.assertTrue(res_list['FLAGS_eager_delete_tensor_gb'], 1.0)
        self.assertTrue(res_list['FLAGS_check_nan_inf'], True)
        self.assertTrue(res['FLAGS_eager_delete_tensor_gb'], 1.0)

class TestGetAndSetFlagsErrors(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        flags_list = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
        flag = 1
        flag_private = {'FLAGS_free_idle_chunk': True}

        def test_set_flags_input_type():
            if False:
                for i in range(10):
                    print('nop')
            base.set_flags(flags_list)
        self.assertRaises(TypeError, test_set_flags_input_type)

        def test_set_private_flag():
            if False:
                while True:
                    i = 10
            base.set_flags(flag_private)
        self.assertRaises(ValueError, test_set_private_flag)

        def test_get_flags_input_type():
            if False:
                for i in range(10):
                    print('nop')
            base.get_flags(flag)
        self.assertRaises(TypeError, test_get_flags_input_type)

        def test_get_private_flag():
            if False:
                while True:
                    i = 10
            base.get_flags('FLAGS_free_idle_chunk')
        self.assertRaises(ValueError, test_get_private_flag)
if __name__ == '__main__':
    unittest.main()