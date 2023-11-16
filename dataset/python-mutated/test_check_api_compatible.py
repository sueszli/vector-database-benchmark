"""
TestCases for check_api_compatible.py
"""
import tempfile
import unittest
from check_api_compatible import check_compatible, check_compatible_str, read_argspec_from_file

class Test_check_compatible(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fullargspec_prefix = 'inspect.Full'
        self.argspec_str_o = self.fullargspec_prefix + "ArgSpec(args=['shape', 'dtype', 'name'], varargs=None, varkw=None, defaults=(None, None), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        return super().setUp()

    def test_normal_not_changed(self):
        if False:
            i = 10
            return i + 15
        argspec_o = eval(self.argspec_str_o)
        argspec_n = eval(self.argspec_str_o)
        self.assertTrue(check_compatible(argspec_o, argspec_n))

    def test_args_added(self):
        if False:
            print('Hello World!')
        argspec_str_n = "ArgSpec(args=['shape', 'dtype', 'name', 'arg4'], varargs=None, varkw=None, defaults=(None, None), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = eval(self.argspec_str_o)
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertFalse(check_compatible(argspec_o, argspec_n))
        argspec_str_n = "ArgSpec(args=['shape', 'dtype', 'name', 'arg4'], varargs=None, varkw=None, defaults=(None, None, 1), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertTrue(check_compatible(argspec_o, argspec_n))
        argspec_str_n = "ArgSpec(args=['shape', 'dtype', 'name', 'arg4'], varargs=None, varkw=None, defaults=(None, None, 1, True), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertFalse(check_compatible(argspec_o, argspec_n))
        argspec_str_n = "ArgSpec(args=['shape', 'dtype', 'name', 'arg4'], varargs=None, varkw=None, defaults=(True, None, None, 1), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertTrue(check_compatible(argspec_o, argspec_n))

    def test_args_places_exchanged(self):
        if False:
            while True:
                i = 10
        argspec_str_n = "ArgSpec(args=['shape', 'name', 'dtype'], varargs=None, varkw=None, defaults=(None, None), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = eval(self.argspec_str_o)
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertFalse(check_compatible(argspec_o, argspec_n))

    def test_args_reduced(self):
        if False:
            return 10
        argspec_str_n = "ArgSpec(args=['shape', 'name'], varargs=None, varkw=None, defaults=(None,), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = eval(self.argspec_str_o)
        argspec_n = eval(self.fullargspec_prefix + argspec_str_n)
        self.assertFalse(check_compatible(argspec_o, argspec_n))

class Test_check_compatible_str(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.fullargspec_prefix = 'inspect.Full'
        self.argspec_str_o = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient', 'type'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        return super().setUp()

    def test_normal_not_changed(self):
        if False:
            for i in range(10):
                print('nop')
        argspec_o = self.argspec_str_o
        argspec_n = self.argspec_str_o
        self.assertTrue(check_compatible_str(argspec_o, argspec_n))

    def test_args_added(self):
        if False:
            return 10
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient', 'type', 'argadded'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = self.argspec_str_o
        self.assertFalse(check_compatible_str(argspec_o, argspec_n))
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient', 'type', 'argadded'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR, argadded), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        self.assertTrue(check_compatible_str(argspec_o, argspec_n))
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient', 'type', 'argadded'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR, argadded, 1), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        self.assertFalse(check_compatible_str(argspec_o, argspec_n))
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient', 'type', 'argadded'], varargs=None, varkw=None, defaults=(1, None, False, None, False, VarType.LOD_TENSOR, argadded), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        self.assertTrue(check_compatible_str(argspec_o, argspec_n))

    def test_args_places_exchanged(self):
        if False:
            for i in range(10):
                print('nop')
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'type', 'stop_gradient'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = self.argspec_str_o
        self.assertFalse(check_compatible_str(argspec_o, argspec_n))

    def test_args_reduced(self):
        if False:
            return 10
        argspec_n = self.fullargspec_prefix + "ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer', 'stop_gradient'], varargs=None, varkw=None, defaults=(None, False, None, False, VarType.LOD_TENSOR), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        argspec_o = self.argspec_str_o
        self.assertFalse(check_compatible_str(argspec_o, argspec_n))

    def test_args_defaults_None(self):
        if False:
            while True:
                i = 10
        argspec_o = "inspect.FullArgSpec(args=['filename'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'filename': <class 'str'>})"
        argspec_n = "inspect.FullArgSpec(args=['filename'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'filename': <class 'str'>})"
        self.assertTrue(check_compatible_str(argspec_o, argspec_n))

class Test_read_argspec_from_file(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.fullargspec_prefix = 'inspect.Full'
        self.argspec_str_o = self.fullargspec_prefix + "ArgSpec(args=['shape', 'dtype', 'name'], varargs=None, varkw=None, defaults=(None, None), kwonlyargs=[], kwonlydefaults=None, annotations={})"
        self.api_spec_file = tempfile.TemporaryFile('w+t')
        if self.api_spec_file:
            self.api_spec_file.write('\n'.join(["paddle.ones (ArgSpec(args=['shape', 'dtype', 'name'], varargs=None, varkw=None, defaults=(None, None), kwonlyargs=[], kwonlydefaults=None, annotations={}), ('document', '50a3b3a77fa13bb2ae4337d8f9d091b7'))", "paddle.five_plus_five (ArgSpec(), ('document', 'ff0f188c95030158cc6398d2a6c5five'))"]))
            self.api_spec_file.seek(0)
        return super().setUp()

    def tearDown(self):
        if False:
            return 10
        if self.api_spec_file:
            self.api_spec_file.close()

    def test_case_normal(self):
        if False:
            while True:
                i = 10
        if self.api_spec_file:
            api_argspec_dict = read_argspec_from_file(self.api_spec_file)
            argspec = eval(self.argspec_str_o)
            self.assertEqual(api_argspec_dict.get('paddle.ones').args, argspec.args)
            self.assertEqual(api_argspec_dict.get('paddle.ones').defaults, argspec.defaults)
            self.assertIsNone(api_argspec_dict.get('paddle.five_plus_five'))
        else:
            self.fail('api_spec_file error')
if __name__ == '__main__':
    unittest.main()