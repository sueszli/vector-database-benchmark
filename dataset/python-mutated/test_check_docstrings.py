import inspect
import os
import sys
import unittest
git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, 'utils'))
from check_docstrings import get_default_description, replace_default_in_arg_description

class CheckDostringsTested(unittest.TestCase):

    def test_replace_default_in_arg_description(self):
        if False:
            for i in range(10):
                print('nop')
        desc_with_default = '`float`, *optional*, defaults to 2.0'
        self.assertEqual(replace_default_in_arg_description(desc_with_default, 2.0), '`float`, *optional*, defaults to 2.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_default, 1.0), '`float`, *optional*, defaults to 1.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_default, inspect._empty), '`float`')
        desc_with_default_typo = '`float`, `optional`, defaults to 2.0'
        self.assertEqual(replace_default_in_arg_description(desc_with_default_typo, 2.0), '`float`, *optional*, defaults to 2.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_default_typo, 1.0), '`float`, *optional*, defaults to 1.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_default, None), '`float`, *optional*, defaults to 2.0')
        desc_with_default = '`float`, *optional*, defaults to None'
        self.assertEqual(replace_default_in_arg_description(desc_with_default, None), '`float`, *optional*')
        desc_with_default = '`float`, *optional*, defaults to `None`'
        self.assertEqual(replace_default_in_arg_description(desc_with_default, None), '`float`, *optional*')
        desc_with_default = '`float`, *optional*, defaults to 1/255'
        self.assertEqual(replace_default_in_arg_description(desc_with_default, 1 / 255), '`float`, *optional*, defaults to `1/255`')
        desc_with_default = '`float`, *optional*, defaults to `1/255`'
        self.assertEqual(replace_default_in_arg_description(desc_with_default, 1 / 255), '`float`, *optional*, defaults to `1/255`')
        desc_with_optional = '`float`, *optional*'
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, 2.0), '`float`, *optional*, defaults to 2.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, 1.0), '`float`, *optional*, defaults to 1.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, None), '`float`, *optional*')
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, inspect._empty), '`float`')
        desc_with_no_optional = '`float`'
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, 2.0), '`float`, *optional*, defaults to 2.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, 1.0), '`float`, *optional*, defaults to 1.0')
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, None), '`float`, *optional*')
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, inspect._empty), '`float`')

    def test_get_default_description(self):
        if False:
            for i in range(10):
                print('nop')

        def _fake_function(a, b: int, c=1, d: float=2.0, e: str='blob'):
            if False:
                print('Hello World!')
            pass
        params = inspect.signature(_fake_function).parameters
        assert get_default_description(params['a']) == '`<fill_type>`'
        assert get_default_description(params['b']) == '`int`'
        assert get_default_description(params['c']) == '`<fill_type>`, *optional*, defaults to 1'
        assert get_default_description(params['d']) == '`float`, *optional*, defaults to 2.0'
        assert get_default_description(params['e']) == '`str`, *optional*, defaults to `"blob"`'