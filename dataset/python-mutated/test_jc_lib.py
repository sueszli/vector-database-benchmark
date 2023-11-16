from copy import deepcopy
import unittest
from typing import Generator
import jc.lib

class MyTests(unittest.TestCase):

    def test_lib_parse_csv(self):
        if False:
            while True:
                i = 10
        data = {'': [], 'a,b,c\n1,2,3': [{'a': '1', 'b': '2', 'c': '3'}]}
        for (test_data, expected_output) in data.items():
            self.assertEqual(jc.lib.parse('csv', test_data), expected_output)

    def test_lib_parse_csv_s_is_generator(self):
        if False:
            while True:
                i = 10
        self.assertIsInstance(jc.lib.parse('csv_s', 'a,b,c\n1,2,3'), Generator)

    def test_lib_parse_kv(self):
        if False:
            print('Hello World!')
        data = {'': {}, 'a=1\nb=2\nc=3': {'a': '1', 'b': '2', 'c': '3'}}
        for (test_data, expected_output) in data.items():
            self.assertEqual(jc.lib.parse('kv', test_data), expected_output)

    def test_lib_parser_mod_list_is_list(self):
        if False:
            print('Hello World!')
        self.assertIsInstance(jc.lib.parser_mod_list(), list)

    def test_lib_parser_mod_list_contains_csv(self):
        if False:
            print('Hello World!')
        self.assertTrue('csv' in jc.lib.parser_mod_list())

    def test_lib_parser_mod_list_length(self):
        if False:
            print('Hello World!')
        self.assertGreaterEqual(len(jc.lib.parser_mod_list()), 80)

    def test_lib_parser_info_is_dict(self):
        if False:
            return 10
        self.assertIsInstance(jc.lib.parser_info('csv'), dict)

    def test_lib_parser_info_csv(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(jc.lib.parser_info('csv')['name'] == 'csv')

    def test_lib_all_parser_info_is_list_of_dicts(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(jc.lib.all_parser_info(), list)
        self.assertIsInstance(jc.lib.all_parser_info()[0], dict)

    def test_lib_all_parser_info_contains_csv(self):
        if False:
            while True:
                i = 10
        p_list = []
        for p in jc.lib.all_parser_info():
            p_list.append(p['name'])
        self.assertTrue('csv' in p_list)

    def test_lib_all_parser_info_length(self):
        if False:
            i = 10
            return i + 15
        self.assertGreaterEqual(len(jc.lib.all_parser_info()), 80)

    def test_lib_all_parser_hidden_length(self):
        if False:
            i = 10
            return i + 15
        reg_length = len(jc.lib.all_parser_info())
        hidden_length = len(jc.lib.all_parser_info(show_hidden=True))
        self.assertGreater(hidden_length, reg_length)

    def test_lib_plugin_parser_mod_list_is_list(self):
        if False:
            return 10
        self.assertIsInstance(jc.lib.plugin_parser_mod_list(), list)

    def test_lib_plugin_parser_mod_list_length_is_zero(self):
        if False:
            while True:
                i = 10
        'Ensure there are no plugin parsers present during test/build.'
        self.assertEqual(len(jc.lib.plugin_parser_mod_list()), 0)

    def test_lib_cliname_to_modname(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(jc.lib._cliname_to_modname('module-name'), 'module_name')

    def test_lib_argumentname_to_modname(self):
        if False:
            print('Hello World!')
        self.assertEqual(jc.lib._cliname_to_modname('--module-name'), 'module_name')

    def test_lib_modname_to_cliname(self):
        if False:
            while True:
                i = 10
        self.assertEqual(jc.lib._modname_to_cliname('module_name'), 'module-name')

    def test_lib_all_parser_info_show_deprecated(self):
        if False:
            while True:
                i = 10
        old_parsers = deepcopy(jc.lib.parsers)
        old_get_parser = deepcopy(jc.lib._get_parser)

        class mock_parser_info:
            version = '1.1'
            description = '`deprecated` command parser'
            author = 'nobody'
            author_email = 'nobody@gmail.com'
            compatible = ['linux', 'darwin']
            magic_commands = ['deprecated']
            deprecated = True

        class mock_parser:
            info = mock_parser_info
        jc.lib.parsers = ['deprecated']
        jc.lib._get_parser = lambda x: mock_parser
        result = jc.lib.all_parser_info(show_deprecated=True)
        jc.lib.parsers = old_parsers
        jc.lib._get_parser = old_get_parser
        self.assertEqual(len(result), 1)

    def test_lib_all_parser_info_show_hidden(self):
        if False:
            print('Hello World!')
        old_parsers = deepcopy(jc.lib.parsers)
        old_get_parser = deepcopy(jc.lib._get_parser)

        class mock_parser_info:
            version = '1.1'
            description = '`deprecated` command parser'
            author = 'nobody'
            author_email = 'nobody@gmail.com'
            compatible = ['linux', 'darwin']
            magic_commands = ['deprecated']
            hidden = True

        class mock_parser:
            info = mock_parser_info
        jc.lib.parsers = ['deprecated']
        jc.lib._get_parser = lambda x: mock_parser
        result = jc.lib.all_parser_info(show_hidden=True)
        jc.lib.parsers = old_parsers
        jc.lib._get_parser = old_get_parser
        self.assertEqual(len(result), 1)
if __name__ == '__main__':
    unittest.main()