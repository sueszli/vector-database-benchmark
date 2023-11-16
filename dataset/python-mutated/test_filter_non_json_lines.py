from __future__ import annotations
import unittest
from ansible.module_utils.json_utils import _filter_non_json_lines

class TestAnsibleModuleExitJson(unittest.TestCase):
    single_line_json_dict = u'{"key": "value", "olá": "mundo"}'
    single_line_json_array = u'["a","b","c"]'
    multi_line_json_dict = u'{\n"key":"value"\n}'
    multi_line_json_array = u'[\n"a",\n"b",\n"c"]'
    all_inputs = [single_line_json_dict, single_line_json_array, multi_line_json_dict, multi_line_json_array]
    junk = [u'single line of junk', u'line 1/2 of junk\nline 2/2 of junk']
    unparsable_cases = (u'No json here', u'"olá": "mundo"', u'{"No json": "ending"', u'{"wrong": "ending"]', u'["wrong": "ending"}')

    def test_just_json(self):
        if False:
            return 10
        for i in self.all_inputs:
            (filtered, warnings) = _filter_non_json_lines(i)
            self.assertEqual(filtered, i)
            self.assertEqual(warnings, [])

    def test_leading_junk(self):
        if False:
            return 10
        for i in self.all_inputs:
            for j in self.junk:
                (filtered, warnings) = _filter_non_json_lines(j + '\n' + i)
                self.assertEqual(filtered, i)
                self.assertEqual(warnings, [])

    def test_trailing_junk(self):
        if False:
            i = 10
            return i + 15
        for i in self.all_inputs:
            for j in self.junk:
                (filtered, warnings) = _filter_non_json_lines(i + '\n' + j)
                self.assertEqual(filtered, i)
                self.assertEqual(warnings, [u'Module invocation had junk after the JSON data: %s' % j.strip()])

    def test_leading_and_trailing_junk(self):
        if False:
            i = 10
            return i + 15
        for i in self.all_inputs:
            for j in self.junk:
                (filtered, warnings) = _filter_non_json_lines('\n'.join([j, i, j]))
                self.assertEqual(filtered, i)
                self.assertEqual(warnings, [u'Module invocation had junk after the JSON data: %s' % j.strip()])

    def test_unparsable_filter_non_json_lines(self):
        if False:
            return 10
        for i in self.unparsable_cases:
            self.assertRaises(ValueError, _filter_non_json_lines, data=i)