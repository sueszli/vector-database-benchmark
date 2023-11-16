import os
import unittest
import json
import jc.parsers.ini_dup
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-test.ini'), 'r', encoding='utf-8') as f:
        generic_ini_test = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-iptelserver.ini'), 'r', encoding='utf-8') as f:
        generic_ini_iptelserver = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-double-quote.ini'), 'r', encoding='utf-8') as f:
        generic_ini_double_quote = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-single-quote.ini'), 'r', encoding='utf-8') as f:
        generic_ini_single_quote = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-dup-test.json'), 'r', encoding='utf-8') as f:
        generic_ini_dup_test_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-dup-iptelserver.json'), 'r', encoding='utf-8') as f:
        generic_ini_dup_iptelserver_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-dup-double-quote.json'), 'r', encoding='utf-8') as f:
        generic_ini_dup_double_quote_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/ini-dup-single-quote.json'), 'r', encoding='utf-8') as f:
        generic_ini_dup_single_quote_json = json.loads(f.read())

    def test_ini_dup_nodata(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the test ini file with no data\n        '
        self.assertEqual(jc.parsers.ini_dup.parse('', quiet=True), {})

    def test_ini_dup_test(self):
        if False:
            while True:
                i = 10
        '\n        Test the test ini file\n        '
        self.assertEqual(jc.parsers.ini_dup.parse(self.generic_ini_test, quiet=True), self.generic_ini_dup_test_json)

    def test_ini_dup_iptelserver(self):
        if False:
            print('Hello World!')
        '\n        Test the iptelserver ini file\n        '
        self.assertEqual(jc.parsers.ini_dup.parse(self.generic_ini_iptelserver, quiet=True), self.generic_ini_dup_iptelserver_json)

    def test_ini_dup_duplicate_keys(self):
        if False:
            i = 10
            return i + 15
        '\n        Test input that contains duplicate keys.\n        '
        data = '\n[section]\nduplicate_key: value1\nanother_key = foo\nduplicate_key = value2\n'
        expected = {'section': {'duplicate_key': ['value1', 'value2'], 'another_key': ['foo']}}
        self.assertEqual(jc.parsers.ini_dup.parse(data, quiet=True), expected)

    def test_ini_dup_missing_top_section(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test INI file missing top-level section header.\n        '
        data = '\nkey: value1\nanother_key = foo\n[section2]\nkey3: bar\nkey4 =\n[section 3]\nkey5 = "quoted"\n'
        expected = {'key': ['value1'], 'another_key': ['foo'], 'section2': {'key3': ['bar'], 'key4': ['']}, 'section 3': {'key5': ['quoted']}}
        self.assertEqual(jc.parsers.ini_dup.parse(data, quiet=True), expected)

    def test_ini_dup_doublequote(self):
        if False:
            print('Hello World!')
        '\n        Test ini file with double quotes around a value\n        '
        self.assertEqual(jc.parsers.ini_dup.parse(self.generic_ini_double_quote, quiet=True), self.generic_ini_dup_double_quote_json)

    def test_ini_dup_singlequote(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ini file with single quotes around a value\n        '
        self.assertEqual(jc.parsers.ini_dup.parse(self.generic_ini_single_quote, quiet=True), self.generic_ini_dup_single_quote_json)
if __name__ == '__main__':
    unittest.main()