import os
import unittest
import json
import jc.parsers.kv
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/keyvalue.txt'), 'r', encoding='utf-8') as f:
        generic_ini_keyvalue = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/keyvalue-ifcfg.txt'), 'r', encoding='utf-8') as f:
        generic_ini_keyvalue_ifcfg = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/keyvalue.json'), 'r', encoding='utf-8') as f:
        generic_ini_keyvalue_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/keyvalue-ifcfg.json'), 'r', encoding='utf-8') as f:
        generic_ini_keyvalue_ifcfg_json = json.loads(f.read())

    def test_kv_nodata(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the test kv file with no data\n        '
        self.assertEqual(jc.parsers.kv.parse('', quiet=True), {})

    def test_kv_keyvalue(self):
        if False:
            return 10
        '\n        Test a file that only includes key/value lines\n        '
        self.assertEqual(jc.parsers.kv.parse(self.generic_ini_keyvalue, quiet=True), self.generic_ini_keyvalue_json)

    def test_kv_keyvalue_ifcfg(self):
        if False:
            print('Hello World!')
        '\n        Test a sample ifcfg key/value file that has quotation marks in the values\n        '
        self.assertEqual(jc.parsers.kv.parse(self.generic_ini_keyvalue_ifcfg, quiet=True), self.generic_ini_keyvalue_ifcfg_json)

    def test_kv_duplicate_keys(self):
        if False:
            while True:
                i = 10
        '\n        Test input that contains duplicate keys. Only the last value should be used.\n        '
        data = '\nduplicate_key: value1\nanother_key = foo\nduplicate_key = value2\n'
        expected = {'duplicate_key': 'value2', 'another_key': 'foo'}
        self.assertEqual(jc.parsers.kv.parse(data, quiet=True), expected)

    def test_kv_doublequote(self):
        if False:
            print('Hello World!')
        '\n        Test kv string with double quotes around a value\n        '
        data = '\nkey1: "value1"\nkey2: value2\n        '
        expected = {'key1': 'value1', 'key2': 'value2'}
        self.assertEqual(jc.parsers.kv.parse(data, quiet=True), expected)

    def test_kv_singlequote(self):
        if False:
            return 10
        '\n        Test kv string with double quotes around a value\n        '
        data = "\nkey1: 'value1'\nkey2: value2\n        "
        expected = {'key1': 'value1', 'key2': 'value2'}
        self.assertEqual(jc.parsers.kv.parse(data, quiet=True), expected)
if __name__ == '__main__':
    unittest.main()