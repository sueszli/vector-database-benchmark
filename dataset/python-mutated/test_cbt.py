import json
import os
import unittest
from jc.exceptions import ParseError
import jc.parsers.cbt
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-single.out'), 'r', encoding='utf-8') as f:
        single = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-multiple-columns.out'), 'r', encoding='utf-8') as f:
        multiple_columns = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-multiple-rows.out'), 'r', encoding='utf-8') as f:
        multiple_rows = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-single.json'), 'r', encoding='utf-8') as f:
        single_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-multiple-columns.json'), 'r', encoding='utf-8') as f:
        multiple_columns_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-multiple-rows.json'), 'r', encoding='utf-8') as f:
        multiple_rows_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cbt-multiple-rows-raw.json'), 'r', encoding='utf-8') as f:
        multiple_rows_raw_json = json.loads(f.read())

    def test_cbt_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'cbt' with no data\n        "
        self.assertEqual(jc.parsers.cbt.parse('', quiet=True), [])

    def test_cbt_single_row(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'cbt' with a single row\n        "
        self.assertEqual(jc.parsers.cbt.parse(self.single, quiet=True), self.single_json)

    def test_cbt_multiple_column_families(self):
        if False:
            while True:
                i = 10
        "\n        Test 'cbt' with multiple columns from multiple column families\n        "
        self.assertEqual(jc.parsers.cbt.parse(self.multiple_columns, quiet=True), self.multiple_columns_json)

    def test_cbt_multiple_rows(self):
        if False:
            while True:
                i = 10
        "\n        Test 'cbt' with multiple rows\n        "
        self.assertEqual(jc.parsers.cbt.parse(self.multiple_rows, quiet=True), self.multiple_rows_json)

    def test_cbt_multiple_rows_raw(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cbt' with multiple rows raw\n        "
        self.assertEqual(jc.parsers.cbt.parse(self.multiple_rows, quiet=True, raw=True), self.multiple_rows_raw_json)
if __name__ == '__main__':
    unittest.main()