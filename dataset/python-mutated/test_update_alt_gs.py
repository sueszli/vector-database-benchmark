import os
import unittest
import json
import jc.parsers.update_alt_gs
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/update-alternatives-get-selections.out'), 'r', encoding='utf-8') as f:
        update_alternatives_get_selections = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/update-alternatives-get-selections.json'), 'r', encoding='utf-8') as f:
        update_alternatives_get_selections_json = json.loads(f.read())

    def test_update_alt_gs_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'update-alternatives --get-selections' with no data\n        "
        self.assertEqual(jc.parsers.update_alt_gs.parse('', quiet=True), [])

    def test_update_alt_gs(self):
        if False:
            while True:
                i = 10
        "\n        Test 'update-alternatives --get-selections'\n        "
        self.assertEqual(jc.parsers.update_alt_gs.parse(self.update_alternatives_get_selections, quiet=True), self.update_alternatives_get_selections_json)
if __name__ == '__main__':
    unittest.main()