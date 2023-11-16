import os
import unittest
import json
from typing import Dict
import jc.parsers.pci_ids
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'pci_ids': ('fixtures/generic/pci.ids', 'fixtures/generic/pci.ids.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_pci_ids_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pci_ids' with no data\n        "
        self.assertEqual(jc.parsers.pci_ids.parse('', quiet=True), {})

    def test_pci_ids(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pci_ids'\n        "
        self.assertEqual(jc.parsers.pci_ids.parse(self.f_in['pci_ids'], quiet=True), self.f_json['pci_ids'])
if __name__ == '__main__':
    unittest.main()