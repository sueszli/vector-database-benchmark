import os
import unittest
import json
from typing import Dict
import jc.parsers.lspci
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'ubuntu_20_10_lspci_mmv': ('fixtures/ubuntu-20.10/lspci-mmv.out', 'fixtures/ubuntu-20.10/lspci-mmv.json'), 'ubuntu_20_10_lspci_nmmv': ('fixtures/ubuntu-20.10/lspci-nmmv.out', 'fixtures/ubuntu-20.10/lspci-nmmv.json'), 'ubuntu_20_10_lspci_nnmmv': ('fixtures/ubuntu-20.10/lspci-nnmmv.out', 'fixtures/ubuntu-20.10/lspci-nnmmv.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_lspci_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'lspci' with no data\n        "
        self.assertEqual(jc.parsers.lspci.parse('', quiet=True), [])

    def test_lspci_mmv_ubuntu_20_10(self):
        if False:
            print('Hello World!')
        "\n        Test 'lspci -mmv' on Ubuntu 20.10\n        "
        self.assertEqual(jc.parsers.lspci.parse(self.f_in['ubuntu_20_10_lspci_mmv'], quiet=True), self.f_json['ubuntu_20_10_lspci_mmv'])

    def test_lspci_nmmv_ubuntu_20_10(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'lspci -nmmv' on Ubuntu 20.10\n        "
        self.assertEqual(jc.parsers.lspci.parse(self.f_in['ubuntu_20_10_lspci_nmmv'], quiet=True), self.f_json['ubuntu_20_10_lspci_nmmv'])

    def test_lspci_nnmmv_ubuntu_20_10(self):
        if False:
            return 10
        "\n        Test 'lspci -nnmmv' on Ubuntu 20.10\n        "
        self.assertEqual(jc.parsers.lspci.parse(self.f_in['ubuntu_20_10_lspci_nnmmv'], quiet=True), self.f_json['ubuntu_20_10_lspci_nnmmv'])
if __name__ == '__main__':
    unittest.main()