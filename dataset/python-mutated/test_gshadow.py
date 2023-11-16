import os
import json
import unittest
import jc.parsers.gshadow
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/gshadow.out'), 'r', encoding='utf-8') as f:
        centos_7_7_gshadow = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/gshadow.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_gshadow = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/gshadow.json'), 'r', encoding='utf-8') as f:
        centos_7_7_gshadow_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/gshadow.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_gshadow_json = json.loads(f.read())

    def test_gshadow_nodata(self):
        if False:
            return 10
        "\n        Test 'cat /etc/gshadow' with no data\n        "
        self.assertEqual(jc.parsers.gshadow.parse('', quiet=True), [])

    def test_gshadow_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'cat /etc/gshadow' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.gshadow.parse(self.centos_7_7_gshadow, quiet=True), self.centos_7_7_gshadow_json)

    def test_gshadow_ubuntu_18_4(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'cat /etc/gshadow' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.gshadow.parse(self.ubuntu_18_4_gshadow, quiet=True), self.ubuntu_18_4_gshadow_json)
if __name__ == '__main__':
    unittest.main()