import os
import json
import unittest
import jc.parsers.systemctl_lj
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/systemctl-lj.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_systemctl_lj = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/systemctl-lj.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_systemctl_lj_json = json.loads(f.read())

    def test_systemctl_lj_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'systemctl -a list-jobs' with no data\n        "
        self.assertEqual(jc.parsers.systemctl_lj.parse('', quiet=True), [])

    def test_systemctl_lj_ubuntu_18_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'systemctl -a list-jobs' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.systemctl_lj.parse(self.ubuntu_18_4_systemctl_lj, quiet=True), self.ubuntu_18_4_systemctl_lj_json)
if __name__ == '__main__':
    unittest.main()