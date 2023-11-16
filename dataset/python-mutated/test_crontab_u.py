import os
import json
import unittest
import jc.parsers.crontab_u
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/crontab-u.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_crontab_u = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/crontab-u.out'), 'r', encoding='utf-8') as f:
        centos_7_7_crontab_u = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/debian10/crontab-u.out'), 'r', encoding='utf-8') as f:
        debian10_crontab_u = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/crontab-u-no-normal-entries.out'), 'r', encoding='utf-8') as f:
        generic_crontab_u_no_normal_entries = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/crontab-u-var-fix.out'), 'r', encoding='utf-8') as f:
        generic_crontab_u_var_fix = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/crontab-u.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_crontab_u_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/crontab-u.json'), 'r', encoding='utf-8') as f:
        centos_7_7_crontab_u_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/debian10/crontab-u.json'), 'r', encoding='utf-8') as f:
        debian10_crontab_u_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/crontab-u-no-normal-entries.json'), 'r', encoding='utf-8') as f:
        generic_crontab_u_no_normal_entries_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/crontab-u-var-fix.json'), 'r', encoding='utf-8') as f:
        generic_crontab_u_var_fix_json = json.loads(f.read())

    def test_crontab_u_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'crontab' with no data (has a user field)\n        "
        self.assertEqual(jc.parsers.crontab_u.parse('', quiet=True), {})

    def test_crontab_u_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test 'crontab' on Ubuntu 18.4 (has a user field)\n        "
        self.assertEqual(jc.parsers.crontab_u.parse(self.ubuntu_18_4_crontab_u, quiet=True), self.ubuntu_18_4_crontab_u_json)

    def test_crontab_u_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'crontab' on Centos 7.7 (has a user field)\n        "
        self.assertEqual(jc.parsers.crontab_u.parse(self.centos_7_7_crontab_u, quiet=True), self.centos_7_7_crontab_u_json)

    def test_crontab_u_debian10(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'crontab' on Debian10 (has a user field)\n        "
        self.assertEqual(jc.parsers.crontab_u.parse(self.debian10_crontab_u, quiet=True), self.debian10_crontab_u_json)

    def test_crontab_u_no_normal_entries(self):
        if False:
            return 10
        "\n        Test 'crontab' with no normal entries - only shortcut entries (has a user field)\n        "
        self.assertEqual(jc.parsers.crontab_u.parse(self.generic_crontab_u_no_normal_entries, quiet=True), self.generic_crontab_u_no_normal_entries_json)

    def test_crontab_u_var_fix(self):
        if False:
            return 10
        "\n        Test 'crontab' with wildcard schedule should not generate variables from command line section\n        "
        self.assertEqual(jc.parsers.crontab_u.parse(self.generic_crontab_u_var_fix, quiet=True), self.generic_crontab_u_var_fix_json)
if __name__ == '__main__':
    unittest.main()