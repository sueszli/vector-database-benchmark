import os
import sys
import time
import unittest
import json
import jc.parsers.rpm_qi
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.platform.startswith('win32'):
    os.environ['TZ'] = 'America/Los_Angeles'
    time.tzset()

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rpm-qai.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rpm_qai = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rpm-qi-package.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rpm_qi_package = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rpm-qai.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rpm_qai_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rpm-qi-package.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rpm_qi_package_json = json.loads(f.read())

    def test_rpm_qi_nodata(self):
        if False:
            return 10
        "\n        Test 'rpm -qi' with no data\n        "
        self.assertEqual(jc.parsers.rpm_qi.parse('', quiet=True), [])

    def test_rpm_qai_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'rpm -qai' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.rpm_qi.parse(self.centos_7_7_rpm_qai, quiet=True), self.centos_7_7_rpm_qai_json)

    def test_rpm_qi_package_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'rpm -qi make' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.rpm_qi.parse(self.centos_7_7_rpm_qi_package, quiet=True), self.centos_7_7_rpm_qi_package_json)
if __name__ == '__main__':
    unittest.main()