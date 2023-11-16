import os
import unittest
import json
import jc.parsers.timedatectl
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/timedatectl.out'), 'r', encoding='utf-8') as f:
        centos_7_7_timedatectl = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/timedatectl.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_timedatectl = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/timedatectl-rtc-local.out'), 'r', encoding='utf-8') as f:
        timedatectl_rtc_local = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/timedatectl-timesync-status.out'), 'r', encoding='utf-8') as f:
        timedatectl_timesync_status = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/timedatectl.json'), 'r', encoding='utf-8') as f:
        centos_7_7_timedatectl_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/timedatectl.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_timedatectl_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/timedatectl-rtc-local.json'), 'r', encoding='utf-8') as f:
        timedatectl_rtc_local_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/timedatectl-timesync-status.json'), 'r', encoding='utf-8') as f:
        timedatectl_timesync_status_json = json.loads(f.read())

    def test_timedatectl_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'timedatectl' with no data\n        "
        self.assertEqual(jc.parsers.timedatectl.parse('', quiet=True), {})

    def test_timedatectl_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'timedatectl' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.timedatectl.parse(self.centos_7_7_timedatectl, quiet=True), self.centos_7_7_timedatectl_json)

    def test_timedatectl_ubuntu_18_4(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'timedatectl' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.timedatectl.parse(self.ubuntu_18_4_timedatectl, quiet=True), self.ubuntu_18_4_timedatectl_json)

    def test_timedatectl_rtc_local(self):
        if False:
            print('Hello World!')
        "\n        Test 'timedatectl' with RTC set to local\n        "
        self.assertEqual(jc.parsers.timedatectl.parse(self.timedatectl_rtc_local, quiet=True), self.timedatectl_rtc_local_json)

    def test_timedatectl_timesync_status(self):
        if False:
            print('Hello World!')
        "\n        Test 'timedatectl timesync-status'\n        "
        self.assertEqual(jc.parsers.timedatectl.parse(self.timedatectl_timesync_status, quiet=True), self.timedatectl_timesync_status_json)
if __name__ == '__main__':
    unittest.main()