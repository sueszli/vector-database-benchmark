import os
import sys
import time
import json
import unittest
import jc.parsers.vmstat
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.platform.startswith('win32'):
    os.environ['TZ'] = 'America/Los_Angeles'
    time.tzset()

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-a.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_a = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-at-5-10.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_at_5_10 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-awt.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_awt = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-d.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_d = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-dt.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_dt = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-w.out'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_w = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/vmstat-1-long.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_04_vmstat_1_long = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-a.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_a_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-at-5-10.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_at_5_10_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-awt.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_awt_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-d.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_d_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-dt.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_dt_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/vmstat-w.json'), 'r', encoding='utf-8') as f:
        centos_7_7_vmstat_w_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/vmstat-1-long.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_04_vmstat_1_long_json = json.loads(f.read())

    def test_vmstat_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'vmstat' with no data\n        "
        self.assertEqual(jc.parsers.vmstat.parse('', quiet=True), [])

    def test_vmstat(self):
        if False:
            print('Hello World!')
        "\n        Test 'vmstat'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat, quiet=True), self.centos_7_7_vmstat_json)

    def test_vmstat_a(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'vmstat -a'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_a, quiet=True), self.centos_7_7_vmstat_a_json)

    def test_vmstat_at_5_10(self):
        if False:
            print('Hello World!')
        "\n        Test 'vmstat -at 5 10'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_at_5_10, quiet=True), self.centos_7_7_vmstat_at_5_10_json)

    def test_vmstat_awt(self):
        if False:
            while True:
                i = 10
        "\n        Test 'vmstat -awt'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_awt, quiet=True), self.centos_7_7_vmstat_awt_json)

    def test_vmstat_d(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'vmstat -d'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_d, quiet=True), self.centos_7_7_vmstat_d_json)

    def test_vmstat_dt(self):
        if False:
            return 10
        "\n        Test 'vmstat -dt'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_dt, quiet=True), self.centos_7_7_vmstat_dt_json)

    def test_vmstat_w(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'vmstat -w'\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.centos_7_7_vmstat_w, quiet=True), self.centos_7_7_vmstat_w_json)

    def test_vmstat_1_long(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'vmstat -1' (on ubuntu) with long output that reprints the header rows\n        "
        self.assertEqual(jc.parsers.vmstat.parse(self.ubuntu_18_04_vmstat_1_long, quiet=True), self.ubuntu_18_04_vmstat_1_long_json)
if __name__ == '__main__':
    unittest.main()