import os
import json
import unittest
import jc.parsers.mpstat_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat.out'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat-A.out'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat_A = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat-A-2-5.out'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat_A_2_5 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/mpstat-A.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_mpstat_A = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat-A-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat_A_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/mpstat-A-2-5-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_mpstat_A_2_5_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/mpstat-A-streaming.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_mpstat_A_streaming_json = json.loads(f.read())

    def test_mpstat_s_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'mpstat' with no data\n        "
        self.assertEqual(list(jc.parsers.mpstat_s.parse([], quiet=True)), [])

    def test_mpstat_s_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'mpstat' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.mpstat_s.parse(self.centos_7_7_mpstat.splitlines(), quiet=True)), self.centos_7_7_mpstat_streaming_json)

    def test_mpstat_s_A_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'mpstat -A' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.mpstat_s.parse(self.centos_7_7_mpstat_A.splitlines(), quiet=True)), self.centos_7_7_mpstat_A_streaming_json)

    def test_mpstat_s_A_2_5_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'mpstat -A 2 5' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.mpstat_s.parse(self.centos_7_7_mpstat_A_2_5.splitlines(), quiet=True)), self.centos_7_7_mpstat_A_2_5_streaming_json)

    def test_mpstat_s_A_ubuntu_18_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'mpstat -A' on Ubuntu 18.4\n        "
        self.assertEqual(list(jc.parsers.mpstat_s.parse(self.ubuntu_18_4_mpstat_A.splitlines(), quiet=True)), self.ubuntu_18_4_mpstat_A_streaming_json)
if __name__ == '__main__':
    unittest.main()