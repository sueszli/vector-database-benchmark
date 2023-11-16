import os
import json
import unittest
import jc.parsers.pidstat_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hl.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hl = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw-2-5.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_2_5 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pidstat-ht.out'), 'r', encoding='utf-8') as f:
        generic_pidstat_ht = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hl-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hl_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw-2-5-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_2_5_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pidstat-ht-streaming.json'), 'r', encoding='utf-8') as f:
        generic_pidstat_ht_streaming_json = json.loads(f.read())

    def test_pidstat_s_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pidstat' with no data\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse([], quiet=True)), [])

    def test_pidstat_s_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'pidstat' on Centos 7.7. Should be no output since only -h is supported\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse(self.centos_7_7_pidstat.splitlines(), quiet=True)), [])

    def test_pidstat_s_hl_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'pidstat -hl' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse(self.centos_7_7_pidstat_hl.splitlines(), quiet=True)), self.centos_7_7_pidstat_hl_streaming_json)

    def test_pidstat_s_hdlrsuw_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pidstat -hdlrsuw' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse(self.centos_7_7_pidstat_hdlrsuw.splitlines(), quiet=True)), self.centos_7_7_pidstat_hdlrsuw_streaming_json)

    def test_pidstat_s_hdlrsuw_2_5_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pidstat -hdlrsuw 2 5' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse(self.centos_7_7_pidstat_hdlrsuw_2_5.splitlines(), quiet=True)), self.centos_7_7_pidstat_hdlrsuw_2_5_streaming_json)

    def test_pidstat_s_ht(self):
        if False:
            return 10
        "\n        Test 'pidstat -hT'\n        "
        self.assertEqual(list(jc.parsers.pidstat_s.parse(self.generic_pidstat_ht.splitlines(), quiet=True)), self.generic_pidstat_ht_streaming_json)
if __name__ == '__main__':
    unittest.main()