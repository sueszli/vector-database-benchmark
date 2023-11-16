import os
import json
import unittest
import jc.parsers.syslog_bsd_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/syslog-3164.out'), 'r', encoding='utf-8') as f:
        syslog_bsd = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/syslog-3164-streaming.json'), 'r', encoding='utf-8') as f:
        syslog_bsd_streaming_json = json.loads(f.read())

    def test_syslog_bsd_s_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'syslog_bsd' with no data\n        "
        self.assertEqual(list(jc.parsers.syslog_bsd_s.parse([], quiet=True)), [])

    def test_syslog_bsd_s(self):
        if False:
            print('Hello World!')
        '\n        Test bsd Syslog\n        '
        self.assertEqual(list(jc.parsers.syslog_bsd_s.parse(self.syslog_bsd.splitlines(), quiet=True)), self.syslog_bsd_streaming_json)
if __name__ == '__main__':
    unittest.main()