import os
import json
import unittest
import jc.parsers.syslog_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/syslog-5424.out'), 'r', encoding='utf-8') as f:
        syslog = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/syslog-5424-streaming.json'), 'r', encoding='utf-8') as f:
        syslog_streaming_json = json.loads(f.read())

    def test_syslog_s_nodata(self):
        if False:
            return 10
        "\n        Test 'syslog' with no data\n        "
        self.assertEqual(list(jc.parsers.syslog_s.parse([], quiet=True)), [])

    def test_syslog_s(self):
        if False:
            return 10
        '\n        Test syslog file\n        '
        self.assertEqual(list(jc.parsers.syslog_s.parse(self.syslog.splitlines(), quiet=True)), self.syslog_streaming_json)
if __name__ == '__main__':
    unittest.main()