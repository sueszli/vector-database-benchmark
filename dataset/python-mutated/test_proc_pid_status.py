import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_pid_status
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        fixtures = {'proc_pid_status': ('fixtures/linux-proc/pid_status', 'fixtures/linux-proc/pid_status.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_pid_status_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_pid_status' with no data\n        "
        self.assertEqual(jc.parsers.proc_pid_status.parse('', quiet=True), {})

    def test_proc_pid_status(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/<pid>/status'\n        "
        self.assertEqual(jc.parsers.proc_pid_status.parse(self.f_in['proc_pid_status'], quiet=True), self.f_json['proc_pid_status'])
if __name__ == '__main__':
    unittest.main()