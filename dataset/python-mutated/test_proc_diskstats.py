import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_diskstats
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'proc_diskstats': ('fixtures/linux-proc/diskstats', 'fixtures/linux-proc/diskstats.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_diskstats_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'proc_diskstats' with no data\n        "
        self.assertEqual(jc.parsers.proc_diskstats.parse('', quiet=True), [])

    def test_proc_diskstats(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/diskstats'\n        "
        self.assertEqual(jc.parsers.proc_diskstats.parse(self.f_in['proc_diskstats'], quiet=True), self.f_json['proc_diskstats'])
if __name__ == '__main__':
    unittest.main()