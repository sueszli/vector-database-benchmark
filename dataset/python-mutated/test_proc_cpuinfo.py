import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_cpuinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'proc_cpuinfo': ('fixtures/linux-proc/cpuinfo', 'fixtures/linux-proc/cpuinfo.json'), 'proc_cpuinfo2': ('fixtures/linux-proc/cpuinfo2', 'fixtures/linux-proc/cpuinfo2.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_cpuinfo_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'proc_cpuinfo' with no data\n        "
        self.assertEqual(jc.parsers.proc_cpuinfo.parse('', quiet=True), [])

    def test_proc_cpuinfo(self):
        if False:
            return 10
        "\n        Test '/proc/buddyinfo'\n        "
        self.assertEqual(jc.parsers.proc_cpuinfo.parse(self.f_in['proc_cpuinfo'], quiet=True), self.f_json['proc_cpuinfo'])

    def test_proc_cpuinfo2(self):
        if False:
            return 10
        "\n        Test '/proc/buddyinfo2'\n        "
        self.assertEqual(jc.parsers.proc_cpuinfo.parse(self.f_in['proc_cpuinfo2'], quiet=True), self.f_json['proc_cpuinfo2'])
if __name__ == '__main__':
    unittest.main()