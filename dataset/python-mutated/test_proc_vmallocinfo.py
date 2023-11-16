import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_vmallocinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        fixtures = {'proc_vmallocinfo': ('fixtures/linux-proc/vmallocinfo', 'fixtures/linux-proc/vmallocinfo.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_vmallocinfo_nodata(self):
        if False:
            return 10
        "\n        Test 'proc_vmallocinfo' with no data\n        "
        self.assertEqual(jc.parsers.proc_vmallocinfo.parse('', quiet=True), [])

    def test_proc_vmallocinfo(self):
        if False:
            return 10
        "\n        Test '/proc/vmallocinfo'\n        "
        self.assertEqual(jc.parsers.proc_vmallocinfo.parse(self.f_in['proc_vmallocinfo'], quiet=True), self.f_json['proc_vmallocinfo'])
if __name__ == '__main__':
    unittest.main()