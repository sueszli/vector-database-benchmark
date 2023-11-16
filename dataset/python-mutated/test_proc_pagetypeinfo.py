import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_pagetypeinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'proc_pagetypeinfo': ('fixtures/linux-proc/pagetypeinfo', 'fixtures/linux-proc/pagetypeinfo.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_pagetypeinfo_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_pagetypeinfo' with no data\n        "
        self.assertEqual(jc.parsers.proc_pagetypeinfo.parse('', quiet=True), {})

    def test_proc_pagetypeinfo(self):
        if False:
            return 10
        "\n        Test '/proc/pagetypeinfo'\n        "
        self.assertEqual(jc.parsers.proc_pagetypeinfo.parse(self.f_in['proc_pagetypeinfo'], quiet=True), self.f_json['proc_pagetypeinfo'])
if __name__ == '__main__':
    unittest.main()