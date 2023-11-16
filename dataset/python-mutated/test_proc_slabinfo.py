import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_slabinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_slabinfo': ('fixtures/linux-proc/slabinfo', 'fixtures/linux-proc/slabinfo.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_slabinfo_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'proc_slabinfo' with no data\n        "
        self.assertEqual(jc.parsers.proc_slabinfo.parse('', quiet=True), [])

    def test_proc_slabinfo(self):
        if False:
            while True:
                i = 10
        "\n        Test '/proc/slabinfo'\n        "
        self.assertEqual(jc.parsers.proc_slabinfo.parse(self.f_in['proc_slabinfo'], quiet=True), self.f_json['proc_slabinfo'])
if __name__ == '__main__':
    unittest.main()