import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_consoles
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_consoles': ('fixtures/linux-proc/consoles', 'fixtures/linux-proc/consoles.json'), 'proc_consoles2': ('fixtures/linux-proc/consoles2', 'fixtures/linux-proc/consoles2.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_consoles_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_consoles' with no data\n        "
        self.assertEqual(jc.parsers.proc_consoles.parse('', quiet=True), [])

    def test_proc_consoles(self):
        if False:
            while True:
                i = 10
        "\n        Test '/proc/consoles'\n        "
        self.assertEqual(jc.parsers.proc_consoles.parse(self.f_in['proc_consoles'], quiet=True), self.f_json['proc_consoles'])

    def test_proc_consoles2(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/consoles2'\n        "
        self.assertEqual(jc.parsers.proc_consoles.parse(self.f_in['proc_consoles2'], quiet=True), self.f_json['proc_consoles2'])
if __name__ == '__main__':
    unittest.main()