import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_version
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_version': ('fixtures/linux-proc/version', 'fixtures/linux-proc/version.json'), 'proc_version2': ('fixtures/linux-proc/version2', 'fixtures/linux-proc/version2.json'), 'proc_version3': ('fixtures/linux-proc/version3', 'fixtures/linux-proc/version3.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_version_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_version' with no data\n        "
        self.assertEqual(jc.parsers.proc_version.parse('', quiet=True), {})

    def test_proc_version(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/version'\n        "
        self.assertEqual(jc.parsers.proc_version.parse(self.f_in['proc_version'], quiet=True), self.f_json['proc_version'])

    def test_proc_version2(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test '/proc/version' #2\n        "
        self.assertEqual(jc.parsers.proc_version.parse(self.f_in['proc_version2'], quiet=True), self.f_json['proc_version2'])

    def test_proc_version3(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/version' #3\n        "
        self.assertEqual(jc.parsers.proc_version.parse(self.f_in['proc_version3'], quiet=True), self.f_json['proc_version3'])
if __name__ == '__main__':
    unittest.main()