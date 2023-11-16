import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_locks
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_locks': ('fixtures/linux-proc/locks', 'fixtures/linux-proc/locks.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_locks_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'proc_locks' with no data\n        "
        self.assertEqual(jc.parsers.proc_locks.parse('', quiet=True), [])

    def test_proc_locks(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/locks'\n        "
        self.assertEqual(jc.parsers.proc_locks.parse(self.f_in['proc_locks'], quiet=True), self.f_json['proc_locks'])
if __name__ == '__main__':
    unittest.main()