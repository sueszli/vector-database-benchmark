import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_dev_mcast
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        fixtures = {'proc_net_dev_mcast': ('fixtures/linux-proc/net_dev_mcast', 'fixtures/linux-proc/net_dev_mcast.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_dev_mcast_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_net_dev_mcast' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_dev_mcast.parse('', quiet=True), [])

    def test_proc_net_dev_mcast(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/net/dev_mcast'\n        "
        self.assertEqual(jc.parsers.proc_net_dev_mcast.parse(self.f_in['proc_net_dev_mcast'], quiet=True), self.f_json['proc_net_dev_mcast'])
if __name__ == '__main__':
    unittest.main()