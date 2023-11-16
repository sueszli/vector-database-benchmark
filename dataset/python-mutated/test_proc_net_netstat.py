import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_netstat
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        fixtures = {'proc_net_netstat': ('fixtures/linux-proc/net_netstat', 'fixtures/linux-proc/net_netstat.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_netstat_nodata(self):
        if False:
            return 10
        "\n        Test 'proc_net_netstat' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_netstat.parse('', quiet=True), [])

    def test_proc_net_netstat(self):
        if False:
            return 10
        "\n        Test '/proc/net/netstat'\n        "
        self.assertEqual(jc.parsers.proc_net_netstat.parse(self.f_in['proc_net_netstat'], quiet=True), self.f_json['proc_net_netstat'])
if __name__ == '__main__':
    unittest.main()