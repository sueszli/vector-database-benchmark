import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_route
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        fixtures = {'proc_net_route': ('fixtures/linux-proc/net_route', 'fixtures/linux-proc/net_route.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_route_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'proc_net_route' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_route.parse('', quiet=True), [])

    def test_proc_net_route(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/net/route'\n        "
        self.assertEqual(jc.parsers.proc_net_route.parse(self.f_in['proc_net_route'], quiet=True), self.f_json['proc_net_route'])
if __name__ == '__main__':
    unittest.main()