import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_igmp6
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_net_igmp6': ('fixtures/linux-proc/net_igmp6', 'fixtures/linux-proc/net_igmp6.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_igmp6_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'proc_net_igmp6' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_igmp6.parse('', quiet=True), [])

    def test_proc_net_igmp6(self):
        if False:
            while True:
                i = 10
        "\n        Test '/proc/net/igmp6'\n        "
        self.assertEqual(jc.parsers.proc_net_igmp6.parse(self.f_in['proc_net_igmp6'], quiet=True), self.f_json['proc_net_igmp6'])
if __name__ == '__main__':
    unittest.main()