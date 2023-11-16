import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_igmp
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'proc_net_igmp': ('fixtures/linux-proc/net_igmp', 'fixtures/linux-proc/net_igmp.json'), 'proc_net_igmp_more': ('fixtures/linux-proc/net_igmp_more', 'fixtures/linux-proc/net_igmp_more.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_igmp_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'proc_net_igmp' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_igmp.parse('', quiet=True), [])

    def test_proc_net_igmp(self):
        if False:
            return 10
        "\n        Test '/proc/net/igmp'\n        "
        self.assertEqual(jc.parsers.proc_net_igmp.parse(self.f_in['proc_net_igmp'], quiet=True), self.f_json['proc_net_igmp'])

    def test_proc_net_igmp_more(self):
        if False:
            while True:
                i = 10
        "\n        Test '/proc/net/igmp' additional file\n        "
        self.assertEqual(jc.parsers.proc_net_igmp.parse(self.f_in['proc_net_igmp_more'], quiet=True), self.f_json['proc_net_igmp_more'])
if __name__ == '__main__':
    unittest.main()