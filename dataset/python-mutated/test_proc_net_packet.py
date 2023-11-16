import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_net_packet
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'proc_net_packet': ('fixtures/linux-proc/net_packet', 'fixtures/linux-proc/net_packet.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_net_packet_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'proc_net_packet' with no data\n        "
        self.assertEqual(jc.parsers.proc_net_packet.parse('', quiet=True), {})

    def test_proc_net_packet(self):
        if False:
            i = 10
            return i + 15
        "\n        Test '/proc/net/packet'\n        "
        self.assertEqual(jc.parsers.proc_net_packet.parse(self.f_in['proc_net_packet'], quiet=True), self.f_json['proc_net_packet'])
if __name__ == '__main__':
    unittest.main()