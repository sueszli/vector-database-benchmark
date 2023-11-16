import os
import unittest
import json
from typing import Dict
import jc.parsers.sshd_conf
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        fixtures = {'sshd_t': ('fixtures/generic/sshd-T.out', 'fixtures/generic/sshd-T.json'), 'sshd_t_2': ('fixtures/generic/sshd-T-2.out', 'fixtures/generic/sshd-T-2.json'), 'sshd_config': ('fixtures/generic/sshd_config', 'fixtures/generic/sshd_config.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_sshd_conf_nodata(self):
        if False:
            return 10
        "\n        Test 'sshd_conf' with no data\n        "
        self.assertEqual(jc.parsers.sshd_conf.parse('', quiet=True), {})

    def test_sshd_T(self):
        if False:
            return 10
        "\n        Test 'sshd -T'\n        "
        self.assertEqual(jc.parsers.sshd_conf.parse(self.f_in['sshd_t'], quiet=True), self.f_json['sshd_t'])

    def test_sshd_T_2(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'sshd -T' with another sample\n        "
        self.assertEqual(jc.parsers.sshd_conf.parse(self.f_in['sshd_t_2'], quiet=True), self.f_json['sshd_t_2'])

    def test_sshd_config(self):
        if False:
            return 10
        "\n        Test 'cat sshd_config'\n        "
        self.assertEqual(jc.parsers.sshd_conf.parse(self.f_in['sshd_config'], quiet=True), self.f_json['sshd_config'])
if __name__ == '__main__':
    unittest.main()