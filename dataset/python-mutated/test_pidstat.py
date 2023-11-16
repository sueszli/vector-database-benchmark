import os
import unittest
import json
import jc.parsers.pidstat
from jc.exceptions import ParseError
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hl.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hl = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw-2-5.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_2_5 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pidstat-ht.out'), 'r', encoding='utf-8') as f:
        generic_pidstat_ht = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hl.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hl_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pidstat-hdlrsuw-2-5.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pidstat_hdlrsuw_2_5_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pidstat-ht.json'), 'r', encoding='utf-8') as f:
        generic_pidstat_ht_json = json.loads(f.read())

    def test_pidstat_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pidstat' with no data\n        "
        self.assertEqual(jc.parsers.pidstat.parse('', quiet=True), [])

    def test_pidstat(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pidstat' without -h... should raise ParseError\n        "
        self.assertRaises(ParseError, jc.parsers.pidstat.parse, self.centos_7_7_pidstat, quiet=True)

    def test_pidstat_hl_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'pidstat -hl' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.pidstat.parse(self.centos_7_7_pidstat_hl, quiet=True), self.centos_7_7_pidstat_hl_json)

    def test_pidstat_hdlrsuw_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'pidstat -hdlrsuw' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.pidstat.parse(self.centos_7_7_pidstat_hdlrsuw, quiet=True), self.centos_7_7_pidstat_hdlrsuw_json)

    def test_pidstat_hdlrsuw_2_5_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pidstat -hdlrsuw 2 5' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.pidstat.parse(self.centos_7_7_pidstat_hdlrsuw_2_5, quiet=True), self.centos_7_7_pidstat_hdlrsuw_2_5_json)

    def test_pidstat_ht(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'pidstat -hT'\n        "
        self.assertEqual(jc.parsers.pidstat.parse(self.generic_pidstat_ht, quiet=True), self.generic_pidstat_ht_json)
if __name__ == '__main__':
    unittest.main()