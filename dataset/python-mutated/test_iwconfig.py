import os
import json
import unittest
import jc.parsers.iwconfig
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class iwconfigTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig.out'), 'r', encoding='utf-8') as f:
        iwconfig_output = f.read()
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig-many.out'), 'r', encoding='utf-8') as f:
        iwconfig_many_output = f.read()
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig-space-dash-ssid.out'), 'r', encoding='utf-8') as f:
        iwconfig_space_dash_ssid = f.read()
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig.json'), 'r', encoding='utf-8') as f:
        iwconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig-raw.json'), 'r', encoding='utf-8') as f:
        iwconfig_raw_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig-many.json'), 'r', encoding='utf-8') as f:
        iwconfig_many_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, 'fixtures/generic/iwconfig-space-dash-ssid.json'), 'r', encoding='utf-8') as f:
        iwconfig_space_dash_ssid_json = json.loads(f.read())

    def test_iwconfig_nodata(self):
        if False:
            return 10
        "\n        Test 'iwconfig' with no data\n        "
        self.assertEqual(jc.parsers.iwconfig.parse('', quiet=True), [])

    def test_iwconfig_raw(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'iwconfig' raw\n        "
        self.assertEqual(jc.parsers.iwconfig.parse(self.iwconfig_output, quiet=True, raw=True), self.iwconfig_raw_json)

    def test_iwconfig(self):
        if False:
            while True:
                i = 10
        "\n        Test 'iwconfig'\n        "
        self.assertEqual(jc.parsers.iwconfig.parse(self.iwconfig_output, quiet=True), self.iwconfig_json)

    def test_iwconfig_many(self):
        if False:
            return 10
        "\n        Test 'iwconfig' many interface\n        "
        self.assertEqual(jc.parsers.iwconfig.parse(self.iwconfig_many_output, quiet=True), self.iwconfig_many_json)

    def test_iwconfig_space_dash_ssid(self):
        if False:
            print('Hello World!')
        "\n        Test 'iwconfig' many spaces and dashes in the SSID\n        "
        self.assertEqual(jc.parsers.iwconfig.parse(self.iwconfig_space_dash_ssid, quiet=True), self.iwconfig_space_dash_ssid_json)
if __name__ == '__main__':
    unittest.main()