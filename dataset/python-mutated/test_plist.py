import os
import unittest
import json
import jc.parsers.plist
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-garageband-info.plist'), 'rb') as f:
        generic_garageband = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-safari-info.plist'), 'r', encoding='utf-8') as f:
        generic_safari = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-alltypes.plist'), 'r', encoding='utf-8') as f:
        generic_alltypes = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-alltypes-bin.plist'), 'rb') as f:
        generic_alltypes_bin = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-nextstep.plist'), 'rb') as f:
        nextstep = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-nextstep2.plist'), 'rb') as f:
        nextstep2 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-garageband-info.json'), 'r', encoding='utf-8') as f:
        generic_garageband_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-safari-info.json'), 'r', encoding='utf-8') as f:
        generic_safari_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-alltypes.json'), 'r', encoding='utf-8') as f:
        generic_alltypes_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-alltypes-bin.json'), 'r', encoding='utf-8') as f:
        generic_alltypes_bin_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-nextstep.json'), 'r', encoding='utf-8') as f:
        nextstep_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/plist-nextstep2.json'), 'r', encoding='utf-8') as f:
        nextstep2_json = json.loads(f.read())

    def test_plist_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'plist' with no data\n        "
        self.assertEqual(jc.parsers.plist.parse('', quiet=True), {})

    def test_plist_binary(self):
        if False:
            return 10
        '\n        Test binary plist file (garage band)\n        '
        self.assertEqual(jc.parsers.plist.parse(self.generic_garageband, quiet=True), self.generic_garageband_json)

    def test_plist_xml(self):
        if False:
            while True:
                i = 10
        '\n        Test XML plist file (safari)\n        '
        self.assertEqual(jc.parsers.plist.parse(self.generic_safari, quiet=True), self.generic_safari_json)

    def test_plist_xml_alltypes(self):
        if False:
            while True:
                i = 10
        '\n        Test XML plist file with all object types\n        '
        self.assertEqual(jc.parsers.plist.parse(self.generic_alltypes, quiet=True), self.generic_alltypes_json)

    def test_plist_bin_alltypes(self):
        if False:
            print('Hello World!')
        '\n        Test binary plist file with all object types\n        '
        self.assertEqual(jc.parsers.plist.parse(self.generic_alltypes_bin, quiet=True), self.generic_alltypes_bin_json)

    def test_plist_nextstep(self):
        if False:
            return 10
        '\n        Test NeXTSTEP style plist file\n        '
        self.assertEqual(jc.parsers.plist.parse(self.nextstep, quiet=True), self.nextstep_json)

    def test_plist_nextstep2(self):
        if False:
            i = 10
            return i + 15
        '\n        Test NeXTSTEP style plist file simple\n        '
        self.assertEqual(jc.parsers.plist.parse(self.nextstep2, quiet=True), self.nextstep2_json)
if __name__ == '__main__':
    unittest.main()