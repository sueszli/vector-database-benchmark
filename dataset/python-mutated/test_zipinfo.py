import os
import unittest
import json
import jc.parsers.zipinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/rhel-8/zipinfo.out'), 'r', encoding='utf-8') as f:
        rhel_8_zipinfo = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/zipinfo-multi.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_zipinfo_multi = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/rhel-8/zipinfo-space-in-name.out'), 'r', encoding='utf-8') as f:
        rhel_8_zipinfo_space_in_name = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/rhel-8/zipinfo.json'), 'r', encoding='utf-8') as f:
        rhel_8_zipinfo_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/zipinfo-multi.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_zipinfo_multi_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/rhel-8/zipinfo-space-in-name.json'), 'r', encoding='utf-8') as f:
        rhel_8_zipinfo_space_in_name_json = json.loads(f.read())

    def test_zipinfo_nodata(self):
        if False:
            return 10
        "\n        Test 'zipinfo' parser with no data\n        "
        self.assertEqual(jc.parsers.zipinfo.parse('', quiet=True), [])

    def test_zipinfo_rhel_8(self):
        if False:
            while True:
                i = 10
        "\n        Test 'zipinfo' on Red Hat 8\n        "
        self.assertEqual(jc.parsers.zipinfo.parse(self.rhel_8_zipinfo, quiet=True), self.rhel_8_zipinfo_json)

    def test_zipinfo_multi_osx_10_14_6(self):
        if False:
            print('Hello World!')
        "\n        Test 'zipinfo' with multiple archives on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.zipinfo.parse(self.osx_10_14_6_zipinfo_multi, quiet=True), self.osx_10_14_6_zipinfo_multi_json)

    def test_zipinfo_rhel_8_space_in_name(self):
        if False:
            print('Hello World!')
        "\n        Test 'zipinfo' on Red Hat 8 with spaces in the file path\n        "
        self.assertEqual(jc.parsers.zipinfo.parse(self.rhel_8_zipinfo_space_in_name, quiet=True), self.rhel_8_zipinfo_space_in_name_json)
if __name__ == '__main__':
    unittest.main()