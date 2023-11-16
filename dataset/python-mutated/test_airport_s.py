import os
import unittest
import json
import jc.parsers.airport_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/airport-s.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_airport_s = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/airport-s.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_airport_s_json = json.loads(f.read())

    def test_airport_s_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'airport -s' with no data\n        "
        self.assertEqual(jc.parsers.airport_s.parse('', quiet=True), [])

    def test_airport_s_osx_10_14_6(self):
        if False:
            return 10
        "\n        Test 'airport -s' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.airport_s.parse(self.osx_10_14_6_airport_s, quiet=True), self.osx_10_14_6_airport_s_json)
if __name__ == '__main__':
    unittest.main()