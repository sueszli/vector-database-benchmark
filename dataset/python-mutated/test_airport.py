import os
import unittest
import json
import jc.parsers.airport
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/airport-I.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_airport_I = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/airport-I.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_airport_I_json = json.loads(f.read())

    def test_airport_I_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'airport -I' with no data\n        "
        self.assertEqual(jc.parsers.airport.parse('', quiet=True), {})

    def test_airport_I_osx_10_14_6(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'airport -I' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.airport.parse(self.osx_10_14_6_airport_I, quiet=True), self.osx_10_14_6_airport_I_json)
if __name__ == '__main__':
    unittest.main()