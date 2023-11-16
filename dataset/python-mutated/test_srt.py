import os
import unittest
import json
import jc.parsers.srt
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/srt-attack_of_the_clones.srt'), 'r', encoding='utf-8') as f:
        generic_attack_of_the_clones = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/srt-complex.srt'), 'r', encoding='utf-8') as f:
        generic_complex = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/srt-attack_of_the_clones_raw.json'), 'r', encoding='utf-8') as f:
        generic_attack_of_the_clones_raw_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/srt-attack_of_the_clones.json'), 'r', encoding='utf-8') as f:
        generic_attack_of_the_clones_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/srt-complex.json'), 'r', encoding='utf-8') as f:
        generic_complex_json = json.loads(f.read())

    def test_srt_nodata(self):
        if False:
            print('Hello World!')
        '\n        Test srt parser with no data\n        '
        self.assertEqual(jc.parsers.srt.parse('', quiet=True), [])

    def test_srt_nodata_r(self):
        if False:
            return 10
        '\n        Test srt parser with no data and raw output\n        '
        self.assertEqual(jc.parsers.srt.parse('', raw=True, quiet=True), [])

    def test_srt_attack_of_the_clones_raw(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the attack of the clones srt file without post processing\n        '
        self.assertEqual(jc.parsers.srt.parse(self.generic_attack_of_the_clones, raw=True, quiet=True), self.generic_attack_of_the_clones_raw_json)

    def test_srt_attack_of_the_clones(self):
        if False:
            return 10
        '\n        Test the attack of the clones srt file\n        '
        self.assertEqual(jc.parsers.srt.parse(self.generic_attack_of_the_clones, quiet=True), self.generic_attack_of_the_clones_json)

    def test_srt_complex(self):
        if False:
            i = 10
            return i + 15
        '\n        Test a complex srt file\n        '
        self.assertEqual(jc.parsers.srt.parse(self.generic_complex, quiet=True), self.generic_complex_json)
if __name__ == '__main__':
    unittest.main()