import json
import unittest
from os.path import dirname, join
from mycroft.util.json_helper import load_commented_json

class TestFileLoad(unittest.TestCase):

    def test_load(self):
        if False:
            return 10
        root_dir = dirname(__file__)
        plainfile = join(root_dir, 'plain.json')
        with open(plainfile, 'r') as f:
            data_from_plain = json.load(f)
        commentedfile = join(root_dir, 'commented.json')
        data_from_commented = load_commented_json(commentedfile)
        self.assertEqual(data_from_commented, data_from_plain)
if __name__ == '__main__':
    unittest.main()