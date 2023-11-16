import os
import unittest
import json
from typing import Dict
from jc.parsers.git_ls_remote import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        fixtures = {'git_ls_remote': ('fixtures/generic/git-ls-remote.out', 'fixtures/generic/git-ls-remote.json'), 'git_ls_remote_raw': ('fixtures/generic/git-ls-remote.out', 'fixtures/generic/git-ls-remote-raw.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_git_ls_remote_nodata(self):
        if False:
            return 10
        "\n        Test 'git_ls_remote' with no data\n        "
        self.assertEqual(parse('', quiet=True), {})

    def test_git_ls_remote(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_ls_remote'\n        "
        self.assertEqual(parse(self.f_in['git_ls_remote'], quiet=True), self.f_json['git_ls_remote'])

    def test_git_ls_remote_raw(self):
        if False:
            while True:
                i = 10
        "\n        Test 'git_ls_remote' with raw option\n        "
        self.assertEqual(parse(self.f_in['git_ls_remote_raw'], quiet=True, raw=True), self.f_json['git_ls_remote_raw'])
if __name__ == '__main__':
    unittest.main()