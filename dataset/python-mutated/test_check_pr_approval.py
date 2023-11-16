"""
TestCases for check_pr_approval.py
"""
import subprocess
import sys
import unittest

class Test_check_approval(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.codeset = 'UTF-8'
        self.jsonstr = '\n[\n  {\n    "id": 688077074,\n    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDc3MDc0",\n    "user": {\n      "login": "wadefelix",\n      "id": 1306724,\n      "type": "User",\n      "site_admin": false\n    },\n    "body": "",\n    "state": "COMMENTED",\n    "author_association": "CONTRIBUTOR"\n  },\n  {\n    "id": 688092580,\n    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDkyNTgw",\n    "user": {\n      "login": "MingMingShangTian",\n      "id": 13469016,\n      "type": "User",\n      "site_admin": false\n    },\n    "body": "LGTM",\n    "state": "APPROVED",\n    "author_association": "CONTRIBUTOR"\n  },\n  {\n    "id": 689175539,\n    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg5MTc1NTM5",\n    "user": {\n      "login": "pangyoki",\n      "id": 26408901,\n      "type": "User",\n      "site_admin": false\n    },\n    "body": "LGTM",\n    "state": "APPROVED",\n    "author_association": "CONTRIBUTOR"\n  }\n]\n'.encode(self.codeset)

    def test_ids(self):
        if False:
            while True:
                i = 10
        cmd = [sys.executable, 'check_pr_approval.py', '1', '26408901']
        subprc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_logins(self):
        if False:
            while True:
                i = 10
        cmd = [sys.executable, 'check_pr_approval.py', '1', 'pangyoki']
        subprc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_ids_and_logins(self):
        if False:
            i = 10
            return i + 15
        cmd = [sys.executable, 'check_pr_approval.py', '2', 'pangyoki', '13469016']
        subprc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_check_with_required_reviewer_not_approved(self):
        if False:
            return 10
        cmd = [sys.executable, 'check_pr_approval.py', '2', 'wadefelix', ' 13469016']
        subprc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = subprc.communicate(input=self.jsonstr)
        self.assertEqual('FALSE', output.decode(self.codeset).rstrip())
if __name__ == '__main__':
    unittest.main()