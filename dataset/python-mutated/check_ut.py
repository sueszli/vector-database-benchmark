""" Get pull requests. """
import os
import os.path
import sys
from github import Github

class PRChecker:
    """PR Checker."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = None

    def check(self, filename, msg):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            filename (str): File to get block names.\n            msg (str): Error message.\n        '
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('No PR ID')
            sys.exit(0)
        print(pr_id)
        if not os.path.isfile(filename):
            print('No author to check')
            sys.exit(0)
        self.repo = self.github.get_repo('PaddlePaddle/Paddle')
        pr = self.repo.get_pull(int(pr_id))
        user = pr.user.login
        with open(filename) as f:
            for l in f:
                if l.rstrip('\r\n') == user:
                    print(f'{user} {msg}')
if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.check('block.txt', 'has unit-test to be fixed, so CI failed.')
    pr_checker.check('bk.txt', 'has benchmark issue to be fixed, so CI failed.')