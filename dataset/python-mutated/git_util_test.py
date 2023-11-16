import unittest
from unittest.mock import patch
from git.exc import InvalidGitRepositoryError
from streamlit.git_util import GITHUB_HTTP_URL, GITHUB_SSH_URL, GitRepo

class GitUtilTest(unittest.TestCase):

    def test_https_url_check(self):
        if False:
            print('Hello World!')
        self.assertRegex('https://github.com/username/repo.git', GITHUB_HTTP_URL)
        self.assertRegex('https://github.com/username/repo', GITHUB_HTTP_URL)
        self.assertRegex('https://www.github.com/username/repo.git', GITHUB_HTTP_URL)
        self.assertRegex('https://www.github.com/username/repo', GITHUB_HTTP_URL)
        self.assertNotRegex('http://www.github.com/username/repo.git', GITHUB_HTTP_URL)

    def test_ssh_url_check(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRegex('git@github.com:username/repo.git', GITHUB_SSH_URL)
        self.assertRegex('git@github.com:username/repo', GITHUB_SSH_URL)

    def test_git_repo_invalid(self):
        if False:
            return 10
        with patch('git.Repo') as mock:
            mock.side_effect = InvalidGitRepositoryError('Not a git repo')
            repo = GitRepo('.')
            self.assertFalse(repo.is_valid())

    def test_old_git_version(self):
        if False:
            i = 10
            return i + 15
        "If the installed git is older than 2.7, certain repo operations\n        prompt the user for credentials. We don't want to do this, so\n        repo.is_valid() returns False for old gits.\n        "
        with patch('git.repo.base.Repo.GitCommandWrapperType') as git_mock, patch('streamlit.git_util.os'):
            git_mock.return_value.version_info = (1, 6, 4)
            repo = GitRepo('.')
            self.assertFalse(repo.is_valid())
            self.assertEqual((1, 6, 4), repo.git_version)

    def test_git_repo_valid(self):
        if False:
            return 10
        with patch('git.repo.base.Repo.GitCommandWrapperType') as git_mock, patch('streamlit.git_util.os'):
            git_mock.return_value.version_info = (2, 20, 3)
            repo = GitRepo('.')
            self.assertTrue(repo.is_valid())
            self.assertEqual((2, 20, 3), repo.git_version)

    def test_gitpython_not_installed(self):
        if False:
            while True:
                i = 10
        with patch.dict('sys.modules', {'git': None}):
            repo = GitRepo('.')
            self.assertFalse(repo.is_valid())