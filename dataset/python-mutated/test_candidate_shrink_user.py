import unittest
from trashcli.put.core.candidate import Candidate

class TestCandidateShrinkUser(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.environ = {}

    def test_should_substitute_tilde_in_place_of_home_dir(self):
        if False:
            return 10
        self.environ['HOME'] = '/home/user'
        self.trash_dir = '/home/user/.local/share/Trash'
        self.assert_name_is('~/.local/share/Trash')

    def test_should_not_substitute(self):
        if False:
            i = 10
            return i + 15
        self.environ['HOME'] = '/home/user'
        self.environ['TRASH_PUT_DISABLE_SHRINK'] = '1'
        self.trash_dir = '/home/user/.local/share/Trash'
        self.assert_name_is('/home/user/.local/share/Trash')

    def test_when_not_in_home_dir(self):
        if False:
            print('Hello World!')
        self.environ['HOME'] = '/home/user'
        self.trash_dir = '/not-in-home/Trash'
        self.assert_name_is('/not-in-home/Trash')

    def test_tilde_works_also_with_trailing_slash(self):
        if False:
            while True:
                i = 10
        self.environ['HOME'] = '/home/user/'
        self.trash_dir = '/home/user/.local/share/Trash'
        self.assert_name_is('~/.local/share/Trash')

    def test_str_uses_tilde_with_many_slashes(self):
        if False:
            i = 10
            return i + 15
        self.environ['HOME'] = '/home/user////'
        self.trash_dir = '/home/user/.local/share/Trash'
        self.assert_name_is('~/.local/share/Trash')

    def test_dont_get_confused_by_empty_home_dir(self):
        if False:
            for i in range(10):
                print('nop')
        self.environ['HOME'] = ''
        self.trash_dir = '/foo/Trash'
        self.assert_name_is('/foo/Trash')

    def test_should_work_even_if_HOME_does_not_exists(self):
        if False:
            i = 10
            return i + 15
        self.trash_dir = '/foo/Trash'
        self.assert_name_is('/foo/Trash')

    def assert_name_is(self, expected_name):
        if False:
            for i in range(10):
                print('nop')
        self.candidate = Candidate(self.trash_dir, '', '', '', None)
        shrinked = self.candidate.shrink_user(self.environ)
        assert expected_name == shrinked