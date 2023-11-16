import unittest
from trashcli.lib.user_info import SingleUserInfoProvider

class TestUserInfoProvider(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.provider = SingleUserInfoProvider()

    def test_getuid(self):
        if False:
            while True:
                i = 10
        info = self.provider.get_user_info({}, 123)
        assert [123] == [i.uid for i in info]

    def test_home(self):
        if False:
            return 10
        info = self.provider.get_user_info({'HOME': '~'}, 123)
        assert [['~/.local/share/Trash']] == [i.home_trash_dir_paths for i in info]