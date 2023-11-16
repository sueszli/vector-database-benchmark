import unittest

from tests.support.fake_volume_of import volume_of_stub
from trashcli.list.trash_dir_selector import TrashDirsSelector
from trashcli.trash_dirs_scanner import trash_dir_found


class MockScanner:
    def __init__(self, name):
        self.name = name

    def scan_trash_dirs(self, environ, uid):
        return [self.name, environ, uid]


class TestTrashDirsSelector(unittest.TestCase):
    def setUp(self):
        volumes = volume_of_stub(lambda x: "volume_of %s" % x)
        self.selector = TrashDirsSelector(MockScanner("user"),
                                          MockScanner("all"),
                                          volumes)

    def test_default(self):
        result = list(self.selector.select(False, [], 'environ', 'uid'))

        assert result == ['user', 'environ', 'uid']

    def test_user_specified(self):
        result = list(self.selector.select(False, ['user-specified-dirs'],
                                           'environ', 'uid'))

        assert result == [(trash_dir_found, ('user-specified-dirs',
                                             'volume_of user-specified-dirs'))]

    def test_all_user_specified(self):
        result = list(self.selector.select(True, ['user-specified-dirs'],
                                           'environ', 'uid'))

        assert result == ['all', 'environ', 'uid']
