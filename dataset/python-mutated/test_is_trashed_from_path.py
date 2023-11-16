import unittest
from trashcli.restore.run_restore_action import original_location_matches_path

class TestOriginalLocationMatchesPath(unittest.TestCase):

    def test1(self):
        if False:
            for i in range(10):
                print('nop')
        assert original_location_matches_path('/full/path', '/full') == True

    def test2(self):
        if False:
            while True:
                i = 10
        assert original_location_matches_path('/full/path', '/full/path') == True

    def test3(self):
        if False:
            while True:
                i = 10
        assert original_location_matches_path('/prefix-extension', '/prefix') == False

    def test_root(self):
        if False:
            while True:
                i = 10
        assert original_location_matches_path('/any/path', '/') == True