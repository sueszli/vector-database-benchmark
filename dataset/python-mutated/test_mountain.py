import unittest
from libs.mock import *
from runner.mountain import Mountain

class TestMountain(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.mountain = Mountain()

    def test_it_gets_test_results(self):
        if False:
            while True:
                i = 10
        with patch_object(self.mountain.stream, 'writeln', Mock()):
            with patch_object(self.mountain.lesson, 'learn', Mock()):
                self.mountain.walk_the_path()
                self.assertTrue(self.mountain.lesson.learn.called)