import unittest
from trashcli.put.parser import make_parser

class Test_make_parser(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.parser = make_parser('program-name')

    def test(self):
        if False:
            i = 10
            return i + 15
        options = self.parser.parse_args([])
        assert options.verbose == 0

    def test2(self):
        if False:
            i = 10
            return i + 15
        options = self.parser.parse_args(['-v'])
        assert options.verbose == 1

    def test3(self):
        if False:
            return 10
        options = self.parser.parse_args(['-vv'])
        assert options.verbose == 2

    def test_trash_dir_not_specified(self):
        if False:
            i = 10
            return i + 15
        options = self.parser.parse_args([])
        assert options.trashdir is None

    def test_trash_dir_specified(self):
        if False:
            print('Hello World!')
        options = self.parser.parse_args(['--trash-dir', '/MyTrash'])
        assert options.trashdir == '/MyTrash'

    def test_force_volume_off(self):
        if False:
            i = 10
            return i + 15
        options = self.parser.parse_args([])
        assert options.forced_volume is None

    def test_force_volume_on(self):
        if False:
            while True:
                i = 10
        options = self.parser.parse_args(['--force-volume', '/fake-vol'])
        assert options.forced_volume == '/fake-vol'

    def test_force_option_default(self):
        if False:
            return 10
        options = self.parser.parse_args([])
        assert options.mode is None

    def test_force_option(self):
        if False:
            while True:
                i = 10
        options = self.parser.parse_args(['-f'])
        assert options.mode == 'force'

    def test_interactive_override_force_option(self):
        if False:
            print('Hello World!')
        options = self.parser.parse_args(['-f', '-i'])
        assert options.mode == 'interactive'

    def test_interactive_option(self):
        if False:
            return 10
        options = self.parser.parse_args(['-i'])
        assert options.mode == 'interactive'