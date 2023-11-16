import unittest
import trashcli.list
import trashcli.list.main
import trashcli.list.parser
from trashcli.lib.print_version import PrintVersionArgs

class TestTrashListParser(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = trashcli.list.parser.Parser('trash-list')

    def test_version(self):
        if False:
            return 10
        args = self.parse(['--version'])
        assert PrintVersionArgs == type(args)

    def test_trash_dir_not_specified(self):
        if False:
            i = 10
            return i + 15
        args = self.parse([])
        assert [] == args.trash_dirs

    def test_trash_dir_specified(self):
        if False:
            while True:
                i = 10
        args = self.parse(['--trash-dir=foo'])
        assert ['foo'] == args.trash_dirs

    def test_size_off(self):
        if False:
            i = 10
            return i + 15
        args = self.parse([])
        assert 'deletion_date' == args.attribute_to_print

    def test_size_on(self):
        if False:
            while True:
                i = 10
        args = self.parse(['--size'])
        assert 'size' == args.attribute_to_print

    def test_files_off(self):
        if False:
            while True:
                i = 10
        args = self.parse([])
        assert False == args.show_files

    def test_files_on(self):
        if False:
            i = 10
            return i + 15
        args = self.parse(['--files'])
        assert True == args.show_files

    def parse(self, args):
        if False:
            while True:
                i = 10
        return self.parser.parse_list_args(args, 'trash-list')