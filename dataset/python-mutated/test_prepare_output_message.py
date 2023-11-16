import unittest
from trashcli.empty.prepare_output_message import prepare_output_message
from trashcli.trash_dirs_scanner import trash_dir_found

class TestPrepareOutputMessage(unittest.TestCase):

    def test_one_dir(self):
        if False:
            while True:
                i = 10
        trash_dirs = [(trash_dir_found, ('/Trash', '/'))]
        result = prepare_output_message(trash_dirs)
        assert 'Would empty the following trash directories:\n    - /Trash\nProceed? (y/n) ' == result

    def test_multiple_dirs(self):
        if False:
            return 10
        trash_dirs = [(trash_dir_found, ('/Trash1', '/')), (trash_dir_found, ('/Trash2', '/'))]
        result = prepare_output_message(trash_dirs)
        assert 'Would empty the following trash directories:\n    - /Trash1\n    - /Trash2\nProceed? (y/n) ' == result

    def test_no_dirs(self):
        if False:
            return 10
        trash_dirs = []
        result = prepare_output_message(trash_dirs)
        assert 'No trash directories to empty.\n' == result