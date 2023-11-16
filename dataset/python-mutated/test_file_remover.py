import unittest
from trashcli.rm.file_remover import FileRemover
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError

class TestFileRemover(unittest.TestCase):

    def test_remove_file_fails_when_file_does_not_exists(self):
        if False:
            while True:
                i = 10
        file_remover = FileRemover()
        self.assertRaises(FileNotFoundError, file_remover.remove_file2, '/non/existing/path')