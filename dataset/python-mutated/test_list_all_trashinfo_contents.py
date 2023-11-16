import unittest
import six
from mock import Mock, call

class TestListing(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.trashdir = Mock()
        self.trashinfo_reader = Mock()
        self.listing = Listing(self.trashdir, self.trashinfo_reader)

    def test_it_should_read_all_trashinfo_from_home_dir(self):
        if False:
            for i in range(10):
                print('nop')
        self.listing.read_home_trashdir('/path/to/trash_dir')
        self.trashdir.list_trashinfos.assert_called_with(trashdir='/path/to/trash_dir', list_to=self.trashinfo_reader)

class TestTrashDirReader(unittest.TestCase):

    def test_should_list_all_trashinfo_found(self):
        if False:
            print('Hello World!')

        def files(path):
            if False:
                while True:
                    i = 10
            yield 'file1'
            yield 'file2'
        os_listdir = Mock(side_effect=files)
        trashdir = TrashDirReader(os_listdir)
        out = Mock()
        trashdir.list_trashinfos(trashdir='/path', list_to=out)
        six.assertCountEqual(self, [call(trashinfo='/path/file1'), call(trashinfo='/path/file2')], out.mock_calls)

class TrashDirReader:

    def __init__(self, os_listdir):
        if False:
            print('Hello World!')
        self.os_listdir = os_listdir

    def list_trashinfos(self, trashdir, list_to):
        if False:
            return 10
        import os
        for entry in self.os_listdir(trashdir):
            full_path = os.path.join(trashdir, entry)
            list_to(trashinfo=full_path)

class Listing:

    def __init__(self, trashdir, trashinfo_reader):
        if False:
            i = 10
            return i + 15
        self.trashdir = trashdir
        self.trashinfo_reader = trashinfo_reader

    def read_home_trashdir(self, path):
        if False:
            while True:
                i = 10
        self.trashdir.list_trashinfos(trashdir=path, list_to=self.trashinfo_reader)