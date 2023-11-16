import unittest
from tests.support.files import make_file
from tests.support.my_path import MyPath
from trashcli import fs

class Test_file_size(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tmp_dir = MyPath.make_temp_dir()

    def test(self):
        if False:
            while True:
                i = 10
        make_file(self.tmp_dir / 'a-file', '123')
        result = fs.file_size(self.tmp_dir / 'a-file')
        assert 3 == result

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tmp_dir.clean_up()