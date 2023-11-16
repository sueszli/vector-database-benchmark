import datetime
import os
import sys
import unittest
from tests.support.files import make_file
from tests.support.my_path import MyPath
from scripts import bump
from trashcli import base_dir
from trashcli.fs import read_file
sys.path.insert(0, os.path.join(base_dir, 'script'))

class Test_version_from_date(unittest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        today = datetime.date(2021, 5, 11)
        result = bump.version_from_date(today)
        assert result == '0.21.5.11'

class Test_save_new_version(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir = MyPath.make_temp_dir()

    def test(self):
        if False:
            print('Hello World!')
        make_file(self.tmp_dir / 'trash.py', 'somecode before\nversion="0.20.1.20"\nsomecode after\ndont change this line: version="0.20.1.20"\n')
        bump.save_new_version('0.21.5.11', self.tmp_dir / 'trash.py')
        result = read_file(self.tmp_dir / 'trash.py')
        assert result == 'somecode before\nversion = \'0.21.5.11\'\nsomecode after\ndont change this line: version="0.20.1.20"\n'

    def test2(self):
        if False:
            while True:
                i = 10
        make_file(self.tmp_dir / 'trash.py', 'somecode before\n    version="0.20.1.20"\nsomecode after\n')
        bump.save_new_version('0.21.5.11', self.tmp_dir / 'trash.py')
        result = read_file(self.tmp_dir / 'trash.py')
        assert result == 'somecode before\n    version="0.20.1.20"\nsomecode after\n'

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tmp_dir.clean_up()