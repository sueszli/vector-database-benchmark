import os
import unittest
import pytest
from tests.run_command import run_command
from tests.support.files import make_file
from tests.support.my_path import MyPath

@pytest.mark.slow
class TestPutScripts(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tmp_dir = MyPath.make_temp_dir()

    def test_trash_put_works(self):
        if False:
            for i in range(10):
                print('nop')
        result = run_command('.', 'trash-put')
        assert 'usage: trash-put [OPTION]... FILE...' in result.stderr.splitlines()

    def test_trash_put_touch_filesystem(self):
        if False:
            return 10
        result = run_command('.', 'trash-put', ['non-existent'])
        assert result.stderr == "trash-put: cannot trash non existent 'non-existent'\n"

    def test_trashes_dangling_symlink(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_dangling_link(self.tmp_dir / 'link')
        result = run_command(self.tmp_dir, 'trash-put', ['-v', '--trash-dir', self.tmp_dir / 'trash-dir', 'link'], env={'TRASH_PUT_DISABLE_SHRINK': '1'})
        self.assertEqual(["trash-put: 'link' trashed in %s" % (self.tmp_dir / 'trash-dir')], self.read_trashed_in_message(result), result.stderr)
        assert not os.path.lexists(self.tmp_dir / 'link')
        assert os.path.lexists(self.tmp_dir / 'trash-dir' / 'files' / 'link')

    def test_trashes_connected_symlink(self):
        if False:
            print('Hello World!')
        self.make_connected_link(self.tmp_dir / 'link')
        result = run_command(self.tmp_dir, 'trash-put', ['-v', '--trash-dir', self.tmp_dir / 'trash-dir', 'link'], env={'TRASH_PUT_DISABLE_SHRINK': '1'})
        self.assertEqual(["trash-put: 'link' trashed in %s" % (self.tmp_dir / 'trash-dir')], self.read_trashed_in_message(result), result.stderr)
        assert result.stdout == ''
        assert not os.path.lexists(self.tmp_dir / 'link')
        assert os.path.lexists(self.tmp_dir / 'trash-dir' / 'files' / 'link')

    def read_trashed_in_message(self, result):
        if False:
            print('Hello World!')
        return list(filter(lambda line: 'trashed in' in line, result.stderr.splitlines()))

    def make_connected_link(self, path):
        if False:
            i = 10
            return i + 15
        make_file(self.tmp_dir / 'link-target')
        os.symlink('link-target', path)

    def make_dangling_link(self, path):
        if False:
            print('Hello World!')
        os.symlink('non-existent', self.tmp_dir / 'link')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmp_dir.clean_up()