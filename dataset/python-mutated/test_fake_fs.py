import unittest
from tests.test_put.support.fake_fs.fake_fs import FakeFs
from tests.test_put.support.format_mode import format_mode
from tests.support.capture_error import capture_error

class TestFakeFs(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.fs = FakeFs('/')

    def test(self):
        if False:
            while True:
                i = 10
        result = self.fs.ls_a('/')
        assert result == ['.', '..']

    def test_create_dir(self):
        if False:
            i = 10
            return i + 15
        self.fs.mkdir('/foo')
        result = self.fs.ls_a('/')
        assert result == ['.', '..', 'foo']

    def test_find_dir_root(self):
        if False:
            print('Hello World!')
        assert '/' == self.fs.find_dir_or_file('/').name

    def test_find_dir_root_subdir(self):
        if False:
            print('Hello World!')
        self.fs.mkdir('/foo')
        assert 'foo' == self.fs.find_dir_or_file('/foo').name

    def test_create_dir_in_dir(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.mkdir('/foo')
        self.fs.mkdir('/foo/bar')
        result = self.fs.ls_a('/foo')
        assert result == ['.', '..', 'bar']

    def test_create_file(self):
        if False:
            while True:
                i = 10
        self.fs.mkdir('/foo')
        self.fs.mkdir('/foo/bar')
        self.fs.atomic_write('/foo/bar/baz', 'content')
        result = self.fs.read('/foo/bar/baz')
        assert result == 'content'

    def test_chmod(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.make_file('/foo')
        self.fs.chmod('/foo', 493)
        assert oct(self.fs.get_mod('/foo')) == oct(493)

    def test_is_dir_when_file(self):
        if False:
            print('Hello World!')
        self.fs.make_file('/foo')
        assert self.fs.isdir('/foo') is False

    def test_is_dir_when_dir(self):
        if False:
            print('Hello World!')
        self.fs.mkdir('/foo')
        assert self.fs.isdir('/foo') is True

    def test_is_dir_when_it_does_not_exists(self):
        if False:
            while True:
                i = 10
        assert self.fs.isdir('/does-not-exists') is False

    def test_exists_false(self):
        if False:
            print('Hello World!')
        assert self.fs.exists('/foo') is False

    def test_exists_true(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.make_file('/foo')
        assert self.fs.exists('/foo') is True

    def test_remove_file(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file('/foo')
        self.fs.remove_file('/foo')
        assert self.fs.exists('/foo') is False

    def test_makedirs(self):
        if False:
            i = 10
            return i + 15
        self.fs.makedirs('/foo/bar/baz', 448)
        assert [self.fs.isdir('/foo/bar/baz'), self.fs.get_mod('/foo/bar/baz')] == [True, 448]

    def test_move(self):
        if False:
            while True:
                i = 10
        self.fs.make_file('/foo')
        self.fs.move('/foo', '/bar')
        assert self.fs.exists('/foo') is False
        assert self.fs.exists('/bar') is True

    def test_move_dir(self):
        if False:
            print('Hello World!')
        self.fs.mkdir('/fruits')
        self.fs.make_file('/apple')
        self.fs.move('/apple', '/fruits')
        assert self.fs.ls_a('/fruits') == ['.', '..', 'apple']

    def test_islink_on_a_file(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file('/foo', 'content')
        assert self.fs.islink('/foo') is False

    def test_islink_on_a_link(self):
        if False:
            return 10
        self.fs.symlink('dest', '/foo')
        assert self.fs.islink('/foo') is True

    def test_set_sticky_bit_when_unset(self):
        if False:
            print('Hello World!')
        self.fs.make_file('/foo')
        assert self.fs.has_sticky_bit('/foo') is False

    def test_set_sticky_bit_when_set(self):
        if False:
            print('Hello World!')
        self.fs.make_file('/foo')
        self.fs.set_sticky_bit('/foo')
        assert self.fs.has_sticky_bit('/foo') is True

    def test_islink_when_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.fs.islink('/foo') is False

    def test_islink_when_directory_not_exisiting(self):
        if False:
            while True:
                i = 10
        assert self.fs.islink('/foo/bar/baz') is False

    def test_absolute_path(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file('/foo')
        assert '' == self.fs.find_dir_or_file('/foo').content

    def test_relativae_path(self):
        if False:
            return 10
        self.fs.make_file('/foo', 'content')
        assert 'content' == self.fs.find_dir_or_file('foo').content

    def test_relativae_path_with_cd(self):
        if False:
            while True:
                i = 10
        self.fs.makedirs('/foo/bar', 493)
        self.fs.make_file('/foo/bar/baz', 'content')
        self.fs.cd('/foo/bar')
        assert 'content' == self.fs.find_dir_or_file('baz').content

    def test_isfile_with_file(self):
        if False:
            print('Hello World!')
        self.fs.make_file('/foo')
        assert self.fs.isfile('/foo') is True

    def test_isfile_with_dir(self):
        if False:
            return 10
        self.fs.mkdir('/foo')
        assert self.fs.isfile('/foo') is False

    def test_getsize_with_empty_file(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file('foo')
        assert 0 == self.fs.getsize('foo')

    def test_getsize_with_non_empty_file(self):
        if False:
            while True:
                i = 10
        self.fs.make_file('foo', '1234')
        assert 4 == self.fs.getsize('foo')

    def test_getsize_with_dir(self):
        if False:
            print('Hello World!')
        self.fs.mkdir('foo')
        self.assertRaises(NotImplementedError, lambda : self.fs.getsize('foo'))

    def test_mode_lets_create_a_file(self):
        if False:
            print('Hello World!')
        self.fs.makedirs('/foo/bar/baz', 493)
        self.fs.make_file('/foo/bar/baz/1', '1')
        assert self.fs.isfile('/foo/bar/baz/1') is True

    def test_mode_does_not_let_create_a_file(self):
        if False:
            return 10
        self.fs.makedirs('/foo/bar/baz', 493)
        self.fs.chmod('/foo/bar/baz', 45)
        error = capture_error(lambda : self.fs.make_file('/foo/bar/baz/1', '1'))
        assert str(error) == "[Errno 13] Permission denied: '/foo/bar/baz/1'"

    def test_makedirs_honor_file_permissions(self):
        if False:
            return 10
        self.fs.makedirs('/foo', 0)
        error = capture_error(lambda : self.fs.makedirs('/foo/bar', 493))
        assert str(error) == "[Errno 13] Permission denied: '/foo/bar'"

    def test_get_mod_s_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.make_file('/foo', 'content')
        assert format_mode(self.fs.get_mod('/foo')) == '0o644'

    def test_get_mod_s_2(self):
        if False:
            return 10
        self.fs.makedirs('/foo', 0)
        assert format_mode(self.fs.get_mod('/foo')) == '0o000'