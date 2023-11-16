import os
import os.path
import shutil
import socket
import tarfile
import tempfile
import unittest
from docker.constants import IS_WINDOWS_PLATFORM
from docker.utils import exclude_paths, tar
import pytest
from ..helpers import make_tree

def convert_paths(collection):
    if False:
        i = 10
        return i + 15
    return set(map(convert_path, collection))

def convert_path(path):
    if False:
        for i in range(10):
            print('nop')
    return path.replace('/', os.path.sep)

class ExcludePathsTest(unittest.TestCase):
    dirs = ['foo', 'foo/bar', 'bar', 'target', 'target/subdir', 'subdir', 'subdir/target', 'subdir/target/subdir', 'subdir/subdir2', 'subdir/subdir2/target', 'subdir/subdir2/target/subdir']
    files = ['Dockerfile', 'Dockerfile.alt', '.dockerignore', 'a.py', 'a.go', 'b.py', 'cde.py', 'foo/a.py', 'foo/b.py', 'foo/bar/a.py', 'bar/a.py', 'foo/Dockerfile3', 'target/file.txt', 'target/subdir/file.txt', 'subdir/file.txt', 'subdir/target/file.txt', 'subdir/target/subdir/file.txt', 'subdir/subdir2/file.txt', 'subdir/subdir2/target/file.txt', 'subdir/subdir2/target/subdir/file.txt']
    all_paths = set(dirs + files)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.base = make_tree(self.dirs, self.files)

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.base)

    def exclude(self, patterns, dockerfile=None):
        if False:
            for i in range(10):
                print('nop')
        return set(exclude_paths(self.base, patterns, dockerfile=dockerfile))

    def test_no_excludes(self):
        if False:
            return 10
        assert self.exclude(['']) == convert_paths(self.all_paths)

    def test_no_dupes(self):
        if False:
            return 10
        paths = exclude_paths(self.base, ['!a.py'])
        assert sorted(paths) == sorted(set(paths))

    def test_wildcard_exclude(self):
        if False:
            print('Hello World!')
        assert self.exclude(['*']) == {'Dockerfile', '.dockerignore'}

    def test_exclude_dockerfile_dockerignore(self):
        if False:
            print('Hello World!')
        "\n        Even if the .dockerignore file explicitly says to exclude\n        Dockerfile and/or .dockerignore, don't exclude them from\n        the actual tar file.\n        "
        assert self.exclude(['Dockerfile', '.dockerignore']) == convert_paths(self.all_paths)

    def test_exclude_custom_dockerfile(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If we're using a custom Dockerfile, make sure that's not\n        excluded.\n        "
        assert self.exclude(['*'], dockerfile='Dockerfile.alt') == {'Dockerfile.alt', '.dockerignore'}
        assert self.exclude(['*'], dockerfile='foo/Dockerfile3') == convert_paths({'foo/Dockerfile3', '.dockerignore'})
        assert self.exclude(['*'], dockerfile='./foo/Dockerfile3') == convert_paths({'foo/Dockerfile3', '.dockerignore'})

    def test_exclude_dockerfile_child(self):
        if False:
            while True:
                i = 10
        includes = self.exclude(['foo/'], dockerfile='foo/Dockerfile3')
        assert convert_path('foo/Dockerfile3') in includes
        assert convert_path('foo/a.py') not in includes

    def test_single_filename(self):
        if False:
            print('Hello World!')
        assert self.exclude(['a.py']) == convert_paths(self.all_paths - {'a.py'})

    def test_single_filename_leading_dot_slash(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['./a.py']) == convert_paths(self.all_paths - {'a.py'})

    def test_single_filename_trailing_slash(self):
        if False:
            while True:
                i = 10
        assert self.exclude(['a.py/']) == convert_paths(self.all_paths - {'a.py'})

    def test_wildcard_filename_start(self):
        if False:
            return 10
        assert self.exclude(['*.py']) == convert_paths(self.all_paths - {'a.py', 'b.py', 'cde.py'})

    def test_wildcard_with_exception(self):
        if False:
            print('Hello World!')
        assert self.exclude(['*.py', '!b.py']) == convert_paths(self.all_paths - {'a.py', 'cde.py'})

    def test_wildcard_with_wildcard_exception(self):
        if False:
            print('Hello World!')
        assert self.exclude(['*.*', '!*.go']) == convert_paths(self.all_paths - {'a.py', 'b.py', 'cde.py', 'Dockerfile.alt'})

    def test_wildcard_filename_end(self):
        if False:
            i = 10
            return i + 15
        assert self.exclude(['a.*']) == convert_paths(self.all_paths - {'a.py', 'a.go'})

    def test_question_mark(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['?.py']) == convert_paths(self.all_paths - {'a.py', 'b.py'})

    def test_single_subdir_single_filename(self):
        if False:
            while True:
                i = 10
        assert self.exclude(['foo/a.py']) == convert_paths(self.all_paths - {'foo/a.py'})

    def test_single_subdir_single_filename_leading_slash(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['/foo/a.py']) == convert_paths(self.all_paths - {'foo/a.py'})

    def test_exclude_include_absolute_path(self):
        if False:
            print('Hello World!')
        base = make_tree([], ['a.py', 'b.py'])
        assert exclude_paths(base, ['/*', '!/*.py']) == {'a.py', 'b.py'}

    def test_single_subdir_with_path_traversal(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['foo/whoops/../a.py']) == convert_paths(self.all_paths - {'foo/a.py'})

    def test_single_subdir_wildcard_filename(self):
        if False:
            print('Hello World!')
        assert self.exclude(['foo/*.py']) == convert_paths(self.all_paths - {'foo/a.py', 'foo/b.py'})

    def test_wildcard_subdir_single_filename(self):
        if False:
            return 10
        assert self.exclude(['*/a.py']) == convert_paths(self.all_paths - {'foo/a.py', 'bar/a.py'})

    def test_wildcard_subdir_wildcard_filename(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['*/*.py']) == convert_paths(self.all_paths - {'foo/a.py', 'foo/b.py', 'bar/a.py'})

    def test_directory(self):
        if False:
            i = 10
            return i + 15
        assert self.exclude(['foo']) == convert_paths(self.all_paths - {'foo', 'foo/a.py', 'foo/b.py', 'foo/bar', 'foo/bar/a.py', 'foo/Dockerfile3'})

    def test_directory_with_trailing_slash(self):
        if False:
            print('Hello World!')
        assert self.exclude(['foo']) == convert_paths(self.all_paths - {'foo', 'foo/a.py', 'foo/b.py', 'foo/bar', 'foo/bar/a.py', 'foo/Dockerfile3'})

    def test_directory_with_single_exception(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['foo', '!foo/bar/a.py']) == convert_paths(self.all_paths - {'foo/a.py', 'foo/b.py', 'foo', 'foo/bar', 'foo/Dockerfile3'})

    def test_directory_with_subdir_exception(self):
        if False:
            return 10
        assert self.exclude(['foo', '!foo/bar']) == convert_paths(self.all_paths - {'foo/a.py', 'foo/b.py', 'foo', 'foo/Dockerfile3'})

    @pytest.mark.skipif(not IS_WINDOWS_PLATFORM, reason='Backslash patterns only on Windows')
    def test_directory_with_subdir_exception_win32_pathsep(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['foo', '!foo\\bar']) == convert_paths(self.all_paths - {'foo/a.py', 'foo/b.py', 'foo', 'foo/Dockerfile3'})

    def test_directory_with_wildcard_exception(self):
        if False:
            print('Hello World!')
        assert self.exclude(['foo', '!foo/*.py']) == convert_paths(self.all_paths - {'foo/bar', 'foo/bar/a.py', 'foo', 'foo/Dockerfile3'})

    def test_subdirectory(self):
        if False:
            return 10
        assert self.exclude(['foo/bar']) == convert_paths(self.all_paths - {'foo/bar', 'foo/bar/a.py'})

    @pytest.mark.skipif(not IS_WINDOWS_PLATFORM, reason='Backslash patterns only on Windows')
    def test_subdirectory_win32_pathsep(self):
        if False:
            return 10
        assert self.exclude(['foo\\bar']) == convert_paths(self.all_paths - {'foo/bar', 'foo/bar/a.py'})

    def test_double_wildcard(self):
        if False:
            return 10
        assert self.exclude(['**/a.py']) == convert_paths(self.all_paths - {'a.py', 'foo/a.py', 'foo/bar/a.py', 'bar/a.py'})
        assert self.exclude(['foo/**/bar']) == convert_paths(self.all_paths - {'foo/bar', 'foo/bar/a.py'})

    def test_single_and_double_wildcard(self):
        if False:
            while True:
                i = 10
        assert self.exclude(['**/target/*/*']) == convert_paths(self.all_paths - {'target/subdir/file.txt', 'subdir/target/subdir/file.txt', 'subdir/subdir2/target/subdir/file.txt'})

    def test_trailing_double_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.exclude(['subdir/**']) == convert_paths(self.all_paths - {'subdir/file.txt', 'subdir/target/file.txt', 'subdir/target/subdir/file.txt', 'subdir/subdir2/file.txt', 'subdir/subdir2/target/file.txt', 'subdir/subdir2/target/subdir/file.txt', 'subdir/target', 'subdir/target/subdir', 'subdir/subdir2', 'subdir/subdir2/target', 'subdir/subdir2/target/subdir'})

    def test_double_wildcard_with_exception(self):
        if False:
            return 10
        assert self.exclude(['**', '!bar', '!foo/bar']) == convert_paths({'foo/bar', 'foo/bar/a.py', 'bar', 'bar/a.py', 'Dockerfile', '.dockerignore'})

    def test_include_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        base = make_tree(['a'], ['a/b.py'])
        assert exclude_paths(base, ['*', '!*/b.py']) == set()

    def test_last_line_precedence(self):
        if False:
            while True:
                i = 10
        base = make_tree([], ['garbage.md', 'trash.md', 'README.md', 'README-bis.md', 'README-secret.md'])
        assert exclude_paths(base, ['*.md', '!README*.md', 'README-secret.md']) == {'README.md', 'README-bis.md'}

    def test_parent_directory(self):
        if False:
            return 10
        base = make_tree([], ['a.py', 'b.py', 'c.py'])
        assert exclude_paths(base, ['../a.py', '/../b.py']) == {'c.py'}

class TarTest(unittest.TestCase):

    def test_tar_with_excludes(self):
        if False:
            return 10
        dirs = ['foo', 'foo/bar', 'bar']
        files = ['Dockerfile', 'Dockerfile.alt', '.dockerignore', 'a.py', 'a.go', 'b.py', 'cde.py', 'foo/a.py', 'foo/b.py', 'foo/bar/a.py', 'bar/a.py']
        exclude = ['*.py', '!b.py', '!a.go', 'foo', 'Dockerfile*', '.dockerignore']
        expected_names = {'Dockerfile', '.dockerignore', 'a.go', 'b.py', 'bar', 'bar/a.py'}
        base = make_tree(dirs, files)
        self.addCleanup(shutil.rmtree, base)
        with tar(base, exclude=exclude) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == sorted(expected_names)

    def test_tar_with_empty_directory(self):
        if False:
            i = 10
            return i + 15
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        for d in ['foo', 'bar']:
            os.makedirs(os.path.join(base, d))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == ['bar', 'foo']

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM or os.geteuid() == 0, reason='root user always has access ; no chmod on Windows')
    def test_tar_with_inaccessible_file(self):
        if False:
            for i in range(10):
                print('nop')
        base = tempfile.mkdtemp()
        full_path = os.path.join(base, 'foo')
        self.addCleanup(shutil.rmtree, base)
        with open(full_path, 'w') as f:
            f.write('content')
        os.chmod(full_path, 146)
        with pytest.raises(IOError) as ei:
            tar(base)
        assert f'Can not read file in context: {full_path}' in ei.exconly()

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM, reason='No symlinks on Windows')
    def test_tar_with_file_symlinks(self):
        if False:
            while True:
                i = 10
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        with open(os.path.join(base, 'foo'), 'w') as f:
            f.write('content')
        os.makedirs(os.path.join(base, 'bar'))
        os.symlink('../foo', os.path.join(base, 'bar/foo'))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == ['bar', 'bar/foo', 'foo']

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM, reason='No symlinks on Windows')
    def test_tar_with_directory_symlinks(self):
        if False:
            print('Hello World!')
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        for d in ['foo', 'bar']:
            os.makedirs(os.path.join(base, d))
        os.symlink('../foo', os.path.join(base, 'bar/foo'))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == ['bar', 'bar/foo', 'foo']

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM, reason='No symlinks on Windows')
    def test_tar_with_broken_symlinks(self):
        if False:
            i = 10
            return i + 15
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        for d in ['foo', 'bar']:
            os.makedirs(os.path.join(base, d))
        os.symlink('../baz', os.path.join(base, 'bar/foo'))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == ['bar', 'bar/foo', 'foo']

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM, reason='No UNIX sockets on Win32')
    def test_tar_socket_file(self):
        if False:
            return 10
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        for d in ['foo', 'bar']:
            os.makedirs(os.path.join(base, d))
        sock = socket.socket(socket.AF_UNIX)
        self.addCleanup(sock.close)
        sock.bind(os.path.join(base, 'test.sock'))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert sorted(tar_data.getnames()) == ['bar', 'foo']

    def tar_test_negative_mtime_bug(self):
        if False:
            i = 10
            return i + 15
        base = tempfile.mkdtemp()
        filename = os.path.join(base, 'th.txt')
        self.addCleanup(shutil.rmtree, base)
        with open(filename, 'w') as f:
            f.write('Invisible Full Moon')
        os.utime(filename, (12345, -3600.0))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            assert tar_data.getnames() == ['th.txt']
            assert tar_data.getmember('th.txt').mtime == -3600

    @pytest.mark.skipif(IS_WINDOWS_PLATFORM, reason='No symlinks on Windows')
    def test_tar_directory_link(self):
        if False:
            for i in range(10):
                print('nop')
        dirs = ['a', 'b', 'a/c']
        files = ['a/hello.py', 'b/utils.py', 'a/c/descend.py']
        base = make_tree(dirs, files)
        self.addCleanup(shutil.rmtree, base)
        os.symlink(os.path.join(base, 'b'), os.path.join(base, 'a/c/b'))
        with tar(base) as archive:
            tar_data = tarfile.open(fileobj=archive)
            names = tar_data.getnames()
            for member in dirs + files:
                assert member in names
            assert 'a/c/b' in names
            assert 'a/c/b/utils.py' not in names