import os
import time
import tempfile
import shutil
from pytest import raises
from pathlib import Path
from hscommon.testutil import eq_
from hscommon.plat import ISWINDOWS
from core.fs import File
from core.directories import Directories, DirectoryState, AlreadyThereError, InvalidPathError
from core.exclude import ExcludeList, ExcludeDict

def create_fake_fs(rootpath):
    if False:
        for i in range(10):
            print('nop')
    rootpath = rootpath.joinpath('fs')
    rootpath.mkdir()
    rootpath.joinpath('dir1').mkdir()
    rootpath.joinpath('dir2').mkdir()
    rootpath.joinpath('dir3').mkdir()
    with rootpath.joinpath('file1.test').open('wt') as fp:
        fp.write('1')
    with rootpath.joinpath('file2.test').open('wt') as fp:
        fp.write('12')
    with rootpath.joinpath('file3.test').open('wt') as fp:
        fp.write('123')
    with rootpath.joinpath('dir1', 'file1.test').open('wt') as fp:
        fp.write('1')
    with rootpath.joinpath('dir2', 'file2.test').open('wt') as fp:
        fp.write('12')
    with rootpath.joinpath('dir3', 'file3.test').open('wt') as fp:
        fp.write('123')
    return rootpath
testpath = None

def setup_module(module):
    if False:
        for i in range(10):
            print('nop')
    testpath = Path(tempfile.mkdtemp())
    module.testpath = testpath
    rootpath = testpath.joinpath('onefile')
    rootpath.mkdir()
    with rootpath.joinpath('test.txt').open('wt') as fp:
        fp.write('test_data')
    create_fake_fs(testpath)

def teardown_module(module):
    if False:
        while True:
            i = 10
    shutil.rmtree(str(module.testpath))

def test_empty():
    if False:
        return 10
    d = Directories()
    eq_(len(d), 0)
    assert 'foobar' not in d

def test_add_path():
    if False:
        print('Hello World!')
    d = Directories()
    p = testpath.joinpath('onefile')
    d.add_path(p)
    eq_(1, len(d))
    assert p in d
    assert p.joinpath('foobar') in d
    assert p.parent not in d
    p = testpath.joinpath('fs')
    d.add_path(p)
    eq_(2, len(d))
    assert p in d

def test_add_path_when_path_is_already_there():
    if False:
        for i in range(10):
            print('nop')
    d = Directories()
    p = testpath.joinpath('onefile')
    d.add_path(p)
    with raises(AlreadyThereError):
        d.add_path(p)
    with raises(AlreadyThereError):
        d.add_path(p.joinpath('foobar'))
    eq_(1, len(d))

def test_add_path_containing_paths_already_there():
    if False:
        return 10
    d = Directories()
    d.add_path(testpath.joinpath('onefile'))
    eq_(1, len(d))
    d.add_path(testpath)
    eq_(len(d), 1)
    eq_(d[0], testpath)

def test_add_path_non_latin(tmpdir):
    if False:
        i = 10
        return i + 15
    p = Path(str(tmpdir))
    to_add = p.joinpath('unicode‚')
    os.mkdir(str(to_add))
    d = Directories()
    try:
        d.add_path(to_add)
    except UnicodeDecodeError:
        assert False

def test_del():
    if False:
        i = 10
        return i + 15
    d = Directories()
    d.add_path(testpath.joinpath('onefile'))
    try:
        del d[1]
        assert False
    except IndexError:
        pass
    d.add_path(testpath.joinpath('fs'))
    del d[1]
    eq_(1, len(d))

def test_states():
    if False:
        i = 10
        return i + 15
    d = Directories()
    p = testpath.joinpath('onefile')
    d.add_path(p)
    eq_(DirectoryState.NORMAL, d.get_state(p))
    d.set_state(p, DirectoryState.REFERENCE)
    eq_(DirectoryState.REFERENCE, d.get_state(p))
    eq_(DirectoryState.REFERENCE, d.get_state(p.joinpath('dir1')))
    eq_(1, len(d.states))
    eq_(p, list(d.states.keys())[0])
    eq_(DirectoryState.REFERENCE, d.states[p])

def test_get_state_with_path_not_there():
    if False:
        print('Hello World!')
    d = Directories()
    d.add_path(testpath.joinpath('onefile'))
    eq_(d.get_state(testpath), DirectoryState.NORMAL)

def test_states_overwritten_when_larger_directory_eat_smaller_ones():
    if False:
        for i in range(10):
            print('nop')
    d = Directories()
    p = testpath.joinpath('onefile')
    d.add_path(p)
    d.set_state(p, DirectoryState.EXCLUDED)
    d.add_path(testpath)
    d.set_state(testpath, DirectoryState.REFERENCE)
    eq_(d.get_state(p), DirectoryState.REFERENCE)
    eq_(d.get_state(p.joinpath('dir1')), DirectoryState.REFERENCE)
    eq_(d.get_state(testpath), DirectoryState.REFERENCE)

def test_get_files():
    if False:
        while True:
            i = 10
    d = Directories()
    p = testpath.joinpath('fs')
    d.add_path(p)
    d.set_state(p.joinpath('dir1'), DirectoryState.REFERENCE)
    d.set_state(p.joinpath('dir2'), DirectoryState.EXCLUDED)
    files = list(d.get_files())
    eq_(5, len(files))
    for f in files:
        if f.path.parent == p.joinpath('dir1'):
            assert f.is_ref
        else:
            assert not f.is_ref

def test_get_files_with_folders():
    if False:
        for i in range(10):
            print('nop')

    class FakeFile(File):

        @classmethod
        def can_handle(cls, path):
            if False:
                i = 10
                return i + 15
            return True
    d = Directories()
    p = testpath.joinpath('fs')
    d.add_path(p)
    files = list(d.get_files(fileclasses=[FakeFile]))
    eq_(6, len(files))

def test_get_folders():
    if False:
        print('Hello World!')
    d = Directories()
    p = testpath.joinpath('fs')
    d.add_path(p)
    d.set_state(p.joinpath('dir1'), DirectoryState.REFERENCE)
    d.set_state(p.joinpath('dir2'), DirectoryState.EXCLUDED)
    folders = list(d.get_folders())
    eq_(len(folders), 3)
    ref = [f for f in folders if f.is_ref]
    not_ref = [f for f in folders if not f.is_ref]
    eq_(len(ref), 1)
    eq_(ref[0].path, p.joinpath('dir1'))
    eq_(len(not_ref), 2)
    eq_(ref[0].size, 1)

def test_get_files_with_inherited_exclusion():
    if False:
        while True:
            i = 10
    d = Directories()
    p = testpath.joinpath('onefile')
    d.add_path(p)
    d.set_state(p, DirectoryState.EXCLUDED)
    eq_([], list(d.get_files()))

def test_save_and_load(tmpdir):
    if False:
        i = 10
        return i + 15
    d1 = Directories()
    d2 = Directories()
    p1 = Path(str(tmpdir.join('p1')))
    p1.mkdir()
    p2 = Path(str(tmpdir.join('p2')))
    p2.mkdir()
    d1.add_path(p1)
    d1.add_path(p2)
    d1.set_state(p1, DirectoryState.REFERENCE)
    d1.set_state(p1.joinpath('dir1'), DirectoryState.EXCLUDED)
    tmpxml = str(tmpdir.join('directories_testunit.xml'))
    d1.save_to_file(tmpxml)
    d2.load_from_file(tmpxml)
    eq_(2, len(d2))
    eq_(DirectoryState.REFERENCE, d2.get_state(p1))
    eq_(DirectoryState.EXCLUDED, d2.get_state(p1.joinpath('dir1')))

def test_invalid_path():
    if False:
        print('Hello World!')
    d = Directories()
    p = Path('does_not_exist')
    with raises(InvalidPathError):
        d.add_path(p)
    eq_(0, len(d))

def test_set_state_on_invalid_path():
    if False:
        while True:
            i = 10
    d = Directories()
    try:
        d.set_state(Path('foobar'), DirectoryState.NORMAL)
    except LookupError:
        assert False

def test_load_from_file_with_invalid_path(tmpdir):
    if False:
        while True:
            i = 10
    d1 = Directories()
    d1.add_path(testpath.joinpath('onefile'))
    p = Path(str(tmpdir.join('toremove')))
    p.mkdir()
    d1.add_path(p)
    p.rmdir()
    tmpxml = str(tmpdir.join('directories_testunit.xml'))
    d1.save_to_file(tmpxml)
    d2 = Directories()
    d2.load_from_file(tmpxml)
    eq_(1, len(d2))

def test_unicode_save(tmpdir):
    if False:
        while True:
            i = 10
    d = Directories()
    p1 = Path(str(tmpdir), 'helloé')
    p1.mkdir()
    p1.joinpath('fooé').mkdir()
    d.add_path(p1)
    d.set_state(p1.joinpath('fooé'), DirectoryState.EXCLUDED)
    tmpxml = str(tmpdir.join('directories_testunit.xml'))
    try:
        d.save_to_file(tmpxml)
    except UnicodeDecodeError:
        assert False

def test_get_files_refreshes_its_directories():
    if False:
        return 10
    d = Directories()
    p = testpath.joinpath('fs')
    d.add_path(p)
    files = d.get_files()
    eq_(6, len(list(files)))
    time.sleep(1)
    os.remove(str(p.joinpath('dir1', 'file1.test')))
    files = d.get_files()
    eq_(5, len(list(files)))

def test_get_files_does_not_choke_on_non_existing_directories(tmpdir):
    if False:
        print('Hello World!')
    d = Directories()
    p = Path(str(tmpdir))
    d.add_path(p)
    shutil.rmtree(str(p))
    eq_([], list(d.get_files()))

def test_get_state_returns_excluded_by_default_for_hidden_directories(tmpdir):
    if False:
        return 10
    d = Directories()
    p = Path(str(tmpdir))
    hidden_dir_path = p.joinpath('.foo')
    p.joinpath('.foo').mkdir()
    d.add_path(p)
    eq_(d.get_state(hidden_dir_path), DirectoryState.EXCLUDED)
    d.set_state(hidden_dir_path, DirectoryState.NORMAL)
    eq_(d.get_state(hidden_dir_path), DirectoryState.NORMAL)

def test_default_path_state_override(tmpdir):
    if False:
        i = 10
        return i + 15

    class MyDirectories(Directories):

        def _default_state_for_path(self, path):
            if False:
                print('Hello World!')
            if 'foobar' in path.parts:
                return DirectoryState.EXCLUDED
            return DirectoryState.NORMAL
    d = MyDirectories()
    p1 = Path(str(tmpdir))
    p1.joinpath('foobar').mkdir()
    p1.joinpath('foobar/somefile').touch()
    p1.joinpath('foobaz').mkdir()
    p1.joinpath('foobaz/somefile').touch()
    d.add_path(p1)
    eq_(d.get_state(p1.joinpath('foobaz')), DirectoryState.NORMAL)
    eq_(d.get_state(p1.joinpath('foobar')), DirectoryState.EXCLUDED)
    eq_(len(list(d.get_files())), 1)
    d.set_state(p1.joinpath('foobar'), DirectoryState.NORMAL)
    eq_(d.get_state(p1.joinpath('foobar')), DirectoryState.NORMAL)
    eq_(len(list(d.get_files())), 2)

class TestExcludeList:

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.d = Directories(exclude_list=ExcludeList(union_regex=False))

    def get_files_and_expect_num_result(self, num_result):
        if False:
            for i in range(10):
                print('nop')
        'Calls get_files(), get the filenames only, print for debugging.\n        num_result is how many files are expected as a result.'
        print(f'EXCLUDED REGEX: paths {self.d._exclude_list.compiled_paths} files: {self.d._exclude_list.compiled_files} all: {self.d._exclude_list.compiled}')
        files = list(self.d.get_files())
        files = [file.name for file in files]
        print(f'FINAL FILES {files}')
        eq_(len(files), num_result)
        return files

    def test_exclude_recycle_bin_by_default(self, tmpdir):
        if False:
            print('Hello World!')
        regex = '^.*Recycle\\.Bin$'
        self.d._exclude_list.add(regex)
        self.d._exclude_list.mark(regex)
        p1 = Path(str(tmpdir))
        p1.joinpath('$Recycle.Bin').mkdir()
        p1.joinpath('$Recycle.Bin', 'subdir').mkdir()
        self.d.add_path(p1)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin')), DirectoryState.EXCLUDED)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.EXCLUDED)
        self.d.set_state(p1.joinpath('$Recycle.Bin', 'subdir'), DirectoryState.NORMAL)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)

    def test_exclude_refined(self, tmpdir):
        if False:
            for i in range(10):
                print('nop')
        regex1 = '^\\$Recycle\\.Bin$'
        self.d._exclude_list.add(regex1)
        self.d._exclude_list.mark(regex1)
        p1 = Path(str(tmpdir))
        p1.joinpath('$Recycle.Bin').mkdir()
        p1.joinpath('$Recycle.Bin', 'somefile.png').touch()
        p1.joinpath('$Recycle.Bin', 'some_unwanted_file.jpg').touch()
        p1.joinpath('$Recycle.Bin', 'subdir').mkdir()
        p1.joinpath('$Recycle.Bin', 'subdir', 'somesubdirfile.png').touch()
        p1.joinpath('$Recycle.Bin', 'subdir', 'unwanted_subdirfile.gif').touch()
        p1.joinpath('$Recycle.Bin', 'subdar').mkdir()
        p1.joinpath('$Recycle.Bin', 'subdar', 'somesubdarfile.jpeg').touch()
        p1.joinpath('$Recycle.Bin', 'subdar', 'unwanted_subdarfile.png').touch()
        self.d.add_path(p1.joinpath('$Recycle.Bin'))
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin')), DirectoryState.EXCLUDED)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.EXCLUDED)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdar')), DirectoryState.EXCLUDED)
        self.d.set_state(p1.joinpath('$Recycle.Bin', 'subdir'), DirectoryState.NORMAL)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin')), DirectoryState.EXCLUDED)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdar')), DirectoryState.EXCLUDED)
        files = self.get_files_and_expect_num_result(2)
        assert 'somefile.png' not in files
        assert 'some_unwanted_file.jpg' not in files
        assert 'somesubdarfile.jpeg' not in files
        assert 'unwanted_subdarfile.png' not in files
        assert 'somesubdirfile.png' in files
        assert 'unwanted_subdirfile.gif' in files
        self.d.set_state(p1.joinpath('$Recycle.Bin'), DirectoryState.NORMAL)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdar')), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(6)
        assert 'somefile.png' in files
        assert 'some_unwanted_file.jpg' in files
        regex2 = '.*unwanted.*'
        self.d._exclude_list.add(regex2)
        self.d._exclude_list.mark(regex2)
        files = self.get_files_and_expect_num_result(3)
        assert 'somefile.png' in files
        assert 'some_unwanted_file.jpg' not in files
        assert 'unwanted_subdirfile.gif' not in files
        assert 'unwanted_subdarfile.png' not in files
        if ISWINDOWS:
            regex3 = '.*Recycle\\.Bin\\\\.*unwanted.*subdirfile.*'
        else:
            regex3 = '.*Recycle\\.Bin\\/.*unwanted.*subdirfile.*'
        self.d._exclude_list.rename(regex2, regex3)
        assert self.d._exclude_list.error(regex3) is None
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(5)
        assert 'unwanted_subdirfile.gif' not in files
        assert 'unwanted_subdarfile.png' in files
        regex4 = '.*subdir$'
        self.d._exclude_list.rename(regex3, regex4)
        assert self.d._exclude_list.error(regex4) is None
        p1.joinpath('$Recycle.Bin', 'subdar', 'file_ending_with_subdir').touch()
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.EXCLUDED)
        files = self.get_files_and_expect_num_result(4)
        assert 'file_ending_with_subdir' not in files
        assert 'somesubdarfile.jpeg' in files
        assert 'somesubdirfile.png' not in files
        assert 'unwanted_subdirfile.gif' not in files
        self.d.set_state(p1.joinpath('$Recycle.Bin', 'subdir'), DirectoryState.NORMAL)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(6)
        assert 'file_ending_with_subdir' not in files
        assert 'somesubdirfile.png' in files
        assert 'unwanted_subdirfile.gif' in files
        regex5 = '.*subdir.*'
        self.d._exclude_list.rename(regex4, regex5)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)
        p1.joinpath('$Recycle.Bin', 'subdir', 'file_which_shouldnt_match').touch()
        files = self.get_files_and_expect_num_result(5)
        assert 'somesubdirfile.png' not in files
        assert 'unwanted_subdirfile.gif' not in files
        assert 'file_ending_with_subdir' not in files
        assert 'file_which_shouldnt_match' in files
        regex6 = '.*/.*subdir.*/.*'
        if ISWINDOWS:
            regex6 = '.*\\\\.*subdir.*\\\\.*'
        assert os.sep in regex6
        self.d._exclude_list.rename(regex5, regex6)
        self.d._exclude_list.remove(regex1)
        eq_(len(self.d._exclude_list.compiled), 1)
        assert regex1 not in self.d._exclude_list
        assert regex5 not in self.d._exclude_list
        assert self.d._exclude_list.error(regex6) is None
        assert regex6 in self.d._exclude_list
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', 'subdir')), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(5)
        assert 'somesubdirfile.png' not in files
        assert 'unwanted_subdirfile.gif' not in files
        assert 'file_ending_with_subdir' in files
        assert 'file_which_shouldnt_match' not in files

    def test_japanese_unicode(self, tmpdir):
        if False:
            while True:
                i = 10
        p1 = Path(str(tmpdir))
        p1.joinpath('$Recycle.Bin').mkdir()
        p1.joinpath('$Recycle.Bin', 'somerecycledfile.png').touch()
        p1.joinpath('$Recycle.Bin', 'some_unwanted_file.jpg').touch()
        p1.joinpath('$Recycle.Bin', 'subdir').mkdir()
        p1.joinpath('$Recycle.Bin', 'subdir', '過去白濁物語～]_カラー.jpg').touch()
        p1.joinpath('$Recycle.Bin', '思叫物語').mkdir()
        p1.joinpath('$Recycle.Bin', '思叫物語', 'なししろ会う前').touch()
        p1.joinpath('$Recycle.Bin', '思叫物語', '堂～ロ').touch()
        self.d.add_path(p1.joinpath('$Recycle.Bin'))
        regex3 = '.*物語.*'
        self.d._exclude_list.add(regex3)
        self.d._exclude_list.mark(regex3)
        eq_(self.d.get_state(p1.joinpath('$Recycle.Bin', '思叫物語')), DirectoryState.EXCLUDED)
        files = self.get_files_and_expect_num_result(2)
        assert '過去白濁物語～]_カラー.jpg' not in files
        assert 'なししろ会う前' not in files
        assert '堂～ロ' not in files
        regex4 = '.*物語$'
        self.d._exclude_list.rename(regex3, regex4)
        assert self.d._exclude_list.error(regex4) is None
        self.d.set_state(p1.joinpath('$Recycle.Bin', '思叫物語'), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(5)
        assert '過去白濁物語～]_カラー.jpg' in files
        assert 'なししろ会う前' in files
        assert '堂～ロ' in files

    def test_get_state_returns_excluded_for_hidden_directories_and_files(self, tmpdir):
        if False:
            print('Hello World!')
        regex = '^\\..*$'
        self.d._exclude_list.add(regex)
        self.d._exclude_list.mark(regex)
        p1 = Path(str(tmpdir))
        p1.joinpath('foobar').mkdir()
        p1.joinpath('foobar', '.hidden_file.txt').touch()
        p1.joinpath('foobar', '.hidden_dir').mkdir()
        p1.joinpath('foobar', '.hidden_dir', 'foobar.jpg').touch()
        p1.joinpath('foobar', '.hidden_dir', '.hidden_subfile.png').touch()
        self.d.add_path(p1.joinpath('foobar'))
        eq_(self.d.get_state(p1.joinpath('foobar', '.hidden_dir')), DirectoryState.EXCLUDED)
        self.d.set_state(p1.joinpath('foobar', '.hidden_dir'), DirectoryState.NORMAL)
        files = self.get_files_and_expect_num_result(1)
        eq_(len(self.d._exclude_list.compiled_paths), 0)
        eq_(len(self.d._exclude_list.compiled_files), 1)
        assert '.hidden_file.txt' not in files
        assert '.hidden_subfile.png' not in files
        assert 'foobar.jpg' in files

class TestExcludeDict(TestExcludeList):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.d = Directories(exclude_list=ExcludeDict(union_regex=False))

class TestExcludeListunion(TestExcludeList):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.d = Directories(exclude_list=ExcludeList(union_regex=True))

class TestExcludeDictunion(TestExcludeList):

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.d = Directories(exclude_list=ExcludeDict(union_regex=True))