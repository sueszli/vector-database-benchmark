import errno
import os
import shutil
import time
from unittest.mock import patch
import pytest
from ... import xattr
from ...chunker import has_seek_hole
from ...constants import *
from ...helpers import EXIT_WARNING
from ...helpers import flags_noatime, flags_normal
from .. import changedir, same_ts_ns
from .. import are_symlinks_supported, are_hardlinks_supported, is_utime_fully_supported, is_birthtime_fully_supported
from ...platformflags import is_darwin, is_win32
from . import RK_ENCRYPTION, requires_hardlinks, cmd, create_test_files, create_regular_file, assert_dirs_equal, _extract_hardlinks_setup, assert_creates_file, generate_archiver_tests
pytest_generate_tests = lambda metafunc: generate_archiver_tests(metafunc, kinds='local,remote,binary')

@pytest.mark.skipif(not are_symlinks_supported(), reason='symlinks not supported')
def test_symlink_extract(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    create_test_files(archiver.input_path)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
        assert os.readlink('input/link1') == 'somewhere'

@pytest.mark.skipif(not are_symlinks_supported() or not are_hardlinks_supported() or is_darwin, reason='symlinks or hardlinks or hardlinked symlinks not supported')
def test_hardlinked_symlinks_extract(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    create_regular_file(archiver.input_path, 'target', size=1024)
    with changedir('input'):
        os.symlink('target', 'symlink1')
        os.link('symlink1', 'symlink2', follow_symlinks=False)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test')
        print(output)
        with changedir('input'):
            assert os.path.exists('target')
            assert os.readlink('symlink1') == 'target'
            assert os.readlink('symlink2') == 'target'
            st1 = os.stat('symlink1', follow_symlinks=False)
            st2 = os.stat('symlink2', follow_symlinks=False)
            assert st1.st_nlink == 2
            assert st2.st_nlink == 2
            assert st1.st_ino == st2.st_ino
            assert st1.st_size == st2.st_size

@pytest.mark.skipif(not is_utime_fully_supported(), reason='cannot properly setup and execute test without utime')
def test_directory_timestamps1(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    create_test_files(archiver.input_path)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert os.path.exists('output/input/dir2/file2')
    sti = os.stat('input/dir2')
    sto = os.stat('output/input/dir2')
    assert same_ts_ns(sti.st_mtime_ns, sto.st_mtime_ns)

@pytest.mark.skipif(not is_utime_fully_supported(), reason='cannot properly setup and execute test without utime')
def test_directory_timestamps2(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    create_test_files(archiver.input_path)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    flist_dir_first = b'input/dir2\ninput/dir2/file2\n'
    cmd(archiver, 'create', '--paths-from-stdin', 'test', input=flist_dir_first)
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert os.path.exists('output/input/dir2/file2')
    sti = os.stat('input/dir2')
    sto = os.stat('output/input/dir2')
    assert same_ts_ns(sti.st_mtime_ns, sto.st_mtime_ns)

@pytest.mark.skipif(not is_utime_fully_supported(), reason='cannot properly setup and execute test without utime')
def test_directory_timestamps3(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    create_test_files(archiver.input_path)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    flist_file_first = b'input/dir2/file2\ninput/dir2\n'
    cmd(archiver, 'create', '--paths-from-stdin', 'test', input=flist_file_first)
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert os.path.exists('output/input/dir2/file2')
    sti = os.stat('input/dir2')
    sto = os.stat('output/input/dir2')
    assert same_ts_ns(sti.st_mtime_ns, sto.st_mtime_ns)

@pytest.mark.skipif(not is_utime_fully_supported(), reason='cannot properly setup and execute test without utime')
def test_atime(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)

    def has_noatime(some_file):
        if False:
            while True:
                i = 10
        atime_before = os.stat(some_file).st_atime_ns
        try:
            with open(os.open(some_file, flags_noatime)) as file:
                file.read()
        except PermissionError:
            return False
        else:
            atime_after = os.stat(some_file).st_atime_ns
            noatime_used = flags_noatime != flags_normal
            return noatime_used and atime_before == atime_after
    create_test_files(archiver.input_path)
    (atime, mtime) = (123456780, 234567890)
    have_noatime = has_noatime('input/file1')
    os.utime('input/file1', (atime, mtime))
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', '--atime', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    sti = os.stat('input/file1')
    sto = os.stat('output/input/file1')
    assert same_ts_ns(sti.st_mtime_ns, sto.st_mtime_ns)
    assert same_ts_ns(sto.st_mtime_ns, mtime * 1000000000.0)
    if have_noatime:
        assert same_ts_ns(sti.st_atime_ns, sto.st_atime_ns)
        assert same_ts_ns(sto.st_atime_ns, atime * 1000000000.0)
    else:
        assert same_ts_ns(sto.st_atime_ns, atime * 1000000000.0)

@pytest.mark.skipif(not is_utime_fully_supported(), reason='cannot setup and execute test without utime')
@pytest.mark.skipif(not is_birthtime_fully_supported(), reason='cannot setup and execute test without birthtime')
def test_birthtime(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    create_test_files(archiver.input_path)
    (birthtime, mtime, atime) = (946598400, 946684800, 946771200)
    os.utime('input/file1', (atime, birthtime))
    os.utime('input/file1', (atime, mtime))
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    sti = os.stat('input/file1')
    sto = os.stat('output/input/file1')
    assert same_ts_ns(sti.st_birthtime * 1000000000.0, sto.st_birthtime * 1000000000.0)
    assert same_ts_ns(sto.st_birthtime * 1000000000.0, birthtime * 1000000000.0)
    assert same_ts_ns(sti.st_mtime_ns, sto.st_mtime_ns)
    assert same_ts_ns(sto.st_mtime_ns, mtime * 1000000000.0)

@pytest.mark.skipif(is_win32, reason='frequent test failures on github CI on win32')
def test_sparse_file(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)

    def is_sparse(fn, total_size, hole_size):
        if False:
            return 10
        st = os.stat(fn)
        assert st.st_size == total_size
        sparse = True
        if sparse and hasattr(st, 'st_blocks') and (st.st_blocks * 512 >= st.st_size):
            sparse = False
        if sparse and has_seek_hole:
            with open(fn, 'rb') as fd:
                try:
                    if fd.seek(0, os.SEEK_HOLE) != 0:
                        sparse = False
                    if fd.seek(0, os.SEEK_DATA) != hole_size:
                        sparse = False
                except OSError:
                    sparse = False
        return sparse
    filename = os.path.join(archiver.input_path, 'sparse')
    content = b'foobar'
    hole_size = 5 * (1 << CHUNK_MAX_EXP)
    total_size = hole_size + len(content) + hole_size
    with open(filename, 'wb') as fd:
        fd.seek(hole_size, 1)
        fd.write(content)
        fd.seek(hole_size, 1)
        pos = fd.tell()
        fd.truncate(pos)
    sparse_support = is_sparse(filename, total_size, hole_size)
    if sparse_support:
        cmd(archiver, 'rcreate', RK_ENCRYPTION)
        cmd(archiver, 'create', 'test', 'input')
        with changedir(archiver.output_path):
            cmd(archiver, 'extract', 'test', '--sparse')
        assert_dirs_equal('input', 'output/input')
        filename = os.path.join(archiver.output_path, 'input', 'sparse')
        with open(filename, 'rb') as fd:
            assert fd.read(hole_size) == b'\x00' * hole_size
            assert fd.read(len(content)) == content
            assert fd.read(hole_size) == b'\x00' * hole_size
        assert is_sparse(filename, total_size, hole_size)

def test_unusual_filenames(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    filenames = ['normal', 'with some blanks', '(with_parens)']
    for filename in filenames:
        filename = os.path.join(archiver.input_path, filename)
        with open(filename, 'wb'):
            pass
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    for filename in filenames:
        with changedir('output'):
            cmd(archiver, 'extract', 'test', os.path.join('input', filename))
        assert os.path.exists(os.path.join('output', 'input', filename))

def test_strip_components(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'dir/file')
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--strip-components', '3')
        assert not os.path.exists('file')
        with assert_creates_file('file'):
            cmd(archiver, 'extract', 'test', '--strip-components', '2')
        with assert_creates_file('dir/file'):
            cmd(archiver, 'extract', 'test', '--strip-components', '1')
        with assert_creates_file('input/dir/file'):
            cmd(archiver, 'extract', 'test', '--strip-components', '0')

@requires_hardlinks
def test_extract_hardlinks1(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    _extract_hardlinks_setup(archiver)
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
        assert os.stat('input/source').st_nlink == 4
        assert os.stat('input/abba').st_nlink == 4
        assert os.stat('input/dir1/hardlink').st_nlink == 4
        assert os.stat('input/dir1/subdir/hardlink').st_nlink == 4
        assert open('input/dir1/subdir/hardlink', 'rb').read() == b'123456'

@requires_hardlinks
def test_extract_hardlinks2(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    _extract_hardlinks_setup(archiver)
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--strip-components', '2')
        assert os.stat('hardlink').st_nlink == 2
        assert os.stat('subdir/hardlink').st_nlink == 2
        assert open('subdir/hardlink', 'rb').read() == b'123456'
        assert os.stat('aaaa').st_nlink == 2
        assert os.stat('source2').st_nlink == 2
    with changedir('output'):
        cmd(archiver, 'extract', 'test', 'input/dir1')
        assert os.stat('input/dir1/hardlink').st_nlink == 2
        assert os.stat('input/dir1/subdir/hardlink').st_nlink == 2
        assert open('input/dir1/subdir/hardlink', 'rb').read() == b'123456'
        assert os.stat('input/dir1/aaaa').st_nlink == 2
        assert os.stat('input/dir1/source2').st_nlink == 2

@requires_hardlinks
def test_extract_hardlinks_twice(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    path_a = os.path.join(archiver.input_path, 'a')
    path_b = os.path.join(archiver.input_path, 'b')
    os.mkdir(path_a)
    os.mkdir(path_b)
    hl_a = os.path.join(path_a, 'hardlink')
    hl_b = os.path.join(path_b, 'hardlink')
    create_regular_file(archiver.input_path, hl_a, contents=b'123456')
    os.link(hl_a, hl_b)
    cmd(archiver, 'rcreate', '--encryption=none')
    cmd(archiver, 'create', 'test', 'input', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
        assert os.stat('input/a/hardlink').st_nlink == 2
        assert os.stat('input/b/hardlink').st_nlink == 2

def test_extract_include_exclude(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file2', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file3', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file4', size=1024 * 80)
    cmd(archiver, 'create', '--exclude=input/file4', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', 'input/file1')
    assert sorted(os.listdir('output/input')) == ['file1']
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude=input/file2')
    assert sorted(os.listdir('output/input')) == ['file1', 'file3']
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude-from=' + archiver.exclude_file_path)
    assert sorted(os.listdir('output/input')) == ['file1', 'file3']

def test_extract_include_exclude_regex(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file2', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file3', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file4', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file333', size=1024 * 80)
    cmd(archiver, 'create', '--exclude=re:input/file4$', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2', 'file3', 'file333']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude=re:file3+')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude=input/file2', '--exclude=re:file[01]')
    assert sorted(os.listdir('output/input')) == ['file3', 'file333']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude-from=' + archiver.exclude_file_path, '--exclude=re:file1', '--exclude=re:file(\\d)\\1\\1$')
    assert sorted(os.listdir('output/input')) == ['file3']

def test_extract_include_exclude_regex_from_file(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file2', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file3', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file4', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file333', size=1024 * 80)
    with open(archiver.exclude_file_path, 'wb') as fd:
        fd.write(b're:input/file4$\n')
        fd.write(b'fm:*file3*\n')
    cmd(archiver, 'create', '--exclude-from=' + archiver.exclude_file_path, 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2']
    shutil.rmtree('output/input')
    with open(archiver.exclude_file_path, 'wb') as fd:
        fd.write(b're:file3+\n')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude-from=' + archiver.exclude_file_path)
    assert sorted(os.listdir('output/input')) == ['file1', 'file2']
    shutil.rmtree('output/input')
    with open(archiver.exclude_file_path, 'wb') as fd:
        fd.write(b're:file(\\d)\\1\\1$\n')
        fd.write(b'fm:nothingwillmatchthis\n')
        fd.write(b'*/file1\n')
        fd.write(b're:file2$\n')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--exclude-from=' + archiver.exclude_file_path)
    assert sorted(os.listdir('output/input')) == []

def test_extract_with_pattern(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file2', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file3', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file4', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file333', size=1024 * 80)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', 're:.*')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2', 'file3', 'file333', 'file4']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', '--exclude=re:file[34]$', 'test', 're:file\\d$')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', '--exclude=input/file1', 'test', 're:file[12]$')
    assert sorted(os.listdir('output/input')) == ['file2']
    shutil.rmtree('output/input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', 'fm:input/file1', 'fm:*file33*', 'input/file2')
    assert sorted(os.listdir('output/input')) == ['file1', 'file2', 'file333']

def test_extract_list_output(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file', size=1024 * 80)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test')
    assert 'input/file' not in output
    shutil.rmtree('output/input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test', '--info')
    assert 'input/file' not in output
    shutil.rmtree('output/input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test', '--list')
    assert 'input/file' in output
    shutil.rmtree('output/input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test', '--list', '--info')
    assert 'input/file' in output

def test_extract_progress(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file', size=1024 * 80)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        output = cmd(archiver, 'extract', 'test', '--progress')
        assert 'Extracting:' in output

def test_extract_pattern_opt(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file2', size=1024 * 80)
    create_regular_file(archiver.input_path, 'file_important', size=1024 * 80)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', '--pattern=+input/file_important', '--pattern=-input/file*')
    assert sorted(os.listdir('output/input')) == ['file_important']

@pytest.mark.skipif(not xattr.XATTR_FAKEROOT, reason='Linux capabilities test, requires fakeroot >= 1.20.2')
def test_extract_capabilities(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    if archiver.EXE:
        pytest.skip('Skipping binary test due to patch objects')
    fchown = os.fchown

    def patched_fchown(fd, uid, gid):
        if False:
            while True:
                i = 10
        xattr.setxattr(fd, b'security.capability', b'', follow_symlinks=False)
        fchown(fd, uid, gid)
    capabilities = b'\x01\x00\x00\x02\x00 \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    create_regular_file(archiver.input_path, 'file')
    xattr.setxattr(b'input/file', b'security.capability', capabilities)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        with patch.object(os, 'fchown', patched_fchown):
            cmd(archiver, 'extract', 'test')
        assert xattr.getxattr(b'input/file', b'security.capability') == capabilities

@pytest.mark.skipif(not xattr.XATTR_FAKEROOT, reason='xattr not supported on this system, or this version of fakeroot')
def test_extract_xattrs_errors(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    if archiver.EXE:
        pytest.skip('Skipping binary test due to patch objects')

    def patched_setxattr_E2BIG(*args, **kwargs):
        if False:
            while True:
                i = 10
        raise OSError(errno.E2BIG, 'E2BIG')

    def patched_setxattr_ENOTSUP(*args, **kwargs):
        if False:
            print('Hello World!')
        raise OSError(errno.ENOTSUP, 'ENOTSUP')

    def patched_setxattr_EACCES(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise OSError(errno.EACCES, 'EACCES')
    create_regular_file(archiver.input_path, 'file')
    xattr.setxattr(b'input/file', b'user.attribute', b'value')
    cmd(archiver, 'rcreate', '-enone')
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        input_abspath = os.path.abspath('input/file')
        with patch.object(xattr, 'setxattr', patched_setxattr_E2BIG):
            out = cmd(archiver, 'extract', 'test', exit_code=EXIT_WARNING)
            assert 'too big for this filesystem' in out
            assert 'when setting extended attribute user.attribute' in out
        os.remove(input_abspath)
        with patch.object(xattr, 'setxattr', patched_setxattr_ENOTSUP):
            out = cmd(archiver, 'extract', 'test', exit_code=EXIT_WARNING)
            assert 'ENOTSUP' in out
            assert 'when setting extended attribute user.attribute' in out
        os.remove(input_abspath)
        with patch.object(xattr, 'setxattr', patched_setxattr_EACCES):
            out = cmd(archiver, 'extract', 'test', exit_code=EXIT_WARNING)
            assert 'EACCES' in out
            assert 'when setting extended attribute user.attribute' in out
        assert os.path.isfile(input_abspath)

@pytest.mark.skipif(not is_darwin, reason='only for macOS')
def test_extract_xattrs_resourcefork(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    create_regular_file(archiver.input_path, 'file')
    cmd(archiver, 'rcreate', '-enone')
    input_path = os.path.abspath('input/file')
    (xa_key, xa_value) = (b'com.apple.ResourceFork', b'whatshouldbehere')
    xattr.setxattr(input_path.encode(), xa_key, xa_value)
    birthtime_expected = os.stat(input_path).st_birthtime
    mtime_expected = os.stat(input_path).st_mtime_ns
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
        extracted_path = os.path.abspath('input/file')
        birthtime_extracted = os.stat(extracted_path).st_birthtime
        mtime_extracted = os.stat(extracted_path).st_mtime_ns
        xa_value_extracted = xattr.getxattr(extracted_path.encode(), xa_key)
    assert xa_value_extracted == xa_value
    assert -1000 <= (birthtime_extracted - birthtime_expected) * 1000000000.0 <= 1000
    assert mtime_extracted == mtime_expected

def test_overwrite(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    if archiver.EXE:
        pytest.skip('Test_overwrite seems incompatible with fakeroot and/or the binary.')
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'dir2/file2', size=1024 * 80)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    os.mkdir('output/input')
    os.mkdir('output/input/file1')
    os.mkdir('output/input/dir2')
    with changedir('output'):
        cmd(archiver, 'extract', 'test')
    assert_dirs_equal('input', 'output/input')
    os.unlink('output/input/file1')
    os.mkdir('output/input/file1')
    os.mkdir('output/input/file1/dir')
    with changedir('output'):
        cmd(archiver, 'extract', 'test', exit_code=1)

@pytest.mark.skipif(not xattr.XATTR_FAKEROOT, reason='xattr not supported on this system, or this version of fakeroot')
def test_do_not_fail_when_percent_is_in_xattr_name(archivers, request):
    if False:
        i = 10
        return i + 15
    'https://github.com/borgbackup/borg/issues/6063'
    archiver = request.getfixturevalue(archivers)
    if archiver.EXE:
        pytest.skip('Skipping binary test due to patch objects')

    def patched_setxattr_EACCES(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise OSError(errno.EACCES, 'EACCES')
    create_regular_file(archiver.input_path, 'file')
    xattr.setxattr(b'input/file', b'user.attribute%p', b'value')
    cmd(archiver, 'rcreate', '-enone')
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        with patch.object(xattr, 'setxattr', patched_setxattr_EACCES):
            cmd(archiver, 'extract', 'test', exit_code=EXIT_WARNING)

@pytest.mark.skipif(not xattr.XATTR_FAKEROOT, reason='xattr not supported on this system, or this version of fakeroot')
def test_do_not_fail_when_percent_is_in_file_name(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    'https://github.com/borgbackup/borg/issues/6063'
    archiver = request.getfixturevalue(archivers)
    if archiver.EXE:
        pytest.skip('Skipping binary test due to patch objects')

    def patched_setxattr_EACCES(*args, **kwargs):
        if False:
            print('Hello World!')
        raise OSError(errno.EACCES, 'EACCES')
    os.makedirs(os.path.join(archiver.input_path, 'dir%p'))
    xattr.setxattr(b'input/dir%p', b'user.attribute', b'value')
    cmd(archiver, 'rcreate', '-enone')
    cmd(archiver, 'create', 'test', 'input')
    with changedir('output'):
        with patch.object(xattr, 'setxattr', patched_setxattr_EACCES):
            cmd(archiver, 'extract', 'test', exit_code=EXIT_WARNING)

def test_extract_continue(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    (CONTENTS1, CONTENTS2, CONTENTS3) = (b'contents1' * 100, b'contents2' * 200, b'contents3' * 300)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'file1', contents=CONTENTS1)
    create_regular_file(archiver.input_path, 'file2', contents=CONTENTS2)
    create_regular_file(archiver.input_path, 'file3', contents=CONTENTS3)
    cmd(archiver, 'create', 'arch', 'input')
    with changedir('output'):
        cmd(archiver, 'extract', 'arch')
        file1_st = os.stat('input/file1')
        file2_st = os.stat('input/file2')
        os.link('input/file2', 'hardlink-to-keep-inode-f2')
        os.truncate('input/file2', 123)
        file3_st = os.stat('input/file3')
        os.link('input/file3', 'hardlink-to-keep-inode-f3')
        os.remove('input/file3')
    time.sleep(1)
    with changedir('output'):
        cmd(archiver, 'extract', 'arch', '--continue')
        now_file1_st = os.stat('input/file1')
        assert file1_st.st_ino == now_file1_st.st_ino
        assert file1_st.st_mtime_ns == now_file1_st.st_mtime_ns
        new_file2_st = os.stat('input/file2')
        assert file2_st.st_ino != new_file2_st.st_ino
        assert file2_st.st_mtime_ns == new_file2_st.st_mtime_ns
        new_file3_st = os.stat('input/file3')
        assert file3_st.st_ino != new_file3_st.st_ino
        assert file3_st.st_mtime_ns == new_file3_st.st_mtime_ns
        if not is_win32:
            assert file1_st.st_ctime_ns == now_file1_st.st_ctime_ns
            assert file2_st.st_ctime_ns != new_file2_st.st_ctime_ns
            assert file3_st.st_ctime_ns != new_file3_st.st_ctime_ns
        with open('input/file1', 'rb') as f:
            assert f.read() == CONTENTS1
        with open('input/file2', 'rb') as f:
            assert f.read() == CONTENTS2
        with open('input/file3', 'rb') as f:
            assert f.read() == CONTENTS3