import io
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple
import pytest
from pip._internal.exceptions import InstallationError
from pip._internal.utils.unpacking import is_within_directory, untar_file, unzip_file
from tests.lib import TestData

class TestUnpackArchives:
    """
    test_tar.tgz/test_tar.zip have content as follows engineered to confirm 3
    things:
     1) confirm that reg files, dirs, and symlinks get unpacked
     2) permissions are not preserved (and go by the 022 umask)
     3) reg files with *any* execute perms, get chmod +x

       file.txt         600 regular file
       symlink.txt      777 symlink to file.txt
       script_owner.sh  700 script where owner can execute
       script_group.sh  610 script where group can execute
       script_world.sh  601 script where world can execute
       dir              744 directory
       dir/dirfile      622 regular file
     4) the file contents are extracted correctly (though the content of
        each file isn't currently unique)

    """

    def setup_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.tempdir = tempfile.mkdtemp()
        self.old_mask = os.umask(18)
        self.symlink_expected_mode = None

    def teardown_method(self) -> None:
        if False:
            print('Hello World!')
        os.umask(self.old_mask)
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def mode(self, path: str) -> int:
        if False:
            while True:
                i = 10
        return stat.S_IMODE(os.stat(path).st_mode)

    def confirm_files(self) -> None:
        if False:
            while True:
                i = 10
        for (fname, expected_mode, test, expected_contents) in [('file.txt', 420, os.path.isfile, b'file\n'), ('symlink.txt', 420, os.path.isfile, None), ('script_owner.sh', 493, os.path.isfile, b'file\n'), ('script_group.sh', 493, os.path.isfile, b'file\n'), ('script_world.sh', 493, os.path.isfile, b'file\n'), ('dir', 493, os.path.isdir, None), (os.path.join('dir', 'dirfile'), 420, os.path.isfile, b'')]:
            path = os.path.join(self.tempdir, fname)
            if path.endswith('symlink.txt') and sys.platform == 'win32':
                continue
            assert test(path), path
            if expected_contents is not None:
                with open(path, mode='rb') as f:
                    contents = f.read()
                assert contents == expected_contents, f'fname: {fname}'
            if sys.platform == 'win32':
                continue
            mode = self.mode(path)
            assert mode == expected_mode, f'mode: {mode}, expected mode: {expected_mode}'

    def make_zip_file(self, filename: str, file_list: List[str]) -> str:
        if False:
            while True:
                i = 10
        '\n        Create a zip file for test case\n        '
        test_zip = os.path.join(self.tempdir, filename)
        with zipfile.ZipFile(test_zip, 'w') as myzip:
            for item in file_list:
                myzip.writestr(item, 'file content')
        return test_zip

    def make_tar_file(self, filename: str, file_list: List[str]) -> str:
        if False:
            while True:
                i = 10
        '\n        Create a tar file for test case\n        '
        test_tar = os.path.join(self.tempdir, filename)
        with tarfile.open(test_tar, 'w') as mytar:
            for item in file_list:
                file_tarinfo = tarfile.TarInfo(item)
                mytar.addfile(file_tarinfo, io.BytesIO(b'file content'))
        return test_tar

    def test_unpack_tgz(self, data: TestData) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test unpacking a *.tgz, and setting execute permissions\n        '
        test_file = data.packages.joinpath('test_tar.tgz')
        untar_file(os.fspath(test_file), self.tempdir)
        self.confirm_files()
        file_txt_path = os.path.join(self.tempdir, 'file.txt')
        mtime = time.gmtime(os.stat(file_txt_path).st_mtime)
        assert mtime[0:6] == (2013, 8, 16, 5, 13, 37), mtime

    def test_unpack_zip(self, data: TestData) -> None:
        if False:
            while True:
                i = 10
        '\n        Test unpacking a *.zip, and setting execute permissions\n        '
        test_file = data.packages.joinpath('test_zip.zip')
        unzip_file(os.fspath(test_file), self.tempdir)
        self.confirm_files()

    def test_unpack_zip_failure(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test unpacking a *.zip with file containing .. path\n        and expect exception\n        '
        files = ['regular_file.txt', os.path.join('..', 'outside_file.txt')]
        test_zip = self.make_zip_file('test_zip.zip', files)
        with pytest.raises(InstallationError) as e:
            unzip_file(test_zip, self.tempdir)
        assert 'trying to install outside target directory' in str(e.value)

    def test_unpack_zip_success(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test unpacking a *.zip with regular files,\n        no file will be installed outside target directory after unpack\n        so no exception raised\n        '
        files = ['regular_file1.txt', os.path.join('dir', 'dir_file1.txt'), os.path.join('dir', '..', 'dir_file2.txt')]
        test_zip = self.make_zip_file('test_zip.zip', files)
        unzip_file(test_zip, self.tempdir)

    def test_unpack_tar_failure(self) -> None:
        if False:
            return 10
        '\n        Test unpacking a *.tar with file containing .. path\n        and expect exception\n        '
        files = ['regular_file.txt', os.path.join('..', 'outside_file.txt')]
        test_tar = self.make_tar_file('test_tar.tar', files)
        with pytest.raises(InstallationError) as e:
            untar_file(test_tar, self.tempdir)
        assert 'trying to install outside target directory' in str(e.value)

    def test_unpack_tar_success(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test unpacking a *.tar with regular files,\n        no file will be installed outside target directory after unpack\n        so no exception raised\n        '
        files = ['regular_file1.txt', os.path.join('dir', 'dir_file1.txt'), os.path.join('dir', '..', 'dir_file2.txt')]
        test_tar = self.make_tar_file('test_tar.tar', files)
        untar_file(test_tar, self.tempdir)

def test_unpack_tar_unicode(tmpdir: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    test_tar = tmpdir / 'test.tar'
    with tarfile.open(test_tar, 'w', format=tarfile.PAX_FORMAT, encoding='utf-8') as f:
        metadata = tarfile.TarInfo('dir/åäö_日本語.py')
        f.addfile(metadata, io.BytesIO(b'hello world'))
    output_dir = tmpdir / 'output'
    output_dir.mkdir()
    untar_file(os.fspath(test_tar), str(output_dir))
    output_dir_name = str(output_dir)
    contents = os.listdir(output_dir_name)
    assert 'åäö_日本語.py' in contents

@pytest.mark.parametrize('args, expected', [(('parent/sub', 'parent/'), False), (('parent', 'parent/foo'), True), (('parent/', 'parent/foo/../bar'), True), (('parent/', 'parent/sub'), True), (('parent/', 'parent/../sub'), False)])
def test_is_within_directory(args: Tuple[str, str], expected: bool) -> None:
    if False:
        i = 10
        return i + 15
    result = is_within_directory(*args)
    assert result == expected