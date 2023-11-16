import itertools
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Union
from unittest import mock
import pytest
from pip._internal.utils import temp_dir
from pip._internal.utils.misc import ensure_dir
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory, _Default, _default, global_tempdir_manager, tempdir_registry

@pytest.mark.skipif("sys.platform == 'win32'")
def test_symlinked_path() -> None:
    if False:
        i = 10
        return i + 15
    with TempDirectory() as tmp_dir:
        assert os.path.exists(tmp_dir.path)
        alt_tmp_dir = tempfile.mkdtemp(prefix='pip-test-')
        assert os.path.dirname(tmp_dir.path) == os.path.dirname(os.path.realpath(alt_tmp_dir))
        if os.path.realpath(alt_tmp_dir) != os.path.abspath(alt_tmp_dir):
            assert os.path.dirname(tmp_dir.path) != os.path.dirname(alt_tmp_dir)
        else:
            assert os.path.dirname(tmp_dir.path) == os.path.dirname(alt_tmp_dir)
        os.rmdir(tmp_dir.path)
        assert not os.path.exists(tmp_dir.path)

def test_deletes_readonly_files() -> None:
    if False:
        return 10

    def create_file(*args: str) -> None:
        if False:
            while True:
                i = 10
        fpath = os.path.join(*args)
        ensure_dir(os.path.dirname(fpath))
        with open(fpath, 'w') as f:
            f.write('Holla!')

    def readonly_file(*args: str) -> None:
        if False:
            return 10
        fpath = os.path.join(*args)
        os.chmod(fpath, stat.S_IREAD)
    with TempDirectory() as tmp_dir:
        create_file(tmp_dir.path, 'normal-file')
        create_file(tmp_dir.path, 'readonly-file')
        readonly_file(tmp_dir.path, 'readonly-file')
        create_file(tmp_dir.path, 'subfolder', 'normal-file')
        create_file(tmp_dir.path, 'subfolder', 'readonly-file')
        readonly_file(tmp_dir.path, 'subfolder', 'readonly-file')

def test_path_access_after_context_raises() -> None:
    if False:
        return 10
    with TempDirectory() as tmp_dir:
        path = tmp_dir.path
    with pytest.raises(AssertionError) as e:
        _ = tmp_dir.path
    assert path in str(e.value)

def test_path_access_after_clean_raises() -> None:
    if False:
        while True:
            i = 10
    tmp_dir = TempDirectory()
    path = tmp_dir.path
    tmp_dir.cleanup()
    with pytest.raises(AssertionError) as e:
        _ = tmp_dir.path
    assert path in str(e.value)

def test_create_and_cleanup_work() -> None:
    if False:
        i = 10
        return i + 15
    tmp_dir = TempDirectory()
    created_path = tmp_dir.path
    assert tmp_dir.path is not None
    assert os.path.exists(created_path)
    tmp_dir.cleanup()
    assert not os.path.exists(created_path)

@pytest.mark.parametrize('name', ['ABC', 'ABC.dist-info', '_+-', '_package', 'A......B', 'AB', 'A', '2'])
def test_adjacent_directory_names(name: str) -> None:
    if False:
        return 10

    def names() -> Iterator[str]:
        if False:
            return 10
        return AdjacentTempDirectory._generate_names(name)
    chars = AdjacentTempDirectory.LEADING_CHARS
    some_names = list(itertools.islice(names(), 1000))
    assert len(some_names) == 1000
    assert name not in some_names
    if len(name) > 2:
        assert len(some_names) > 0.9 * len(set(some_names))
        same_len = list(itertools.takewhile(lambda x: len(x) == len(name), some_names))
        assert len(same_len) > 10
        expected_names = ['~' + name[1:]]
        expected_names.extend(('~' + c + name[2:] for c in chars))
        for (x, y) in zip(some_names, expected_names):
            assert x == y
    else:
        assert min((len(x) for x in some_names)) > 1
        assert len(some_names) == len(set(some_names))
        if len(name) == 2:
            assert all((x.endswith(name) for x in some_names[1:]))
        else:
            assert all((x.endswith(name) for x in some_names))

@pytest.mark.parametrize('name', ['A', 'ABC', 'ABC.dist-info', '_+-', '_package'])
def test_adjacent_directory_exists(name: str, tmpdir: Path) -> None:
    if False:
        print('Hello World!')
    (block_name, expect_name) = itertools.islice(AdjacentTempDirectory._generate_names(name), 2)
    original = os.path.join(tmpdir, name)
    blocker = os.path.join(tmpdir, block_name)
    ensure_dir(original)
    ensure_dir(blocker)
    with AdjacentTempDirectory(original) as atmp_dir:
        assert expect_name == os.path.split(atmp_dir.path)[1]

def test_adjacent_directory_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        return 10
    name = 'ABC'

    def raising_mkdir(*args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        raise OSError('Unknown OSError')
    with TempDirectory() as tmp_dir:
        original = os.path.join(tmp_dir.path, name)
        ensure_dir(original)
        monkeypatch.setattr('os.mkdir', raising_mkdir)
        with pytest.raises(OSError):
            with AdjacentTempDirectory(original):
                pass

def test_global_tempdir_manager() -> None:
    if False:
        return 10
    with global_tempdir_manager():
        d = TempDirectory(globally_managed=True)
        path = d.path
        assert os.path.exists(path)
    assert not os.path.exists(path)

def test_tempdirectory_asserts_global_tempdir(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        return 10
    monkeypatch.setattr(temp_dir, '_tempdir_manager', None)
    with pytest.raises(AssertionError):
        TempDirectory(globally_managed=True)
deleted_kind = 'deleted'
not_deleted_kind = 'not-deleted'

@pytest.mark.parametrize('delete,kind,exists', [(None, deleted_kind, False), (_default, deleted_kind, False), (True, deleted_kind, False), (False, deleted_kind, True), (None, not_deleted_kind, True), (_default, not_deleted_kind, True), (True, not_deleted_kind, False), (False, not_deleted_kind, True), (None, 'unspecified', False), (_default, 'unspecified', False), (True, 'unspecified', False), (False, 'unspecified', True)])
def test_tempdir_registry(delete: Union[bool, _Default], kind: str, exists: bool) -> None:
    if False:
        i = 10
        return i + 15
    with tempdir_registry() as registry:
        registry.set_delete(deleted_kind, True)
        registry.set_delete(not_deleted_kind, False)
        with TempDirectory(delete=delete, kind=kind) as d:
            path = d.path
            assert os.path.exists(path)
        assert os.path.exists(path) == exists

@pytest.mark.parametrize('delete,exists', [(_default, True), (None, False)])
def test_temp_dir_does_not_delete_explicit_paths_by_default(tmpdir: Path, delete: Optional[_Default], exists: bool) -> None:
    if False:
        print('Hello World!')
    p = tmpdir / 'example'
    p.mkdir()
    path = os.fspath(p)
    with tempdir_registry() as registry:
        registry.set_delete(deleted_kind, True)
        with TempDirectory(path=path, delete=delete, kind=deleted_kind) as d:
            assert str(d.path) == path
            assert os.path.exists(path)
        assert os.path.exists(path) == exists

@pytest.mark.parametrize('should_delete', [True, False])
def test_tempdir_registry_lazy(should_delete: bool) -> None:
    if False:
        return 10
    '\n    Test the registry entry can be updated after a temp dir is created,\n    to change whether a kind should be deleted or not.\n    '
    with tempdir_registry() as registry:
        with TempDirectory(delete=None, kind='test-for-lazy') as d:
            path = d.path
            registry.set_delete('test-for-lazy', should_delete)
            assert os.path.exists(path)
        assert os.path.exists(path) == (not should_delete)

def test_tempdir_cleanup_ignore_errors() -> None:
    if False:
        print('Hello World!')
    os_unlink = os.unlink

    def unlink(name: str, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'bomb' in name:
            raise PermissionError(name)
        else:
            os_unlink(name)
    with mock.patch('os.unlink', unlink):
        with TempDirectory(ignore_cleanup_errors=True) as tmp_dir:
            path = tmp_dir.path
            with open(os.path.join(path, 'bomb'), 'a'):
                pass
    filename = os.path.join(path, 'bomb')
    assert os.path.isfile(filename)
    os.unlink(filename)