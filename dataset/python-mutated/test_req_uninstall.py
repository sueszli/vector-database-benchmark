import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from unittest.mock import Mock
import pytest
import pip._internal.req.req_uninstall
from pip._internal.req.req_uninstall import StashedUninstallPathSet, UninstallPathSet, UninstallPthEntries, compact, compress_for_output_listing, compress_for_rename, uninstallation_paths
from tests.lib import create_file

def mock_permitted(ups: UninstallPathSet, path: str) -> bool:
    if False:
        i = 10
        return i + 15
    return True

def test_uninstallation_paths() -> None:
    if False:
        print('Hello World!')

    class dist:

        def iter_declared_entries(self) -> Optional[Iterator[str]]:
            if False:
                for i in range(10):
                    print('nop')
            yield 'file.py'
            yield 'file.pyc'
            yield 'file.so'
            yield 'nopyc.py'
        location = ''
    d = dist()
    paths = list(uninstallation_paths(d))
    expected = ['file.py', 'file.pyc', 'file.pyo', 'file.so', 'nopyc.py', 'nopyc.pyc', 'nopyc.pyo']
    assert paths == expected
    paths2 = list(uninstallation_paths(d))
    assert paths2 == paths

def test_compressed_listing(tmpdir: Path) -> None:
    if False:
        i = 10
        return i + 15

    def in_tmpdir(paths: List[str]) -> List[str]:
        if False:
            print('Hello World!')
        return [str(os.path.join(tmpdir, path.replace('/', os.path.sep))) for path in paths]
    sample = in_tmpdir(['lib/mypkg.dist-info/METADATA', 'lib/mypkg.dist-info/PKG-INFO', 'lib/mypkg/would_be_removed.txt', 'lib/mypkg/would_be_skipped.skip.txt', 'lib/mypkg/__init__.py', 'lib/mypkg/my_awesome_code.py', 'lib/mypkg/__pycache__/my_awesome_code-magic.pyc', 'lib/mypkg/support/support_file.py', 'lib/mypkg/support/more_support.py', 'lib/mypkg/support/would_be_skipped.skip.py', 'lib/mypkg/support/__pycache__/support_file-magic.pyc', 'lib/random_other_place/file_without_a_dot_pyc', 'bin/mybin'])
    for fname in sample:
        create_file(fname, 'random blub')
    sample = [path for path in sample if '.skip.' not in path]
    expected_remove = in_tmpdir(['bin/mybin', 'lib/mypkg.dist-info/*', 'lib/mypkg/*', 'lib/random_other_place/file_without_a_dot_pyc'])
    expected_skip = in_tmpdir(['lib/mypkg/would_be_skipped.skip.txt', 'lib/mypkg/support/would_be_skipped.skip.py'])
    expected_rename = in_tmpdir(['bin/', 'lib/mypkg.dist-info/', 'lib/mypkg/would_be_removed.txt', 'lib/mypkg/__init__.py', 'lib/mypkg/my_awesome_code.py', 'lib/mypkg/__pycache__/', 'lib/mypkg/support/support_file.py', 'lib/mypkg/support/more_support.py', 'lib/mypkg/support/__pycache__/', 'lib/random_other_place/'])
    (will_remove, will_skip) = compress_for_output_listing(sample)
    will_rename = compress_for_rename(sample)
    assert sorted(expected_skip) == sorted(compact(will_skip))
    assert sorted(expected_remove) == sorted(compact(will_remove))
    assert sorted(expected_rename) == sorted(compact(will_rename))

class TestUninstallPathSet:

    def test_add(self, tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            return 10
        monkeypatch.setattr(pip._internal.req.req_uninstall.UninstallPathSet, '_permitted', mock_permitted)
        file_extant = os.path.normcase(os.path.join(tmpdir, 'foo'))
        file_nonexistent = os.path.normcase(os.path.join(tmpdir, 'nonexistent'))
        with open(file_extant, 'w'):
            pass
        ups = UninstallPathSet(dist=Mock())
        assert ups._paths == set()
        ups.add(file_extant)
        assert ups._paths == {file_extant}
        ups.add(file_nonexistent)
        assert ups._paths == {file_extant}

    def test_add_pth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.setattr(pip._internal.req.req_uninstall.UninstallPathSet, '_permitted', mock_permitted)
        tmpdir = os.path.normcase(tmp_path)
        on_windows = sys.platform == 'win32'
        pth_file = os.path.join(tmpdir, 'foo.pth')
        relative = '../../example'
        if on_windows:
            share = '\\\\example\\share\\'
            share_com = '\\\\example.com\\share\\'
        with open(pth_file, 'w') as f:
            f.writelines([tmpdir, '\n', relative, '\n'])
            if on_windows:
                f.writelines([share, '\n', share_com, '\n'])
        pth = UninstallPthEntries(pth_file)
        pth.add(tmpdir)
        pth.add(relative)
        if on_windows:
            pth.add(share)
            pth.add(share_com)
        if on_windows:
            check = {tmpdir, relative, share, share_com}
        else:
            check = {tmpdir, relative}
        assert pth.entries == check

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_add_symlink(self, tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            return 10
        monkeypatch.setattr(pip._internal.req.req_uninstall.UninstallPathSet, '_permitted', mock_permitted)
        f = os.path.join(tmpdir, 'foo')
        with open(f, 'w'):
            pass
        foo_link = os.path.join(tmpdir, 'foo_link')
        os.symlink(f, foo_link)
        ups = UninstallPathSet(dist=Mock())
        ups.add(foo_link)
        assert ups._paths == {foo_link}

    def test_compact_shorter_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        monkeypatch.setattr(pip._internal.req.req_uninstall.UninstallPathSet, '_permitted', mock_permitted)
        monkeypatch.setattr('os.path.exists', lambda p: True)
        short_path = os.path.normcase(os.path.abspath(os.path.join(os.path.sep, 'path')))
        ups = UninstallPathSet(dist=Mock())
        ups.add(short_path)
        ups.add(os.path.join(short_path, 'longer'))
        assert compact(ups._paths) == {short_path}

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_detect_symlink_dirs(self, monkeypatch: pytest.MonkeyPatch, tmpdir: Path) -> None:
        if False:
            print('Hello World!')
        monkeypatch.setattr(pip._internal.req.req_uninstall.UninstallPathSet, '_permitted', mock_permitted)
        d = tmpdir.joinpath('dir')
        d.mkdir()
        dlink = tmpdir.joinpath('dirlink')
        os.symlink(d, dlink)
        d.joinpath('file').touch()
        path1 = str(d.joinpath('file'))
        path2 = str(dlink.joinpath('file'))
        ups = UninstallPathSet(dist=Mock())
        ups.add(path1)
        ups.add(path2)
        assert ups._paths == {path1}

class TestStashedUninstallPathSet:
    WALK_RESULT: List[Tuple[str, List[str], List[str]]] = [('A', ['B', 'C'], ['a.py']), ('A/B', ['D'], ['b.py']), ('A/B/D', [], ['c.py']), ('A/C', [], ['d.py', 'e.py']), ('A/E', ['F'], ['f.py']), ('A/E/F', [], []), ('A/G', ['H'], ['g.py']), ('A/G/H', [], ['h.py'])]

    @classmethod
    def mock_walk(cls, root: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        if False:
            for i in range(10):
                print('nop')
        for (dirname, subdirs, files) in cls.WALK_RESULT:
            dirname = os.path.sep.join(dirname.split('/'))
            if dirname.startswith(root):
                yield (dirname[len(root) + 1:], subdirs, files)

    def test_compress_for_rename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        paths = [os.path.sep.join(p.split('/')) for p in ['A/B/b.py', 'A/B/D/c.py', 'A/C/d.py', 'A/E/f.py', 'A/G/g.py']]
        expected_paths = [os.path.sep.join(p.split('/')) for p in ['A/B/', 'A/C/d.py', 'A/E/', 'A/G/g.py']]
        monkeypatch.setattr('os.walk', self.mock_walk)
        actual_paths = compress_for_rename(paths)
        assert set(expected_paths) == set(actual_paths)

    @classmethod
    def make_stash(cls, tmpdir: Path, paths: List[str]) -> Tuple[StashedUninstallPathSet, List[Tuple[str, str]]]:
        if False:
            i = 10
            return i + 15
        for (dirname, subdirs, files) in cls.WALK_RESULT:
            root = os.path.join(tmpdir, *dirname.split('/'))
            if not os.path.exists(root):
                os.mkdir(root)
            for d in subdirs:
                os.mkdir(os.path.join(root, d))
            for f in files:
                with open(os.path.join(root, f), 'wb'):
                    pass
        pathset = StashedUninstallPathSet()
        paths = [os.path.join(tmpdir, *p.split('/')) for p in paths]
        stashed_paths = [(p, pathset.stash(p)) for p in paths]
        return (pathset, stashed_paths)

    def test_stash(self, tmpdir: Path) -> None:
        if False:
            while True:
                i = 10
        (pathset, stashed_paths) = self.make_stash(tmpdir, ['A/B/', 'A/C/d.py', 'A/E/', 'A/G/g.py'])
        for (old_path, new_path) in stashed_paths:
            assert not os.path.exists(old_path)
            assert os.path.exists(new_path)
        assert stashed_paths == pathset._moves

    def test_commit(self, tmpdir: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        (pathset, stashed_paths) = self.make_stash(tmpdir, ['A/B/', 'A/C/d.py', 'A/E/', 'A/G/g.py'])
        pathset.commit()
        for (old_path, new_path) in stashed_paths:
            assert not os.path.exists(old_path)
            assert not os.path.exists(new_path)

    def test_rollback(self, tmpdir: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        (pathset, stashed_paths) = self.make_stash(tmpdir, ['A/B/', 'A/C/d.py', 'A/E/', 'A/G/g.py'])
        pathset.rollback()
        for (old_path, new_path) in stashed_paths:
            assert os.path.exists(old_path)
            assert not os.path.exists(new_path)

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_commit_symlinks(self, tmpdir: Path) -> None:
        if False:
            return 10
        adir = tmpdir / 'dir'
        adir.mkdir()
        dirlink = tmpdir / 'dirlink'
        dirlink.symlink_to(adir)
        afile = tmpdir / 'file'
        afile.write_text('...')
        filelink = tmpdir / 'filelink'
        filelink.symlink_to(afile)
        pathset = StashedUninstallPathSet()
        stashed_paths = []
        stashed_paths.append(pathset.stash(os.fspath(dirlink)))
        stashed_paths.append(pathset.stash(os.fspath(filelink)))
        for stashed_path in stashed_paths:
            assert os.path.lexists(stashed_path)
        assert not os.path.exists(dirlink)
        assert not os.path.exists(filelink)
        pathset.commit()
        for stashed_path in stashed_paths:
            assert not os.path.lexists(stashed_path)
        assert not os.path.lexists(dirlink) and (not os.path.isdir(dirlink))
        assert not os.path.lexists(filelink) and (not os.path.isfile(filelink))
        assert os.path.isdir(adir)
        assert os.path.isfile(afile)

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_rollback_symlinks(self, tmpdir: Path) -> None:
        if False:
            i = 10
            return i + 15
        adir = tmpdir / 'dir'
        adir.mkdir()
        dirlink = tmpdir / 'dirlink'
        dirlink.symlink_to(adir)
        afile = tmpdir / 'file'
        afile.write_text('...')
        filelink = tmpdir / 'filelink'
        filelink.symlink_to(afile)
        pathset = StashedUninstallPathSet()
        stashed_paths = []
        stashed_paths.append(pathset.stash(os.fspath(dirlink)))
        stashed_paths.append(pathset.stash(os.fspath(filelink)))
        for stashed_path in stashed_paths:
            assert os.path.lexists(stashed_path)
        assert not os.path.lexists(dirlink)
        assert not os.path.lexists(filelink)
        pathset.rollback()
        for stashed_path in stashed_paths:
            assert not os.path.lexists(stashed_path)
        assert os.path.lexists(dirlink) and os.path.isdir(dirlink)
        assert os.path.lexists(filelink) and os.path.isfile(filelink)
        assert os.path.isdir(adir)
        assert os.path.isfile(afile)