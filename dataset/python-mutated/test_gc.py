import datetime
import logging
import os
import shutil
import textwrap
import pytest
from dvc.cli import main
from dvc.exceptions import CollectCacheError, InvalidArgumentError
from dvc.fs import LocalFileSystem
from dvc.utils.fs import remove
from dvc_data.hashfile.db.local import LocalHashFileDB

@pytest.fixture
def good_and_bad_cache(tmp_dir, dvc):
    if False:
        for i in range(10):
            print('nop')
    tmp_dir.dvc_gen('foo', 'foo')
    (stage,) = tmp_dir.dvc_gen('data', {'sub': {'data_sub': 'data_sub', 'data': 'data', 'тест': 'проверка'}})
    raw_dir_hash = stage.outs[0].hash_info.as_raw().value
    odb = dvc.cache.local
    bad_cache = {raw_dir_hash}
    for i in ['123', '234', '345']:
        odb.add_bytes(i, i.encode('utf8'))
        bad_cache.add(i)
    good_cache = {md5 for md5 in odb.all() if md5 not in bad_cache}
    return (good_cache, bad_cache)

def test_gc_api(dvc, good_and_bad_cache):
    if False:
        while True:
            i = 10
    dvc.gc(workspace=True)
    odb = dvc.cache.local
    (good_cache, bad_cache) = good_and_bad_cache
    assert set(odb.all()) == good_cache

def test_gc_cli(dvc, good_and_bad_cache):
    if False:
        i = 10
        return i + 15
    assert main(['gc', '-wf']) == 0
    odb = dvc.cache.local
    (good_cache, bad_cache) = good_and_bad_cache
    assert set(odb.all()) == good_cache

def test_gc_branches_tags(tmp_dir, dvc, scm):
    if False:
        for i in range(10):
            print('nop')
    tmp_dir.dvc_gen('file', 'v1.0', commit='v1.0')
    scm.tag('v1.0')
    with tmp_dir.branch('test', new=True):
        dvc.remove('file.dvc')
        tmp_dir.dvc_gen('file', 'test', commit='test')
    dvc.remove('file.dvc')
    tmp_dir.dvc_gen('file', 'trash', commit='trash')
    dvc.remove('file.dvc')
    tmp_dir.dvc_gen('file', 'master', commit='trash')
    odb = dvc.cache.local
    assert len(list(odb.all())) == 4
    dvc.gc(all_tags=True, all_branches=True)
    assert len(list(odb.all())) == 3
    dvc.gc(all_tags=False, all_branches=True)
    assert len(list(odb.all())) == 2
    dvc.gc(all_tags=True, all_branches=False)
    assert len(list(odb.all())) == 1

def test_gc_multiple_dvc_repos(tmp_dir, scm, dvc, erepo_dir):
    if False:
        return 10
    tmp_dir.dvc_gen('only_in_first', 'only in main repo')
    tmp_dir.dvc_gen('in_both', 'in both repos')
    erepo_dir.dvc.cache.local.path = dvc.cache.local.path
    with erepo_dir.chdir():
        erepo_dir.dvc_gen('in_both', 'in both repos')
        erepo_dir.dvc_gen('only_in_second', 'only in additional repo')
    odb = dvc.cache.local
    assert len(list(odb.all())) == 3
    dvc.gc(repos=[erepo_dir], workspace=True)
    assert len(list(odb.all())) == 3
    dvc.gc(workspace=True)
    assert len(list(odb.all())) == 2

def test_all_commits(tmp_dir, scm, dvc):
    if False:
        while True:
            i = 10
    tmp_dir.dvc_gen('testfile', 'uncommitted')
    tmp_dir.dvc_gen('testfile', 'committed', commit='committed')
    tmp_dir.dvc_gen('testfile', 'modified', commit='modified')
    tmp_dir.dvc_gen('testfile', 'workspace')
    n = _count_files(dvc.cache.local.path)
    dvc.gc(all_commits=True)
    assert _count_files(dvc.cache.local.path) == n - 1

def test_gc_no_dir_cache(tmp_dir, dvc):
    if False:
        print('Hello World!')
    tmp_dir.dvc_gen({'foo': 'foo', 'bar': 'bar'})
    (dir_stage,) = tmp_dir.dvc_gen({'dir': {'x': 'x', 'subdir': {'y': 'y'}}})
    remove(dir_stage.outs[0].cache_path)
    with pytest.raises(CollectCacheError):
        dvc.gc(workspace=True)
    assert _count_files(dvc.cache.local.path) == 4
    dvc.gc(force=True, workspace=True)
    assert _count_files(dvc.cache.local.path) == 2

def _count_files(path):
    if False:
        return 10
    return sum((len(files) for (_, _, files) in os.walk(path)))

def test_gc_no_unpacked_dir(tmp_dir, dvc):
    if False:
        for i in range(10):
            print('nop')
    dir_stages = tmp_dir.dvc_gen({'dir': {'file': 'text'}})
    dvc.status()
    os.remove('dir.dvc')
    unpackeddir = dir_stages[0].outs[0].cache_path + LocalHashFileDB.UNPACKED_DIR_SUFFIX
    shutil.copytree('dir', unpackeddir)
    assert os.path.exists(unpackeddir)
    dvc.gc(force=True, workspace=True)
    assert not os.path.exists(unpackeddir)

def test_gc_without_workspace_raises_error(tmp_dir, dvc):
    if False:
        while True:
            i = 10
    dvc.gc(force=True, workspace=True)
    from dvc.exceptions import InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        dvc.gc(force=True)
    with pytest.raises(InvalidArgumentError):
        dvc.gc(force=True, workspace=False)

def test_gc_cloud_with_or_without_specifier(tmp_dir, erepo_dir, local_cloud):
    if False:
        print('Hello World!')
    erepo_dir.add_remote(config=local_cloud.config)
    dvc = erepo_dir.dvc
    from dvc.exceptions import InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        dvc.gc(force=True, cloud=True)
    dvc.gc(cloud=True, all_tags=True)
    dvc.gc(cloud=True, all_commits=True)
    dvc.gc(cloud=True, all_branches=True)
    dvc.gc(cloud=True, all_commits=False, all_branches=True, all_tags=True)

def test_gc_without_workspace_on_tags_branches_commits(tmp_dir, dvc):
    if False:
        while True:
            i = 10
    dvc.gc(force=True, all_tags=True)
    dvc.gc(force=True, all_commits=True)
    dvc.gc(force=False, all_branches=True)
    dvc.gc(force=False, all_branches=True, all_commits=False, workspace=False)

@pytest.mark.parametrize('cloud', ['c', ''])
def test_gc_without_workspace(tmp_dir, dvc, caplog, cloud):
    if False:
        i = 10
        return i + 15
    with caplog.at_level(logging.WARNING, logger='dvc'):
        assert main(['gc', f'-{cloud}vf']) == 255
    assert 'Either of `-w|--workspace`, `-a|--all-branches`, `-T|--all-tags` `--all-experiments`, `--all-commits`, `--date` or `--rev` needs to be set.' in caplog.text

def test_gc_with_possible_args_positive(tmp_dir, dvc):
    if False:
        while True:
            i = 10
    for flag in ['-w', '-a', '-T', '--all-commits', '-aT', '-wa', '-waT']:
        assert main(['gc', '-vf', flag]) == 0

def test_gc_cloud_positive(tmp_dir, dvc, tmp_path_factory, local_remote):
    if False:
        i = 10
        return i + 15
    for flag in ['-cw', '-ca', '-cT', '-caT', '-cwT']:
        assert main(['gc', '-vf', flag]) == 0

def test_gc_cloud_remove_order(tmp_dir, scm, dvc, mocker, local_remote):
    if False:
        for i in range(10):
            print('nop')
    (standalone, dir1, dir2) = tmp_dir.dvc_gen({'file1': 'standalone', 'dir1': {'file2': 'file2'}, 'dir2': {'file3': 'file3', 'file4': 'file4'}})
    dvc.push()
    dvc.remove(standalone.relpath)
    dvc.remove(dir1.relpath)
    dvc.remove(dir2.relpath)
    dvc.gc(workspace=True)
    mocked_remove = mocker.patch.object(LocalFileSystem, 'remove', autospec=True)
    dvc.gc(workspace=True, cloud=True)
    assert len(mocked_remove.mock_calls) == 4
    for args in mocked_remove.call_args_list[:2]:
        checksum = str(args[0][1])
        assert checksum.endswith('.dir.unpacked')
    checksums = mocked_remove.call_args_list[2][0][1]
    assert isinstance(checksums, list)
    assert all((x.endswith('.dir') for x in checksums))
    checksums = mocked_remove.call_args_list[3][0][1]
    assert isinstance(checksums, list)
    assert not any((x.endswith('.dir') for x in checksums))

def test_gc_not_collect_pipeline_tracked_files(tmp_dir, dvc, run_copy):
    if False:
        for i in range(10):
            print('nop')
    from dvc.dvcfile import PROJECT_FILE, load_file
    tmp_dir.gen('foo', 'foo')
    tmp_dir.gen('bar', 'bar')
    run_copy('foo', 'foo2', name='copy')
    shutil.rmtree(dvc.stage_cache.cache_dir)
    assert _count_files(dvc.cache.local.path) == 1
    dvc.gc(workspace=True, force=True)
    assert _count_files(dvc.cache.local.path) == 1
    load_file(dvc, PROJECT_FILE).remove(force=True)
    dvc.gc(workspace=True, force=True)
    assert _count_files(dvc.cache.local.path) == 0

def test_gc_all_experiments(tmp_dir, scm, dvc):
    if False:
        while True:
            i = 10
    from dvc.repo.experiments.refs import ExpRefInfo
    (foo,) = tmp_dir.dvc_gen('foo', 'foo', commit='foo')
    foo_hash = foo.outs[0].hash_info.value
    tmp_dir.dvc_gen('foo', 'bar', commit='bar')
    baseline = scm.get_rev()
    (baz,) = tmp_dir.dvc_gen('foo', 'baz', commit='baz')
    baz_hash = baz.outs[0].hash_info.value
    ref = ExpRefInfo(baseline, 'exp')
    scm.set_ref(str(ref), scm.get_rev())
    dvc.gc(all_experiments=True, force=True)
    assert not (tmp_dir / '.dvc' / 'cache' / 'files' / 'md5' / foo_hash[:2] / foo_hash[2:]).exists()
    assert (tmp_dir / '.dvc' / 'cache' / 'files' / 'md5' / baz_hash[:2] / baz_hash[2:]).read_text() == 'baz'

def test_gc_rev_num(tmp_dir, scm, dvc):
    if False:
        i = 10
        return i + 15
    num = 2
    hashes = {}
    for i in range(4):
        i_str = str(i)
        f = tmp_dir.dvc_gen('foo', i_str, commit=i_str)
        hashes[i] = f[0].outs[0].hash_info.value
    dvc.gc(rev='HEAD', num=num, force=True)
    for (n, i) in enumerate(reversed(range(4))):
        cache = tmp_dir / '.dvc' / 'cache' / 'files' / 'md5' / hashes[i][:2] / hashes[i][2:]
        if n >= num:
            assert not cache.exists()
        else:
            assert cache.read_text() == str(i)

def test_date(tmp_dir, scm, dvc):
    if False:
        print('Hello World!')
    tmp_dir.dvc_gen('testfile', 'content', commit='add testfile')
    now = datetime.datetime.now()
    datestamp = (now.date() + datetime.timedelta(days=1)).isoformat()
    tmp_dir.dvc_gen('testfile', 'modified', commit='modified')
    dvc.gc(commit_date=datestamp)
    assert _count_files(dvc.cache.local.path) == 1
    assert dvc.cache.local.exists('9ae73c65f418e6f79ceb4f0e4a4b98d5')
    tmp_dir.dvc_gen('testfile', 'modified, again', commit='modify')
    datestamp = (now.date() - datetime.timedelta(days=1)).isoformat()
    dvc.gc(commit_date=datestamp)
    assert _count_files(dvc.cache.local.path) == 2
    assert dvc.cache.local.exists('9ae73c65f418e6f79ceb4f0e4a4b98d5')
    assert dvc.cache.local.exists('3bcf3b1be3e794a97a5a6b93a005784c')

def test_gc_not_in_remote(tmp_dir, scm, dvc, mocker, local_remote):
    if False:
        i = 10
        return i + 15
    (standalone, dir1, dir2) = tmp_dir.dvc_gen({'file1': 'standalone', 'dir1': {'file2': 'file2'}, 'dir2': {'file3': 'file3', 'file4': 'file4'}})
    mocked_remove = mocker.spy(LocalFileSystem, 'remove')
    dvc.gc(workspace=True)
    assert not mocked_remove.call_args_list
    dvc.push(['file1', 'dir1'])
    dvc.gc(workspace=True, not_in_remote=True)
    assert len(mocked_remove.mock_calls) == 3
    arg_list = mocked_remove.call_args_list
    standalone_hash = standalone.outs[0].hash_info.value
    dir1_hash = dir1.outs[0].hash_info.value
    assert f'{dir1_hash[2:]}.unpacked' in arg_list[0][0][1]
    assert f'{dir1_hash[2:]}' in arg_list[1][0][1][0]
    assert len(arg_list[2][0][1]) == 2
    assert f'{standalone_hash[2:]}' in arg_list[2][0][1][0] or f'{standalone_hash[2:]}' in arg_list[2][0][1][1]

def test_gc_not_in_remote_remote_arg(tmp_dir, scm, dvc, mocker, make_remote):
    if False:
        print('Hello World!')
    make_remote('local_remote', typ='local')
    make_remote('other_remote', typ='local', default=False)
    tmp_dir.dvc_gen({'file1': 'standalone', 'dir1': {'file2': 'file2'}, 'dir2': {'file3': 'file3', 'file4': 'file4'}})
    mocked_remove = mocker.spy(LocalFileSystem, 'remove')
    dvc.push(['file1', 'dir1'], remote='other_remote')
    dvc.gc(workspace=True, not_in_remote=True)
    assert not mocked_remove.mock_calls
    dvc.gc(workspace=True, not_in_remote=True, remote='other_remote')
    assert len(mocked_remove.mock_calls) == 3

def test_gc_not_in_remote_with_remote_field(tmp_dir, scm, dvc, mocker, make_remote):
    if False:
        print('Hello World!')
    make_remote('local_remote', typ='local')
    make_remote('other_remote', typ='local', default=False)
    text = textwrap.dedent('        outs:\n        - path: foo\n          remote: other_remote\n          hash: md5\n    ')
    tmp_dir.gen('foo.dvc', text)
    tmp_dir.dvc_gen('foo', 'foo')
    dvc.push()
    mocked_remove = mocker.spy(LocalFileSystem, 'remove')
    dvc.gc(workspace=True, not_in_remote=True)
    assert len(mocked_remove.mock_calls) == 1

def test_gc_not_in_remote_cloud(tmp_dir, scm, dvc):
    if False:
        return 10
    with pytest.raises(InvalidArgumentError, match='`--not-in-remote` and `--cloud` are mutually exclusive'):
        dvc.gc(workspace=True, not_in_remote=True, cloud=True)

def test_gc_cloud_remote_field(tmp_dir, scm, dvc, mocker, make_remote):
    if False:
        for i in range(10):
            print('nop')
    make_remote('local_remote', typ='local')
    make_remote('other_remote', typ='local', default=False)
    text = textwrap.dedent('        outs:\n        - path: foo\n          remote: other_remote\n          hash: md5\n    ')
    tmp_dir.gen('foo.dvc', text)
    tmp_dir.dvc_gen('foo', 'foo')
    dvc.push()
    tmp_dir.dvc_gen('foo', 'bar')
    mocked_remove = mocker.spy(LocalFileSystem, 'remove')
    dvc.gc(workspace=True, cloud=True)
    assert len(mocked_remove.mock_calls) == 2

def test_gc_dry(dvc, good_and_bad_cache):
    if False:
        return 10
    dvc.gc(workspace=True, dry=True)
    odb = dvc.cache.local
    (good_cache, bad_cache) = good_and_bad_cache
    assert set(odb.all()) != good_cache

def test_gc_logging(caplog, dvc, good_and_bad_cache):
    if False:
        while True:
            i = 10
    with caplog.at_level(logging.INFO, logger='dvc'):
        dvc.gc(workspace=True)
    assert 'Removed 3 objects from repo cache.' in caplog.text
    assert "No unused 'local' cache to remove." in caplog.text
    assert "No unused 'legacy' cache to remove." in caplog.text