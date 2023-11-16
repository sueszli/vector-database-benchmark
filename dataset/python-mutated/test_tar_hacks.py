import os
import tarfile
from wal_e import tar_partition

def test_fsync_tar_members(monkeypatch, tmpdir):
    if False:
        return 10
    'Test that _fsync_files() syncs all files and directories\n\n    Syncing directories is a platform specific feature, so it is\n    optional.\n\n    There is a separate test in test_blackbox that tar_file_extract()\n    actually calls _fsync_files and passes it the expected list of\n    files.\n\n    '
    dira = tmpdir.join('dira').ensure(dir=True)
    dirb = tmpdir.join('dirb').ensure(dir=True)
    foo = dira.join('foo').ensure()
    bar = dirb.join('bar').ensure()
    baz = dirb.join('baz').ensure()
    open_descriptors = {}
    synced_filenames = set()
    tmproot = str(tmpdir)
    real_open = os.open
    real_close = os.close
    real_fsync = os.fsync

    def fake_open(filename, flags, mode=511):
        if False:
            for i in range(10):
                print('nop')
        fd = real_open(filename, flags, mode)
        if filename.startswith(tmproot):
            open_descriptors[fd] = filename
        return fd

    def fake_close(fd):
        if False:
            print('Hello World!')
        if fd in open_descriptors:
            del open_descriptors[fd]
        real_close(fd)
        return

    def fake_fsync(fd):
        if False:
            print('Hello World!')
        if fd in open_descriptors:
            synced_filenames.add(open_descriptors[fd])
        real_fsync(fd)
        return
    monkeypatch.setattr(os, 'open', fake_open)
    monkeypatch.setattr(os, 'close', fake_close)
    monkeypatch.setattr(os, 'fsync', fake_fsync)
    filenames = [str(filename) for filename in [foo, bar, baz]]
    tar_partition._fsync_files(filenames)
    for filename in filenames:
        assert filename in synced_filenames
    if hasattr(os, 'O_DIRECTORY'):
        assert str(dira) in synced_filenames
        assert str(dirb) in synced_filenames

def test_dynamically_emptied_directories(tmpdir):
    if False:
        return 10
    'Ensure empty directories in the base backup are created\n\n    Particularly in the case when PostgreSQL empties the files in\n    those directories in parallel.  This emptying can happen after the\n    files are partitioned into their tarballs but before the tar and\n    upload process is complete.\n\n    '
    adir = tmpdir.join('adir').ensure(dir=True)
    bdir = adir.join('bdir').ensure(dir=True)
    some_file = bdir.join('afile')
    some_file.write('1234567890')
    base_dir = adir.strpath
    (spec, parts) = tar_partition.partition(base_dir)
    tar_paths = []
    for part in parts:
        for tar_info in part:
            rel_path = os.path.relpath(tar_info.submitted_path, base_dir)
            tar_paths.append(rel_path)
    assert 'bdir' in tar_paths

def test_creation_upper_dir(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    'Check for upper-directory creation in untarring\n\n    This affected the special "cat" based extraction works when no\n    upper level directory is present.  Using that path depends on\n    PIPE_BUF_BYTES, so test that integration via monkey-patching it to\n    a small value.\n\n    '
    from wal_e import pipebuf
    adir = tmpdir.join('adir').ensure(dir=True)
    some_file = adir.join('afile')
    some_file.write('1234567890')
    tar_path = str(tmpdir.join('foo.tar'))
    tar = tarfile.open(name=tar_path, mode='w')
    tar.add(str(some_file))
    tar.close()
    original_cat_extract = tar_partition.cat_extract

    class CheckCatExtract(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.called = False

        def __call__(self, *args, **kwargs):
            if False:
                return 10
            self.called = True
            return original_cat_extract(*args, **kwargs)
    check = CheckCatExtract()
    monkeypatch.setattr(tar_partition, 'cat_extract', check)
    monkeypatch.setattr(pipebuf, 'PIPE_BUF_BYTES', 1)
    dest_dir = tmpdir.join('dest')
    dest_dir.ensure(dir=True)
    with open(tar_path, 'rb') as f:
        tar_partition.TarPartition.tarfile_extract(f, str(dest_dir))
    assert check.called