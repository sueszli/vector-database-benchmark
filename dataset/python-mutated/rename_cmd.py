from ...constants import *
from ...manifest import Manifest
from ...repository import Repository
from . import cmd, create_regular_file, generate_archiver_tests, RK_ENCRYPTION
pytest_generate_tests = lambda metafunc: generate_archiver_tests(metafunc, kinds='local,remote,binary')

def test_rename(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    create_regular_file(archiver.input_path, 'file1', size=1024 * 80)
    create_regular_file(archiver.input_path, 'dir2/file2', size=1024 * 80)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test', 'input')
    cmd(archiver, 'create', 'test.2', 'input')
    cmd(archiver, 'extract', 'test', '--dry-run')
    cmd(archiver, 'extract', 'test.2', '--dry-run')
    cmd(archiver, 'rename', 'test', 'test.3')
    cmd(archiver, 'extract', 'test.2', '--dry-run')
    cmd(archiver, 'rename', 'test.2', 'test.4')
    cmd(archiver, 'extract', 'test.3', '--dry-run')
    cmd(archiver, 'extract', 'test.4', '--dry-run')
    with Repository(archiver.repository_path) as repository:
        manifest = Manifest.load(repository, Manifest.NO_OPERATION_CHECK)
    assert len(manifest.archives) == 2
    assert 'test.3' in manifest.archives
    assert 'test.4' in manifest.archives