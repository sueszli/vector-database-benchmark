import os
import random
import shutil
import socket
import string
import sys
import tempfile
import uuid
from filecmp import dircmp
from pathlib import Path
from shutil import copytree, make_archive, rmtree
import zipfile
import ray
import pytest
from ray._private.gcs_utils import GcsAioClient
from ray._private.ray_constants import KV_NAMESPACE_PACKAGE, RAY_RUNTIME_ENV_IGNORE_GITIGNORE
from ray._private.runtime_env.packaging import GCS_STORAGE_MAX_SIZE, MAC_OS_ZIP_HIDDEN_DIR_NAME, Protocol, _dir_travel, _get_excludes, _store_package_in_gcs, download_and_unpack_package, get_local_dir_from_uri, get_top_level_dir_from_compressed_package, get_uri_for_directory, get_uri_for_package, is_whl_uri, is_zip_uri, parse_uri, remove_dir_from_filepaths, unzip_package, upload_package_if_needed, _get_gitignore, upload_package_to_gcs
from ray.experimental.internal_kv import _initialize_internal_kv, _internal_kv_del, _internal_kv_exists, _internal_kv_get, _internal_kv_reset
TOP_LEVEL_DIR_NAME = 'top_level'
ARCHIVE_NAME = 'archive.zip'
HTTPS_PACKAGE_URI = 'https://github.com/shrekris-anyscale/test_module/archive/HEAD.zip'
S3_PACKAGE_URI = 's3://runtime-env-test/test_runtime_env.zip'
S3_WHL_PACKAGE_URI = 's3://runtime-env-test/test_module-0.0.1-py3-none-any.whl'
GS_PACKAGE_URI = 'gs://public-runtime-env-test/test_module.zip'

def random_string(size: int=10):
    if False:
        while True:
            i = 10
    return ''.join((random.choice(string.ascii_uppercase) for _ in range(size)))

@pytest.fixture
def random_dir(tmp_path) -> Path:
    if False:
        return 10
    subdir = tmp_path / 'subdir'
    subdir.mkdir()
    for _ in range(10):
        p1 = tmp_path / random_string(10)
        with p1.open('w') as f1:
            f1.write(random_string(100))
        p2 = tmp_path / random_string(10)
        with p2.open('w') as f2:
            f2.write(random_string(200))
    yield tmp_path

@pytest.fixture
def short_path_dir():
    if False:
        for i in range(10):
            print('nop')
    'A directory with a short path.\n\n    This directory is used to test the case where a socket file is in the\n    directory.  Socket files have a maximum length of 108 characters, so the\n    path from the built-in pytest fixture tmp_path is too long.\n    '
    dir = Path('short_path')
    dir.mkdir()
    yield dir
    shutil.rmtree(str(dir))

@pytest.fixture
def random_zip_file_without_top_level_dir(random_dir):
    if False:
        for i in range(10):
            print('nop')
    make_archive(random_dir / ARCHIVE_NAME[:ARCHIVE_NAME.rfind('.')], 'zip', random_dir)
    yield str(random_dir / ARCHIVE_NAME)

@pytest.fixture
def random_zip_file_with_top_level_dir(tmp_path):
    if False:
        i = 10
        return i + 15
    path = tmp_path
    top_level_dir = path / TOP_LEVEL_DIR_NAME
    top_level_dir.mkdir(parents=True)
    next_level_dir = top_level_dir
    for _ in range(10):
        p1 = next_level_dir / random_string(10)
        with p1.open('w') as f1:
            f1.write(random_string(100))
        p2 = next_level_dir / random_string(10)
        with p2.open('w') as f2:
            f2.write(random_string(200))
        dir1 = next_level_dir / random_string(15)
        dir1.mkdir(parents=True)
        dir2 = next_level_dir / random_string(15)
        dir2.mkdir(parents=True)
        next_level_dir = dir2
    macos_dir = path / MAC_OS_ZIP_HIDDEN_DIR_NAME
    macos_dir.mkdir(parents=True)
    with (macos_dir / 'file').open('w') as f:
        f.write('macos file')
    make_archive(path / ARCHIVE_NAME[:ARCHIVE_NAME.rfind('.')], 'zip', path, TOP_LEVEL_DIR_NAME)
    yield str(path / ARCHIVE_NAME)

class TestGetURIForDirectory:

    def test_invalid_directory(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            get_uri_for_directory('/does/not/exist')
        with pytest.raises(ValueError):
            get_uri_for_directory('does/not/exist')

    def test_determinism(self, random_dir):
        if False:
            return 10
        uris = {get_uri_for_directory(random_dir) for _ in range(10)}
        assert len(uris) == 1
        with open(random_dir / f'test_{random_string()}', 'w') as f:
            f.write(random_string())
        assert {get_uri_for_directory(random_dir)} != uris

    def test_relative_paths(self, random_dir):
        if False:
            return 10
        p = Path(random_dir)
        relative_uri = get_uri_for_directory(os.path.relpath(p))
        absolute_uri = get_uri_for_directory(p.resolve())
        assert relative_uri == absolute_uri

    def test_excludes(self, random_dir):
        if False:
            print('Hello World!')
        included_uri = get_uri_for_directory(random_dir)
        excluded_uri = get_uri_for_directory(random_dir, excludes=['subdir'])
        assert included_uri != excluded_uri
        rmtree((Path(random_dir) / 'subdir').resolve())
        deleted_uri = get_uri_for_directory(random_dir)
        assert deleted_uri == excluded_uri

    def test_empty_directory(self):
        if False:
            print('Hello World!')
        try:
            os.mkdir('d1')
            os.mkdir('d2')
            assert get_uri_for_directory('d1') == get_uri_for_directory('d2')
        finally:
            os.rmdir('d1')
            os.rmdir('d2')

    def test_uri_hash_length(self, random_dir):
        if False:
            i = 10
            return i + 15
        uri = get_uri_for_directory(random_dir)
        hex_hash = uri.split('_')[-1][:-len('.zip')]
        assert len(hex_hash) == 16

    @pytest.mark.skipif(sys.platform == 'win32', reason='Unix sockets not available on windows')
    def test_unopenable_files_skipped(self, random_dir, short_path_dir):
        if False:
            while True:
                i = 10
        'Test that unopenable files can be present in the working_dir.\n\n        Some files such as `.sock` files are unopenable. This test ensures that\n        we skip those files when generating the content hash. Previously this\n        would raise an exception, see #25411.\n        '
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(short_path_dir / 'test_socket'))
        with pytest.raises(OSError):
            (short_path_dir / 'test_socket').open()
        get_uri_for_directory(short_path_dir)

class TestUploadPackageIfNeeded:

    def test_create_upload_once(self, tmp_path, random_dir, ray_start_regular):
        if False:
            while True:
                i = 10
        uri = get_uri_for_directory(random_dir)
        uploaded = upload_package_if_needed(uri, tmp_path, random_dir)
        assert uploaded
        assert _internal_kv_exists(uri, namespace=KV_NAMESPACE_PACKAGE)
        uploaded = upload_package_if_needed(uri, tmp_path, random_dir)
        assert not uploaded
        assert _internal_kv_exists(uri, namespace=KV_NAMESPACE_PACKAGE)
        _internal_kv_del(uri, namespace=KV_NAMESPACE_PACKAGE)
        assert not _internal_kv_exists(uri, namespace=KV_NAMESPACE_PACKAGE)
        uploaded = upload_package_if_needed(uri, tmp_path, random_dir)
        assert uploaded

class TestStorePackageInGcs:

    class DisconnectedClient:
        """Mock GcsClient that fails cannot put in the GCS."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

        def internal_kv_put(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            raise RuntimeError('Cannot reach GCS!')

    def raise_runtime_error(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise RuntimeError('Raised a runtime error!')

    def test_upload_succeeds(self, ray_start_regular):
        if False:
            for i in range(10):
                print('nop')
        'Check function behavior when upload succeeds.'
        uri = 'gcs://test.zip'
        bytes = b'test'
        assert len(bytes) < GCS_STORAGE_MAX_SIZE
        assert not _internal_kv_exists(uri, namespace=KV_NAMESPACE_PACKAGE)
        assert _store_package_in_gcs(uri, bytes) == len(bytes)
        assert bytes == _internal_kv_get(uri, namespace=KV_NAMESPACE_PACKAGE)

    def test_upload_fails(self):
        if False:
            print('Hello World!')
        'Check that function throws useful error when upload fails.'
        uri = 'gcs://test.zip'
        bytes = b'test'
        assert len(bytes) < GCS_STORAGE_MAX_SIZE
        _internal_kv_reset()
        _initialize_internal_kv(self.DisconnectedClient())
        with pytest.raises(RuntimeError, match='Failed to store package in the GCS'):
            _store_package_in_gcs(uri, bytes)

    def test_package_size_too_large(self):
        if False:
            print('Hello World!')
        'Check that function throws useful error when package is too large.'
        uri = 'gcs://test.zip'
        bytes = b'a' * (GCS_STORAGE_MAX_SIZE + 1)
        with pytest.raises(ValueError, match='Package size'):
            _store_package_in_gcs(uri, bytes)

class TestGetTopLevelDirFromCompressedPackage:

    def test_get_top_level_valid(self, random_zip_file_with_top_level_dir):
        if False:
            for i in range(10):
                print('nop')
        top_level_dir_name = get_top_level_dir_from_compressed_package(str(random_zip_file_with_top_level_dir))
        assert top_level_dir_name == TOP_LEVEL_DIR_NAME

    def test_get_top_level_invalid(self, random_zip_file_without_top_level_dir):
        if False:
            while True:
                i = 10
        top_level_dir_name = get_top_level_dir_from_compressed_package(str(random_zip_file_without_top_level_dir))
        assert top_level_dir_name is None

class TestRemoveDirFromFilepaths:

    def test_valid_removal(self, random_zip_file_with_top_level_dir):
        if False:
            return 10
        archive_path = random_zip_file_with_top_level_dir
        tmp_path = archive_path[:archive_path.rfind(os.path.sep)]
        original_dir_path = os.path.join(tmp_path, TOP_LEVEL_DIR_NAME)
        copy_dir_path = os.path.join(tmp_path, TOP_LEVEL_DIR_NAME + '_copy')
        copytree(original_dir_path, copy_dir_path)
        remove_dir_from_filepaths(tmp_path, TOP_LEVEL_DIR_NAME + '_copy')
        dcmp = dircmp(tmp_path, os.path.join(tmp_path, TOP_LEVEL_DIR_NAME))
        assert set(dcmp.left_only) == {ARCHIVE_NAME, TOP_LEVEL_DIR_NAME, MAC_OS_ZIP_HIDDEN_DIR_NAME}
        assert len(dcmp.right_only) == 0

@pytest.mark.parametrize('remove_top_level_directory', [False, True])
@pytest.mark.parametrize('unlink_zip', [False, True])
class TestUnzipPackage:

    def dcmp_helper(self, remove_top_level_directory, unlink_zip, tmp_subdir, tmp_path, archive_path):
        if False:
            return 10
        dcmp = None
        if remove_top_level_directory:
            dcmp = dircmp(tmp_subdir, os.path.join(tmp_path, TOP_LEVEL_DIR_NAME))
        else:
            dcmp = dircmp(os.path.join(tmp_subdir, TOP_LEVEL_DIR_NAME), os.path.join(tmp_path, TOP_LEVEL_DIR_NAME))
        assert len(dcmp.left_only) == 0
        assert len(dcmp.right_only) == 0
        if unlink_zip:
            assert not Path(archive_path).is_file()
        else:
            assert Path(archive_path).is_file()

    def test_unzip_package(self, random_zip_file_with_top_level_dir, remove_top_level_directory, unlink_zip):
        if False:
            return 10
        archive_path = random_zip_file_with_top_level_dir
        tmp_path = archive_path[:archive_path.rfind(os.path.sep)]
        tmp_subdir = os.path.join(tmp_path, TOP_LEVEL_DIR_NAME + '_tmp')
        unzip_package(package_path=archive_path, target_dir=tmp_subdir, remove_top_level_directory=remove_top_level_directory, unlink_zip=unlink_zip)
        self.dcmp_helper(remove_top_level_directory, unlink_zip, tmp_subdir, tmp_path, archive_path)

    def test_unzip_with_matching_subdirectory_names(self, remove_top_level_directory, unlink_zip, tmp_path):
        if False:
            return 10
        path = tmp_path
        top_level_dir = path / TOP_LEVEL_DIR_NAME
        top_level_dir.mkdir(parents=True)
        next_level_dir = top_level_dir
        for _ in range(10):
            dir1 = next_level_dir / TOP_LEVEL_DIR_NAME
            dir1.mkdir(parents=True)
            next_level_dir = dir1
        make_archive(path / ARCHIVE_NAME[:ARCHIVE_NAME.rfind('.')], 'zip', path, TOP_LEVEL_DIR_NAME)
        archive_path = str(path / ARCHIVE_NAME)
        tmp_path = archive_path[:archive_path.rfind(os.path.sep)]
        tmp_subdir = os.path.join(tmp_path, TOP_LEVEL_DIR_NAME + '_tmp')
        unzip_package(package_path=archive_path, target_dir=tmp_subdir, remove_top_level_directory=remove_top_level_directory, unlink_zip=unlink_zip)
        self.dcmp_helper(remove_top_level_directory, unlink_zip, tmp_subdir, tmp_path, archive_path)

    def test_unzip_package_with_multiple_top_level_dirs(self, remove_top_level_directory, unlink_zip, random_zip_file_without_top_level_dir):
        if False:
            i = 10
            return i + 15
        "Test unzipping a package with multiple top level directories (not counting __MACOSX).\n\n        Tests that we don't remove the top level directory, regardless of the\n        value of remove_top_level_directory.\n        "
        archive_path = random_zip_file_without_top_level_dir
        tmp_path = archive_path[:archive_path.rfind(os.path.sep)]
        target_dir = os.path.join(tmp_path, 'target_dir')
        print(os.listdir(tmp_path))
        unzip_package(package_path=archive_path, target_dir=target_dir, remove_top_level_directory=remove_top_level_directory, unlink_zip=unlink_zip)
        print(os.listdir(target_dir))
        dcmp = dircmp(tmp_path, target_dir)
        print(dcmp.report())
        assert dcmp.left_only == ['target_dir']
        assert dcmp.right_only == ([ARCHIVE_NAME] if unlink_zip else [])
        if unlink_zip:
            assert not Path(archive_path).is_file()
        else:
            assert Path(archive_path).is_file()

class TestParseUri:

    @pytest.mark.parametrize('parsing_tuple', [('gcs://file.zip', Protocol.GCS, 'file.zip'), ('s3://bucket/file.zip', Protocol.S3, 's3_bucket_file.zip'), ('https://test.com/file.zip', Protocol.HTTPS, 'https_test_com_file.zip'), ('gs://bucket/file.zip', Protocol.GS, 'gs_bucket_file.zip')])
    def test_parsing_remote_basic(self, parsing_tuple):
        if False:
            for i in range(10):
                print('nop')
        (uri, protocol, package_name) = parsing_tuple
        (parsed_protocol, parsed_package_name) = parse_uri(uri)
        assert protocol == parsed_protocol
        assert package_name == parsed_package_name

    @pytest.mark.parametrize('parsing_tuple', [('https://username:PAT@github.com/repo/archive/commit_hash.zip', 'https_username_PAT_github_com_repo_archive_commit_hash.zip'), ('https://un:pwd@gitlab.com/user/repo/-/archive/commit_hash/repo-commit_hash.zip', 'https_un_pwd_gitlab_com_user_repo_-_archive_commit_hash_repo-commit_hash.zip')])
    def test_parse_private_git_https_uris(self, parsing_tuple):
        if False:
            print('Hello World!')
        (raw_uri, parsed_uri) = parsing_tuple
        (parsed_protocol, parsed_package_name) = parse_uri(raw_uri)
        assert parsed_protocol == Protocol.HTTPS
        assert parsed_package_name == parsed_uri

    @pytest.mark.parametrize('parsing_tuple', [('https://username:PAT@github.com/repo/archive:2/commit_hash.zip', Protocol.HTTPS, 'https_username_PAT_github_com_repo_archive_2_commit_hash.zip'), ('gs://fake/2022-10-21T13:11:35+00:00/package.zip', Protocol.GS, 'gs_fake_2022-10-21T13_11_35_00_00_package.zip'), ('s3://fake/2022-10-21T13:11:35+00:00/package.zip', Protocol.S3, 's3_fake_2022-10-21T13_11_35_00_00_package.zip'), ('file:///fake/2022-10-21T13:11:35+00:00/package.zip', Protocol.FILE, 'file__fake_2022-10-21T13_11_35_00_00_package.zip')])
    def test_parse_uris_with_disallowed_chars(self, parsing_tuple):
        if False:
            for i in range(10):
                print('nop')
        (raw_uri, protocol, parsed_uri) = parsing_tuple
        (parsed_protocol, parsed_package_name) = parse_uri(raw_uri)
        assert parsed_protocol == protocol
        assert parsed_package_name == parsed_uri

    @pytest.mark.parametrize('parsing_tuple', [('https://username:PAT@github.com/repo/archive:2/commit_hash.whl', Protocol.HTTPS, 'commit_hash.whl'), ('gs://fake/2022-10-21T13:11:35+00:00/package.whl', Protocol.GS, 'package.whl'), ('s3://fake/2022-10-21T13:11:35+00:00/package.whl', Protocol.S3, 'package.whl'), ('file:///fake/2022-10-21T13:11:35+00:00/package.whl', Protocol.FILE, 'package.whl')])
    def test_parse_remote_whl_uris(self, parsing_tuple):
        if False:
            return 10
        (raw_uri, protocol, parsed_uri) = parsing_tuple
        (parsed_protocol, parsed_package_name) = parse_uri(raw_uri)
        assert parsed_protocol == protocol
        assert parsed_package_name == parsed_uri

    @pytest.mark.parametrize('gcs_uri', ['gcs://pip_install_test-0.5-py3-none-any.whl', 'gcs://storing@here.zip'])
    def test_parse_gcs_uri(self, gcs_uri):
        if False:
            return 10
        'GCS URIs should not be modified in this function.'
        (protocol, package_name) = parse_uri(gcs_uri)
        assert protocol == Protocol.GCS
        assert package_name == gcs_uri.split('/')[-1]

@pytest.mark.asyncio
class TestDownloadAndUnpackPackage:

    async def test_download_and_unpack_package_with_gcs_uri_without_gcs_client(self, ray_start_regular):
        with tempfile.TemporaryDirectory() as temp_dir:
            zipfile_path = Path(temp_dir) / 'test-zip-file.zip'
            with zipfile.ZipFile(zipfile_path, 'x') as zip:
                zip.writestr('file.txt', 'Hello, world!')
            pkg_uri = 'gcs://my-zipfile.zip'
            upload_package_to_gcs(pkg_uri, zipfile_path.read_bytes())
            with pytest.raises(ValueError):
                await download_and_unpack_package(pkg_uri=pkg_uri, base_directory=temp_dir, gcs_aio_client=None)

    async def test_download_and_unpack_package_with_gcs_uri(self, ray_start_regular):
        gcs_aio_client = GcsAioClient(address=ray._private.worker.global_worker.gcs_client.address)
        with tempfile.TemporaryDirectory() as temp_dir:
            zipfile_path = Path(temp_dir) / 'test-zip-file.zip'
            with zipfile.ZipFile(zipfile_path, 'x') as zip:
                zip.writestr('file.txt', 'Hello, world!')
            pkg_uri = 'gcs://my-zipfile.zip'
            upload_package_to_gcs(pkg_uri, zipfile_path.read_bytes())
            local_dir = await download_and_unpack_package(pkg_uri=pkg_uri, base_directory=temp_dir, gcs_aio_client=gcs_aio_client)
            assert (Path(local_dir) / 'file.txt').exists()

    async def test_download_and_unpack_package_with_https_uri(self):
        with tempfile.TemporaryDirectory() as temp_dest_dir:
            local_dir = await download_and_unpack_package(pkg_uri=HTTPS_PACKAGE_URI, base_directory=temp_dest_dir)
            assert (Path(local_dir) / 'test_module').exists()

    async def test_download_and_unpack_package_with_s3_uri(self):
        with tempfile.TemporaryDirectory() as temp_dest_dir:
            local_dir = await download_and_unpack_package(pkg_uri=S3_PACKAGE_URI, base_directory=temp_dest_dir)
            assert (Path(local_dir) / 'test_module').exists()
        with tempfile.TemporaryDirectory() as temp_dest_dir:
            wheel_uri = await download_and_unpack_package(pkg_uri=S3_WHL_PACKAGE_URI, base_directory=temp_dest_dir)
            assert (Path(local_dir) / wheel_uri).exists()

    async def test_download_and_unpack_package_with_file_uri(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zipfile_path = Path(temp_dir) / 'test-zip-file.zip'
            with zipfile.ZipFile(zipfile_path, 'x') as zip:
                zip.writestr('file.txt', 'Hello, world!')
            from urllib.request import pathname2url
            from urllib.parse import urljoin
            file_path = pathname2url(str(zipfile_path))
            pkg_uri = urljoin('file:', file_path[1:])
            local_dir = await download_and_unpack_package(pkg_uri=pkg_uri, base_directory=temp_dir)
            assert (Path(local_dir) / 'file.txt').exists()

    @pytest.mark.parametrize('protocol', [Protocol.CONDA, Protocol.PIP])
    async def test_download_and_unpack_package_with_unsupported_protocol(self, protocol: Protocol):
        pkg_uri = f'{protocol.value}://some-package.zip'
        with pytest.raises(NotImplementedError) as excinfo:
            await download_and_unpack_package(pkg_uri=pkg_uri, base_directory='/tmp')
        assert f'{protocol.name} is not supported' in str(excinfo.value)

    @pytest.mark.parametrize('invalid_pkg_uri', ['gcs://gcs-cannot-have-a-folder/my-zipfile.zip', 's3://file-wihout-file-extension'])
    async def test_download_and_unpack_package_with_invalid_uri(self, invalid_pkg_uri: str):
        with pytest.raises(ValueError) as excinfo:
            await download_and_unpack_package(pkg_uri=invalid_pkg_uri, base_directory='/tmp')
        assert 'Invalid package URI' in str(excinfo.value)

def test_get_gitignore(tmp_path):
    if False:
        while True:
            i = 10
    gitignore_path = tmp_path / '.gitignore'
    gitignore_path.write_text('*.pyc')
    assert _get_gitignore(tmp_path)(Path(tmp_path / 'foo.pyc')) is True
    assert _get_gitignore(tmp_path)(Path(tmp_path / 'foo.py')) is False

@pytest.mark.parametrize('ignore_gitignore', [True, False])
@pytest.mark.skipif(sys.platform == 'win32', reason='Fails on windows')
def test_travel(tmp_path, ignore_gitignore, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    dir_paths = set()
    file_paths = set()
    item_num = 0
    excludes = []
    root = tmp_path / 'test'
    if ignore_gitignore:
        monkeypatch.setenv(RAY_RUNTIME_ENV_IGNORE_GITIGNORE, '1')
    else:
        monkeypatch.delenv(RAY_RUNTIME_ENV_IGNORE_GITIGNORE, raising=False)

    def construct(path, excluded=False, depth=0):
        if False:
            print('Hello World!')
        nonlocal item_num
        path.mkdir(parents=True)
        if not excluded:
            dir_paths.add(str(path))
        if depth > 8:
            return
        if item_num > 500:
            return
        dir_num = random.randint(0, 10)
        file_num = random.randint(0, 10)
        for _ in range(dir_num):
            uid = str(uuid.uuid4()).split('-')[0]
            dir_path = path / uid
            exclud_sub = random.randint(0, 5) == 0
            if not excluded and exclud_sub:
                excludes.append(str(dir_path.relative_to(root)))
            if not excluded:
                construct(dir_path, exclud_sub or excluded, depth + 1)
            item_num += 1
        if item_num > 1000:
            return
        for _ in range(file_num):
            uid = str(uuid.uuid4()).split('-')[0]
            v = random.randint(0, 1000)
            with (path / uid).open('w') as f:
                f.write(str(v))
            if not excluded:
                if random.randint(0, 5) == 0:
                    excludes.append(str((path / uid).relative_to(root)))
                else:
                    file_paths.add((str(path / uid), str(v)))
            item_num += 1
        gitignore = root / '.gitignore'
        gitignore.write_text('*.pyc')
        file_paths.add((str(gitignore), '*.pyc'))
        with (root / 'foo.pyc').open('w') as f:
            f.write('foo')
        if ignore_gitignore:
            file_paths.add((str(root / 'foo.pyc'), 'foo'))
    construct(root)
    exclude_spec = _get_excludes(root, excludes)
    visited_dir_paths = set()
    visited_file_paths = set()

    def handler(path):
        if False:
            i = 10
            return i + 15
        if path.is_dir():
            visited_dir_paths.add(str(path))
        else:
            with open(path) as f:
                visited_file_paths.add((str(path), f.read()))
    _dir_travel(root, [exclude_spec], handler)
    assert file_paths == visited_file_paths
    assert dir_paths == visited_dir_paths

def test_is_whl_uri():
    if False:
        i = 10
        return i + 15
    assert is_whl_uri('gcs://my-package.whl')
    assert not is_whl_uri('gcs://asdf.zip')
    assert not is_whl_uri('invalid_format')

def test_is_zip_uri():
    if False:
        while True:
            i = 10
    assert is_zip_uri('s3://my-package.zip')
    assert is_zip_uri('gcs://asdf.zip')
    assert not is_zip_uri('invalid_format')
    assert not is_zip_uri('gcs://a.whl')

def test_get_uri_for_package():
    if False:
        for i in range(10):
            print('nop')
    assert get_uri_for_package(Path('/tmp/my-pkg.whl')) == 'gcs://my-pkg.whl'

def test_get_local_dir_from_uri():
    if False:
        i = 10
        return i + 15
    uri = 'gcs://<working_dir_content_hash>.zip'
    assert get_local_dir_from_uri(uri, 'base_dir') == Path('base_dir/<working_dir_content_hash>')
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))