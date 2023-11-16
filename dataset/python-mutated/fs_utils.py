import contextlib
import errno
import functools
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
from typing import List, Optional
from urllib.parse import unquote, urlparse
import certifi
import fsspec
import h5py
import pyarrow.fs
import urllib3
from filelock import FileLock
from fsspec.core import split_protocol
from ludwig.api_annotations import DeveloperAPI
logger = logging.getLogger(__name__)

@DeveloperAPI
def get_default_cache_location() -> str:
    if False:
        print('Hello World!')
    'Returns a path to the default LUDWIG_CACHE location, or $HOME/.ludwig_cache.'
    cache_path = None
    if 'LUDWIG_CACHE' in os.environ and os.environ['LUDWIG_CACHE']:
        cache_path = os.environ['LUDWIG_CACHE']
    else:
        cache_path = str(pathlib.Path.home().joinpath('.ludwig_cache'))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path

@DeveloperAPI
def get_fs_and_path(url):
    if False:
        while True:
            i = 10
    (protocol, path) = split_protocol(url)
    path = unquote(urlparse(path).path)
    path = os.fspath(pathlib.PurePosixPath(path))
    fs = fsspec.filesystem(protocol)
    return (fs, path)

@DeveloperAPI
def has_remote_protocol(url):
    if False:
        i = 10
        return i + 15
    (protocol, _) = split_protocol(url)
    return protocol and protocol != 'file'

@DeveloperAPI
def is_http(urlpath):
    if False:
        i = 10
        return i + 15
    (protocol, _) = split_protocol(urlpath)
    return protocol == 'http' or protocol == 'https'

@DeveloperAPI
def upgrade_http(urlpath):
    if False:
        print('Hello World!')
    (protocol, url) = split_protocol(urlpath)
    if protocol == 'http':
        return 'https://' + url
    return None

@DeveloperAPI
@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_path(path: str) -> Optional[bytes]:
    if False:
        return 10
    if is_http(path):
        try:
            return get_bytes_obj_from_http_path(path)
        except Exception as e:
            logger.warning(e)
            return None
    else:
        try:
            with open_file(path) as f:
                return f.read()
        except OSError as e:
            logger.warning(e)
            return None

@DeveloperAPI
def stream_http_get_request(path: str) -> urllib3.response.HTTPResponse:
    if False:
        for i in range(10):
            print('nop')
    if upgrade_http(path):
        http = urllib3.PoolManager()
    else:
        http = urllib3.PoolManager(ca_certs=certifi.where())
    resp = http.request('GET', path, preload_content=False)
    return resp

@DeveloperAPI
@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_http_path(path: str) -> bytes:
    if False:
        while True:
            i = 10
    resp = stream_http_get_request(path)
    if resp.status == 404:
        upgraded = upgrade_http(path)
        if upgraded:
            logger.info(f'reading url {path} failed. upgrading to https and retrying')
            return get_bytes_obj_from_http_path(upgraded)
        else:
            raise urllib3.exceptions.HTTPError(f'reading url {path} failed and cannot be upgraded to https')
    data = b''
    for chunk in resp.stream(1024):
        data += chunk
    return data

@DeveloperAPI
def find_non_existing_dir_by_adding_suffix(directory_name):
    if False:
        print('Hello World!')
    (fs, _) = get_fs_and_path(directory_name)
    suffix = 0
    curr_directory_name = directory_name
    while fs.exists(curr_directory_name):
        curr_directory_name = directory_name + '_' + str(suffix)
        suffix += 1
    return curr_directory_name

@DeveloperAPI
def abspath(url):
    if False:
        return 10
    (protocol, _) = split_protocol(url)
    if protocol is not None:
        return url
    return os.path.abspath(url)

@DeveloperAPI
def path_exists(url):
    if False:
        return 10
    (fs, path) = get_fs_and_path(url)
    return fs.exists(path)

@DeveloperAPI
def listdir(url):
    if False:
        i = 10
        return i + 15
    (fs, path) = get_fs_and_path(url)
    return fs.listdir(path)

@DeveloperAPI
def safe_move_file(src, dst):
    if False:
        print('Hello World!')
    'Rename a file from `src` to `dst`. Inspired by: https://alexwlchan.net/2019/03/atomic-cross-filesystem-\n    moves-in-python/\n\n    *   Moves must be atomic.  `shutil.move()` is not atomic.\n\n    *   Moves must work across filesystems.  Sometimes temp directories and the\n        model directories live on different filesystems.  `os.replace()` will\n        throw errors if run across filesystems.\n\n    So we try `os.replace()`, but if we detect a cross-filesystem copy, we\n    switch to `shutil.move()` with some wrappers to make it atomic.\n    '
    try:
        os.replace(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            copy_id = uuid.uuid4()
            tmp_dst = f'{dst}.{copy_id}.tmp'
            shutil.copyfile(src, tmp_dst)
            os.replace(tmp_dst, dst)
            os.unlink(src)
        else:
            raise

@DeveloperAPI
def safe_move_directory(src, dst):
    if False:
        while True:
            i = 10
    'Recursively moves files from src directory to dst directory and removes src directory.\n\n    If dst directory does not exist, it will be created.\n    '
    try:
        os.replace(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            copy_id = uuid.uuid4()
            tmp_dst = f'{dst}.{copy_id}.tmp'
            shutil.copytree(src, tmp_dst)
            os.replace(tmp_dst, dst)
            os.unlink(src)
        else:
            raise

@DeveloperAPI
def rename(src, tgt):
    if False:
        i = 10
        return i + 15
    (protocol, _) = split_protocol(tgt)
    if protocol is not None:
        fs = fsspec.filesystem(protocol)
        fs.mv(src, tgt, recursive=True)
    else:
        safe_move_file(src, tgt)

@DeveloperAPI
def upload_file(src, tgt):
    if False:
        return 10
    (protocol, _) = split_protocol(tgt)
    fs = fsspec.filesystem(protocol)
    fs.put(src, tgt)

@DeveloperAPI
def copy(src, tgt, recursive=False):
    if False:
        i = 10
        return i + 15
    (protocol, _) = split_protocol(tgt)
    fs = fsspec.filesystem(protocol)
    fs.copy(src, tgt, recursive=recursive)

@DeveloperAPI
def makedirs(url, exist_ok=False):
    if False:
        while True:
            i = 10
    (fs, path) = get_fs_and_path(url)
    fs.makedirs(path, exist_ok=exist_ok)

@DeveloperAPI
def delete(url, recursive=False):
    if False:
        return 10
    (fs, path) = get_fs_and_path(url)
    return fs.delete(path, recursive=recursive)

@DeveloperAPI
def upload(lpath, rpath):
    if False:
        return 10
    (fs, path) = get_fs_and_path(rpath)
    pyarrow.fs.copy_files(lpath, path, destination_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs)))

@DeveloperAPI
def download(rpath, lpath):
    if False:
        while True:
            i = 10
    (fs, path) = get_fs_and_path(rpath)
    pyarrow.fs.copy_files(path, lpath, source_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs)))

@DeveloperAPI
def checksum(url):
    if False:
        while True:
            i = 10
    (fs, path) = get_fs_and_path(url)
    return fs.checksum(path)

@DeveloperAPI
def to_url(path):
    if False:
        while True:
            i = 10
    (protocol, _) = split_protocol(path)
    if protocol is not None:
        return path
    return pathlib.Path(os.path.abspath(path)).as_uri()

@DeveloperAPI
@contextlib.contextmanager
def upload_output_directory(url):
    if False:
        i = 10
        return i + 15
    if url is None:
        yield (None, None)
        return
    (protocol, _) = split_protocol(url)
    if protocol is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (fs, remote_path) = get_fs_and_path(url)
            if path_exists(url):
                fs.get(url, tmpdir + '/', recursive=True)

            def put_fn():
                if False:
                    print('Hello World!')
                pyarrow.fs.copy_files(tmpdir, remote_path, destination_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs)))
            yield (tmpdir, put_fn)
            put_fn()
    else:
        makedirs(url, exist_ok=True)
        yield (url, None)

@DeveloperAPI
@contextlib.contextmanager
def open_file(url, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    (fs, path) = get_fs_and_path(url)
    with fs.open(path, *args, **kwargs) as f:
        yield f

@DeveloperAPI
@contextlib.contextmanager
def download_h5(url):
    if False:
        return 10
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(url))
        (fs, path) = get_fs_and_path(url)
        fs.get(path, local_path)
        with h5py.File(local_path, 'r') as f:
            yield f

@DeveloperAPI
@contextlib.contextmanager
def upload_h5(url):
    if False:
        for i in range(10):
            print('nop')
    with upload_output_file(url) as local_fname:
        mode = 'w'
        if url == local_fname and path_exists(url):
            mode = 'r+'
        with h5py.File(local_fname, mode) as f:
            yield f

@DeveloperAPI
@contextlib.contextmanager
def upload_output_file(url):
    if False:
        i = 10
        return i + 15
    'Takes a remote URL as input, returns a temp filename, then uploads it when done.'
    (protocol, _) = split_protocol(url)
    if protocol is not None:
        fs = fsspec.filesystem(protocol)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_fname = os.path.join(tmpdir, 'tmpfile')
            yield local_fname
            fs.put(local_fname, url, recursive=True)
    else:
        yield url

@DeveloperAPI
class file_lock(contextlib.AbstractContextManager):
    """File lock based on filelock package."""

    def __init__(self, path: str, ignore_remote_protocol: bool=True, lock_file: str='.lock') -> None:
        if False:
            print('Hello World!')
        if not isinstance(path, (str, os.PathLike, pathlib.Path)):
            self.lock = None
        else:
            path = os.path.join(path, lock_file) if os.path.isdir(path) else f'{path}./{lock_file}'
            if ignore_remote_protocol and has_remote_protocol(path):
                self.lock = None
            else:
                self.lock = FileLock(path, timeout=-1)

    def __enter__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.lock:
            return self.lock.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.lock:
            return self.lock.__exit__(*args, **kwargs)

@DeveloperAPI
def list_file_names_in_directory(directory_name: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    file_path: pathlib.Path
    file_names: List[str] = [file_path.name for file_path in pathlib.Path(directory_name).iterdir() if file_path.is_file()]
    return file_names