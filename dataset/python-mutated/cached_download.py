from __future__ import print_function
import hashlib
import os
import os.path as osp
import shutil
import sys
import tempfile
import filelock
from .download import download
cache_root = osp.join(osp.expanduser('~'), '.cache/gdown')
if not osp.exists(cache_root):
    try:
        os.makedirs(cache_root)
    except OSError:
        pass

def md5sum(filename, blocksize=None):
    if False:
        for i in range(10):
            print('nop')
    if blocksize is None:
        blocksize = 65536
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda : f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()

def assert_md5sum(filename, md5, quiet=False, blocksize=None):
    if False:
        return 10
    if not (isinstance(md5, str) and len(md5) == 32):
        raise ValueError('MD5 must be 32 chars: {}'.format(md5))
    if not quiet:
        print('Computing MD5: {}'.format(filename))
    md5_actual = md5sum(filename)
    if md5_actual == md5:
        if not quiet:
            print('MD5 matches: {}'.format(filename))
        return True
    raise AssertionError("MD5 doesn't match:\nactual: {}\nexpected: {}".format(md5_actual, md5))

def cached_download(url=None, path=None, md5=None, quiet=False, postprocess=None, **kwargs):
    if False:
        print('Hello World!')
    'Cached download from URL.\n\n    Parameters\n    ----------\n    url: str\n        URL. Google Drive URL is also supported.\n    path: str, optional\n        Output filename. Default is basename of URL.\n    md5: str, optional\n        Expected MD5 for specified file.\n    quiet: bool\n        Suppress terminal output. Default is False.\n    postprocess: callable\n        Function called with filename as postprocess.\n    kwargs: dict\n        Keyword arguments to be passed to `download`.\n\n    Returns\n    -------\n    path: str\n        Output filename.\n    '
    if path is None:
        path = url.replace('/', '-SLASH-').replace(':', '-COLON-').replace('=', '-EQUAL-').replace('?', '-QUESTION-')
        path = osp.join(cache_root, path)
    if osp.exists(path) and (not md5):
        if not quiet:
            print('File exists: {}'.format(path))
        return path
    elif osp.exists(path) and md5:
        try:
            assert_md5sum(path, md5, quiet=quiet)
            return path
        except AssertionError as e:
            print(e, file=sys.stderr)
    lock_path = osp.join(cache_root, '_dl_lock')
    try:
        os.makedirs(osp.dirname(path))
    except OSError:
        pass
    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = osp.join(temp_root, 'dl')
        if not quiet:
            msg = 'Cached Downloading'
            if path:
                msg = '{}: {}'.format(msg, path)
            else:
                msg = '{}...'.format(msg)
            print(msg, file=sys.stderr)
        download(url, temp_path, quiet=quiet, **kwargs)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, path)
    except Exception:
        shutil.rmtree(temp_root)
        raise
    if md5:
        assert_md5sum(path, md5, quiet=quiet)
    if postprocess is not None:
        postprocess(path)
    return path