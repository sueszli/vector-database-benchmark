import logging
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Optional
import tqdm
__all__ = ['NNI_BLOB', 'load_or_download_file', 'upload_file', 'nni_cache_home']
NNI_BLOB = 'https://nni.blob.core.windows.net'
ENV_NNI_HOME = 'NNI_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

def nni_cache_home() -> str:
    if False:
        i = 10
        return i + 15
    return os.path.expanduser(os.getenv(ENV_NNI_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'nni')))

def load_or_download_file(local_path: str, download_url: str, download: bool=False, progress: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Download the ``download_url`` to ``local_path``, and check its hash.\n\n    If ``local_path`` already exists, and hash is checked, do nothing.\n    '
    f = None
    hash_prefix = Path(local_path).stem.split('-')[-1]
    _logger = logging.getLogger(__name__)
    try:
        sha256 = hashlib.sha256()
        if Path(local_path).exists():
            _logger.info('"%s" already exists. Checking hash.', local_path)
            with Path(local_path).open('rb') as fr:
                while True:
                    chunk = fr.read(8192)
                    if len(chunk) == 0:
                        break
                    sha256.update(chunk)
        elif download:
            _logger.info('"%s" does not exist. Downloading "%s"', local_path, download_url)
            dst_dir = Path(local_path).parent
            dst_dir.mkdir(exist_ok=True, parents=True)
            if which('azcopy') is not None:
                output_level = []
                if not progress:
                    output_level = ['--output-level', 'quiet']
                subprocess.run(['azcopy', 'copy', download_url, local_path] + output_level, check=True)
                with Path(local_path).open('rb') as fr:
                    while True:
                        chunk = fr.read(8192)
                        if len(chunk) == 0:
                            break
                        sha256.update(chunk)
            else:
                _logger.info('azcopy is not installed. Fall back to use requests.')
                import requests
                f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
                r = requests.get(download_url, stream=True)
                total_length: Optional[str] = r.headers.get('content-length')
                assert total_length is not None, f'Content length is not found in the response of {download_url}'
                with tqdm.tqdm(total=int(total_length), disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        sha256.update(chunk)
                        pbar.update(len(chunk))
                        f.flush()
                f.close()
        else:
            raise FileNotFoundError('Download is not enabled, and file does not exist: {}. Please set download=True.'.format(local_path))
        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError(f'Invalid hash value (expected "{hash_prefix}", got "{digest}") for {local_path}. Please delete the file and try re-downloading.')
        if f is not None:
            shutil.move(f.name, local_path)
    finally:
        if f is not None:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)

def upload_file(local_path: str, destination_path: str, sas_token: str) -> str:
    if False:
        return 10
    "For NNI maintainers to add updated static files to the Azure blob easily.\n    In most cases, you don't need to calculate the hash on your own, it will be automatically inserted.\n    For example, if you write ``https://xxx.com/myfile.zip``, the uploaded file will look like\n    ``https://xxx.com/myfile-da5f43b7.zip``.\n\n    Need to have `azcopy installed <https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy>`_,\n    and a SAS token for the destination storage (``?`` should be included as prefix of token).\n\n    Returns a string which is the uploaded path.\n    "
    _logger = logging.getLogger(__name__)
    sha256 = hashlib.sha256()
    with Path(local_path).open('rb') as fr:
        while True:
            chunk = fr.read(8192)
            if len(chunk) == 0:
                break
            sha256.update(chunk)
    digest = sha256.hexdigest()
    hash_prefix = digest[:8]
    _logger.info('Hash of %s is %s', local_path, digest)
    (stem, suffix) = destination_path.rsplit('.', 1)
    if not stem.endswith('-' + hash_prefix):
        destination_path = stem + '-' + hash_prefix + '.' + suffix
    subprocess.run(['azcopy', 'copy', local_path, destination_path + sas_token], check=True)
    return destination_path