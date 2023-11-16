import os
import logging
import requests
import math
import zipfile
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from tqdm import tqdm
from retrying import retry
log = logging.getLogger(__name__)

@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def maybe_download(url, filename=None, work_directory='.', expected_bytes=None):
    if False:
        for i in range(10):
            print('nop')
    'Download a file if it is not already downloaded.\n\n    Args:\n        filename (str): File name.\n        work_directory (str): Working directory.\n        url (str): URL of the file to download.\n        expected_bytes (int): Expected file size in bytes.\n\n    Returns:\n        str: File path of the file downloaded.\n    '
    if filename is None:
        filename = url.split('/')[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            log.info(f'Downloading {url}')
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, 'wb') as file:
                for data in tqdm(r.iter_content(block_size), total=num_iterables, unit='KB', unit_scale=True):
                    file.write(data)
        else:
            log.error(f'Problem downloading {url}')
            r.raise_for_status()
    else:
        log.info(f'File {filepath} already downloaded')
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError(f'Failed to verify {filepath}')
    return filepath

@contextmanager
def download_path(path=None):
    if False:
        return 10
    'Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,\n    otherwise the real path of the input.\n\n    Args:\n        path (str): Path to download data.\n\n    Returns:\n        str: Real path where the data is stored.\n\n    Examples:\n        >>> with download_path() as path:\n        >>> ... maybe_download(url="http://example.com/file.zip", work_directory=path)\n\n    '
    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path

def unzip_file(zip_src, dst_dir, clean_zip_file=False):
    if False:
        for i in range(10):
            print('nop')
    'Unzip a file\n\n    Args:\n        zip_src (str): Zip file.\n        dst_dir (str): Destination folder.\n        clean_zip_file (bool): Whether or not to clean the zip file.\n    '
    fz = zipfile.ZipFile(zip_src, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)