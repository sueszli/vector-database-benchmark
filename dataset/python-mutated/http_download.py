import hashlib
import os
import shutil
from tempfile import NamedTemporaryFile
import requests
from megfile import smart_copy, smart_getmd5, smart_getsize
from tqdm import tqdm
from ..logger import get_logger
logger = get_logger(__name__)
CHUNK_SIZE = 1024
HTTP_CONNECTION_TIMEOUT = 5

class HTTPDownloadError(BaseException):
    """The class that represents http request error."""

class Bar:

    def __init__(self, total=100):
        if False:
            print('Hello World!')
        self._bar = tqdm(total=total, unit='iB', unit_scale=True, ncols=80)

    def __call__(self, bytes_num):
        if False:
            print('Hello World!')
        self._bar.update(bytes_num)

def download_from_url(url: str, dst: str):
    if False:
        i = 10
        return i + 15
    'Downloads file from given url to ``dst``.\n\n    Args:\n        url: source URL.\n        dst: saving path.\n    '
    dst = os.path.expanduser(dst)
    smart_copy(url, dst, callback=Bar(total=smart_getsize(url)))
    return smart_getmd5(dst)