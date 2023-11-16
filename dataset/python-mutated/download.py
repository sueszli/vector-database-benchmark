"""Download utility based on wget.

The utility is a real simple implementation leveraging the wget command line
application supporting resume on failed download.
"""
from glob import glob
import os
from os.path import exists, dirname
import subprocess
from threading import Thread
from .file_utils import ensure_directory_exists
_running_downloads = {}

def _get_download_tmp(dest):
    if False:
        i = 10
        return i + 15
    'Get temporary file for download.\n\n    Args:\n        dest (str): path to download location\n\n    Returns:\n        (str) path to temporary download location\n    '
    tmp_base = dest + '.part'
    existing = glob(tmp_base + '*')
    if len(existing) > 0:
        return '{}.{}'.format(tmp_base, len(existing))
    else:
        return tmp_base

class Downloader(Thread):
    """Simple file downloader.

    Downloader is a thread based downloader instance when instanciated
    it will download the provided url to a file on disk.

    When the download is complete or failed the `.done` property will
    be set to true and the `.status` will indicate the HTTP status code.
    200 = Success.

    Args:
        url (str): Url to download
        dest (str): Path to save data to
        complete_action (callable): Function to run when download is complete
                                    `func(dest)`
        header: any special header needed for starting the transfer
    """

    def __init__(self, url, dest, complete_action=None, header=None):
        if False:
            for i in range(10):
                print('nop')
        super(Downloader, self).__init__()
        self.url = url
        self.dest = dest
        self.complete_action = complete_action
        self.status = None
        self.done = False
        self._abort = False
        self.header = header
        ensure_directory_exists(dirname(dest), permissions=509)
        self.daemon = True
        self.start()

    def perform_download(self, dest):
        if False:
            while True:
                i = 10
        'Handle the download through wget.\n\n        Args:\n            dest (str): Save location\n        '
        cmd = ['wget', '-c', self.url, '-O', dest, '--tries=20', '--read-timeout=5']
        if self.header:
            cmd += ['--header={}'.format(self.header)]
        return subprocess.call(cmd)

    def run(self):
        if False:
            while True:
                i = 10
        'Do the actual download.'
        tmp = _get_download_tmp(self.dest)
        self.status = self.perform_download(tmp)
        if not self._abort and self.status == 0:
            self.finalize(tmp)
        else:
            self.cleanup(tmp)
        self.done = True
        arg_hash = hash(self.url + self.dest)
        if arg_hash in _running_downloads:
            _running_downloads.pop(arg_hash)

    def finalize(self, tmp):
        if False:
            while True:
                i = 10
        'Move temporary download data to final location.\n\n        Move the .part file to the final destination and perform any\n        actions that should be performed at completion.\n\n        Args:\n            tmp(str): temporary file path\n        '
        os.rename(tmp, self.dest)
        if self.complete_action:
            self.complete_action(self.dest)

    def cleanup(self, tmp):
        if False:
            i = 10
            return i + 15
        'Cleanup after download attempt.'
        if exists(tmp):
            os.remove(self.dest + '.part')
        if self.status == 200:
            self.status = -1

    def abort(self):
        if False:
            while True:
                i = 10
        'Abort download process.'
        self._abort = True

def download(url, dest, complete_action=None, header=None):
    if False:
        print('Hello World!')
    'Start a download or fetch an already running.\n\n    Args:\n        url (str): url to download\n        dest (str): path to save download to\n        complete_action (callable): Optional function to call on completion\n        header (str): Optional header to use for the download\n\n    Returns:\n        Downloader object\n    '
    global _running_downloads
    arg_hash = hash(url + dest)
    if arg_hash not in _running_downloads:
        _running_downloads[arg_hash] = Downloader(url, dest, complete_action, header)
    return _running_downloads[arg_hash]