"""
Tools for pulling an upstream typeshed zip archive from github, cleaning out
irrelevant data, and producing either a typeshed.Typeshed object or a
directory.
"""
from __future__ import annotations
import io
import json
import logging
import pathlib
import shutil
import urllib.request
import zipfile
from . import typeshed
LOG: logging.Logger = logging.getLogger(__name__)
LATEST: str = 'LATEST'

def get_default_typeshed_url() -> str:
    if False:
        i = 10
        return i + 15
    commit_hash = json.loads(urllib.request.urlopen('https://api.github.com/repos/python/typeshed/commits/main').read().decode('utf-8'))['sha']
    LOG.info(f'Found typeshed main at commit {commit_hash}')
    return f'https://api.github.com/repos/python/typeshed/zipball/{commit_hash}'

def get_typeshed_url(specified_url: str) -> str:
    if False:
        return 10
    if specified_url != LATEST:
        return specified_url
    LOG.info('Typeshed URL not specified. Trying to auto-determine it...')
    default_url = get_default_typeshed_url()
    if default_url is None:
        raise RuntimeError('Cannot determine the default typeshed URL. ' + 'Please manually specify one with `--url` argument. ' + 'If the download still fails, please check network connectivity.')
    return default_url

def fetch_as_zipped_bytes(url: str) -> zipfile.ZipFile:
    if False:
        print('Hello World!')
    raw_bytes = io.BytesIO()
    with urllib.request.urlopen(url) as response:
        shutil.copyfileobj(response, raw_bytes)
    return zipfile.ZipFile(raw_bytes)

def should_include_zipinfo(info: zipfile.ZipInfo) -> bool:
    if False:
        while True:
            i = 10
    if info.is_dir():
        return False
    parts = pathlib.Path(info.filename).parts
    if len(parts) < 2 or parts[1] not in ('stubs', 'stdlib'):
        return False
    if parts[-1].endswith('.txt') or parts[-1].endswith('.toml'):
        return False
    if '@python2' in parts:
        return False
    return True

def relative_path_for_zipinfo(info: zipfile.ZipInfo) -> pathlib.Path:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a filename within a zipped typeshed into a path relative to the\n    top of the typeshed repository.\n    '
    return pathlib.Path(*pathlib.Path(info.filename).parts[1:])

def fetch_as_typeshed(url: str) -> typeshed.Typeshed:
    if False:
        for i in range(10):
            print('nop')
    url = get_typeshed_url(url)
    with fetch_as_zipped_bytes(url) as zipped_bytes:
        contents = {relative_path_for_zipinfo(info): zipped_bytes.read(info).decode('utf-8') for info in zipped_bytes.infolist() if should_include_zipinfo(info)}
        contents[pathlib.Path('source_url')] = f'{url}\n'
        return typeshed.MemoryBackedTypeshed(contents)

def fetch_as_directory(url: str, target: pathlib.Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    upstream_typeshed = fetch_as_typeshed(url)
    typeshed.write_to_directory(typeshed=upstream_typeshed, target=target)