"""
Script to download, trim, and patch typeshed zip files.

- By "trimming" we mean removing files only needed for python2
- By "patching" patching we mean altering the stubs. Pyre's patches are
  stored in ./typeshed-patches. These patches are needed because:
  - off-the-shelf typeshed sometimes doesn't play well with Pyre
  - Pyre has some custom extensions (e.g. type arithmetic) that require
    modifying standard library stubs.
"""
from __future__ import annotations
import argparse
import dataclasses
import io
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional
LOG: logging.Logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class FileEntry:
    path: str
    data: Optional[bytes]

@dataclasses.dataclass(frozen=True)
class PatchResult:
    entry: FileEntry
    failed: bool

@dataclasses.dataclass(frozen=True)
class FileCount:
    kept: int
    dropped: int

@dataclasses.dataclass(frozen=True)
class Statistics:
    stdlib: FileCount
    third_party: FileCount
    third_party_package_count: int

@dataclasses.dataclass(frozen=True)
class TrimmedTypeshed:
    entries: List[FileEntry]
    statistics: Statistics

    @classmethod
    def _from_zip_file(cls, zip_file: zipfile.ZipFile) -> TrimmedTypeshed:
        if False:
            for i in range(10):
                print('nop')
        result_info = []
        stdlib_kept_count = 0
        stdlib_dropped_count = 0
        third_party_kept_count = 0
        third_party_dropped_count = 0
        third_party_package_count = 0
        for info in zip_file.infolist():
            parts = pathlib.Path(info.filename).parts
            if parts[-1].endswith('.txt') or parts[-1].endswith('.toml'):
                continue
            if len(parts) <= 1:
                result_info.append(info)
            elif parts[1] == 'stdlib':
                if '@python2' in parts:
                    if not info.is_dir():
                        stdlib_dropped_count += 1
                else:
                    if not info.is_dir():
                        stdlib_kept_count += 1
                    result_info.append(info)
            elif parts[1] == 'stubs':
                if '@python2' in parts:
                    if not info.is_dir():
                        third_party_dropped_count += 1
                else:
                    if not info.is_dir():
                        third_party_kept_count += 1
                    elif len(parts) == 3:
                        third_party_package_count += 1
                    result_info.append(info)
        return cls(entries=[FileEntry(path=str(pathlib.Path('typeshed').joinpath(*pathlib.Path(info.filename).parts[1:])), data=None if info.is_dir() else zip_file.read(info)) for info in result_info], statistics=Statistics(stdlib=FileCount(kept=stdlib_kept_count, dropped=stdlib_dropped_count), third_party=FileCount(kept=third_party_kept_count, dropped=third_party_dropped_count), third_party_package_count=third_party_package_count))

    @classmethod
    def from_raw_zip(cls, downloaded: io.BytesIO) -> TrimmedTypeshed:
        if False:
            for i in range(10):
                print('nop')
        with zipfile.ZipFile(downloaded) as zip_file:
            return cls._from_zip_file(zip_file)

    def log_statistics(self) -> None:
        if False:
            while True:
                i = 10
        statistics = self.statistics
        LOG.info(f'Kept {statistics.stdlib.kept} files and dropped ' + f'{statistics.stdlib.dropped} files from stdlib.')
        LOG.info(f'Kept {statistics.third_party.kept} files and dropped ' + f'{statistics.third_party.dropped} files from third-party libraries.')
        LOG.info('Total number of third-party packages is ' + f'{statistics.third_party_package_count}.')

@dataclasses.dataclass(frozen=True)
class PatchedTypeshed:
    results: List[PatchResult]

    @staticmethod
    def _patch_entry(entry: FileEntry, temporary_path: Path, patch_path: Path) -> PatchResult:
        if False:
            i = 10
            return i + 15
        if entry.data is None or not patch_path.is_file():
            return PatchResult(entry, False)
        new_filepath = temporary_path / 'tempfile'
        new_filepath.write_bytes(entry.data)
        LOG.info(f'Applying patch {patch_path}')
        result = subprocess.run(['patch', '-u', new_filepath, '-i', patch_path])
        if result.returncode != 0:
            try:
                reject_file = temporary_path / 'tempfile.rej'
                rejected_patch = reject_file.read_text()
                LOG.error(f'Failed to apply patch. Rejected patch: {rejected_patch}')
            except FileNotFoundError:
                pass
            finally:
                raise RuntimeError(f'Failed to apply patch at {patch_path}')
        new_data = new_filepath.read_bytes()
        new_filepath.unlink()
        return PatchResult(FileEntry(entry.path, new_data), False)

    @staticmethod
    def _entry_path_to_patch_path(input: str) -> Path:
        if False:
            return 10
        'Removes the first component of the path, and changes the suffix to `.patch`.'
        parts = Path(input).with_suffix('.patch').parts
        return Path('/'.join(parts[1:]))

    @classmethod
    def from_trimmed_typeshed(cls, patch_directory: Path, trimmed_typeshed: TrimmedTypeshed) -> PatchedTypeshed:
        if False:
            return 10
        if not patch_directory.is_dir():
            raise ValueError(f'Specified patch directory {patch_directory} is not a directory')
        with tempfile.TemporaryDirectory() as temporary_root:
            temporary_root_path = Path(temporary_root)
            return cls([cls._patch_entry(entry, temporary_root_path, patch_directory / cls._entry_path_to_patch_path(entry.path)) for entry in trimmed_typeshed.entries])

def get_default_typeshed_url() -> str:
    if False:
        for i in range(10):
            print('nop')
    commit_hash = json.loads(urllib.request.urlopen('https://api.github.com/repos/python/typeshed/commits/main').read().decode('utf-8'))['sha']
    LOG.info(f'Found typeshed main at commit {commit_hash}')
    return f'https://api.github.com/repos/python/typeshed/zipball/{commit_hash}'

def get_typeshed_url(specified_url: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    if specified_url is not None:
        return specified_url
    LOG.info('Typeshed URL not specified. Trying to auto-determine it...')
    default_url = get_default_typeshed_url()
    if default_url is None:
        raise RuntimeError('Cannot determine the default typeshed URL. ' + 'Please manually specify one with `--url` argument. ' + 'If the download still fails, please check network connectivity.')
    return default_url

def download_typeshed(url: str) -> io.BytesIO:
    if False:
        while True:
            i = 10
    downloaded = io.BytesIO()
    with urllib.request.urlopen(url) as response:
        shutil.copyfileobj(response, downloaded)
    return downloaded

def write_output_to_directory(patched_typeshed: PatchedTypeshed, output_directory_path: Path, url: str) -> None:
    if False:
        print('Hello World!')
    os.makedirs(output_directory_path, exist_ok=False)
    for patch_result in patched_typeshed.results:
        path = output_directory_path / Path(patch_result.entry.path)
        data = patch_result.entry.data
        if data is not None:
            os.makedirs(path.parent, exist_ok=True)
            with path.open('wb') as output_file:
                if data is not None:
                    output_file.write(data)
                elif patch_result.failed:
                    LOG.warning(f'Failed to apply patch to {patch_result.entry.path}!')
                else:
                    pass
    with (output_directory_path / 'typeshed' / 'source_url').open('w') as output_file:
        output_file.write(f'{url}\n')

def _find_entry(typeshed_path: Path, entries: List[FileEntry]) -> Optional[FileEntry]:
    if False:
        while True:
            i = 10
    'Finds a particular entry in typeshed, given its path relative to\n    `typeshed-master`, possibly while having a different suffix.'
    for entry in entries:
        if entry.path == f"typeshed-master/{typeshed_path.with_suffix('.pyi')}" and entry.data is not None:
            return entry
    return None

def main() -> None:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='A script to download and trim typeshed zip file.')
    parser.add_argument('-u', '--url', type=str, help='URL from which typeshed zip file is downloaded. ' + 'If not set, default to the current typeshed mater on github.')
    parser.add_argument('-o', '--output', required=True, type=str, help='Where to store the downloaded typeshed (as either a directory or zip file).')
    parser.add_argument('-p', '--patch-directory', required=True, type=str, help='Where the .patch files for amending typeshed are located.')
    parser.set_defaults(as_directory=False)
    arguments = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s][%(levelname)s]: %(message)s', level=logging.INFO)
    patch_directory = Path(arguments.patch_directory)
    url = get_typeshed_url(arguments.url)
    downloaded = download_typeshed(url)
    LOG.info(f'{downloaded.getbuffer().nbytes} bytes downloaded from {url}')
    trimmed_typeshed = TrimmedTypeshed.from_raw_zip(downloaded)
    trimmed_typeshed.log_statistics()
    patched_typeshed = PatchedTypeshed.from_trimmed_typeshed(patch_directory, trimmed_typeshed)
    write_output_to_directory(patched_typeshed, Path(arguments.output), url)
    LOG.info(f'Patched typeshed directory written to {arguments.output}')
if __name__ == '__main__':
    main()