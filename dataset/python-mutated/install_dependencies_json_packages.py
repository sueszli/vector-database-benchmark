"""Installation script for Oppia third-party libraries."""
from __future__ import annotations
import contextlib
from http import client
import io
import json
import os
import ssl
import sys
import tarfile
import urllib
from urllib import error as urlerror
from urllib import request as urlrequest
import zipfile
import certifi
from typing import BinaryIO, Dict, Final, List, Literal, TextIO, TypedDict, Union, cast, overload
DEPENDENCIES_FILE_PATH: Final = os.path.join(os.getcwd(), 'dependencies.json')
TOOLS_DIR: Final = os.path.join('..', 'oppia_tools')
THIRD_PARTY_DIR: Final = os.path.join('.', 'third_party')
THIRD_PARTY_STATIC_DIR: Final = os.path.join(THIRD_PARTY_DIR, 'static')
TMP_UNZIP_PATH: Final = os.path.join('.', 'tmp_unzip.zip')
TARGET_DOWNLOAD_DIRS: Final = {'proto': THIRD_PARTY_DIR, 'frontend': THIRD_PARTY_STATIC_DIR, 'oppiaTools': TOOLS_DIR}
_DOWNLOAD_FORMAT_ZIP: Final = 'zip'
_DOWNLOAD_FORMAT_TAR: Final = 'tar'
_DOWNLOAD_FORMAT_FILES: Final = 'files'
DownloadFormatType = Literal['zip', 'files', 'tar']

class DownloadFormatToDependenciesKeysDict(TypedDict):
    """TypeDict for download format to dependencies keys dict."""
    mandatory_keys: List[str]
    optional_key_pairs: List[List[str]]
DOWNLOAD_FORMATS_TO_DEPENDENCIES_KEYS: Dict[DownloadFormatType, DownloadFormatToDependenciesKeysDict] = {'zip': {'mandatory_keys': ['version', 'url', 'downloadFormat'], 'optional_key_pairs': [['rootDir', 'rootDirPrefix'], ['targetDir', 'targetDirPrefix']]}, 'files': {'mandatory_keys': ['version', 'url', 'files', 'targetDirPrefix', 'downloadFormat'], 'optional_key_pairs': []}, 'tar': {'mandatory_keys': ['version', 'url', 'tarRootDirPrefix', 'targetDirPrefix', 'downloadFormat'], 'optional_key_pairs': []}}
TextModeTypes = Literal['r', 'w', 'a', 'x', 'r+', 'w+', 'a+']
BinaryModeTypes = Literal['rb', 'wb', 'ab', 'xb', 'r+b', 'w+b', 'a+b', 'x+b']

def ensure_directory_exists(d: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates the given directory if it does not already exist.'
    if not os.path.exists(d):
        os.makedirs(d)

def url_retrieve(url: str, output_path: str, max_attempts: int=2, enforce_https: bool=True) -> None:
    if False:
        return 10
    "Retrieve a file from a URL and write the file to the file system.\n\n    Note that we use Python's recommended default settings for verifying SSL\n    connections, which are documented here:\n    https://docs.python.org/3/library/ssl.html#best-defaults.\n\n    Args:\n        url: str. The URL to retrieve the data from.\n        output_path: str. Path to the destination file where the data from the\n            URL will be written.\n        max_attempts: int. The maximum number of attempts that will be made to\n            download the data. For failures before the maximum number of\n            attempts, a message describing the error will be printed. Once the\n            maximum is hit, any errors will be raised.\n        enforce_https: bool. Whether to require that the provided URL starts\n            with 'https://' to ensure downloads are secure.\n\n    Raises:\n        Exception. Raised when the provided URL does not use HTTPS but\n            enforce_https is True.\n    "
    failures = 0
    success = False
    if enforce_https and (not url.startswith('https://')):
        raise Exception('The URL %s should use HTTPS.' % url)
    while not success and failures < max_attempts:
        try:
            with urlrequest.urlopen(url, context=ssl.create_default_context()) as response:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.read())
        except (urlerror.URLError, ssl.SSLError, client.IncompleteRead) as exception:
            failures += 1
            print('Attempt %d of %d failed when downloading %s.' % (failures, max_attempts, url))
            if failures >= max_attempts:
                raise exception
            print('Error: %s' % exception)
            print('Retrying download.')
        else:
            success = True

def url_open(source_url: Union[str, urllib.request.Request]) -> urllib.request._UrlopenRet:
    if False:
        i = 10
        return i + 15
    "Opens a URL and returns the response.\n\n    Args:\n        source_url: Union[str, Request]. The URL.\n\n    Returns:\n        urlopen. The 'urlopen' object.\n    "
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(source_url, context=context)

class DependencyDict(TypedDict, total=False):
    """Dict representation of dependency."""
    version: str
    downloadFormat: DownloadFormatType
    url: str
    rootDirPrefix: str
    rootDir: str
    targetDirPrefix: str
    targetDir: str
    tarRootDirPrefix: str
    files: List[str]
    bundle: Dict[str, List[str]]

class DependenciesDict(TypedDict):
    """Dict representation of dependencies."""
    dependencies: Dict[str, Dict[str, DependencyDict]]

@overload
def open_file(filename: str, mode: TextModeTypes, encoding: str='utf-8', newline: Union[str, None]=None) -> TextIO:
    if False:
        return 10
    ...

@overload
def open_file(filename: str, mode: BinaryModeTypes, encoding: Union[str, None]='utf-8', newline: Union[str, None]=None) -> BinaryIO:
    if False:
        for i in range(10):
            print('nop')
    ...

def open_file(filename: str, mode: Union[TextModeTypes, BinaryModeTypes], encoding: Union[str, None]='utf-8', newline: Union[str, None]=None) -> Union[BinaryIO, TextIO]:
    if False:
        print('Hello World!')
    'Open file and return a corresponding file object.\n\n    Args:\n        filename: str. The file to be opened.\n        mode: Literal. Mode in which the file is opened.\n        encoding: str. Encoding in which the file is opened.\n        newline: None|str. Controls how universal newlines work.\n\n    Returns:\n        IO[Any]. The file object.\n\n    Raises:\n        FileNotFoundError. The file cannot be found.\n    '
    file = cast(Union[BinaryIO, TextIO], open(filename, mode, encoding=encoding, newline=newline))
    return file

def download_files(source_url_root: str, target_dir: str, source_filenames: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Downloads a group of files and saves them to a given directory.\n\n    Each file is downloaded only if it does not already exist.\n\n    Args:\n        source_url_root: str. The URL to prepend to all the filenames.\n        target_dir: str. The directory to save the files to.\n        source_filenames: list(str). Each filename is appended to the\n            end of the source_url_root in order to give the URL from which to\n            download the file. The downloaded file is then placed in target_dir,\n            and retains the same filename.\n    '
    assert isinstance(source_filenames, list), "Expected list of filenames, got '%s'" % source_filenames
    ensure_directory_exists(target_dir)
    for filename in source_filenames:
        if not os.path.exists(os.path.join(target_dir, filename)):
            print('Downloading file %s to %s ...' % (filename, target_dir))
            url_retrieve('%s/%s' % (source_url_root, filename), os.path.join(target_dir, filename))
            print('Download of %s succeeded.' % filename)

def download_and_unzip_files(source_url: str, target_parent_dir: str, zip_root_name: str, target_root_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Downloads a zip file, unzips it, and saves the result in a given dir.\n\n    The download occurs only if the target directory that the zip file unzips\n    to does not exist.\n\n    NB: This function assumes that the root level of the zip file has exactly\n    one folder.\n\n    Args:\n        source_url: str. The URL from which to download the zip file.\n        target_parent_dir: str. The directory to save the contents of the zip\n            file to.\n        zip_root_name: str. The name of the top-level folder in the zip\n            directory.\n        target_root_name: str. The name that the top-level folder should be\n            renamed to in the local directory.\n    '
    if not os.path.exists(os.path.join(target_parent_dir, target_root_name)):
        print('Downloading and unzipping file %s to %s ...' % (zip_root_name, target_parent_dir))
        ensure_directory_exists(target_parent_dir)
        url_retrieve(source_url, TMP_UNZIP_PATH)
        try:
            with zipfile.ZipFile(TMP_UNZIP_PATH, 'r') as zfile:
                zfile.extractall(path=target_parent_dir)
            os.remove(TMP_UNZIP_PATH)
        except Exception:
            if os.path.exists(TMP_UNZIP_PATH):
                os.remove(TMP_UNZIP_PATH)
            req = urllib.request.Request(source_url, None, {})
            req.add_header('User-agent', 'python')
            file_stream = io.BytesIO(url_open(req).read())
            with zipfile.ZipFile(file_stream, 'r') as zfile:
                zfile.extractall(path=target_parent_dir)
        os.rename(os.path.join(target_parent_dir, zip_root_name), os.path.join(target_parent_dir, target_root_name))
        print('Download of %s succeeded.' % zip_root_name)

def download_and_untar_files(source_url: str, target_parent_dir: str, tar_root_name: str, target_root_name: str) -> None:
    if False:
        return 10
    'Downloads a tar file, untars it, and saves the result in a given dir.\n\n    The download occurs only if the target directory that the tar file untars\n    to does not exist.\n\n    NB: This function assumes that the root level of the tar file has exactly\n    one folder.\n\n    Args:\n        source_url: str. The URL from which to download the tar file.\n        target_parent_dir: str. The directory to save the contents of the tar\n            file to.\n        tar_root_name: str. The name of the top-level folder in the tar\n            directory.\n        target_root_name: str. The name that the top-level folder should be\n            renamed to in the local directory.\n    '
    if not os.path.exists(os.path.join(target_parent_dir, target_root_name)):
        print('Downloading and untarring file %s to %s ...' % (tar_root_name, target_parent_dir))
        ensure_directory_exists(target_parent_dir)
        url_retrieve(source_url, TMP_UNZIP_PATH)
        with contextlib.closing(tarfile.open(name=TMP_UNZIP_PATH, mode='r:gz')) as tfile:
            tfile.extractall(target_parent_dir)
        os.remove(TMP_UNZIP_PATH)
        os.rename(os.path.join(target_parent_dir, tar_root_name), os.path.join(target_parent_dir, target_root_name))
        print('Download of %s succeeded.' % tar_root_name)

def get_file_contents(filepath: str, mode: TextModeTypes='r') -> str:
    if False:
        for i in range(10):
            print('nop')
    'Gets the contents of a file, given a relative filepath from oppia/.'
    with open_file(filepath, mode) as f:
        return f.read()

def return_json(filepath: str) -> DependenciesDict:
    if False:
        while True:
            i = 10
    'Return json object when provided url\n\n    Args:\n        filepath: str. The path to the json file.\n\n    Returns:\n        *. A parsed json object. Actual conversion is different based on input\n        to json.loads. More details can be found here:\n            https://docs.python.org/3/library/json.html#encoders-and-decoders.\n    '
    response = get_file_contents(filepath)
    return cast(DependenciesDict, json.loads(response))

def test_dependencies_syntax(dependency_type: DownloadFormatType, dependency_dict: DependencyDict) -> None:
    if False:
        return 10
    'This checks syntax of the dependencies.json dependencies.\n    Display warning message when there is an error and terminate the program.\n\n    Args:\n        dependency_type: DownloadFormatType. Dependency download format.\n        dependency_dict: dict. A dependencies.json dependency dict.\n    '
    keys = list(dependency_dict.keys())
    mandatory_keys = DOWNLOAD_FORMATS_TO_DEPENDENCIES_KEYS[dependency_type]['mandatory_keys']
    optional_key_pairs = DOWNLOAD_FORMATS_TO_DEPENDENCIES_KEYS[dependency_type]['optional_key_pairs']
    for key in mandatory_keys:
        if key not in keys:
            print('------------------------------------------')
            print('There is syntax error in this dependency')
            print(dependency_dict)
            print('This key is missing or misspelled: "%s".' % key)
            print('Exiting')
            sys.exit(1)
    if optional_key_pairs:
        for optional_keys in optional_key_pairs:
            optional_keys_in_dict = [key for key in optional_keys if key in keys]
            if len(optional_keys_in_dict) != 1:
                print('------------------------------------------')
                print('There is syntax error in this dependency')
                print(dependency_dict)
                print('Only one of these keys pair must be used: "%s".' % ', '.join(optional_keys))
                print('Exiting')
                sys.exit(1)
    dependency_url = dependency_dict['url']
    if '#' in dependency_url:
        dependency_url = dependency_url.rpartition('#')[0]
    is_zip_file_format = dependency_type == _DOWNLOAD_FORMAT_ZIP
    is_tar_file_format = dependency_type == _DOWNLOAD_FORMAT_TAR
    if dependency_url.endswith('.zip') and (not is_zip_file_format) or (is_zip_file_format and (not dependency_url.endswith('.zip'))) or (dependency_url.endswith('.tar.gz') and (not is_tar_file_format)) or (is_tar_file_format and (not dependency_url.endswith('.tar.gz'))):
        print('------------------------------------------')
        print('There is syntax error in this dependency')
        print(dependency_dict)
        print('This url %s is invalid for %s file format.' % (dependency_url, dependency_type))
        print('Exiting.')
        sys.exit(1)

def validate_dependencies(filepath: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "This validates syntax of the dependencies.json\n\n    Args:\n        filepath: str. The path to the json file.\n\n    Raises:\n        Exception. The 'downloadFormat' not specified.\n    "
    dependencies_data = return_json(filepath)
    dependencies = dependencies_data['dependencies']
    for (_, dependency) in dependencies.items():
        for (_, dependency_contents) in dependency.items():
            if 'downloadFormat' not in dependency_contents:
                raise Exception('downloadFormat not specified in %s' % dependency_contents)
            download_format = dependency_contents['downloadFormat']
            test_dependencies_syntax(download_format, dependency_contents)

def download_all_dependencies(filepath: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'This download all files to the required folders.\n\n    Args:\n        filepath: str. The path to the json file.\n    '
    validate_dependencies(filepath)
    dependencies_data = return_json(filepath)
    dependencies = dependencies_data['dependencies']
    for (data, dependency) in dependencies.items():
        for (_, dependency_contents) in dependency.items():
            dependency_rev = dependency_contents['version']
            dependency_url = dependency_contents['url']
            download_format = dependency_contents['downloadFormat']
            if download_format == _DOWNLOAD_FORMAT_FILES:
                dependency_files = dependency_contents['files']
                target_dirname = dependency_contents['targetDirPrefix'] + dependency_rev
                dependency_dst = os.path.join(TARGET_DOWNLOAD_DIRS[data], target_dirname)
                download_files(dependency_url, dependency_dst, dependency_files)
            elif download_format == _DOWNLOAD_FORMAT_ZIP:
                if 'rootDir' in dependency_contents:
                    dependency_zip_root_name = dependency_contents['rootDir']
                else:
                    dependency_zip_root_name = dependency_contents['rootDirPrefix'] + dependency_rev
                if 'targetDir' in dependency_contents:
                    dependency_target_root_name = dependency_contents['targetDir']
                else:
                    dependency_target_root_name = dependency_contents['targetDirPrefix'] + dependency_rev
                download_and_unzip_files(dependency_url, TARGET_DOWNLOAD_DIRS[data], dependency_zip_root_name, dependency_target_root_name)
            elif download_format == _DOWNLOAD_FORMAT_TAR:
                dependency_tar_root_name = dependency_contents['tarRootDirPrefix'] + dependency_rev
                dependency_target_root_name = dependency_contents['targetDirPrefix'] + dependency_rev
                download_and_untar_files(dependency_url, TARGET_DOWNLOAD_DIRS[data], dependency_tar_root_name, dependency_target_root_name)

def main() -> None:
    if False:
        print('Hello World!')
    'Installs all the packages from the dependencies.json file.'
    download_all_dependencies(DEPENDENCIES_FILE_PATH)
if __name__ == '__main__':
    main()