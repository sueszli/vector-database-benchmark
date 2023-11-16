"""Installation script for Oppia third-party libraries."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib
import zipfile
from core import utils
from scripts import install_dependencies_json_packages
from typing import Dict, Final, List, Literal, Optional, TypedDict, cast
from . import common
from . import install_python_prod_dependencies
TOOLS_DIR: Final = os.path.join('..', 'oppia_tools')
THIRD_PARTY_DIR: Final = os.path.join('.', 'third_party')
THIRD_PARTY_STATIC_DIR: Final = os.path.join(THIRD_PARTY_DIR, 'static')
DEPENDENCIES_FILE_PATH: Final = os.path.join(os.getcwd(), 'dependencies.json')
TMP_UNZIP_PATH: Final = os.path.join('.', 'tmp_unzip.zip')
common.require_cwd_to_be_oppia(allow_deploy_dir=True)
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
_PARSER = argparse.ArgumentParser(description='\nInstallation script for Oppia third-party libraries.\n')

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

def download_files(source_url_root: str, target_dir: str, source_filenames: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Downloads a group of files and saves them to a given directory.\n\n    Each file is downloaded only if it does not already exist.\n\n    Args:\n        source_url_root: str. The URL to prepend to all the filenames.\n        target_dir: str. The directory to save the files to.\n        source_filenames: list(str). Each filename is appended to the\n            end of the source_url_root in order to give the URL from which to\n            download the file. The downloaded file is then placed in target_dir,\n            and retains the same filename.\n    '
    assert isinstance(source_filenames, list), "Expected list of filenames, got '%s'" % source_filenames
    common.ensure_directory_exists(target_dir)
    for filename in source_filenames:
        if not os.path.exists(os.path.join(target_dir, filename)):
            print('Downloading file %s to %s ...' % (filename, target_dir))
            common.url_retrieve('%s/%s' % (source_url_root, filename), os.path.join(target_dir, filename))
            print('Download of %s succeeded.' % filename)

def download_and_unzip_files(source_url: str, target_parent_dir: str, zip_root_name: str, target_root_name: str) -> None:
    if False:
        print('Hello World!')
    'Downloads a zip file, unzips it, and saves the result in a given dir.\n\n    The download occurs only if the target directory that the zip file unzips\n    to does not exist.\n\n    NB: This function assumes that the root level of the zip file has exactly\n    one folder.\n\n    Args:\n        source_url: str. The URL from which to download the zip file.\n        target_parent_dir: str. The directory to save the contents of the zip\n            file to.\n        zip_root_name: str. The name of the top-level folder in the zip\n            directory.\n        target_root_name: str. The name that the top-level folder should be\n            renamed to in the local directory.\n    '
    if not os.path.exists(os.path.join(target_parent_dir, target_root_name)):
        print('Downloading and unzipping file %s to %s ...' % (zip_root_name, target_parent_dir))
        common.ensure_directory_exists(target_parent_dir)
        common.url_retrieve(source_url, TMP_UNZIP_PATH)
        try:
            with zipfile.ZipFile(TMP_UNZIP_PATH, 'r') as zfile:
                zfile.extractall(path=target_parent_dir)
            os.remove(TMP_UNZIP_PATH)
        except Exception:
            if os.path.exists(TMP_UNZIP_PATH):
                os.remove(TMP_UNZIP_PATH)
            req = urllib.request.Request(source_url, None, {})
            req.add_header('User-agent', 'python')
            file_stream = io.BytesIO(utils.url_open(req).read())
            with zipfile.ZipFile(file_stream, 'r') as zfile:
                zfile.extractall(path=target_parent_dir)
        os.rename(os.path.join(target_parent_dir, zip_root_name), os.path.join(target_parent_dir, target_root_name))
        print('Download of %s succeeded.' % zip_root_name)

def download_and_untar_files(source_url: str, target_parent_dir: str, tar_root_name: str, target_root_name: str) -> None:
    if False:
        i = 10
        return i + 15
    'Downloads a tar file, untars it, and saves the result in a given dir.\n\n    The download occurs only if the target directory that the tar file untars\n    to does not exist.\n\n    NB: This function assumes that the root level of the tar file has exactly\n    one folder.\n\n    Args:\n        source_url: str. The URL from which to download the tar file.\n        target_parent_dir: str. The directory to save the contents of the tar\n            file to.\n        tar_root_name: str. The name of the top-level folder in the tar\n            directory.\n        target_root_name: str. The name that the top-level folder should be\n            renamed to in the local directory.\n    '
    if not os.path.exists(os.path.join(target_parent_dir, target_root_name)):
        print('Downloading and untarring file %s to %s ...' % (tar_root_name, target_parent_dir))
        common.ensure_directory_exists(target_parent_dir)
        common.url_retrieve(source_url, TMP_UNZIP_PATH)
        with contextlib.closing(tarfile.open(name=TMP_UNZIP_PATH, mode='r:gz')) as tfile:
            tfile.extractall(target_parent_dir)
        os.remove(TMP_UNZIP_PATH)
        os.rename(os.path.join(target_parent_dir, tar_root_name), os.path.join(target_parent_dir, target_root_name))
        print('Download of %s succeeded.' % tar_root_name)

def get_file_contents(filepath: str, mode: utils.TextModeTypes='r') -> str:
    if False:
        return 10
    'Gets the contents of a file, given a relative filepath from oppia/.'
    with utils.open_file(filepath, mode) as f:
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
        while True:
            i = 10
    "This validates syntax of the dependencies.json\n\n    Args:\n        filepath: str. The path to the json file.\n\n    Raises:\n        Exception. The 'downloadFormat' not specified.\n    "
    dependencies_data = return_json(filepath)
    dependencies = dependencies_data['dependencies']
    for (_, dependency) in dependencies.items():
        for (_, dependency_contents) in dependency.items():
            if 'downloadFormat' not in dependency_contents:
                raise Exception('downloadFormat not specified in %s' % dependency_contents)
            download_format = dependency_contents['downloadFormat']
            test_dependencies_syntax(download_format, dependency_contents)

def install_elasticsearch_dev_server() -> None:
    if False:
        return 10
    'This installs a local ElasticSearch server to the oppia_tools\n    directory to be used by development servers and backend tests.\n    '
    try:
        subprocess.call(['%s/bin/elasticsearch' % common.ES_PATH, '--version'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'ES_JAVA_OPTS': '-Xms100m -Xmx500m'})
        print('ElasticSearch is already installed.')
        return
    except OSError:
        print('Installing ElasticSearch...')
    if common.is_mac_os() or common.is_linux_os():
        file_ext = 'tar.gz'

        def download_and_extract(*args: str) -> None:
            if False:
                while True:
                    i = 10
            'This downloads and extracts the elasticsearch files.'
            download_and_untar_files(*args)
    elif common.is_windows_os():
        file_ext = 'zip'

        def download_and_extract(*args: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'This downloads and extracts the elasticsearch files.'
            download_and_unzip_files(*args)
    else:
        raise Exception('Unrecognized or unsupported operating system.')
    download_and_extract('https://artifacts.elastic.co/downloads/elasticsearch/' + 'elasticsearch-%s-%s-x86_64.%s' % (common.ELASTICSEARCH_VERSION, common.OS_NAME.lower(), file_ext), TARGET_DOWNLOAD_DIRS['oppiaTools'], 'elasticsearch-%s' % common.ELASTICSEARCH_VERSION, 'elasticsearch-%s' % common.ELASTICSEARCH_VERSION)
    print('ElasticSearch installed successfully.')

def install_redis_cli() -> None:
    if False:
        i = 10
        return i + 15
    'This installs the redis-cli to the local oppia third_party directory so\n    that development servers and backend tests can make use of a local redis\n    cache. Redis-cli installed here (redis-cli-6.0.6) is different from the\n    redis package installed in dependencies.json (redis-3.5.3). The redis-3.5.3\n    package detailed in dependencies.json is the Python library that allows\n    users to communicate with any Redis cache using Python. The redis-cli-6.0.6\n    package installed in this function contains C++ scripts for the redis-cli\n    and redis-server programs detailed below.\n\n    The redis-cli program is the command line interface that serves up an\n    interpreter that allows users to connect to a redis database cache and\n    query the cache using the Redis CLI API. It also contains functionality to\n    shutdown the redis server. We need to install redis-cli separately from the\n    default installation of backend libraries since it is a system program and\n    we need to build the program files after the library is untarred.\n\n    The redis-server starts a Redis database on the local machine that can be\n    queried using either the Python redis library or the redis-cli interpreter.\n    '
    try:
        subprocess.call([common.REDIS_SERVER_PATH, '--version'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('Redis-cli is already installed.')
    except OSError:
        print('Installing redis-cli...')
        download_and_untar_files('https://download.redis.io/releases/redis-%s.tar.gz' % common.REDIS_CLI_VERSION, TARGET_DOWNLOAD_DIRS['oppiaTools'], 'redis-%s' % common.REDIS_CLI_VERSION, 'redis-cli-%s' % common.REDIS_CLI_VERSION)
        with common.CD(os.path.join(TARGET_DOWNLOAD_DIRS['oppiaTools'], 'redis-cli-%s' % common.REDIS_CLI_VERSION)):
            subprocess.call(['make'])
        subprocess.call(['chmod', '+x', common.REDIS_SERVER_PATH])
        subprocess.call(['chmod', '+x', common.REDIS_CLI_PATH])
        print('Redis-cli installed successfully.')

def main(args: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Installs all the third party libraries.'
    if common.is_windows_os():
        raise Exception('The redis command line interface will not be installed because your machine is on the Windows operating system.')
    unused_parsed_args = _PARSER.parse_args(args=args)
    install_python_prod_dependencies.main()
    install_dependencies_json_packages.download_all_dependencies(DEPENDENCIES_FILE_PATH)
    install_redis_cli()
    install_elasticsearch_dev_server()
if __name__ == '__main__':
    main()