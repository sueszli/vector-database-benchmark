"""Utility functions for handling and fetching repo archives in zip format."""
import os
import tempfile
from pathlib import Path
from typing import Optional
from zipfile import BadZipFile, ZipFile
import requests
from cookiecutter.exceptions import InvalidZipRepository
from cookiecutter.prompt import read_repo_password
from cookiecutter.utils import make_sure_path_exists, prompt_and_delete

def unzip(zip_uri: str, is_url: bool, clone_to_dir: 'os.PathLike[str]'='.', no_input: bool=False, password: Optional[str]=None):
    if False:
        print('Hello World!')
    'Download and unpack a zipfile at a given URI.\n\n    This will download the zipfile to the cookiecutter repository,\n    and unpack into a temporary directory.\n\n    :param zip_uri: The URI for the zipfile.\n    :param is_url: Is the zip URI a URL or a file?\n    :param clone_to_dir: The cookiecutter repository directory\n        to put the archive into.\n    :param no_input: Do not prompt for user input and eventually force a refresh of\n        cached resources.\n    :param password: The password to use when unpacking the repository.\n    '
    clone_to_dir = Path(clone_to_dir).expanduser()
    make_sure_path_exists(clone_to_dir)
    if is_url:
        identifier = zip_uri.rsplit('/', 1)[1]
        zip_path = os.path.join(clone_to_dir, identifier)
        if os.path.exists(zip_path):
            download = prompt_and_delete(zip_path, no_input=no_input)
        else:
            download = True
        if download:
            r = requests.get(zip_uri, stream=True, timeout=100)
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
    else:
        zip_path = os.path.abspath(zip_uri)
    try:
        zip_file = ZipFile(zip_path)
        if len(zip_file.namelist()) == 0:
            raise InvalidZipRepository(f'Zip repository {zip_uri} is empty')
        first_filename = zip_file.namelist()[0]
        if not first_filename.endswith('/'):
            raise InvalidZipRepository(f'Zip repository {zip_uri} does not include a top-level directory')
        project_name = first_filename[:-1]
        unzip_base = tempfile.mkdtemp()
        unzip_path = os.path.join(unzip_base, project_name)
        try:
            zip_file.extractall(path=unzip_base)
        except RuntimeError:
            if password is not None:
                try:
                    zip_file.extractall(path=unzip_base, pwd=password.encode('utf-8'))
                except RuntimeError:
                    raise InvalidZipRepository('Invalid password provided for protected repository')
            elif no_input:
                raise InvalidZipRepository('Unable to unlock password protected repository')
            else:
                retry = 0
                while retry is not None:
                    try:
                        password = read_repo_password('Repo password')
                        zip_file.extractall(path=unzip_base, pwd=password.encode('utf-8'))
                        retry = None
                    except RuntimeError:
                        retry += 1
                        if retry == 3:
                            raise InvalidZipRepository('Invalid password provided for protected repository')
    except BadZipFile:
        raise InvalidZipRepository(f'Zip repository {zip_uri} is not a valid zip archive:')
    return unzip_path