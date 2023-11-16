"""
Initialize an arbitrary project
"""
import functools
import logging
import shutil
from cookiecutter import config, exceptions, repository
from samcli.lib.utils import osutils
from .exceptions import ArbitraryProjectDownloadFailed
LOG = logging.getLogger(__name__)
BAD_LOCATION_ERROR_MSG = 'Please verify your location. The following types of location are supported:\n\n* Github: gh:user/repo (or) https://github.com/user/repo (or) git@github.com:user/repo.git\n          For Git repositories, you must use location of the root of the repository.\n\n* Mercurial: hg+ssh://hg@bitbucket.org/repo\n\n* Http(s): https://example.com/code.zip\n\n* Local Path: /path/to/code.zip'

def generate_non_cookiecutter_project(location, output_dir):
    if False:
        for i in range(10):
            print('nop')
    "\n    Uses Cookiecutter APIs to download a project at given ``location`` to the ``output_dir``.\n    This does *not* run cookiecutter on the downloaded project.\n\n    Parameters\n    ----------\n    location : str\n        Path to where the project is. This supports all formats of location cookiecutter supports\n        (ex: zip, git, ssh, hg, local zipfile)\n\n        NOTE: This value *cannot* be a local directory. We didn't see a value in simply copying the directory\n        contents to ``output_dir`` without any processing.\n\n    output_dir : str\n        Directory where the project should be downloaded to\n\n    Returns\n    -------\n    str\n        Name of the directory where the project was downloaded to.\n\n    Raises\n    ------\n    cookiecutter.exception.CookiecutterException if download failed for some reason\n    "
    LOG.debug('Downloading project from %s to %s', location, output_dir)
    no_input = True
    location = repository.expand_abbreviations(location, config.BUILTIN_ABBREVIATIONS)
    if repository.is_zip_file(location):
        LOG.debug('%s location is a zip file', location)
        download_fn = functools.partial(repository.unzip, zip_uri=location, is_url=repository.is_repo_url(location), no_input=no_input)
    elif repository.is_repo_url(location):
        LOG.debug('%s location is a source control repository', location)
        download_fn = functools.partial(repository.clone, repo_url=location, no_input=no_input)
    else:
        raise ArbitraryProjectDownloadFailed(msg=BAD_LOCATION_ERROR_MSG)
    try:
        return _download_and_copy(download_fn, output_dir)
    except exceptions.RepositoryNotFound as ex:
        raise ArbitraryProjectDownloadFailed(msg=BAD_LOCATION_ERROR_MSG) from ex

def _download_and_copy(download_fn, output_dir):
    if False:
        while True:
            i = 10
    '\n    Runs the download function to download files into a temporary directory and then copy the files over to\n    the ``output_dir``\n\n    Parameters\n    ----------\n    download_fn : function\n        Method to be called to download. It needs to accept a parameter called `clone_to_dir`. This will be\n        set to the temporary directory\n\n    output_dir : str\n        Path to the directory where files will be copied to\n\n    Returns\n    -------\n    output_dir\n    '
    with osutils.mkdir_temp(ignore_errors=True) as tempdir:
        downloaded_dir = download_fn(clone_to_dir=tempdir)
        osutils.copytree(downloaded_dir, output_dir, ignore=shutil.ignore_patterns('*.git'))
    return output_dir