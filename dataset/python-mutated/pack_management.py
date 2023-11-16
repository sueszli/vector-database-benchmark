"""
Module containing pack management related functions.
"""
from __future__ import absolute_import
import os
import shutil
import hashlib
import stat
import re
from st2common.util.monkey_patch import use_select_poll_workaround
use_select_poll_workaround()
import six
from git.repo import Repo
from gitdb.exc import BadName, BadObject
from lockfile import LockFile
from shutil import which as shutil_which
from st2common import log as logging
from st2common.content import utils
from st2common.constants.pack import MANIFEST_FILE_NAME
from st2common.constants.pack import PACK_RESERVED_CHARACTERS
from st2common.constants.pack import PACK_VERSION_SEPARATOR
from st2common.constants.pack import PACK_VERSION_REGEX
from st2common.services.packs import get_pack_from_index
from st2common.util.pack import get_pack_metadata
from st2common.util.pack import get_pack_ref_from_metadata
from st2common.util.pack import get_pack_warnings
from st2common.util.green import shell
from st2common.util.versioning import complex_semver_match
from st2common.util.versioning import get_stackstorm_version
from st2common.util.versioning import get_python_version
__all__ = ['download_pack', 'get_repo_url', 'eval_repo_url', 'apply_pack_owner_group', 'apply_pack_permissions', 'get_and_set_proxy_config']
LOG = logging.getLogger(__name__)
CONFIG_FILE = 'config.yaml'
CURRENT_STACKSTORM_VERSION = get_stackstorm_version()
CURRENT_PYTHON_VERSION = get_python_version()
SUDO_BINARY = shutil_which('sudo')

def download_pack(pack, abs_repo_base='/opt/stackstorm/packs', verify_ssl=True, force=False, proxy_config=None, force_owner_group=True, force_permissions=True, logger=LOG):
    if False:
        for i in range(10):
            print('nop')
    '\n    Download the pack and move it to /opt/stackstorm/packs.\n\n    :param abs_repo_base: Path where the pack should be installed to.\n    :type abs_repo_base: ``str``\n\n    :param pack: Pack name.\n    :rtype pack: ``str``\n\n    :param force_owner_group: Set owner group of the pack directory to the value defined in the\n                              config.\n    :type force_owner_group: ``bool``\n\n    :param force_permissions: True to force 770 permission on all the pack content.\n    :type force_permissions: ``bool``\n\n    :param force: Force the installation and ignore / delete the lock file if it already exists.\n    :type force: ``bool``\n\n    :return: (pack_url, pack_ref, result)\n    :rtype: tuple\n    '
    proxy_config = proxy_config or {}
    try:
        (pack_url, pack_version) = get_repo_url(pack, proxy_config=proxy_config)
    except Exception as e:
        result = [None, pack, (False, six.text_type(e))]
        return result
    result = [pack_url, None, None]
    temp_dir_name = hashlib.md5(pack_url.encode()).hexdigest()
    lock_file = LockFile('/tmp/%s' % temp_dir_name)
    lock_file_path = lock_file.lock_file
    if force:
        logger.debug('Force mode is enabled, deleting lock file...')
        try:
            os.unlink(lock_file_path)
        except OSError:
            pass
    with lock_file:
        try:
            user_home = os.path.expanduser('~')
            abs_local_path = os.path.join(user_home, '.st2packs', temp_dir_name)
            if pack_url.startswith('file://'):
                local_pack_directory = os.path.abspath(os.path.join(pack_url.split('file://')[1]))
            else:
                local_pack_directory = None
            if local_pack_directory and (not os.path.isdir(os.path.join(local_pack_directory, '.git'))):
                if not os.path.isdir(local_pack_directory):
                    raise ValueError('Local pack directory "%s" doesn\'t exist' % local_pack_directory)
                logger.debug('Detected local pack directory which is not a git repository, just copying files over...')
                shutil.copytree(local_pack_directory, abs_local_path)
            else:
                clone_repo(temp_dir=abs_local_path, repo_url=pack_url, verify_ssl=verify_ssl, ref=pack_version)
            pack_metadata = get_pack_metadata(pack_dir=abs_local_path)
            pack_ref = get_pack_ref(pack_dir=abs_local_path)
            result[1] = pack_ref
            if not force:
                verify_pack_version(pack_metadata=pack_metadata)
            move_result = move_pack(abs_repo_base=abs_repo_base, pack_name=pack_ref, abs_local_path=abs_local_path, pack_metadata=pack_metadata, force_owner_group=force_owner_group, force_permissions=force_permissions, logger=logger)
            result[2] = move_result
        finally:
            cleanup_repo(abs_local_path=abs_local_path)
    return tuple(result)

def clone_repo(temp_dir, repo_url, verify_ssl=True, ref='master'):
    if False:
        return 10
    os.environ['GIT_TERMINAL_PROMPT'] = '0'
    os.environ['GIT_ASKPASS'] = '/bin/echo'
    if not verify_ssl:
        os.environ['GIT_SSL_NO_VERIFY'] = 'true'
    repo = Repo.clone_from(repo_url, temp_dir)
    is_local_repo = repo_url.startswith('file://')
    try:
        active_branch = repo.active_branch
    except TypeError as e:
        if is_local_repo:
            active_branch = None
        else:
            raise e
    if is_local_repo and (not active_branch) and (not ref):
        LOG.debug('Installing pack from git repo on disk, skipping branch checkout')
        return temp_dir
    use_branch = False
    if (not ref or ref == active_branch.name) and repo.active_branch.object == repo.head.commit:
        gitref = repo.active_branch.object
    else:
        gitref = get_gitref(repo, 'origin/%s' % ref)
        if gitref:
            use_branch = True
    if not gitref:
        gitref = get_gitref(repo, ref)
    if not gitref and re.match(PACK_VERSION_REGEX, ref):
        gitref = get_gitref(repo, 'v%s' % ref)
    if not gitref:
        format_values = [ref, repo_url]
        msg = '"%s" is not a valid version, hash, tag or branch in %s.'
        valid_versions = get_valid_versions_for_repo(repo=repo)
        if len(valid_versions) >= 1:
            valid_versions_string = ', '.join(valid_versions)
            msg += ' Available versions are: %s.'
            format_values.append(valid_versions_string)
        raise ValueError(msg % tuple(format_values))
    branches = repo.git.branch('-a', '--contains', gitref.hexsha)
    if branches:
        branches = branches.replace('*', '').split()
        if active_branch.name not in branches or use_branch:
            branch = 'origin/%s' % ref if use_branch else branches[0]
            short_branch = ref if use_branch else branches[0].split('/')[-1]
            repo.git.checkout('-b', short_branch, branch)
            branch = repo.head.reference
        else:
            branch = repo.active_branch.name
        repo.git.checkout(gitref.hexsha)
        repo.git.branch('-f', branch, gitref.hexsha)
        repo.git.checkout(branch)
    else:
        repo.git.checkout('v%s' % ref)
    return temp_dir

def move_pack(abs_repo_base, pack_name, abs_local_path, pack_metadata, force_owner_group=True, force_permissions=True, logger=LOG):
    if False:
        i = 10
        return i + 15
    '\n    Move pack directory into the final location.\n    '
    (desired, message) = is_desired_pack(abs_local_path, pack_name)
    if desired:
        to = abs_repo_base
        dest_pack_path = os.path.join(abs_repo_base, pack_name)
        if os.path.exists(dest_pack_path):
            logger.debug('Removing existing pack %s in %s to replace.', pack_name, dest_pack_path)
            old_config_file = os.path.join(dest_pack_path, CONFIG_FILE)
            new_config_file = os.path.join(abs_local_path, CONFIG_FILE)
            if os.path.isfile(old_config_file):
                shutil.move(old_config_file, new_config_file)
            shutil.rmtree(dest_pack_path)
        logger.debug('Moving pack from %s to %s.', abs_local_path, to)
        shutil.move(abs_local_path, dest_pack_path)
        if force_owner_group:
            apply_pack_owner_group(pack_path=dest_pack_path)
        if force_permissions:
            apply_pack_permissions(pack_path=dest_pack_path)
        warning = get_pack_warnings(pack_metadata)
        if warning:
            logger.warning(warning)
        message = 'Success.'
    elif message:
        message = 'Failure : %s' % message
    return (desired, message)

def apply_pack_owner_group(pack_path):
    if False:
        while True:
            i = 10
    '\n    Switch owner group of the pack / virtualenv directory to the configured\n    group.\n\n    NOTE: This requires sudo access.\n    '
    pack_group = utils.get_pack_group()
    if pack_group:
        LOG.debug('Changing owner group of "{}" directory to {}'.format(pack_path, pack_group))
        if SUDO_BINARY:
            args = ['sudo', 'chgrp', '-R', pack_group, pack_path]
        else:
            args = ['chgrp', '-R', pack_group, pack_path]
        (exit_code, _, stderr, _) = shell.run_command(args)
        if exit_code != 0:
            LOG.debug('Failed to change owner group on directory "{}" to "{}": {}'.format(pack_path, pack_group, stderr))
    return True

def apply_pack_permissions(pack_path):
    if False:
        print('Hello World!')
    '\n    Recursively apply permission 775 to pack and its contents.\n    '
    mode = stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
    os.chmod(pack_path, mode)
    for (root, dirs, files) in os.walk(pack_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

def cleanup_repo(abs_local_path):
    if False:
        for i in range(10):
            print('nop')
    if os.path.isdir(abs_local_path):
        shutil.rmtree(abs_local_path)

def get_repo_url(pack, proxy_config=None):
    if False:
        return 10
    '\n    Retrieve pack repo url.\n\n    :rtype: ``str``\n\n    :return: (repo_url, version)\n    :rtype: tuple\n    '
    pack_and_version = pack.split(PACK_VERSION_SEPARATOR)
    name_or_url = pack_and_version[0]
    version = pack_and_version[1] if len(pack_and_version) > 1 else None
    if len(name_or_url.split('/')) == 1:
        pack = get_pack_from_index(name_or_url, proxy_config=proxy_config)
        if not pack:
            raise Exception('No record of the "%s" pack in the index.' % name_or_url)
        return (pack['repo_url'], version or pack['version'])
    else:
        return (eval_repo_url(name_or_url), version)

def eval_repo_url(repo_url):
    if False:
        for i in range(10):
            print('nop')
    '\n    Allow passing short GitHub or GitLab SSH style URLs.\n    '
    if not repo_url:
        raise Exception('No valid repo_url provided or could be inferred.')
    if repo_url.startswith('gitlab@') or repo_url.startswith('file://'):
        return repo_url
    else:
        if len(repo_url.split('/')) == 2 and 'git@' not in repo_url:
            url = 'https://github.com/{}'.format(repo_url)
        else:
            url = repo_url
        return url

def is_desired_pack(abs_pack_path, pack_name):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(abs_pack_path):
        return (False, 'Pack "%s" not found or it\'s missing a "pack.yaml" file.' % pack_name)
    for character in PACK_RESERVED_CHARACTERS:
        if character in pack_name:
            return (False, 'Pack name "%s" contains reserved character "%s"' % (pack_name, character))
    if not os.path.isfile(os.path.join(abs_pack_path, MANIFEST_FILE_NAME)):
        return (False, 'Pack is missing a manifest file (%s).' % MANIFEST_FILE_NAME)
    return (True, '')

def verify_pack_version(pack_metadata):
    if False:
        return 10
    '\n    Verify that the pack works with the currently running StackStorm version.\n    '
    pack_name = pack_metadata.get('name', None)
    required_stackstorm_version = pack_metadata.get('stackstorm_version', None)
    supported_python_versions = pack_metadata.get('python_versions', None)
    if required_stackstorm_version:
        if not complex_semver_match(CURRENT_STACKSTORM_VERSION, required_stackstorm_version):
            msg = 'Pack "%s" requires StackStorm "%s", but current version is "%s". You can override this restriction by providing the "force" flag, but the pack is not guaranteed to work.' % (pack_name, required_stackstorm_version, CURRENT_STACKSTORM_VERSION)
            raise ValueError(msg)
    if supported_python_versions:
        if set(supported_python_versions) == set(['2']) and (not six.PY2):
            msg = 'Pack "%s" requires Python 2.x, but current Python version is "%s". You can override this restriction by providing the "force" flag, but the pack is not guaranteed to work.' % (pack_name, CURRENT_PYTHON_VERSION)
            raise ValueError(msg)
        elif set(supported_python_versions) == set(['3']) and (not six.PY3):
            msg = 'Pack "%s" requires Python 3.x, but current Python version is "%s". You can override this restriction by providing the "force" flag, but the pack is not guaranteed to work.' % (pack_name, CURRENT_PYTHON_VERSION)
            raise ValueError(msg)
        else:
            pass
    return True

def get_gitref(repo, ref):
    if False:
        while True:
            i = 10
    '\n    Retrieve git repo reference if available.\n    '
    try:
        return repo.commit(ref)
    except (BadName, BadObject):
        return False

def get_valid_versions_for_repo(repo):
    if False:
        return 10
    '\n    Retrieve valid versions (tags) for a particular repo (pack).\n\n    It does so by introspecting available tags.\n\n    :rtype: ``list`` of ``str``\n    '
    valid_versions = []
    for tag in repo.tags:
        if tag.name.startswith('v') and re.match(PACK_VERSION_REGEX, tag.name[1:]):
            valid_versions.append(tag.name[1:])
    return valid_versions

def get_pack_ref(pack_dir):
    if False:
        i = 10
        return i + 15
    '\n    Read pack reference from the metadata file and sanitize it.\n    '
    metadata = get_pack_metadata(pack_dir=pack_dir)
    pack_ref = get_pack_ref_from_metadata(metadata=metadata, pack_directory_name=None)
    return pack_ref

def get_and_set_proxy_config():
    if False:
        while True:
            i = 10
    https_proxy = os.environ.get('https_proxy', None)
    http_proxy = os.environ.get('http_proxy', None)
    proxy_ca_bundle_path = os.environ.get('proxy_ca_bundle_path', None)
    no_proxy = os.environ.get('no_proxy', None)
    proxy_config = {}
    if http_proxy or https_proxy:
        LOG.debug('Using proxy %s', http_proxy if http_proxy else https_proxy)
        proxy_config = {'https_proxy': https_proxy, 'http_proxy': http_proxy, 'proxy_ca_bundle_path': proxy_ca_bundle_path, 'no_proxy': no_proxy}
    if https_proxy and (not os.environ.get('https_proxy', None)):
        os.environ['https_proxy'] = https_proxy
    if http_proxy and (not os.environ.get('http_proxy', None)):
        os.environ['http_proxy'] = http_proxy
    if no_proxy and (not os.environ.get('no_proxy', None)):
        os.environ['no_proxy'] = no_proxy
    if proxy_ca_bundle_path and (not os.environ.get('proxy_ca_bundle_path', None)):
        os.environ['no_proxy'] = no_proxy
    return proxy_config