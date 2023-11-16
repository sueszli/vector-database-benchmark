"""
Aptly Debian repository manager.

.. versionadded:: 2018.3.0
"""
import logging
import os
import re
import salt.utils.json
import salt.utils.path
import salt.utils.stringutils
from salt.exceptions import SaltInvocationError
_DEFAULT_CONFIG_PATH = '/etc/aptly.conf'
log = logging.getLogger(__name__)
__virtualname__ = 'aptly'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only works on systems with the aptly binary in the system path.\n    '
    if salt.utils.path.which('aptly'):
        return __virtualname__
    return (False, 'The aptly binaries required cannot be found or are not installed.')

def _cmd_run(cmd):
    if False:
        print('Hello World!')
    '\n    Run the aptly command.\n\n    :return: The string output of the command.\n    :rtype: str\n    '
    cmd.insert(0, 'aptly')
    cmd_ret = __salt__['cmd.run_all'](cmd, ignore_retcode=True)
    if cmd_ret['retcode'] != 0:
        log.debug('Unable to execute command: %s\nError: %s', cmd, cmd_ret['stderr'])
    return cmd_ret['stdout']

def _format_repo_args(comment=None, component=None, distribution=None, uploaders_file=None, saltenv='base'):
    if False:
        i = 10
        return i + 15
    '\n    Format the common arguments for creating or editing a repository.\n\n    :param str comment: The description of the repository.\n    :param str component: The default component to use when publishing.\n    :param str distribution: The default distribution to use when publishing.\n    :param str uploaders_file: The repository upload restrictions config.\n    :param str saltenv: The environment the file resides in.\n\n    :return: A list of the arguments formatted as aptly arguments.\n    :rtype: list\n    '
    ret = list()
    cached_uploaders_path = None
    settings = {'comment': comment, 'component': component, 'distribution': distribution}
    if uploaders_file:
        cached_uploaders_path = __salt__['cp.cache_file'](uploaders_file, saltenv)
        if not cached_uploaders_path:
            log.error('Unable to get cached copy of file: %s', uploaders_file)
            return False
    for setting in settings:
        if settings[setting] is not None:
            ret.append('-{}={}'.format(setting, settings[setting]))
    if cached_uploaders_path:
        ret.append('-uploaders-file={}'.format(cached_uploaders_path))
    return ret

def _validate_config(config_path):
    if False:
        return 10
    '\n    Validate that the configuration file exists and is readable.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n\n    :return: None\n    :rtype: None\n    '
    log.debug('Checking configuration file: %s', config_path)
    if not os.path.isfile(config_path):
        message = 'Unable to get configuration file: {}'.format(config_path)
        log.error(message)
        raise SaltInvocationError(message)

def get_config(config_path=_DEFAULT_CONFIG_PATH):
    if False:
        while True:
            i = 10
    "\n    Get the configuration data.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n\n    :return: A dictionary containing the configuration data.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.get_config\n    "
    _validate_config(config_path)
    cmd = ['config', 'show', '-config={}'.format(config_path)]
    cmd_ret = _cmd_run(cmd)
    return salt.utils.json.loads(cmd_ret)

def list_repos(config_path=_DEFAULT_CONFIG_PATH, with_packages=False):
    if False:
        while True:
            i = 10
    "\n    List all of the repos.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param bool with_packages: Return a list of packages in the repo.\n\n    :return: A dictionary of the repositories.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.list_repos\n    "
    _validate_config(config_path)
    ret = dict()
    cmd = ['repo', 'list', '-config={}'.format(config_path), '-raw=true']
    cmd_ret = _cmd_run(cmd)
    repos = [line.strip() for line in cmd_ret.splitlines()]
    log.debug('Found repositories: %s', len(repos))
    for name in repos:
        ret[name] = get_repo(name=name, config_path=config_path, with_packages=with_packages)
    return ret

def get_repo(name, config_path=_DEFAULT_CONFIG_PATH, with_packages=False):
    if False:
        return 10
    '\n    Get the details of the repository.\n\n    :param str name: The name of the repository.\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param bool with_packages: Return a list of packages in the repo.\n\n    :return: A dictionary containing information about the repository.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' aptly.get_repo name="test-repo"\n    '
    _validate_config(config_path)
    with_packages = str(bool(with_packages)).lower()
    ret = dict()
    cmd = ['repo', 'show', '-config={}'.format(config_path), '-with-packages={}'.format(with_packages), name]
    cmd_ret = _cmd_run(cmd)
    for line in cmd_ret.splitlines():
        try:
            items = line.split(':')
            key = items[0].lower().replace('default', '').strip()
            key = ' '.join(key.split()).replace(' ', '_')
            ret[key] = salt.utils.stringutils.to_none(salt.utils.stringutils.to_num(items[1].strip()))
        except (AttributeError, IndexError):
            log.debug('Skipping line: %s', line)
    if ret:
        log.debug('Found repository: %s', name)
    else:
        log.debug('Unable to find repository: %s', name)
    return ret

def new_repo(name, config_path=_DEFAULT_CONFIG_PATH, comment=None, component=None, distribution=None, uploaders_file=None, from_snapshot=None, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create the new repository.\n\n    :param str name: The name of the repository.\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param str comment: The description of the repository.\n    :param str component: The default component to use when publishing.\n    :param str distribution: The default distribution to use when publishing.\n    :param str uploaders_file: The repository upload restrictions config.\n    :param str from_snapshot: The snapshot to initialize the repository contents from.\n    :param str saltenv: The environment the file resides in.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' aptly.new_repo name="test-repo" comment="Test main repo" component="main" distribution="trusty"\n    '
    _validate_config(config_path)
    current_repo = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    if current_repo:
        log.debug('Repository already exists: %s', name)
        return True
    cmd = ['repo', 'create', '-config={}'.format(config_path)]
    repo_params = _format_repo_args(comment=comment, component=component, distribution=distribution, uploaders_file=uploaders_file, saltenv=saltenv)
    cmd.extend(repo_params)
    cmd.append(name)
    if from_snapshot:
        cmd.extend(['from', 'snapshot', from_snapshot])
    _cmd_run(cmd)
    repo = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    if repo:
        log.debug('Created repo: %s', name)
        return True
    log.error('Unable to create repo: %s', name)
    return False

def set_repo(name, config_path=_DEFAULT_CONFIG_PATH, comment=None, component=None, distribution=None, uploaders_file=None, saltenv='base'):
    if False:
        print('Hello World!')
    '\n    Configure the repository settings.\n\n    :param str name: The name of the repository.\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param str comment: The description of the repository.\n    :param str component: The default component to use when publishing.\n    :param str distribution: The default distribution to use when publishing.\n    :param str uploaders_file: The repository upload restrictions config.\n    :param str from_snapshot: The snapshot to initialize the repository contents from.\n    :param str saltenv: The environment the file resides in.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' aptly.set_repo name="test-repo" comment="Test universe repo" component="universe" distribution="xenial"\n    '
    _validate_config(config_path)
    failed_settings = dict()
    settings = {'comment': comment, 'component': component, 'distribution': distribution}
    for setting in list(settings):
        if settings[setting] is None:
            settings.pop(setting, None)
    current_settings = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    if not current_settings:
        log.error('Unable to get repo: %s', name)
        return False
    for current_setting in list(current_settings):
        if current_setting not in settings:
            current_settings.pop(current_setting, None)
    if settings == current_settings:
        log.debug('Settings already have the desired values for repository: %s', name)
        return True
    cmd = ['repo', 'edit', '-config={}'.format(config_path)]
    repo_params = _format_repo_args(comment=comment, component=component, distribution=distribution, uploaders_file=uploaders_file, saltenv=saltenv)
    cmd.extend(repo_params)
    cmd.append(name)
    _cmd_run(cmd)
    new_settings = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    for setting in settings:
        if settings[setting] != new_settings[setting]:
            failed_settings.update({setting: settings[setting]})
    if failed_settings:
        log.error('Unable to change settings for the repository: %s', name)
        return False
    log.debug('Settings successfully changed to the desired values for repository: %s', name)
    return True

def delete_repo(name, config_path=_DEFAULT_CONFIG_PATH, force=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove the repository.\n\n    :param str name: The name of the repository.\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param bool force: Whether to remove the repository even if it is used as the source\n        of an existing snapshot.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' aptly.delete_repo name="test-repo"\n    '
    _validate_config(config_path)
    force = str(bool(force)).lower()
    current_repo = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    if not current_repo:
        log.debug('Repository already absent: %s', name)
        return True
    cmd = ['repo', 'drop', '-config={}'.format(config_path), '-force={}'.format(force), name]
    _cmd_run(cmd)
    repo = __salt__['aptly.get_repo'](name=name, config_path=config_path)
    if repo:
        log.error('Unable to remove repo: %s', name)
        return False
    log.debug('Removed repo: %s', name)
    return True

def list_mirrors(config_path=_DEFAULT_CONFIG_PATH):
    if False:
        while True:
            i = 10
    "\n    Get a list of all the mirrors.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n\n    :return: A list of the mirror names.\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.list_mirrors\n    "
    _validate_config(config_path)
    cmd = ['mirror', 'list', '-config={}'.format(config_path), '-raw=true']
    cmd_ret = _cmd_run(cmd)
    ret = [line.strip() for line in cmd_ret.splitlines()]
    log.debug('Found mirrors: %s', len(ret))
    return ret

def list_published(config_path=_DEFAULT_CONFIG_PATH):
    if False:
        while True:
            i = 10
    "\n    Get a list of all the published repositories.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n\n    :return: A list of the published repository names.\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.list_published\n    "
    _validate_config(config_path)
    cmd = ['publish', 'list', '-config={}'.format(config_path), '-raw=true']
    cmd_ret = _cmd_run(cmd)
    ret = [line.strip() for line in cmd_ret.splitlines()]
    log.debug('Found published repositories: %s', len(ret))
    return ret

def list_snapshots(config_path=_DEFAULT_CONFIG_PATH, sort_by_time=False):
    if False:
        while True:
            i = 10
    "\n    Get a list of all the snapshots.\n\n    :param str config_path: The path to the configuration file for the aptly instance.\n    :param bool sort_by_time: Whether to sort by creation time instead of by name.\n\n    :return: A list of the snapshot names.\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.list_snapshots\n    "
    _validate_config(config_path)
    cmd = ['snapshot', 'list', '-config={}'.format(config_path), '-raw=true']
    if sort_by_time:
        cmd.append('-sort=time')
    else:
        cmd.append('-sort=name')
    cmd_ret = _cmd_run(cmd)
    ret = [line.strip() for line in cmd_ret.splitlines()]
    log.debug('Found snapshots: %s', len(ret))
    return ret

def cleanup_db(config_path=_DEFAULT_CONFIG_PATH, dry_run=False):
    if False:
        while True:
            i = 10
    "\n    Remove data regarding unreferenced packages and delete files in the package pool that\n        are no longer being used by packages.\n\n    :param bool dry_run: Report potential changes without making any changes.\n\n    :return: A dictionary of the package keys and files that were removed.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aptly.cleanup_db\n    "
    _validate_config(config_path)
    dry_run = str(bool(dry_run)).lower()
    ret = {'deleted_keys': list(), 'deleted_files': list()}
    cmd = ['db', 'cleanup', '-config={}'.format(config_path), '-dry-run={}'.format(dry_run), '-verbose=true']
    cmd_ret = _cmd_run(cmd)
    type_pattern = '^List\\s+[\\w\\s]+(?P<package_type>(file|key)s)[\\w\\s]+:$'
    list_pattern = '^\\s+-\\s+(?P<package>.*)$'
    current_block = None
    for line in cmd_ret.splitlines():
        if current_block:
            match = re.search(list_pattern, line)
            if match:
                package_type = 'deleted_{}'.format(current_block)
                ret[package_type].append(match.group('package'))
            else:
                current_block = None
        if not current_block:
            match = re.search(type_pattern, line)
            if match:
                current_block = match.group('package_type')
    log.debug('Package keys identified for deletion: %s', len(ret['deleted_keys']))
    log.debug('Package files identified for deletion: %s', len(ret['deleted_files']))
    return ret