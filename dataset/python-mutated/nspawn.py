"""
Manage nspawn containers

.. versionadded:: 2015.8.0

`systemd-nspawn(1)`__ is a tool used to manage lightweight namespace
containers. This execution module provides several functions to help manage
these containers.

.. __: http://www.freedesktop.org/software/systemd/man/systemd-nspawn.html

Minions running systemd >= 219 will place new containers in
``/var/lib/machines``, while those running systemd < 219 will place them in
``/var/lib/container``.

.. note:

    ``nsenter(1)`` is required to run commands within containers. It should
    already be present on any systemd host, as part of the **util-linux**
    package.
"""
import errno
import functools
import logging
import os
import re
import shutil
import tempfile
import time
import salt.defaults.exitcodes
import salt.utils.args
import salt.utils.functools
import salt.utils.path
import salt.utils.systemd
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'nspawn'
SEED_MARKER = '/nspawn.initial_seed'
WANT = '/etc/systemd/system/multi-user.target.wants/systemd-nspawn@{0}.service'
EXEC_DRIVER = 'nsenter'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on systems that have been booted with systemd\n    '
    if __grains__['kernel'] == 'Linux' and salt.utils.systemd.booted(__context__):
        if salt.utils.systemd.version() is None:
            log.error('nspawn: Unable to determine systemd version')
        else:
            return __virtualname__
    return (False, 'The nspawn execution module failed to load: only work on systems that have been booted with systemd.')

def _sd_version():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns __context__.get('systemd.version', 0), avoiding duplication of the\n    call to dict.get and making it easier to change how we handle this context\n    var in the future\n    "
    return salt.utils.systemd.version(__context__)

def _ensure_exists(wrapped):
    if False:
        return 10
    '\n    Decorator to ensure that the named container exists.\n    '

    @functools.wraps(wrapped)
    def check_exists(name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not exists(name):
            raise CommandExecutionError(f"Container '{name}' does not exist")
        return wrapped(name, *args, **salt.utils.args.clean_kwargs(**kwargs))
    return check_exists

def _root(name='', all_roots=False):
    if False:
        return 10
    '\n    Return the container root directory. Starting with systemd 219, new\n    images go into /var/lib/machines.\n    '
    if _sd_version() >= 219:
        if all_roots:
            return [os.path.join(x, name) for x in ('/var/lib/machines', '/var/lib/container')]
        else:
            return os.path.join('/var/lib/machines', name)
    else:
        ret = os.path.join('/var/lib/container', name)
        if all_roots:
            return [ret]
        else:
            return ret

def _make_container_root(name):
    if False:
        print('Hello World!')
    '\n    Make the container root directory\n    '
    path = _root(name)
    if os.path.exists(path):
        __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
        raise CommandExecutionError(f'Container {name} already exists')
    else:
        try:
            os.makedirs(path)
            return path
        except OSError as exc:
            raise CommandExecutionError(f'Unable to make container root directory {name}: {exc}')

def _build_failed(dst, name):
    if False:
        print('Hello World!')
    try:
        __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
        shutil.rmtree(dst)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise CommandExecutionError(f'Unable to cleanup container root dir {dst}')
    raise CommandExecutionError(f'Container {name} failed to build')

def _bootstrap_arch(name, **kwargs):
    if False:
        print('Hello World!')
    '\n    Bootstrap an Arch Linux container\n    '
    if not salt.utils.path.which('pacstrap'):
        raise CommandExecutionError('pacstrap not found, is the arch-install-scripts package installed?')
    dst = _make_container_root(name)
    cmd = f'pacstrap -c -d {dst} base'
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        _build_failed(dst, name)
    return ret

def _bootstrap_debian(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Bootstrap a Debian Linux container\n    '
    version = kwargs.get('version', False)
    if not version:
        if __grains__['os'].lower() == 'debian':
            version = __grains__['osrelease']
        else:
            version = 'stable'
    release_blacklist = ['hamm', 'slink', 'potato', 'woody', 'sarge', 'etch', 'lenny', 'squeeze', 'wheezy']
    if version in release_blacklist:
        raise CommandExecutionError('Unsupported Debian version "{}". Only "stable" or "jessie" and newer are supported'.format(version))
    dst = _make_container_root(name)
    cmd = f'debootstrap --arch=amd64 {version} {dst}'
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        _build_failed(dst, name)
    return ret

def _bootstrap_fedora(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Bootstrap a Fedora container\n    '
    dst = _make_container_root(name)
    if not kwargs.get('version', False):
        if __grains__['os'].lower() == 'fedora':
            version = __grains__['osrelease']
        else:
            version = '21'
    else:
        version = '21'
    cmd = 'yum -y --releasever={} --nogpg --installroot={} --disablerepo="*" --enablerepo=fedora install systemd passwd yum fedora-release vim-minimal'.format(version, dst)
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        _build_failed(dst, name)
    return ret

def _bootstrap_ubuntu(name, **kwargs):
    if False:
        print('Hello World!')
    '\n    Bootstrap a Ubuntu Linux container\n    '
    version = kwargs.get('version', False)
    if not version:
        if __grains__['os'].lower() == 'ubuntu':
            version = __grains__['oscodename']
        else:
            version = 'xenial'
    dst = _make_container_root(name)
    cmd = f'debootstrap --arch=amd64 {version} {dst}'
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        _build_failed(dst, name)
    return ret

def _clear_context():
    if False:
        i = 10
        return i + 15
    '\n    Clear any lxc variables set in __context__\n    '
    for var in [x for x in __context__ if x.startswith('nspawn.')]:
        log.trace("Clearing __context__['%s']", var)
        __context__.pop(var, None)

def _ensure_running(name):
    if False:
        print('Hello World!')
    '\n    Raise an exception if the container does not exist\n    '
    if state(name) != 'running':
        return True
    else:
        return start(name)

def _ensure_systemd(version):
    if False:
        while True:
            i = 10
    '\n    Raises an exception if the systemd version is not greater than the\n    passed version.\n    '
    try:
        version = int(version)
    except ValueError:
        raise CommandExecutionError(f"Invalid version '{version}'")
    try:
        installed = _sd_version()
        log.debug('nspawn: detected systemd %s', installed)
    except (IndexError, ValueError):
        raise CommandExecutionError('nspawn: Unable to get systemd version')
    if installed < version:
        raise CommandExecutionError('This function requires systemd >= {} (Detected version: {}).'.format(version, installed))

def _machinectl(cmd, output_loglevel='debug', ignore_retcode=False, use_vt=False):
    if False:
        print('Hello World!')
    '\n    Helper function to run machinectl\n    '
    prefix = 'machinectl --no-legend --no-pager'
    return __salt__['cmd.run_all'](f'{prefix} {cmd}', output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt)

@_ensure_exists
def _run(name, cmd, output=None, no_start=False, stdin=None, python_shell=True, preserve_state=False, output_loglevel='debug', ignore_retcode=False, use_vt=False, keep_env=None):
    if False:
        return 10
    '\n    Common logic for nspawn.run functions\n    '
    orig_state = state(name)
    exc = None
    try:
        ret = __salt__['container_resource.run'](name, cmd, container_type=__virtualname__, exec_driver=EXEC_DRIVER, output=output, no_start=no_start, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt, keep_env=keep_env)
    finally:
        if preserve_state and orig_state == 'stopped' and (state(name) != 'stopped'):
            stop(name)
    if output in (None, 'all'):
        return ret
    else:
        return ret[output]

@_ensure_exists
def pid(name):
    if False:
        print('Hello World!')
    '\n    Returns the PID of a container\n\n    name\n        Container name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.pid arch1\n    '
    try:
        return int(info(name).get('PID'))
    except (TypeError, ValueError) as exc:
        raise CommandExecutionError(f"Unable to get PID for container '{name}': {exc}")

def run(name, cmd, no_start=False, preserve_state=True, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        i = 10
        return i + 15
    "\n    Run :mod:`cmd.run <salt.modules.cmdmod.run>` within a container\n\n    name\n        Name of the container in which to run the command\n\n    cmd\n        Command to run\n\n    no_start : False\n        If the container is not running, don't start it\n\n    preserve_state : True\n        After running the command, return the container to its previous state\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.run mycontainer 'ip addr show'\n    "
    return _run(name, cmd, output=None, no_start=no_start, preserve_state=preserve_state, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_stdout(name, cmd, no_start=False, preserve_state=True, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        while True:
            i = 10
    "\n    Run :mod:`cmd.run_stdout <salt.modules.cmdmod.run_stdout>` within a container\n\n    name\n        Name of the container in which to run the command\n\n    cmd\n        Command to run\n\n    no_start : False\n        If the container is not running, don't start it\n\n    preserve_state : True\n        After running the command, return the container to its previous state\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console. Assumes\n        ``output=all``.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.run_stdout mycontainer 'ip addr show'\n    "
    return _run(name, cmd, output='stdout', no_start=no_start, preserve_state=preserve_state, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_stderr(name, cmd, no_start=False, preserve_state=True, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        i = 10
        return i + 15
    "\n    Run :mod:`cmd.run_stderr <salt.modules.cmdmod.run_stderr>` within a container\n\n    name\n        Name of the container in which to run the command\n\n    cmd\n        Command to run\n\n    no_start : False\n        If the container is not running, don't start it\n\n    preserve_state : True\n        After running the command, return the container to its previous state\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console. Assumes\n        ``output=all``.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.run_stderr mycontainer 'ip addr show'\n    "
    return _run(name, cmd, output='stderr', no_start=no_start, preserve_state=preserve_state, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def retcode(name, cmd, no_start=False, preserve_state=True, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run :mod:`cmd.retcode <salt.modules.cmdmod.retcode>` within a container\n\n    name\n        Name of the container in which to run the command\n\n    cmd\n        Command to run\n\n    no_start : False\n        If the container is not running, don't start it\n\n    preserve_state : True\n        After running the command, return the container to its previous state\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console. Assumes\n        ``output=all``.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.retcode mycontainer 'ip addr show'\n    "
    return _run(name, cmd, output='retcode', no_start=no_start, preserve_state=preserve_state, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_all(name, cmd, no_start=False, preserve_state=True, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        print('Hello World!')
    "\n    Run :mod:`cmd.run_all <salt.modules.cmdmod.run_all>` within a container\n\n    .. note::\n\n        While the command is run within the container, it is initiated from the\n        host. Therefore, the PID in the return dict is from the host, not from\n        the container.\n\n    name\n        Name of the container in which to run the command\n\n    cmd\n        Command to run\n\n    no_start : False\n        If the container is not running, don't start it\n\n    preserve_state : True\n        After running the command, return the container to its previous state\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console. Assumes\n        ``output=all``.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.run_all mycontainer 'ip addr show'\n    "
    return _run(name, cmd, output='all', no_start=no_start, preserve_state=preserve_state, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def bootstrap_container(name, dist=None, version=None):
    if False:
        return 10
    '\n    Bootstrap a container from package servers, if dist is None the os the\n    minion is running as will be created, otherwise the needed bootstrapping\n    tools will need to be available on the host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.bootstrap_container <name>\n    '
    if not dist:
        dist = __grains__['os'].lower()
        log.debug("nspawn.bootstrap: no dist provided, defaulting to '%s'", dist)
    try:
        return globals()[f'_bootstrap_{dist}'](name, version=version)
    except KeyError:
        raise CommandExecutionError(f'Unsupported distribution "{dist}"')

def _needs_install(name):
    if False:
        print('Hello World!')
    ret = 0
    has_minion = retcode(name, 'command -v salt-minion')
    if has_minion:
        processes = run_stdout(name, 'ps aux')
        if 'salt-minion' not in processes:
            ret = 1
        else:
            retcode(name, 'salt-call --local service.stop salt-minion')
    else:
        ret = 1
    return ret

def bootstrap_salt(name, config=None, approve_key=True, install=True, pub_key=None, priv_key=None, bootstrap_url=None, force_install=False, unconditional_install=False, bootstrap_delay=None, bootstrap_args=None, bootstrap_shell=None):
    if False:
        print('Hello World!')
    "\n    Bootstrap a container from package servers, if dist is None the os the\n    minion is running as will be created, otherwise the needed bootstrapping\n    tools will need to be available on the host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nspawn.bootstrap_salt arch1\n    "
    if bootstrap_delay is not None:
        try:
            time.sleep(bootstrap_delay)
        except TypeError:
            time.sleep(5)
    c_info = info(name)
    if not c_info:
        return None
    if bootstrap_args:
        if '{0}' not in bootstrap_args:
            bootstrap_args += ' -c {0}'
    else:
        bootstrap_args = '-c {0}'
    if not bootstrap_shell:
        bootstrap_shell = 'sh'
    orig_state = _ensure_running(name)
    if not orig_state:
        return orig_state
    if not force_install:
        needs_install = _needs_install(name)
    else:
        needs_install = True
    seeded = retcode(name, f"test -e '{SEED_MARKER}'") == 0
    tmp = tempfile.mkdtemp()
    if seeded and (not unconditional_install):
        ret = True
    else:
        ret = False
        cfg_files = __salt__['seed.mkconfig'](config, tmp=tmp, id_=name, approve_key=approve_key, pub_key=pub_key, priv_key=priv_key)
        if needs_install or force_install or unconditional_install:
            if install:
                rstr = __salt__['test.random_hash']()
                configdir = f'/tmp/.c_{rstr}'
                run(name, f'install -m 0700 -d {configdir}', python_shell=False)
                bs_ = __salt__['config.gather_bootstrap_script'](bootstrap=bootstrap_url)
                dest_dir = os.path.join('/tmp', rstr)
                for cmd in [f'mkdir -p {dest_dir}', f'chmod 700 {dest_dir}']:
                    if run_stdout(name, cmd):
                        log.error('tmpdir %s creation failed (%s)', dest_dir, cmd)
                        return False
                copy_to(name, bs_, f'{dest_dir}/bootstrap.sh', makedirs=True)
                copy_to(name, cfg_files['config'], os.path.join(configdir, 'minion'))
                copy_to(name, cfg_files['privkey'], os.path.join(configdir, 'minion.pem'))
                copy_to(name, cfg_files['pubkey'], os.path.join(configdir, 'minion.pub'))
                bootstrap_args = bootstrap_args.format(configdir)
                cmd = '{0} {2}/bootstrap.sh {1}'.format(bootstrap_shell, bootstrap_args.replace("'", "''"), dest_dir)
                log.info("Running %s in LXC container '%s'", cmd, name)
                ret = retcode(name, cmd, output_loglevel='info', use_vt=True) == 0
            else:
                ret = False
        else:
            minion_config = salt.config.minion_config(cfg_files['config'])
            pki_dir = minion_config['pki_dir']
            copy_to(name, cfg_files['config'], '/etc/salt/minion')
            copy_to(name, cfg_files['privkey'], os.path.join(pki_dir, 'minion.pem'))
            copy_to(name, cfg_files['pubkey'], os.path.join(pki_dir, 'minion.pub'))
            run(name, 'salt-call --local service.enable salt-minion', python_shell=False)
            ret = True
        shutil.rmtree(tmp)
        if orig_state == 'stopped':
            stop(name)
        if ret:
            run(name, f"touch '{SEED_MARKER}'", python_shell=False)
    return ret

def list_all():
    if False:
        i = 10
        return i + 15
    '\n    Lists all nspawn containers\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.list_all\n    '
    ret = []
    if _sd_version() >= 219:
        for line in _machinectl('list-images')['stdout'].splitlines():
            try:
                ret.append(line.split()[0])
            except IndexError:
                continue
    else:
        rootdir = _root()
        try:
            for dirname in os.listdir(rootdir):
                if os.path.isdir(os.path.join(rootdir, dirname)):
                    ret.append(dirname)
        except OSError:
            pass
    return ret

def list_running():
    if False:
        print('Hello World!')
    '\n    Lists running nspawn containers\n\n    .. note::\n\n        ``nspawn.list`` also works to list running containers\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.list_running\n        salt myminion nspawn.list\n    '
    ret = []
    for line in _machinectl('list')['stdout'].splitlines():
        try:
            ret.append(line.split()[0])
        except IndexError:
            pass
    return sorted(ret)
list_ = salt.utils.functools.alias_function(list_running, 'list_')

def list_stopped():
    if False:
        while True:
            i = 10
    '\n    Lists stopped nspawn containers\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.list_stopped\n    '
    return sorted(set(list_all()) - set(list_running()))

def exists(name):
    if False:
        i = 10
        return i + 15
    '\n    Returns true if the named container exists\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.exists <name>\n    '
    contextkey = f'nspawn.exists.{name}'
    if contextkey in __context__:
        return __context__[contextkey]
    __context__[contextkey] = name in list_all()
    return __context__[contextkey]

@_ensure_exists
def state(name):
    if False:
        print('Hello World!')
    '\n    Return state of container (running or stopped)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.state <name>\n    '
    try:
        cmd = f'show {name} --property=State'
        return _machinectl(cmd, ignore_retcode=True)['stdout'].split('=')[-1]
    except IndexError:
        return 'stopped'

def info(name, **kwargs):
    if False:
        return 10
    '\n    Return info about a container\n\n    .. note::\n\n        The container must be running for ``machinectl`` to gather information\n        about it. If the container is stopped, then this function will start\n        it.\n\n    start : False\n        If ``True``, then the container will be started to retrieve the info. A\n        ``Started`` key will be in the return data if the container was\n        started.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.info arch1\n        salt myminion nspawn.info arch1 force_start=False\n    '
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    start_ = kwargs.pop('start', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    if not start_:
        _ensure_running(name)
    elif name not in list_running():
        start(name)
    c_info = _machinectl(f'status {name}')
    if c_info['retcode'] != 0:
        raise CommandExecutionError(f"Unable to get info for container '{name}'")
    key_name_map = {'Iface': 'Network Interface', 'Leader': 'PID', 'Service': False, 'Since': 'Running Since'}
    ret = {}
    kv_pair = re.compile('^\\s+([A-Za-z]+): (.+)$')
    tree = re.compile('[|`]')
    lines = c_info['stdout'].splitlines()
    multiline = False
    cur_key = None
    for (idx, line) in enumerate(lines):
        match = kv_pair.match(line)
        if match:
            (key, val) = match.groups()
            key = key_name_map.get(key, key)
            if key is False:
                continue
            elif key == 'PID':
                try:
                    val = val.split()[0]
                except IndexError:
                    pass
            cur_key = key
            if multiline:
                multiline = False
            ret[key] = val
        else:
            if cur_key is None:
                continue
            if tree.search(lines[idx]):
                break
            if multiline:
                ret[cur_key].append(lines[idx].strip())
            else:
                ret[cur_key] = [ret[key], lines[idx].strip()]
                multiline = True
    return ret

@_ensure_exists
def enable(name):
    if False:
        print('Hello World!')
    '\n    Set the named container to be launched at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.enable <name>\n    '
    cmd = f'systemctl enable systemd-nspawn@{name}'
    if __salt__['cmd.retcode'](cmd, python_shell=False) != 0:
        __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
        return False
    return True

@_ensure_exists
def disable(name):
    if False:
        while True:
            i = 10
    '\n    Set the named container to *not* be launched at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.enable <name>\n    '
    cmd = f'systemctl disable systemd-nspawn@{name}'
    if __salt__['cmd.retcode'](cmd, python_shell=False) != 0:
        __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
        return False
    return True

@_ensure_exists
def start(name):
    if False:
        return 10
    '\n    Start the named container\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.start <name>\n    '
    if _sd_version() >= 219:
        ret = _machinectl(f'start {name}')
    else:
        cmd = f'systemctl start systemd-nspawn@{name}'
        ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
        return False
    return True

@_ensure_exists
def stop(name, kill=False):
    if False:
        print('Hello World!')
    '\n    This is a compatibility function which provides the logic for\n    nspawn.poweroff and nspawn.terminate.\n    '
    if _sd_version() >= 219:
        if kill:
            action = 'terminate'
        else:
            action = 'poweroff'
        ret = _machinectl(f'{action} {name}')
    else:
        cmd = f'systemctl stop systemd-nspawn@{name}'
        ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
        return False
    return True

def poweroff(name):
    if False:
        i = 10
        return i + 15
    '\n    Issue a clean shutdown to the container.  Equivalent to running\n    ``machinectl poweroff`` on the named container.\n\n    For convenience, running ``nspawn.stop``(as shown in the CLI examples\n    below) is equivalent to running ``nspawn.poweroff``.\n\n    .. note::\n\n        ``machinectl poweroff`` is only supported in systemd >= 219. On earlier\n        systemd versions, running this function will simply issue a clean\n        shutdown via ``systemctl``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.poweroff arch1\n        salt myminion nspawn.stop arch1\n    '
    return stop(name, kill=False)

def terminate(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Kill all processes in the container without issuing a clean shutdown.\n    Equivalent to running ``machinectl terminate`` on the named container.\n\n    For convenience, running ``nspawn.stop`` and passing ``kill=True`` (as\n    shown in the CLI examples below) is equivalent to running\n    ``nspawn.terminate``.\n\n    .. note::\n\n        ``machinectl terminate`` is only supported in systemd >= 219. On\n        earlier systemd versions, running this function will simply issue a\n        clean shutdown via ``systemctl``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.terminate arch1\n        salt myminion nspawn.stop arch1 kill=True\n    '
    return stop(name, kill=True)

def restart(name):
    if False:
        while True:
            i = 10
    '\n    This is a compatibility function which simply calls nspawn.reboot.\n    '
    return reboot(name)

@_ensure_exists
def reboot(name, kill=False):
    if False:
        return 10
    '\n    Reboot the container by sending a SIGINT to its init process. Equivalent\n    to running ``machinectl reboot`` on the named container.\n\n    For convenience, running ``nspawn.restart`` (as shown in the CLI examples\n    below) is equivalent to running ``nspawn.reboot``.\n\n    .. note::\n\n        ``machinectl reboot`` is only supported in systemd >= 219. On earlier\n        systemd versions, running this function will instead restart the\n        container via ``systemctl``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.reboot arch1\n        salt myminion nspawn.restart arch1\n    '
    if _sd_version() >= 219:
        if state(name) == 'running':
            ret = _machinectl(f'reboot {name}')
        else:
            return start(name)
    else:
        cmd = f'systemctl stop systemd-nspawn@{name}'
        ret = __salt__['cmd.run_all'](cmd, python_shell=False)
        if ret['retcode'] != 0:
            __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
            return False
        cmd = f'systemctl start systemd-nspawn@{name}'
        ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
        return False
    return True

@_ensure_exists
def remove(name, stop=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the named container\n\n    .. warning::\n\n        This function will remove all data associated with the container. It\n        will not, however, remove the btrfs subvolumes created by pulling\n        container images (:mod:`nspawn.pull_raw\n        <salt.modules.nspawn.pull_raw>`, :mod:`nspawn.pull_tar\n        <salt.modules.nspawn.pull_tar>`, :mod:`nspawn.pull_dkr\n        <salt.modules.nspawn.pull_dkr>`).\n\n    stop : False\n        If ``True``, the container will be destroyed even if it is\n        running/frozen.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' nspawn.remove foo\n        salt '*' nspawn.remove foo stop=True\n    "
    if not stop and state(name) != 'stopped':
        raise CommandExecutionError(f"Container '{name}' is not stopped")

    def _failed_remove(name, exc):
        if False:
            print('Hello World!')
        raise CommandExecutionError(f"Unable to remove container '{name}': {exc}")
    if _sd_version() >= 219:
        ret = _machinectl(f'remove {name}')
        if ret['retcode'] != 0:
            __context__['retcode'] = salt.defaults.exitcodes.EX_UNAVAILABLE
            _failed_remove(name, ret['stderr'])
    else:
        try:
            shutil.rmtree(os.path.join(_root(), name))
        except OSError as exc:
            _failed_remove(name, exc)
    return True
destroy = salt.utils.functools.alias_function(remove, 'destroy')

@_ensure_exists
def copy_to(name, source, dest, overwrite=False, makedirs=False):
    if False:
        print('Hello World!')
    "\n    Copy a file from the host into a container\n\n    name\n        Container name\n\n    source\n        File to be copied to the container\n\n    dest\n        Destination on the container. Must be an absolute path.\n\n    overwrite : False\n        Unless this option is set to ``True``, then if a file exists at the\n        location specified by the ``dest`` argument, an error will be raised.\n\n    makedirs : False\n\n        Create the parent directory on the container if it does not already\n        exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion' nspawn.copy_to /tmp/foo /root/foo\n    "
    path = source
    try:
        if source.startswith('salt://'):
            cached_source = __salt__['cp.cache_file'](source)
            if not cached_source:
                raise CommandExecutionError(f'Unable to cache {source}')
            path = cached_source
    except AttributeError:
        raise SaltInvocationError(f'Invalid source file {source}')
    if _sd_version() >= 219:
        pass
    return __salt__['container_resource.copy_to'](name, path, dest, container_type=__virtualname__, exec_driver=EXEC_DRIVER, overwrite=overwrite, makedirs=makedirs)
cp = salt.utils.functools.alias_function(copy_to, 'cp')

def _pull_image(pull_type, image, name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Common logic for machinectl pull-* commands\n    '
    _ensure_systemd(219)
    if exists(name):
        raise SaltInvocationError(f"Container '{name}' already exists")
    if pull_type in ('raw', 'tar'):
        valid_kwargs = ('verify',)
    elif pull_type == 'dkr':
        valid_kwargs = ('index',)
    else:
        raise SaltInvocationError(f"Unsupported image type '{pull_type}'")
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    bad_kwargs = {x: y for (x, y) in salt.utils.args.clean_kwargs(**kwargs).items() if x not in valid_kwargs}
    if bad_kwargs:
        salt.utils.args.invalid_kwargs(bad_kwargs)
    pull_opts = []
    if pull_type in ('raw', 'tar'):
        verify = kwargs.get('verify', False)
        if not verify:
            pull_opts.append('--verify=no')
        else:

            def _bad_verify():
                if False:
                    for i in range(10):
                        print('nop')
                raise SaltInvocationError("'verify' must be one of the following: signature, checksum")
            try:
                verify = verify.lower()
            except AttributeError:
                _bad_verify()
            else:
                if verify not in ('signature', 'checksum'):
                    _bad_verify()
                pull_opts.append(f'--verify={verify}')
    elif pull_type == 'dkr':
        if 'index' in kwargs:
            pull_opts.append('--dkr-index-url={}'.format(kwargs['index']))
    cmd = 'pull-{} {} {} {}'.format(pull_type, ' '.join(pull_opts), image, name)
    result = _machinectl(cmd, use_vt=True)
    if result['retcode'] != 0:
        msg = 'Error occurred pulling image. Stderr from the pull command (if any) follows: '
        if result['stderr']:
            msg += '\n\n{}'.format(result['stderr'])
        raise CommandExecutionError(msg)
    return True

def pull_raw(url, name, verify=False):
    if False:
        i = 10
        return i + 15
    '\n    Execute a ``machinectl pull-raw`` to download a .qcow2 or raw disk image,\n    and add it to /var/lib/machines as a new container.\n\n    .. note::\n\n        **Requires systemd >= 219**\n\n    url\n        URL from which to download the container\n\n    name\n        Name for the new container\n\n    verify : False\n        Perform signature or checksum verification on the container. See the\n        ``machinectl(1)`` man page (section titled "Image Transfer Commands")\n        for more information on requirements for image verification. To perform\n        signature verification, use ``verify=signature``. For checksum\n        verification, use ``verify=checksum``. By default, no verification will\n        be performed.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.pull_raw http://ftp.halifax.rwth-aachen.de/fedora/linux/releases/21/Cloud/Images/x86_64/Fedora-Cloud-Base-20141203-21.x86_64.raw.xz fedora21\n    '
    return _pull_image('raw', url, name, verify=verify)

def pull_tar(url, name, verify=False):
    if False:
        print('Hello World!')
    '\n    Execute a ``machinectl pull-raw`` to download a .tar container image,\n    and add it to /var/lib/machines as a new container.\n\n    .. note::\n\n        **Requires systemd >= 219**\n\n    url\n        URL from which to download the container\n\n    name\n        Name for the new container\n\n    verify : False\n        Perform signature or checksum verification on the container. See the\n        ``machinectl(1)`` man page (section titled "Image Transfer Commands")\n        for more information on requirements for image verification. To perform\n        signature verification, use ``verify=signature``. For checksum\n        verification, use ``verify=checksum``. By default, no verification will\n        be performed.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.pull_tar http://foo.domain.tld/containers/archlinux-2015.02.01.tar.gz arch2\n    '
    return _pull_image('tar', url, name, verify=verify)

def pull_dkr(url, name, index):
    if False:
        i = 10
        return i + 15
    '\n    Execute a ``machinectl pull-dkr`` to download a docker image and add it to\n    /var/lib/machines as a new container.\n\n    .. note::\n\n        **Requires systemd >= 219**\n\n    url\n        URL from which to download the container\n\n    name\n        Name for the new container\n\n    index\n        URL of the Docker index server from which to pull (must be an\n        ``http://`` or ``https://`` URL).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion nspawn.pull_dkr centos/centos6 cent6 index=https://get.docker.com\n        salt myminion nspawn.pull_docker centos/centos6 cent6 index=https://get.docker.com\n    '
    return _pull_image('dkr', url, name, index=index)
pull_docker = salt.utils.functools.alias_function(pull_dkr, 'pull_docker')