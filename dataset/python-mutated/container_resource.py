"""
Common resources for LXC and systemd-nspawn containers

.. versionadded:: 2015.8.0

These functions are not designed to be called directly, but instead from the
:mod:`lxc <salt.modules.lxc>`, :mod:`nspawn <salt.modules.nspawn>`, and
:mod:`docker <salt.modules.docker>` execution modules. They provide for
common logic to be re-used for common actions.
"""
import copy
import functools
import logging
import os
import shlex
import time
import traceback
import salt.utils.args
import salt.utils.path
import salt.utils.vt
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
PATH = 'PATH=/bin:/usr/bin:/sbin:/usr/sbin:/opt/bin:/usr/local/bin:/usr/local/sbin'

def _validate(wrapped):
    if False:
        return 10
    '\n    Decorator for common function argument validation\n    '

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        container_type = kwargs.get('container_type')
        exec_driver = kwargs.get('exec_driver')
        valid_driver = {'docker': ('lxc-attach', 'nsenter', 'docker-exec'), 'lxc': ('lxc-attach',), 'nspawn': ('nsenter',)}
        if container_type not in valid_driver:
            raise SaltInvocationError("Invalid container type '{}'. Valid types are: {}".format(container_type, ', '.join(sorted(valid_driver))))
        if exec_driver not in valid_driver[container_type]:
            raise SaltInvocationError('Invalid command execution driver. Valid drivers are: {}'.format(', '.join(valid_driver[container_type])))
        if exec_driver == 'lxc-attach' and (not salt.utils.path.which('lxc-attach')):
            raise SaltInvocationError("The 'lxc-attach' execution driver has been chosen, but lxc-attach is not available. LXC may not be installed.")
        return wrapped(*args, **salt.utils.args.clean_kwargs(**kwargs))
    return wrapper

def _nsenter(pid):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the nsenter command to attach to the named container\n    '
    return f'nsenter --target {pid} --mount --uts --ipc --net --pid'

def _get_md5(name, path, run_func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the MD5 checksum of a file from a container\n    '
    output = run_func(name, f'md5sum {shlex.quote(path)}', ignore_retcode=True)['stdout']
    try:
        return output.split()[0]
    except IndexError:
        return None

def cache_file(source):
    if False:
        print('Hello World!')
    '\n    Wrapper for cp.cache_file which raises an error if the file was unable to\n    be cached.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion container_resource.cache_file salt://foo/bar/baz.txt\n    '
    try:
        if source.startswith('salt://'):
            cached_source = __salt__['cp.cache_file'](source)
            if not cached_source:
                raise CommandExecutionError(f'Unable to cache {source}')
            return cached_source
    except AttributeError:
        raise SaltInvocationError(f'Invalid source file {source}')
    return source

@_validate
def run(name, cmd, container_type=None, exec_driver=None, output=None, no_start=False, stdin=None, python_shell=True, output_loglevel='debug', ignore_retcode=False, path=None, use_vt=False, keep_env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Common logic for running shell commands in containers\n\n    path\n        path to the container parent (for LXC only)\n        default: /var/lib/lxc (system default)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion container_resource.run mycontainer 'ps aux' container_type=docker exec_driver=nsenter output=stdout\n    "
    valid_output = ('stdout', 'stderr', 'retcode', 'all')
    if output is None:
        cmd_func = 'cmd.run'
    elif output not in valid_output:
        raise SaltInvocationError("'output' param must be one of the following: {}".format(', '.join(valid_output)))
    else:
        cmd_func = 'cmd.run_all'
    if keep_env is None or isinstance(keep_env, bool):
        to_keep = []
    elif not isinstance(keep_env, (list, tuple)):
        try:
            to_keep = keep_env.split(',')
        except AttributeError:
            log.warning('Invalid keep_env value, ignoring')
            to_keep = []
    else:
        to_keep = keep_env
    if exec_driver == 'lxc-attach':
        full_cmd = 'lxc-attach '
        if path:
            full_cmd += f'-P {shlex.quote(path)} '
        if keep_env is not True:
            full_cmd += '--clear-env '
            if 'PATH' not in to_keep:
                full_cmd += f'--set-var {PATH} '
        full_cmd += ' '.join([f'--set-var {x}={shlex.quote(os.environ[x])}' for x in to_keep if x in os.environ])
        full_cmd += f' -n {shlex.quote(name)} -- {cmd}'
    elif exec_driver == 'nsenter':
        pid = __salt__[f'{container_type}.pid'](name)
        full_cmd = f'nsenter --target {pid} --mount --uts --ipc --net --pid -- '
        if keep_env is not True:
            full_cmd += 'env -i '
            if 'PATH' not in to_keep:
                full_cmd += f'{PATH} '
        full_cmd += ' '.join([f'{x}={shlex.quote(os.environ[x])}' for x in to_keep if x in os.environ])
        full_cmd += f' {cmd}'
    elif exec_driver == 'docker-exec':
        full_cmd = 'docker exec '
        if stdin:
            full_cmd += '-i '
        full_cmd += f'{name} '
        if keep_env is not True:
            full_cmd += 'env -i '
            if 'PATH' not in to_keep:
                full_cmd += f'{PATH} '
        full_cmd += ' '.join([f'{x}={shlex.quote(os.environ[x])}' for x in to_keep if x in os.environ])
        full_cmd += f' {cmd}'
    if not use_vt:
        ret = __salt__[cmd_func](full_cmd, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode)
    else:
        (stdout, stderr) = ('', '')
        proc = salt.utils.vt.Terminal(full_cmd, shell=python_shell, log_stdin_level='quiet' if output_loglevel == 'quiet' else 'info', log_stdout_level=output_loglevel, log_stderr_level=output_loglevel, log_stdout=True, log_stderr=True, stream_stdout=False, stream_stderr=False)
        try:
            while proc.has_unread_data:
                try:
                    (cstdout, cstderr) = proc.recv()
                    if cstdout:
                        stdout += cstdout
                    if cstderr:
                        if output is None:
                            stdout += cstderr
                        else:
                            stderr += cstderr
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break
            ret = stdout if output is None else {'retcode': proc.exitstatus, 'pid': 2, 'stdout': stdout, 'stderr': stderr}
        except salt.utils.vt.TerminalException:
            trace = traceback.format_exc()
            log.error(trace)
            ret = stdout if output is None else {'retcode': 127, 'pid': 2, 'stdout': stdout, 'stderr': stderr}
        finally:
            proc.terminate()
    return ret

@_validate
def copy_to(name, source, dest, container_type=None, path=None, exec_driver=None, overwrite=False, makedirs=False):
    if False:
        while True:
            i = 10
    '\n    Common logic for copying files to containers\n\n    path\n        path to the container parent (for LXC only)\n        default: /var/lib/lxc (system default)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion container_resource.copy_to mycontainer /local/file/path /container/file/path container_type=docker exec_driver=nsenter\n    '
    state = __salt__[f'{container_type}.state']

    def run_all(*args, **akwargs):
        if False:
            for i in range(10):
                print('nop')
        akwargs = copy.deepcopy(akwargs)
        if container_type in ['lxc'] and 'path' not in akwargs:
            akwargs['path'] = path
        return __salt__[f'{container_type}.run_all'](*args, **akwargs)
    state_kwargs = {}
    cmd_kwargs = {'ignore_retcode': True}
    if container_type in ['lxc']:
        cmd_kwargs['path'] = path
        state_kwargs['path'] = path

    def _state(name):
        if False:
            print('Hello World!')
        if state_kwargs:
            return state(name, **state_kwargs)
        else:
            return state(name)
    c_state = _state(name)
    if c_state != 'running':
        raise CommandExecutionError(f"Container '{name}' is not running")
    local_file = cache_file(source)
    (source_dir, source_name) = os.path.split(local_file)
    if not os.path.isabs(local_file):
        raise SaltInvocationError('Source path must be absolute')
    elif not os.path.exists(local_file):
        raise SaltInvocationError(f'Source file {local_file} does not exist')
    elif not os.path.isfile(local_file):
        raise SaltInvocationError('Source must be a regular file')
    if not os.path.isabs(dest):
        raise SaltInvocationError('Destination path must be absolute')
    if run_all(name, f'test -d {shlex.quote(dest)}', **cmd_kwargs)['retcode'] == 0:
        dest = os.path.join(dest, source_name)
    else:
        (dest_dir, dest_name) = os.path.split(dest)
        if run_all(name, f'test -d {shlex.quote(dest_dir)}', **cmd_kwargs)['retcode'] != 0:
            if makedirs:
                result = run_all(name, f'mkdir -p {shlex.quote(dest_dir)}', **cmd_kwargs)
                if result['retcode'] != 0:
                    error = "Unable to create destination directory {} in container '{}'".format(dest_dir, name)
                    if result['stderr']:
                        error += ': {}'.format(result['stderr'])
                    raise CommandExecutionError(error)
            else:
                raise SaltInvocationError("Directory {} does not exist on {} container '{}'".format(dest_dir, container_type, name))
    if not overwrite and run_all(name, f'test -e {shlex.quote(dest)}', **cmd_kwargs)['retcode'] == 0:
        raise CommandExecutionError('Destination path {} already exists. Use overwrite=True to overwrite it'.format(dest))
    source_md5 = __salt__['file.get_sum'](local_file, 'md5')
    if source_md5 == _get_md5(name, dest, run_all):
        log.debug('%s and %s:%s are the same file, skipping copy', source, name, dest)
        return True
    log.debug("Copying %s to %s container '%s' as %s", source, container_type, name, dest)
    if exec_driver == 'lxc-attach':
        lxcattach = 'lxc-attach'
        if path:
            lxcattach += f' -P {shlex.quote(path)}'
        copy_cmd = 'cat "{0}" | {4} --clear-env --set-var {1} -n {2} -- tee "{3}"'.format(local_file, PATH, name, dest, lxcattach)
    elif exec_driver == 'nsenter':
        pid = __salt__[f'{container_type}.pid'](name)
        copy_cmd = 'cat "{}" | {} env -i {} tee "{}"'.format(local_file, _nsenter(pid), PATH, dest)
    elif exec_driver == 'docker-exec':
        copy_cmd = 'cat "{}" | docker exec -i {} env -i {} tee "{}"'.format(local_file, name, PATH, dest)
    __salt__['cmd.run'](copy_cmd, python_shell=True, output_loglevel='quiet')
    return source_md5 == _get_md5(name, dest, run_all)