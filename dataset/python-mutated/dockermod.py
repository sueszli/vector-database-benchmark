"""
Management of Docker Containers

.. versionadded:: 2015.8.0
.. versionchanged:: 2017.7.0
    This module has replaced the legacy docker execution module.

:depends: docker_ Python module

.. _`create_container()`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.create_container
.. _`create_host_config()`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.create_host_config
.. _`connect_container_to_network()`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.network.NetworkApiMixin.connect_container_to_network
.. _`create_network()`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.network.NetworkApiMixin.create_network
.. _`logs()`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.logs
.. _`IPAM pool`: http://docker-py.readthedocs.io/en/stable/api.html#docker.types.IPAMPool
.. _docker: https://pypi.python.org/pypi/docker
.. _docker-py: https://pypi.python.org/pypi/docker-py
.. _lxc-attach: https://linuxcontainers.org/lxc/manpages/man1/lxc-attach.1.html
.. _nsenter: http://man7.org/linux/man-pages/man1/nsenter.1.html
.. _docker-exec: http://docs.docker.com/reference/commandline/cli/#exec
.. _`docker-py Low-level API`: http://docker-py.readthedocs.io/en/stable/api.html
.. _timelib: https://pypi.python.org/pypi/timelib
.. _`trusted builds`: https://blog.docker.com/2013/11/introducing-trusted-builds/
.. _`Docker Engine API`: https://docs.docker.com/engine/api/v1.33/#operation/ContainerCreate

.. note::
    Older releases of the Python bindings for Docker were called docker-py_ in
    PyPI. All releases of docker_, and releases of docker-py_ >= 1.6.0 are
    supported. These python bindings can easily be installed using
    :py:func:`pip.install <salt.modules.pip.install>`:

    .. code-block:: bash

        salt myminion pip.install docker

    To upgrade from docker-py_ to docker_, you must first uninstall docker-py_,
    and then install docker_:

    .. code-block:: bash

        salt myminion pip.uninstall docker-py
        salt myminion pip.install docker

.. _docker-authentication:

Authentication
--------------

If you have previously performed a ``docker login`` from the minion, then the
credentials saved in ``~/.docker/config.json`` will be used for any actions
which require authentication. If not, then credentials can be configured in
any of the following locations:

- Minion config file
- Grains
- Pillar data
- Master config file (requires :conf_minion:`pillar_opts` to be set to ``True``
  in Minion config file in order to work)

.. important::
    Versions prior to 3000 require that Docker credentials are configured in
    Pillar data. Be advised that Pillar data is still recommended though,
    because this keeps the configuration from being stored on the Minion.

    Also, keep in mind that if one gets your ``~/.docker/config.json``, the
    password can be decoded from its contents.

The configuration schema is as follows:

.. code-block:: yaml

    docker-registries:
      <registry_url>:
        username: <username>
        password: <password>

For example:

.. code-block:: yaml

    docker-registries:
      hub:
        username: foo
        password: s3cr3t

.. note::
    As of the 2016.3.7, 2016.11.4, and 2017.7.0 releases of Salt, credentials
    for the Docker Hub can be configured simply by specifying ``hub`` in place
    of the registry URL. In earlier releases, it is necessary to specify the
    actual registry URL for the Docker Hub (i.e.
    ``https://index.docker.io/v1/``).

More than one registry can be configured. Salt will look for Docker credentials
in the ``docker-registries`` Pillar key, as well as any key ending in
``-docker-registries``. For example:

.. code-block:: yaml

    docker-registries:
      'https://mydomain.tld/registry:5000':
        username: foo
        password: s3cr3t

    foo-docker-registries:
      https://index.foo.io/v1/:
        username: foo
        password: s3cr3t

    bar-docker-registries:
      https://index.bar.io/v1/:
        username: foo
        password: s3cr3t

To login to the configured registries, use the :py:func:`docker.login
<salt.modules.dockermod.login>` function. This only needs to be done once for a
given registry, and it will store/update the credentials in
``~/.docker/config.json``.

.. note::
    For Salt releases before 2016.3.7 and 2016.11.4, :py:func:`docker.login
    <salt.modules.dockermod.login>` is not available. Instead, Salt will try to
    authenticate using each of your configured registries for each push/pull,
    behavior which is not correct and has been resolved in newer releases.


Configuration Options
---------------------

The following configuration options can be set to fine-tune how Salt uses
Docker:

- ``docker.url``: URL to the docker service (default: local socket).
- ``docker.version``: API version to use (should not need to be set manually in
  the vast majority of cases)
- ``docker.exec_driver``: Execution driver to use, one of ``nsenter``,
  ``lxc-attach``, or ``docker-exec``. See the :ref:`Executing Commands Within a
  Running Container <docker-execution-driver>` section for more details on how
  this config parameter is used.

These configuration options are retrieved using :py:mod:`config.get
<salt.modules.config.get>` (click the link for further information).

.. _docker-execution-driver:

Executing Commands Within a Running Container
---------------------------------------------

.. note::
    With the release of Docker 1.13.1, the Execution Driver has been removed.
    Starting in versions 2016.3.6, 2016.11.4, and 2017.7.0, Salt defaults to
    using ``docker exec`` to run commands in containers, however for older Salt
    releases it will be necessary to set the ``docker.exec_driver`` config
    option to either ``docker-exec`` or ``nsenter`` for Docker versions 1.13.1
    and newer.

Multiple methods exist for executing commands within Docker containers:

- lxc-attach_: Default for older versions of docker
- nsenter_: Enters container namespace to run command
- docker-exec_: Native support for executing commands in Docker containers
  (added in Docker 1.3)

Adding a configuration option (see :py:func:`config.get
<salt.modules.config.get>`) called ``docker.exec_driver`` will tell Salt which
execution driver to use:

.. code-block:: yaml

    docker.exec_driver: docker-exec

If this configuration option is not found, Salt will use the appropriate
interface (either nsenter_ or lxc-attach_) based on the ``Execution Driver``
value returned from ``docker info``. docker-exec_ will not be used by default,
as it is presently (as of version 1.6.2) only able to execute commands as the
effective user of the container. Thus, if a ``USER`` directive was used to run
as a non-privileged user, docker-exec_ would be unable to perform the action as
root. Salt can still use docker-exec_ as an execution driver, but must be
explicitly configured (as in the example above) to do so at this time.

If possible, try to manually specify the execution driver, as it will save Salt
a little work.

This execution module provides functions that shadow those from the :mod:`cmd
<salt.modules.cmdmod>` module. They are as follows:

- :py:func:`docker.retcode <salt.modules.dockermod.retcode>`
- :py:func:`docker.run <salt.modules.dockermod.run>`
- :py:func:`docker.run_all <salt.modules.dockermod.run_all>`
- :py:func:`docker.run_stderr <salt.modules.dockermod.run_stderr>`
- :py:func:`docker.run_stdout <salt.modules.dockermod.run_stdout>`
- :py:func:`docker.script <salt.modules.dockermod.script>`
- :py:func:`docker.script_retcode <salt.modules.dockermod.script_retcode>`


Detailed Function Documentation
-------------------------------
"""
import bz2
import copy
import fnmatch
import functools
import gzip
import json
import logging
import os
import re
import shlex
import shutil
import string
import subprocess
import time
import uuid
import salt.client.ssh.state
import salt.exceptions
import salt.fileclient
import salt.pillar
import salt.utils.dockermod.translate.container
import salt.utils.dockermod.translate.network
import salt.utils.functools
import salt.utils.json
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError
from salt.state import HighState
__docformat__ = 'restructuredtext en'
__deprecated__ = (3009, 'docker', 'https://github.com/saltstack/saltext-docker')
try:
    import docker
    HAS_DOCKER_PY = True
except ImportError:
    HAS_DOCKER_PY = False
try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False
try:
    import timelib
    HAS_TIMELIB = True
except ImportError:
    HAS_TIMELIB = False
HAS_NSENTER = bool(salt.utils.path.which('nsenter'))
log = logging.getLogger(__name__)
__func_alias__ = {'import_': 'import', 'ps_': 'ps', 'rm_': 'rm', 'signal_': 'signal', 'start_': 'start', 'tag_': 'tag', 'apply_': 'apply'}
MIN_DOCKER = (1, 9, 0)
MIN_DOCKER_PY = (1, 6, 0)
VERSION_RE = '([\\d.]+)'
NOTSET = object()
__virtualname__ = 'docker'
__virtual_aliases__ = ('dockerng', 'moby')
__proxyenabled__ = ['docker']
__outputter__ = {'sls': 'highstate', 'apply_': 'highstate', 'highstate': 'highstate'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if docker libs are present\n    '
    if HAS_DOCKER_PY:
        try:
            docker_py_versioninfo = _get_docker_py_versioninfo()
        except Exception:
            return (False, 'Docker module found, but no version could be extracted')
        if docker_py_versioninfo is None:
            return (False, 'Docker module found, but no version could be extracted')
        if docker_py_versioninfo >= MIN_DOCKER_PY:
            try:
                docker_versioninfo = version().get('VersionInfo')
            except Exception:
                docker_versioninfo = None
            if docker_versioninfo is None or docker_versioninfo >= MIN_DOCKER:
                return __virtualname__
            else:
                return (False, 'Insufficient Docker version (required: {}, installed: {})'.format('.'.join(map(str, MIN_DOCKER)), '.'.join(map(str, docker_versioninfo))))
        return (False, 'Insufficient docker-py version (required: {}, installed: {})'.format('.'.join(map(str, MIN_DOCKER_PY)), '.'.join(map(str, docker_py_versioninfo))))
    return (False, 'Could not import docker module, is docker-py installed?')

class DockerJSONDecoder(json.JSONDecoder):

    def decode(self, s, _w=None):
        if False:
            print('Hello World!')
        objs = []
        for line in s.splitlines():
            if not line:
                continue
            (obj, _) = self.raw_decode(line)
            objs.append(obj)
        return objs

def _get_docker_py_versioninfo():
    if False:
        print('Hello World!')
    '\n    Returns the version_info tuple from docker-py\n    '
    try:
        return docker.version_info
    except AttributeError:
        try:
            docker_version = docker.__version__.split('.')
            return tuple((int(n) for n in docker_version))
        except AttributeError:
            pass

def _get_client(timeout=NOTSET, **kwargs):
    if False:
        while True:
            i = 10
    client_kwargs = {}
    if timeout is not NOTSET:
        client_kwargs['timeout'] = timeout
    for (key, val) in (('base_url', 'docker.url'), ('version', 'docker.version')):
        param = __salt__['config.option'](val, NOTSET)
        if param is not NOTSET:
            client_kwargs[key] = param
    if 'base_url' not in client_kwargs and 'DOCKER_HOST' in os.environ:
        client_kwargs['base_url'] = os.environ.get('DOCKER_HOST')
    if 'version' not in client_kwargs:
        client_kwargs['version'] = 'auto'
    docker_machine = __salt__['config.option']('docker.machine', NOTSET)
    if docker_machine is not NOTSET:
        docker_machine_json = __salt__['cmd.run'](['docker-machine', 'inspect', docker_machine], python_shell=False)
        try:
            docker_machine_json = salt.utils.json.loads(docker_machine_json)
            docker_machine_tls = docker_machine_json['HostOptions']['AuthOptions']
            docker_machine_ip = docker_machine_json['Driver']['IPAddress']
            client_kwargs['base_url'] = 'https://' + docker_machine_ip + ':2376'
            client_kwargs['tls'] = docker.tls.TLSConfig(client_cert=(docker_machine_tls['ClientCertPath'], docker_machine_tls['ClientKeyPath']), ca_cert=docker_machine_tls['CaCertPath'], assert_hostname=False, verify=True)
        except Exception as exc:
            raise CommandExecutionError(f'Docker machine {docker_machine} failed: {exc}')
    try:
        ret = docker.APIClient(**client_kwargs)
    except AttributeError:
        ret = docker.Client(**client_kwargs)
    log.debug('docker-py API version: %s', getattr(ret, 'api_version', None))
    return ret

def _get_state(inspect_results):
    if False:
        return 10
    '\n    Helper for deriving the current state of the container from the inspect\n    results.\n    '
    if inspect_results.get('State', {}).get('Paused', False):
        return 'paused'
    elif inspect_results.get('State', {}).get('Running', False):
        return 'running'
    else:
        return 'stopped'

def _docker_client(wrapped):
    if False:
        print('Hello World!')
    '\n    Decorator to run a function that requires the use of a docker.Client()\n    instance.\n    '

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that the client is present\n        '
        kwargs = __utils__['args.clean_kwargs'](**kwargs)
        timeout = kwargs.pop('client_timeout', NOTSET)
        if 'docker.client' not in __context__ or not hasattr(__context__['docker.client'], 'timeout'):
            __context__['docker.client'] = _get_client(timeout=timeout, **kwargs)
        orig_timeout = None
        if timeout is not NOTSET and hasattr(__context__['docker.client'], 'timeout') and (__context__['docker.client'].timeout != timeout):
            orig_timeout = __context__['docker.client'].timeout
            __context__['docker.client'].timeout = timeout
        ret = wrapped(*args, **kwargs)
        if orig_timeout is not None:
            __context__['docker.client'].timeout = orig_timeout
        return ret
    return wrapper

def _refresh_mine_cache(wrapped):
    if False:
        while True:
            i = 10
    '\n    Decorator to trigger a refresh of salt mine data.\n    '

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        refresh salt mine on exit.\n        '
        returned = wrapped(*args, **__utils__['args.clean_kwargs'](**kwargs))
        if _check_update_mine():
            __salt__['mine.send']('docker.ps', verbose=True, all=True, host=True)
        return returned
    return wrapper

def _check_update_mine():
    if False:
        while True:
            i = 10
    try:
        ret = __context__['docker.update_mine']
    except KeyError:
        ret = __context__['docker.update_mine'] = __salt__['config.option']('docker.update_mine', default=True)
    return ret

def _change_state(name, action, expected, *args, **kwargs):
    if False:
        return 10
    '\n    Change the state of a container\n    '
    pre = state(name)
    if action != 'restart' and pre == expected:
        return {'result': False, 'state': {'old': expected, 'new': expected}, 'comment': f"Container '{name}' already {expected}"}
    _client_wrapper(action, name, *args, **kwargs)
    _clear_context()
    try:
        post = state(name)
    except CommandExecutionError:
        post = None
    ret = {'result': post == expected, 'state': {'old': pre, 'new': post}}
    return ret

def _clear_context():
    if False:
        print('Hello World!')
    '\n    Clear the state/exists values stored in context\n    '
    keep_context = ('docker.client', 'docker.exec_driver', 'docker._pull_status', 'docker.docker_version', 'docker.docker_py_version')
    for key in list(__context__):
        try:
            if key.startswith('docker.') and key not in keep_context:
                __context__.pop(key)
        except AttributeError:
            pass

def _get_md5(name, path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the MD5 checksum of a file from a container\n    '
    output = run_stdout(name, f'md5sum {shlex.quote(path)}', ignore_retcode=True)
    try:
        return output.split()[0]
    except IndexError:
        return None

def _get_exec_driver():
    if False:
        return 10
    '\n    Get the method to be used in shell commands\n    '
    contextkey = 'docker.exec_driver'
    if contextkey not in __context__:
        from_config = __salt__['config.option'](contextkey, None)
        if from_config is not None:
            __context__[contextkey] = from_config
            return from_config
        driver = info().get('ExecutionDriver', 'docker-exec')
        if driver == 'docker-exec':
            __context__[contextkey] = driver
        elif driver.startswith('lxc-'):
            __context__[contextkey] = 'lxc-attach'
        elif driver.startswith('native-') and HAS_NSENTER:
            __context__[contextkey] = 'nsenter'
        elif not driver.strip() and HAS_NSENTER:
            log.warning("ExecutionDriver from 'docker info' is blank, falling back to using 'nsenter'. To squelch this warning, set docker.exec_driver. See the Salt documentation for the docker module for more information.")
            __context__[contextkey] = 'nsenter'
        else:
            raise NotImplementedError("Unknown docker ExecutionDriver '{}', or didn't find command to attach to the container".format(driver))
    return __context__[contextkey]

def _get_top_level_images(imagedata, subset=None):
    if False:
        return 10
    '\n    Returns a list of the top-level images (those which are not parents). If\n    ``subset`` (an iterable) is passed, the top-level images in the subset will\n    be returned, otherwise all top-level images will be returned.\n    '
    try:
        parents = [imagedata[x]['ParentId'] for x in imagedata]
        filter_ = subset if subset is not None else imagedata
        return [x for x in filter_ if x not in parents]
    except (KeyError, TypeError):
        raise CommandExecutionError('Invalid image data passed to _get_top_level_images(). Please report this issue. Full image data: {}'.format(imagedata))

def _prep_pull():
    if False:
        while True:
            i = 10
    '\n    Populate __context__ with the current (pre-pull) image IDs (see the\n    docstring for _pull_status for more information).\n    '
    __context__['docker._pull_status'] = [x[:12] for x in images(all=True)]

def _scrub_links(links, name):
    if False:
        return 10
    '\n    Remove container name from HostConfig:Links values to enable comparing\n    container configurations correctly.\n    '
    if isinstance(links, list):
        ret = []
        for l in links:
            ret.append(l.replace(f'/{name}/', '/', 1))
    else:
        ret = links
    return ret

def _ulimit_sort(ulimit_val):
    if False:
        while True:
            i = 10
    if isinstance(ulimit_val, list):
        return sorted(ulimit_val, key=lambda x: (x.get('Name'), x.get('Hard', 0), x.get('Soft', 0)))
    return ulimit_val

def _size_fmt(num):
    if False:
        return 10
    '\n    Format bytes as human-readable file sizes\n    '
    try:
        num = int(num)
        if num < 1024:
            return f'{num} bytes'
        num /= 1024.0
        for unit in ('KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
            if num < 1024.0:
                return f'{num:3.1f} {unit}'
            num /= 1024.0
    except Exception:
        log.error("Unable to format file size for '%s'", num)
        return 'unknown'

@_docker_client
def _client_wrapper(attr, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Common functionality for running low-level API calls\n    '
    catch_api_errors = kwargs.pop('catch_api_errors', True)
    func = getattr(__context__['docker.client'], attr, None)
    if func is None or not hasattr(func, '__call__'):
        raise SaltInvocationError(f"Invalid client action '{attr}'")
    if attr in ('push', 'pull'):
        try:
            __context__['docker.client'].reload_config()
        except AttributeError:
            pass
    err = ''
    try:
        log.debug('Attempting to run docker-py\'s "%s" function with args=%s and kwargs=%s', attr, args, kwargs)
        ret = func(*args, **kwargs)
    except docker.errors.APIError as exc:
        if catch_api_errors:
            raise CommandExecutionError(f'Error {exc.response.status_code}: {exc.explanation}')
        else:
            raise
    except docker.errors.DockerException as exc:
        raise CommandExecutionError(exc.__str__())
    except Exception as exc:
        err = exc.__str__()
    else:
        return ret
    msg = f'Unable to perform {attr}'
    if err:
        msg += f': {err}'
    raise CommandExecutionError(msg)

def _build_status(data, item):
    if False:
        return 10
    '\n    Process a status update from a docker build, updating the data structure\n    '
    stream = item['stream']
    if 'Running in' in stream:
        data.setdefault('Intermediate_Containers', []).append(stream.rstrip().split()[-1])
    if 'Successfully built' in stream:
        data['Id'] = stream.rstrip().split()[-1]

def _import_status(data, item, repo_name, repo_tag):
    if False:
        print('Hello World!')
    '\n    Process a status update from docker import, updating the data structure\n    '
    status = item['status']
    try:
        if 'Downloading from' in status:
            return
        elif all((x in string.hexdigits for x in status)):
            data['Image'] = f'{repo_name}:{repo_tag}'
            data['Id'] = status
    except (AttributeError, TypeError):
        pass

def _pull_status(data, item):
    if False:
        for i in range(10):
            print('nop')
    "\n    Process a status update from a docker pull, updating the data structure.\n\n    For containers created with older versions of Docker, there is no\n    distinction in the status updates between layers that were already present\n    (and thus not necessary to download), and those which were actually\n    downloaded. Because of this, any function that needs to invoke this\n    function needs to pre-fetch the image IDs by running _prep_pull() in any\n    function that calls _pull_status(). It is important to grab this\n    information before anything is pulled so we aren't looking at the state of\n    the images post-pull.\n\n    We can't rely on the way that __context__ is utilized by the images()\n    function, because by design we clear the relevant context variables once\n    we've made changes to allow the next call to images() to pick up any\n    changes that were made.\n    "

    def _already_exists(id_):
        if False:
            for i in range(10):
                print('nop')
        '\n        Layer already exists\n        '
        already_pulled = data.setdefault('Layers', {}).setdefault('Already_Pulled', [])
        if id_ not in already_pulled:
            already_pulled.append(id_)

    def _new_layer(id_):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pulled a new layer\n        '
        pulled = data.setdefault('Layers', {}).setdefault('Pulled', [])
        if id_ not in pulled:
            pulled.append(id_)
    if 'docker._pull_status' not in __context__:
        log.warning('_pull_status context variable was not populated, information on downloaded layers may be inaccurate. Please report this to the SaltStack development team, and if possible include the image (and tag) that was being pulled.')
        __context__['docker._pull_status'] = NOTSET
    status = item['status']
    if status == 'Already exists':
        _already_exists(item['id'])
    elif status in 'Pull complete':
        _new_layer(item['id'])
    elif status.startswith('Status: '):
        data['Status'] = status[8:]
    elif status == 'Download complete':
        if __context__['docker._pull_status'] is not NOTSET:
            id_ = item['id']
            if id_ in __context__['docker._pull_status']:
                _already_exists(id_)
            else:
                _new_layer(id_)

def _push_status(data, item):
    if False:
        while True:
            i = 10
    '\n    Process a status update from a docker push, updating the data structure\n    '
    status = item['status'].lower()
    if 'id' in item:
        if 'already pushed' in status or 'already exists' in status:
            already_pushed = data.setdefault('Layers', {}).setdefault('Already_Pushed', [])
            already_pushed.append(item['id'])
        elif 'successfully pushed' in status or status == 'pushed':
            pushed = data.setdefault('Layers', {}).setdefault('Pushed', [])
            pushed.append(item['id'])

def _error_detail(data, item):
    if False:
        return 10
    '\n    Process an API error, updating the data structure\n    '
    err = item['errorDetail']
    if 'code' in err:
        try:
            msg = ': '.join((item['errorDetail']['code'], item['errorDetail']['message']))
        except TypeError:
            msg = '{}: {}'.format(item['errorDetail']['code'], item['errorDetail']['message'])
    else:
        msg = item['errorDetail']['message']
    data.append(msg)

def get_client_args(limit=None):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.6,2016.11.4,2017.7.0\n    .. versionchanged:: 2017.7.0\n        Replaced the container config args with the ones from the API's\n        ``create_container`` function.\n    .. versionchanged:: 2018.3.0\n        Added ability to limit the input to specific client functions\n\n    Many functions in Salt have been written to support the full list of\n    arguments for a given function in the `docker-py Low-level API`_. However,\n    depending on the version of docker-py installed on the minion, the\n    available arguments may differ. This function will get the arguments for\n    various functions in the installed version of docker-py, to be used as a\n    reference.\n\n    limit\n        An optional list of categories for which to limit the return. This is\n        useful if only a specific set of arguments is desired, and also keeps\n        other function's argspecs from needlessly being examined.\n\n    **AVAILABLE LIMITS**\n\n    - ``create_container`` - arguments accepted by `create_container()`_ (used\n      by :py:func:`docker.create <salt.modules.dockermod.create>`)\n    - ``host_config`` - arguments accepted by `create_host_config()`_ (used to\n      build the host config for :py:func:`docker.create\n      <salt.modules.dockermod.create>`)\n    - ``connect_container_to_network`` - arguments used by\n      `connect_container_to_network()`_ to construct an endpoint config when\n      connecting to a network (used by\n      :py:func:`docker.connect_container_to_network\n      <salt.modules.dockermod.connect_container_to_network>`)\n    - ``create_network`` - arguments accepted by `create_network()`_ (used by\n      :py:func:`docker.create_network <salt.modules.dockermod.create_network>`)\n    - ``ipam_config`` - arguments used to create an `IPAM pool`_ (used by\n      :py:func:`docker.create_network <salt.modules.dockermod.create_network>`\n      in the process of constructing an IPAM config dictionary)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.get_client_args\n        salt myminion docker.get_client_args logs\n        salt myminion docker.get_client_args create_container,connect_container_to_network\n    "
    return __utils__['docker.get_client_args'](limit=limit)

def _get_create_kwargs(skip_translate=None, ignore_collisions=False, validate_ip_addrs=True, client_args=None, **kwargs):
    if False:
        return 10
    "\n    Take input kwargs and return a kwargs dict to pass to docker-py's\n    create_container() function.\n    "
    networks = kwargs.pop('networks', {})
    if kwargs.get('network_mode', '') in networks:
        networks = {kwargs['network_mode']: networks[kwargs['network_mode']]}
    else:
        networks = {}
    kwargs = __utils__['docker.translate_input'](salt.utils.dockermod.translate.container, skip_translate=skip_translate, ignore_collisions=ignore_collisions, validate_ip_addrs=validate_ip_addrs, **__utils__['args.clean_kwargs'](**kwargs))
    if networks:
        kwargs['networking_config'] = _create_networking_config(networks)
    if client_args is None:
        try:
            client_args = get_client_args(['create_container', 'host_config'])
        except CommandExecutionError as exc:
            log.error("docker.create: Error getting client args: '%s'", exc, exc_info=True)
            raise CommandExecutionError(f'Failed to get client args: {exc}')
    full_host_config = {}
    host_kwargs = {}
    create_kwargs = {}
    for arg in list(kwargs):
        if arg in client_args['host_config']:
            host_kwargs[arg] = kwargs.pop(arg)
            continue
        if arg in client_args['create_container']:
            if arg == 'host_config':
                full_host_config.update(kwargs.pop(arg))
            else:
                create_kwargs[arg] = kwargs.pop(arg)
            continue
    create_kwargs['host_config'] = _client_wrapper('create_host_config', **host_kwargs)
    create_kwargs['host_config'].update(full_host_config)
    return (create_kwargs, kwargs)

def compare_containers(first, second, ignore=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n    .. versionchanged:: 2018.3.0\n        Renamed from ``docker.compare_container`` to\n        ``docker.compare_containers`` (old function name remains as an alias)\n\n    Compare two containers' Config and and HostConfig and return any\n    differences between the two.\n\n    first\n        Name or ID of first container\n\n    second\n        Name or ID of second container\n\n    ignore\n        A comma-separated list (or Python list) of keys to ignore when\n        comparing. This is useful when comparing two otherwise identical\n        containers which have different hostnames.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.compare_containers foo bar\n        salt myminion docker.compare_containers foo bar ignore=Hostname\n    "
    ignore = __utils__['args.split_input'](ignore or [])
    result1 = inspect_container(first)
    result2 = inspect_container(second)
    ret = {}
    for conf_dict in ('Config', 'HostConfig'):
        for item in result1[conf_dict]:
            if item in ignore:
                continue
            val1 = result1[conf_dict][item]
            val2 = result2[conf_dict].get(item)
            if item in ('OomKillDisable',) or (val1 is None or val2 is None):
                if bool(val1) != bool(val2):
                    ret.setdefault(conf_dict, {})[item] = {'old': val1, 'new': val2}
            elif item == 'Image':
                image1 = inspect_image(val1)['Id']
                image2 = inspect_image(val2)['Id']
                if image1 != image2:
                    ret.setdefault(conf_dict, {})[item] = {'old': image1, 'new': image2}
            else:
                if item == 'Links':
                    val1 = sorted(_scrub_links(val1, first))
                    val2 = sorted(_scrub_links(val2, second))
                if item == 'Ulimits':
                    val1 = _ulimit_sort(val1)
                    val2 = _ulimit_sort(val2)
                if item == 'Env':
                    val1 = sorted(val1)
                    val2 = sorted(val2)
                if val1 != val2:
                    ret.setdefault(conf_dict, {})[item] = {'old': val1, 'new': val2}
        for item in result2[conf_dict]:
            if item in ignore or item in ret.get(conf_dict, {}):
                continue
            val1 = result1[conf_dict].get(item)
            val2 = result2[conf_dict][item]
            if item in ('OomKillDisable',) or (val1 is None or val2 is None):
                if bool(val1) != bool(val2):
                    ret.setdefault(conf_dict, {})[item] = {'old': val1, 'new': val2}
            elif item == 'Image':
                image1 = inspect_image(val1)['Id']
                image2 = inspect_image(val2)['Id']
                if image1 != image2:
                    ret.setdefault(conf_dict, {})[item] = {'old': image1, 'new': image2}
            else:
                if item == 'Links':
                    val1 = sorted(_scrub_links(val1, first))
                    val2 = sorted(_scrub_links(val2, second))
                if item == 'Ulimits':
                    val1 = _ulimit_sort(val1)
                    val2 = _ulimit_sort(val2)
                if item == 'Env':
                    val1 = sorted(val1)
                    val2 = sorted(val2)
                if val1 != val2:
                    ret.setdefault(conf_dict, {})[item] = {'old': val1, 'new': val2}
    return ret
compare_container = salt.utils.functools.alias_function(compare_containers, 'compare_container')

def compare_container_networks(first, second):
    if False:
        return 10
    "\n    .. versionadded:: 2018.3.0\n\n    Returns the differences between two containers' networks. When a network is\n    only present one of the two containers, that network's diff will simply be\n    represented with ``True`` for the side of the diff in which the network is\n    present) and ``False`` for the side of the diff in which the network is\n    absent.\n\n    This function works by comparing the contents of both containers'\n    ``Networks`` keys (under ``NetworkSettings``) in the return data from\n    :py:func:`docker.inspect_container\n    <salt.modules.dockermod.inspect_container>`. Because each network contains\n    some items that either A) only set at runtime, B) naturally varying from\n    container to container, or both, by default the following keys in each\n    network are examined:\n\n    - **Aliases**\n    - **Links**\n    - **IPAMConfig**\n\n    The exception to this is if ``IPAMConfig`` is unset (i.e. null) in one\n    container but not the other. This happens when no static IP configuration\n    is set, and automatic IP configuration is in effect. So, in order to report\n    on changes between automatic IP configuration in one container and static\n    IP configuration in another container (as we need to do for the\n    :py:func:`docker_container.running <salt.states.docker_container.running>`\n    state), automatic IP configuration will also be checked in these cases.\n\n    This function uses the :conf_minion:`docker.compare_container_networks`\n    minion config option to determine which keys to examine. This provides\n    flexibility in the event that features added in a future Docker release\n    necessitate changes to how Salt compares networks. In these cases, rather\n    than waiting for a new Salt release one can just set\n    :conf_minion:`docker.compare_container_networks`.\n\n    .. versionchanged:: 3000\n        This config option can now also be set in pillar data and grains.\n        Additionally, it can be set in the master config file, provided that\n        :conf_minion:`pillar_opts` is enabled on the minion.\n\n    .. note::\n        The checks for automatic IP configuration described above only apply if\n        ``IPAMConfig`` is among the keys set for static IP checks in\n        :conf_minion:`docker.compare_container_networks`.\n\n    first\n        Name or ID of first container (old)\n\n    second\n        Name or ID of second container (new)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.compare_container_networks foo bar\n    "

    def _get_nets(data):
        if False:
            return 10
        return data.get('NetworkSettings', {}).get('Networks', {})
    compare_keys = __salt__['config.option']('docker.compare_container_networks')
    result1 = inspect_container(first) if not isinstance(first, dict) else first
    result2 = inspect_container(second) if not isinstance(second, dict) else second
    nets1 = _get_nets(result1)
    nets2 = _get_nets(result2)
    state1 = state(first)
    state2 = state(second)
    all_nets = set(nets1)
    all_nets.update(nets2)
    for net_name in all_nets:
        try:
            connected_containers = inspect_network(net_name).get('Containers', {})
        except Exception as exc:
            log.warning('Failed to inspect Docker network %s: %s', net_name, exc)
            continue
        else:
            if state1 == 'running' and net_name in nets1 and (result1['Id'] not in connected_containers):
                del nets1[net_name]
            if state2 == 'running' and net_name in nets2 and (result2['Id'] not in connected_containers):
                del nets2[net_name]
    ret = {}

    def _check_ipconfig(ret, net_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        nets1_missing = 'old' not in kwargs
        if nets1_missing:
            nets1_static = False
        else:
            nets1_static = bool(kwargs['old'])
        nets1_autoip = not nets1_static and (not nets1_missing)
        nets2_missing = 'new' not in kwargs
        if nets2_missing:
            nets2_static = False
        else:
            nets2_static = bool(kwargs['new'])
        nets2_autoip = not nets2_static and (not nets2_missing)
        autoip_keys = compare_keys.get('automatic', [])
        if nets1_autoip and (nets2_static or nets2_missing):
            for autoip_key in autoip_keys:
                autoip_val = nets1[net_name].get(autoip_key)
                if autoip_val:
                    ret.setdefault(net_name, {})[autoip_key] = {'old': autoip_val, 'new': None}
            if nets2_static:
                ret.setdefault(net_name, {})['IPAMConfig'] = {'old': None, 'new': kwargs['new']}
            if not any((x in ret.get(net_name, {}) for x in autoip_keys)):
                ret.setdefault(net_name, {})['IPConfiguration'] = {'old': 'automatic', 'new': 'static' if nets2_static else 'not connected'}
        elif nets2_autoip and (nets1_static or nets1_missing):
            for autoip_key in autoip_keys:
                autoip_val = nets2[net_name].get(autoip_key)
                if autoip_val:
                    ret.setdefault(net_name, {})[autoip_key] = {'old': None, 'new': autoip_val}
            if not any((x in ret.get(net_name, {}) for x in autoip_keys)):
                ret.setdefault(net_name, {})['IPConfiguration'] = {'old': 'static' if nets1_static else 'not connected', 'new': 'automatic'}
            if nets1_static:
                ret.setdefault(net_name, {})['IPAMConfig'] = {'old': kwargs['old'], 'new': None}
        else:
            old_val = kwargs.get('old')
            new_val = kwargs.get('new')
            if old_val != new_val:
                ret.setdefault(net_name, {})['IPAMConfig'] = {'old': old_val, 'new': new_val}
    for net_name in (x for x in nets1 if x not in nets2):
        for key in compare_keys.get('static', []):
            val = nets1[net_name].get(key)
            if key == 'IPAMConfig':
                _check_ipconfig(ret, net_name, old=val)
            if val:
                if key == 'Aliases':
                    try:
                        val.remove(result1['Config']['Hostname'])
                    except (ValueError, AttributeError):
                        pass
                    else:
                        if not val:
                            continue
                ret.setdefault(net_name, {})[key] = {'old': val, 'new': None}
    for net_name in nets2:
        if net_name not in nets1:
            for key in compare_keys.get('static', []):
                val = nets2[net_name].get(key)
                if key == 'IPAMConfig':
                    _check_ipconfig(ret, net_name, new=val)
                    continue
                elif val:
                    if key == 'Aliases':
                        try:
                            val.remove(result2['Config']['Hostname'])
                        except (ValueError, AttributeError):
                            pass
                        else:
                            if not val:
                                continue
                    ret.setdefault(net_name, {})[key] = {'old': None, 'new': val}
        else:
            for key in compare_keys.get('static', []):
                old_val = nets1[net_name][key]
                new_val = nets2[net_name][key]
                for item in (old_val, new_val):
                    try:
                        item.sort()
                    except AttributeError:
                        pass
                if key == 'Aliases':
                    try:
                        old_val.remove(result1['Config']['Hostname'])
                    except (AttributeError, ValueError):
                        pass
                    try:
                        old_val.remove(result1['Id'][:12])
                    except (AttributeError, ValueError):
                        pass
                    if not old_val:
                        old_val = None
                    try:
                        new_val.remove(result2['Config']['Hostname'])
                    except (AttributeError, ValueError):
                        pass
                    try:
                        new_val.remove(result2['Id'][:12])
                    except (AttributeError, ValueError):
                        pass
                    if not new_val:
                        new_val = None
                elif key == 'IPAMConfig':
                    _check_ipconfig(ret, net_name, old=old_val, new=new_val)
                    continue
                if bool(old_val) is bool(new_val) is False:
                    continue
                elif old_val != new_val:
                    ret.setdefault(net_name, {})[key] = {'old': old_val, 'new': new_val}
    return ret

def compare_networks(first, second, ignore='Name,Id,Created,Containers'):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2018.3.0\n\n    Compare two networks and return any differences between the two\n\n    first\n        Name or ID of first container\n\n    second\n        Name or ID of second container\n\n    ignore : Name,Id,Created,Containers\n        A comma-separated list (or Python list) of keys to ignore when\n        comparing.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.compare_network foo bar\n    '
    ignore = __utils__['args.split_input'](ignore or [])
    net1 = inspect_network(first) if not isinstance(first, dict) else first
    net2 = inspect_network(second) if not isinstance(second, dict) else second
    ret = {}
    for item in net1:
        if item in ignore:
            continue
        else:
            ignore.append(item)
        val1 = net1[item]
        val2 = net2.get(item)
        if bool(val1) is bool(val2) is False:
            continue
        elif item == 'IPAM':
            for subkey in val1:
                subval1 = val1[subkey]
                subval2 = val2.get(subkey)
                if bool(subval1) is bool(subval2) is False:
                    continue
                elif subkey == 'Config':
                    kvsort = lambda x: (list(x.keys()), list(x.values()))
                    config1 = sorted(val1['Config'], key=kvsort)
                    config2 = sorted(val2.get('Config', []), key=kvsort)
                    if config1 != config2:
                        ret.setdefault('IPAM', {})['Config'] = {'old': config1, 'new': config2}
                elif subval1 != subval2:
                    ret.setdefault('IPAM', {})[subkey] = {'old': subval1, 'new': subval2}
        elif item == 'Options':
            for subkey in val1:
                subval1 = val1[subkey]
                subval2 = val2.get(subkey)
                if subkey == 'com.docker.network.bridge.name':
                    continue
                elif subval1 != subval2:
                    ret.setdefault('Options', {})[subkey] = {'old': subval1, 'new': subval2}
        elif val1 != val2:
            ret[item] = {'old': val1, 'new': val2}
    for item in (x for x in net2 if x not in ignore):
        val1 = net1.get(item)
        val2 = net2[item]
        if bool(val1) is bool(val2) is False:
            continue
        elif val1 != val2:
            ret[item] = {'old': val1, 'new': val2}
    return ret

def connected(name, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2018.3.0\n\n    Return a list of running containers attached to the specified network\n\n    name\n        Network name\n\n    verbose : False\n        If ``True``, return extended info about each container (IP\n        configuration, etc.)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.connected net_name\n    '
    containers = inspect_network(name).get('Containers', {})
    ret = {}
    for (cid, cinfo) in containers.items():
        try:
            name = cinfo.pop('Name')
        except (KeyError, AttributeError):
            log.warning("'Name' key not present in container definition for container ID '%s' within inspect results for Docker network '%s'. Full container definition: %s", cid, name, cinfo)
            continue
        else:
            cinfo['Id'] = cid
            ret[name] = cinfo
    if not verbose:
        return list(ret)
    return ret

def login(*registries):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2016.3.7,2016.11.4,2017.7.0\n\n    Performs a ``docker login`` to authenticate to one or more configured\n    repositories. See the documentation at the top of this page to configure\n    authentication credentials.\n\n    Multiple registry URLs (matching those configured in Pillar) can be passed,\n    and Salt will attempt to login to *just* those registries. If no registry\n    URLs are provided, Salt will attempt to login to *all* configured\n    registries.\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Results`` - A dictionary mapping registry URLs to the authentication\n      result. ``True`` means a successful login, ``False`` means a failed\n      login.\n    - ``Errors`` - A list of errors encountered during the course of this\n      function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.login\n        salt myminion docker.login hub\n        salt myminion docker.login hub https://mydomain.tld/registry/\n    '
    registry_auth = __salt__['config.get']('docker-registries', {})
    ret = {'retcode': 0}
    errors = ret.setdefault('Errors', [])
    if not isinstance(registry_auth, dict):
        errors.append("'docker-registries' Pillar value must be a dictionary")
        registry_auth = {}
    for (reg_name, reg_conf) in __salt__['config.option']('*-docker-registries', wildcard=True).items():
        try:
            registry_auth.update(reg_conf)
        except TypeError:
            errors.append("Docker registry '{}' was not specified as a dictionary".format(reg_name))
    if not registries:
        registries = list(registry_auth)
    results = ret.setdefault('Results', {})
    for registry in registries:
        if registry not in registry_auth:
            errors.append(f"No match found for registry '{registry}'")
            continue
        try:
            username = registry_auth[registry]['username']
            password = registry_auth[registry]['password']
        except TypeError:
            errors.append(f"Invalid configuration for registry '{registry}'")
        except KeyError as exc:
            errors.append(f"Missing {exc} for registry '{registry}'")
        else:
            cmd = ['docker', 'login', '-u', username, '-p', password]
            if registry.lower() != 'hub':
                cmd.append(registry)
            log.debug("Attempting to login to docker registry '%s' as user '%s'", registry, username)
            login_cmd = __salt__['cmd.run_all'](cmd, python_shell=False, output_loglevel='quiet')
            results[registry] = login_cmd['retcode'] == 0
            if not results[registry]:
                if login_cmd['stderr']:
                    errors.append(login_cmd['stderr'])
                elif login_cmd['stdout']:
                    errors.append(login_cmd['stdout'])
    if errors:
        ret['retcode'] = 1
    return ret

def logout(*registries):
    if False:
        return 10
    '\n    .. versionadded:: 3001\n\n    Performs a ``docker logout`` to remove the saved authentication details for\n    one or more configured repositories.\n\n    Multiple registry URLs (matching those configured in Pillar) can be passed,\n    and Salt will attempt to logout of *just* those registries. If no registry\n    URLs are provided, Salt will attempt to logout of *all* configured\n    registries.\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Results`` - A dictionary mapping registry URLs to the authentication\n      result. ``True`` means a successful logout, ``False`` means a failed\n      logout.\n    - ``Errors`` - A list of errors encountered during the course of this\n      function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.logout\n        salt myminion docker.logout hub\n        salt myminion docker.logout hub https://mydomain.tld/registry/\n    '
    registry_auth = __salt__['config.get']('docker-registries', {})
    ret = {'retcode': 0}
    errors = ret.setdefault('Errors', [])
    if not isinstance(registry_auth, dict):
        errors.append("'docker-registries' Pillar value must be a dictionary")
        registry_auth = {}
    for (reg_name, reg_conf) in __salt__['config.option']('*-docker-registries', wildcard=True).items():
        try:
            registry_auth.update(reg_conf)
        except TypeError:
            errors.append("Docker registry '{}' was not specified as a dictionary".format(reg_name))
    if not registries:
        registries = list(registry_auth)
    results = ret.setdefault('Results', {})
    for registry in registries:
        if registry not in registry_auth:
            errors.append(f"No match found for registry '{registry}'")
            continue
        else:
            cmd = ['docker', 'logout']
            if registry.lower() != 'hub':
                cmd.append(registry)
            log.debug("Attempting to logout of docker registry '%s'", registry)
            logout_cmd = __salt__['cmd.run_all'](cmd, python_shell=False, output_loglevel='quiet')
            results[registry] = logout_cmd['retcode'] == 0
            if not results[registry]:
                if logout_cmd['stderr']:
                    errors.append(logout_cmd['stderr'])
                elif logout_cmd['stdout']:
                    errors.append(logout_cmd['stdout'])
    if errors:
        ret['retcode'] = 1
    return ret

def depends(name):
    if False:
        return 10
    '\n    Returns the containers and images, if any, which depend on the given image\n\n    name\n        Name or ID of image\n\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Containers`` - A list of containers which depend on the specified image\n    - ``Images`` - A list of IDs of images which depend on the specified image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.depends myimage\n        salt myminion docker.depends 0123456789ab\n    '
    image_id = inspect_image(name)['Id']
    container_depends = []
    for container in ps_(all=True, verbose=True).values():
        if container['Info']['Image'] == image_id:
            container_depends.extend([x.lstrip('/') for x in container['Names']])
    return {'Containers': container_depends, 'Images': [x[:12] for (x, y) in images(all=True).items() if y['ParentId'] == image_id]}

def diff(name):
    if False:
        i = 10
        return i + 15
    "\n    Get information on changes made to container's filesystem since it was\n    created. Equivalent to running the ``docker diff`` Docker CLI command.\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary containing any of the following keys:\n\n    - ``Added`` - A list of paths that were added.\n    - ``Changed`` - A list of paths that were changed.\n    - ``Deleted`` - A list of paths that were deleted.\n\n    These keys will only be present if there were changes, so if the container\n    has no differences the return dict will be empty.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.diff mycontainer\n    "
    changes = _client_wrapper('diff', name)
    kind_map = {0: 'Changed', 1: 'Added', 2: 'Deleted'}
    ret = {}
    for change in changes:
        key = kind_map.get(change['Kind'], 'Unknown')
        ret.setdefault(key, []).append(change['Path'])
    if 'Unknown' in ret:
        log.error('Unknown changes detected in docker.diff of container %s. This is probably due to a change in the Docker API. Please report this to the SaltStack developers', name)
    return ret

def exists(name):
    if False:
        while True:
            i = 10
    '\n    Check if a given container exists\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A boolean (``True`` if the container exists, otherwise ``False``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.exists mycontainer\n    '
    contextkey = f'docker.exists.{name}'
    if contextkey in __context__:
        return __context__[contextkey]
    try:
        c_info = _client_wrapper('inspect_container', name, catch_api_errors=False)
    except docker.errors.APIError:
        __context__[contextkey] = False
    else:
        __context__[contextkey] = True
    return __context__[contextkey]

def history(name, quiet=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the history for an image. Equivalent to running the ``docker\n    history`` Docker CLI command.\n\n    name\n        Container name or ID\n\n    quiet : False\n        If ``True``, the return data will simply be a list of the commands run\n        to build the container.\n\n        .. code-block:: bash\n\n            $ salt myminion docker.history nginx:latest quiet=True\n            myminion:\n                - FROM scratch\n                - ADD file:ef063ed0ae9579362871b9f23d2bc0781ef7cd4de6ac822052cf6c9c5a12b1e2 in /\n                - CMD [/bin/bash]\n                - MAINTAINER NGINX Docker Maintainers "docker-maint@nginx.com"\n                - apt-key adv --keyserver pgp.mit.edu --recv-keys 573BFD6B3D8FBC641079A6ABABF5BD827BD9BF62\n                - echo "deb http://nginx.org/packages/mainline/debian/ wheezy nginx" >> /etc/apt/sources.list\n                - ENV NGINX_VERSION=1.7.10-1~wheezy\n                - apt-get update &&     apt-get install -y ca-certificates nginx=${NGINX_VERSION} &&     rm -rf /var/lib/apt/lists/*\n                - ln -sf /dev/stdout /var/log/nginx/access.log\n                - ln -sf /dev/stderr /var/log/nginx/error.log\n                - VOLUME [/var/cache/nginx]\n                - EXPOSE map[80/tcp:{} 443/tcp:{}]\n                - CMD [nginx -g daemon off;]\n                        https://github.com/saltstack/salt/pull/22421\n\n\n    **RETURN DATA**\n\n    If ``quiet=False``, the return value will be a list of dictionaries\n    containing information about each step taken to build the image. The keys\n    in each step include the following:\n\n    - ``Command`` - The command executed in this build step\n    - ``Id`` - Layer ID\n    - ``Size`` - Cumulative image size, in bytes\n    - ``Size_Human`` - Cumulative image size, in human-readable units\n    - ``Tags`` - Tag(s) assigned to this layer\n    - ``Time_Created_Epoch`` - Time this build step was completed (Epoch\n      time)\n    - ``Time_Created_Local`` - Time this build step was completed (Minion\'s\n      local timezone)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.exists mycontainer\n    '
    response = _client_wrapper('history', name)
    key_map = {'CreatedBy': 'Command', 'Created': 'Time_Created_Epoch'}
    command_prefix = re.compile('^/bin/sh -c (?:#\\(nop\\) )?')
    ret = []
    for item in reversed(response):
        step = {}
        for (key, val) in item.items():
            step_key = key_map.get(key, key)
            if step_key == 'Command':
                if not val:
                    val = 'FROM scratch'
                else:
                    val = command_prefix.sub('', val)
            step[step_key] = val
        if 'Time_Created_Epoch' in step:
            step['Time_Created_Local'] = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(step['Time_Created_Epoch']))
        for param in ('Size',):
            if param in step:
                step[f'{param}_Human'] = _size_fmt(step[param])
        ret.append(copy.deepcopy(step))
    if quiet:
        return [x.get('Command') for x in ret]
    return ret

def images(verbose=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns information about the Docker images on the Minion. Equivalent to\n    running the ``docker images`` Docker CLI command.\n\n    all : False\n        If ``True``, untagged images will also be returned\n\n    verbose : False\n        If ``True``, a ``docker inspect`` will be run on each image returned.\n\n\n    **RETURN DATA**\n\n    A dictionary with each key being an image ID, and each value some general\n    info about that image (time created, size, tags associated with the image,\n    etc.)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.images\n        salt myminion docker.images all=True\n    '
    if 'docker.images' not in __context__:
        response = _client_wrapper('images', all=kwargs.get('all', False))
        key_map = {'Created': 'Time_Created_Epoch'}
        for img in response:
            img_id = img.pop('Id', None)
            if img_id is None:
                continue
            for item in img:
                img_state = 'untagged' if img['RepoTags'] in (['<none>:<none>'], None) else 'tagged'
                bucket = __context__.setdefault('docker.images', {})
                bucket = bucket.setdefault(img_state, {})
                img_key = key_map.get(item, item)
                bucket.setdefault(img_id, {})[img_key] = img[item]
            if 'Time_Created_Epoch' in bucket.get(img_id, {}):
                bucket[img_id]['Time_Created_Local'] = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(bucket[img_id]['Time_Created_Epoch']))
            for param in ('Size', 'VirtualSize'):
                if param in bucket.get(img_id, {}):
                    bucket[img_id][f'{param}_Human'] = _size_fmt(bucket[img_id][param])
    context_data = __context__.get('docker.images', {})
    ret = copy.deepcopy(context_data.get('tagged', {}))
    if kwargs.get('all', False):
        ret.update(copy.deepcopy(context_data.get('untagged', {})))
    if verbose:
        for img_id in ret:
            ret[img_id]['Info'] = inspect_image(img_id)
    return ret

def info():
    if False:
        while True:
            i = 10
    '\n    Returns a dictionary of system-wide information. Equivalent to running\n    the ``docker info`` Docker CLI command.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.info\n    '
    return _client_wrapper('info')

def inspect(name):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2017.7.0\n        Volumes and networks are now checked, in addition to containers and\n        images.\n\n    This is a generic container/image/volume/network inspecton function. It\n    will run the following functions in order:\n\n    - :py:func:`docker.inspect_container\n      <salt.modules.dockermod.inspect_container>`\n    - :py:func:`docker.inspect_image <salt.modules.dockermod.inspect_image>`\n    - :py:func:`docker.inspect_volume <salt.modules.dockermod.inspect_volume>`\n    - :py:func:`docker.inspect_network <salt.modules.dockermod.inspect_network>`\n\n    The first of these to find a match will be returned.\n\n    name\n        Container/image/volume/network name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary of container/image/volume/network information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.inspect mycontainer\n        salt myminion docker.inspect busybox\n    '
    try:
        return inspect_container(name)
    except CommandExecutionError as exc:
        if 'does not exist' not in exc.strerror:
            raise
    try:
        return inspect_image(name)
    except CommandExecutionError as exc:
        if not exc.strerror.startswith('Error 404'):
            raise
    try:
        return inspect_volume(name)
    except CommandExecutionError as exc:
        if not exc.strerror.startswith('Error 404'):
            raise
    try:
        return inspect_network(name)
    except CommandExecutionError as exc:
        if not exc.strerror.startswith('Error 404'):
            raise
    raise CommandExecutionError(f'Error 404: No such image/container/volume/network: {name}')

def inspect_container(name):
    if False:
        i = 10
        return i + 15
    '\n    Retrieves container information. Equivalent to running the ``docker\n    inspect`` Docker CLI command, but will only look for container information.\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary of container information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.inspect_container mycontainer\n        salt myminion docker.inspect_container 0123456789ab\n    '
    return _client_wrapper('inspect_container', name)

def inspect_image(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieves image information. Equivalent to running the ``docker inspect``\n    Docker CLI command, but will only look for image information.\n\n    .. note::\n        To inspect an image, it must have been pulled from a registry or built\n        locally. Images on a Docker registry which have not been pulled cannot\n        be inspected.\n\n    name\n        Image name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary of image information\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.inspect_image busybox\n        salt myminion docker.inspect_image centos:6\n        salt myminion docker.inspect_image 0123456789ab\n    '
    ret = _client_wrapper('inspect_image', name)
    for param in ('Size', 'VirtualSize'):
        if param in ret:
            ret[f'{param}_Human'] = _size_fmt(ret[param])
    return ret

def list_containers(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Returns a list of containers by name. This is different from\n    :py:func:`docker.ps <salt.modules.dockermod.ps_>` in that\n    :py:func:`docker.ps <salt.modules.dockermod.ps_>` returns its results\n    organized by container ID.\n\n    all : False\n        If ``True``, stopped containers will be included in return data\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.list_containers\n    '
    ret = set()
    for item in ps_(all=kwargs.get('all', False)).values():
        names = item.get('Names')
        if not names:
            continue
        for c_name in [x.lstrip('/') for x in names or []]:
            ret.add(c_name)
    return sorted(ret)

def list_tags():
    if False:
        print('Hello World!')
    '\n    Returns a list of tagged images\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.list_tags\n    '
    ret = set()
    for item in images().values():
        if not item.get('RepoTags'):
            continue
        ret.update(set(item['RepoTags']))
    return sorted(ret)

def resolve_image_id(name):
    if False:
        return 10
    '\n    .. versionadded:: 2018.3.0\n\n    Given an image name (or partial image ID), return the full image ID. If no\n    match is found among the locally-pulled images, then ``False`` will be\n    returned.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.resolve_image_id foo\n        salt myminion docker.resolve_image_id foo:bar\n        salt myminion docker.resolve_image_id 36540f359ca3\n    '
    try:
        inspect_result = inspect_image(name)
        return inspect_result['Id']
    except CommandExecutionError:
        pass
    except KeyError:
        log.error("Inspecting docker image '%s' returned an unexpected data structure: %s", name, inspect_result)
    return False

def resolve_tag(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2017.7.2\n    .. versionchanged:: 2018.3.0\n        Instead of matching against pulled tags using\n        :py:func:`docker.list_tags <salt.modules.dockermod.list_tags>`, this\n        function now simply inspects the passed image name using\n        :py:func:`docker.inspect_image <salt.modules.dockermod.inspect_image>`\n        and returns the first matching tag. If no matching tags are found, it\n        is assumed that the passed image is an untagged image ID, and the full\n        ID is returned.\n\n    Inspects the specified image name and returns the first matching tag in the\n    inspect results. If the specified image is not pulled locally, this\n    function will return ``False``.\n\n    name\n        Image name to resolve. If the image is found but there are no tags,\n        this means that the image name passed was an untagged image. In this\n        case the image ID will be returned.\n\n    all : False\n        If ``True``, a list of all matching tags will be returned. If the image\n        is found but there are no tags, then a list will still be returned, but\n        it will simply contain the image ID.\n\n        .. versionadded:: 2018.3.0\n\n    tags\n        .. deprecated:: 2018.3.0\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.resolve_tag busybox\n        salt myminion docker.resolve_tag centos:7 all=True\n        salt myminion docker.resolve_tag c9f378ac27d9\n    '
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    all_ = kwargs.pop('all', False)
    if kwargs:
        __utils__['args.invalid_kwargs'](kwargs)
    try:
        inspect_result = inspect_image(name)
        tags = inspect_result['RepoTags']
        if all_:
            if tags:
                return tags
        else:
            return tags[0]
    except CommandExecutionError:
        return False
    except KeyError:
        log.error("Inspecting docker image '%s' returned an unexpected data structure: %s", name, inspect_result)
    except IndexError:
        pass
    return [inspect_result['Id']] if all_ else inspect_result['Id']

def logs(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2018.3.0\n        Support for all of docker-py\'s `logs()`_ function\'s arguments, with the\n        exception of ``stream``.\n\n    Returns the logs for the container. An interface to docker-py\'s `logs()`_\n    function.\n\n    name\n        Container name or ID\n\n    stdout : True\n        Return stdout lines\n\n    stderr : True\n        Return stdout lines\n\n    timestamps : False\n        Show timestamps\n\n    tail : all\n        Output specified number of lines at the end of logs. Either an integer\n        number of lines or the string ``all``.\n\n    since\n        Show logs since the specified time, passed as a UNIX epoch timestamp.\n        Optionally, if timelib_ is installed on the minion the timestamp can be\n        passed as a string which will be resolved to a date using\n        ``timelib.strtodatetime()``.\n\n    follow : False\n        If ``True``, this function will block until the container exits and\n        return the logs when it does. The default behavior is to return what is\n        in the log at the time this function is executed.\n\n        .. note:\n            Since it blocks, this option should be used with caution.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # All logs\n        salt myminion docker.logs mycontainer\n        # Last 100 lines of log\n        salt myminion docker.logs mycontainer tail=100\n        # Just stderr\n        salt myminion docker.logs mycontainer stdout=False\n        # Logs since a specific UNIX timestamp\n        salt myminion docker.logs mycontainer since=1511688459\n        # Flexible format for "since" argument (requires timelib)\n        salt myminion docker.logs mycontainer since=\'1 hour ago\'\n        salt myminion docker.logs mycontainer since=\'1 week ago\'\n        salt myminion docker.logs mycontainer since=\'1 fortnight ago\'\n    '
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    if 'stream' in kwargs:
        raise SaltInvocationError("The 'stream' argument is not supported")
    try:
        kwargs['since'] = int(kwargs['since'])
    except KeyError:
        pass
    except (ValueError, TypeError):
        if HAS_TIMELIB:
            try:
                kwargs['since'] = timelib.strtodatetime(kwargs['since'])
            except Exception as exc:
                log.warning("docker.logs: Failed to parse '%s' using timelib: %s", kwargs['since'], exc)
    return salt.utils.stringutils.to_unicode(_client_wrapper('logs', name, **kwargs))

def pid(name):
    if False:
        while True:
            i = 10
    '\n    Returns the PID of a container\n\n    name\n        Container name or ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.pid mycontainer\n        salt myminion docker.pid 0123456789ab\n    '
    return inspect_container(name)['State']['Pid']

def port(name, private_port=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns port mapping information for a given container. Equivalent to\n    running the ``docker port`` Docker CLI command.\n\n    name\n        Container name or ID\n\n        .. versionchanged:: 2019.2.0\n            This value can now be a pattern expression (using the\n            pattern-matching characters defined in fnmatch_). If a pattern\n            expression is used, this function will return a dictionary mapping\n            container names which match the pattern to the mappings for those\n            containers. When no pattern expression is used, a dictionary of the\n            mappings for the specified container name will be returned.\n\n        .. _fnmatch: https://docs.python.org/2/library/fnmatch.html\n\n    private_port : None\n        If specified, get information for that specific port. Can be specified\n        either as a port number (i.e. ``5000``), or as a port number plus the\n        protocol (i.e. ``5000/udp``).\n\n        If this argument is omitted, all port mappings will be returned.\n\n\n    **RETURN DATA**\n\n    A dictionary of port mappings, with the keys being the port and the values\n    being the mapping(s) for that port.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.port mycontainer\n        salt myminion docker.port mycontainer 5000\n        salt myminion docker.port mycontainer 5000/udp\n    '
    pattern_used = bool(re.search('[*?\\[]', name))
    names = fnmatch.filter(list_containers(all=True), name) if pattern_used else [name]
    if private_port is None:
        pattern = '*'
    elif isinstance(private_port, int):
        pattern = f'{private_port}/*'
    else:
        err = "Invalid private_port '{}'. Must either be a port number, or be in port/protocol notation (e.g. 5000/tcp)".format(private_port)
        try:
            (port_num, _, protocol) = private_port.partition('/')
            protocol = protocol.lower()
            if not port_num.isdigit() or protocol not in ('tcp', 'udp'):
                raise SaltInvocationError(err)
            pattern = port_num + '/' + protocol
        except AttributeError:
            raise SaltInvocationError(err)
    ret = {}
    for c_name in names:
        mappings = inspect_container(c_name).get('NetworkSettings', {}).get('Ports', {})
        ret[c_name] = {x: mappings[x] for x in fnmatch.filter(mappings, pattern)}
    return ret.get(name, {}) if not pattern_used else ret

def ps_(filters=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Returns information about the Docker containers on the Minion. Equivalent\n    to running the ``docker ps`` Docker CLI command.\n\n    all : False\n        If ``True``, stopped containers will also be returned\n\n    host: False\n        If ``True``, local host\'s network topology will be included\n\n    verbose : False\n        If ``True``, a ``docker inspect`` will be run on each container\n        returned.\n\n    filters: None\n        A dictionary of filters to be processed on the container list.\n        Available filters:\n\n          - exited (int): Only containers with specified exit code\n          - status (str): One of restarting, running, paused, exited\n          - label (str): format either "key" or "key=value"\n\n    **RETURN DATA**\n\n    A dictionary with each key being an container ID, and each value some\n    general info about that container (time created, name, command, etc.)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.ps\n        salt myminion docker.ps all=True\n        salt myminion docker.ps filters="{\'label\': \'role=web\'}"\n    '
    response = _client_wrapper('containers', all=True, filters=filters)
    key_map = {'Created': 'Time_Created_Epoch'}
    context_data = {}
    for container in response:
        c_id = container.pop('Id', None)
        if c_id is None:
            continue
        for item in container:
            c_state = 'running' if container.get('Status', '').lower().startswith('up ') else 'stopped'
            bucket = context_data.setdefault(c_state, {})
            c_key = key_map.get(item, item)
            bucket.setdefault(c_id, {})[c_key] = container[item]
        if 'Time_Created_Epoch' in bucket.get(c_id, {}):
            bucket[c_id]['Time_Created_Local'] = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(bucket[c_id]['Time_Created_Epoch']))
    ret = copy.deepcopy(context_data.get('running', {}))
    if kwargs.get('all', False):
        ret.update(copy.deepcopy(context_data.get('stopped', {})))
    if kwargs.get('verbose', False):
        for c_id in ret:
            ret[c_id]['Info'] = inspect_container(c_id)
    if kwargs.get('host', False):
        ret.setdefault('host', {}).setdefault('interfaces', {}).update(__salt__['network.interfaces']())
    return ret

def state(name):
    if False:
        while True:
            i = 10
    '\n    Returns the state of the container\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A string representing the current state of the container (either\n    ``running``, ``paused``, or ``stopped``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.state mycontainer\n    '
    contextkey = f'docker.state.{name}'
    if contextkey in __context__:
        return __context__[contextkey]
    __context__[contextkey] = _get_state(inspect_container(name))
    return __context__[contextkey]

def search(name, official=False, trusted=False):
    if False:
        return 10
    '\n    Searches the registry for an image\n\n    name\n        Search keyword\n\n    official : False\n        Limit results to official builds\n\n    trusted : False\n        Limit results to `trusted builds`_\n\n    **RETURN DATA**\n\n    A dictionary with each key being the name of an image, and the following\n    information for each image:\n\n    - ``Description`` - Image description\n    - ``Official`` - A boolean (``True`` if an official build, ``False`` if\n      not)\n    - ``Stars`` - Number of stars the image has on the registry\n    - ``Trusted`` - A boolean (``True`` if a trusted build, ``False`` if not)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.search centos\n        salt myminion docker.search centos official=True\n    '
    response = _client_wrapper('search', name)
    if not response:
        raise CommandExecutionError(f"No images matched the search string '{name}'")
    key_map = {'description': 'Description', 'is_official': 'Official', 'is_trusted': 'Trusted', 'star_count': 'Stars'}
    limit = []
    if official:
        limit.append('Official')
    if trusted:
        limit.append('Trusted')
    results = {}
    for item in response:
        c_name = item.pop('name', None)
        if c_name is not None:
            for key in item:
                mapped_key = key_map.get(key, key)
                results.setdefault(c_name, {})[mapped_key] = item[key]
    if not limit:
        return results
    ret = {}
    for (key, val) in results.items():
        for item in limit:
            if val.get(item, False):
                ret[key] = val
                break
    return ret

def top(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Runs the `docker top` command on a specific container\n\n    name\n        Container name or ID\n\n    CLI Example:\n\n    **RETURN DATA**\n\n    A list of dictionaries containing information about each process\n\n\n    .. code-block:: bash\n\n        salt myminion docker.top mycontainer\n        salt myminion docker.top 0123456789ab\n    '
    response = _client_wrapper('top', name)
    columns = {}
    for (idx, col_name) in enumerate(response['Titles']):
        columns[idx] = col_name
    ret = []
    for process in response['Processes']:
        cur_proc = {}
        for (idx, val) in enumerate(process):
            cur_proc[columns[idx]] = val
        ret.append(cur_proc)
    return ret

def version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a dictionary of Docker version information. Equivalent to running\n    the ``docker version`` Docker CLI command.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.version\n    '
    ret = _client_wrapper('version')
    version_re = re.compile(VERSION_RE)
    if 'Version' in ret:
        match = version_re.match(str(ret['Version']))
        if match:
            ret['VersionInfo'] = tuple((int(x) for x in match.group(1).split('.')))
    if 'ApiVersion' in ret:
        match = version_re.match(str(ret['ApiVersion']))
        if match:
            ret['ApiVersionInfo'] = tuple((int(x) for x in match.group(1).split('.')))
    return ret

def _create_networking_config(networks):
    if False:
        return 10
    log.debug('creating networking config from %s', networks)
    return _client_wrapper('create_networking_config', {k: _client_wrapper('create_endpoint_config', **v) for (k, v) in networks.items()})

@_refresh_mine_cache
def create(image, name=None, start=False, skip_translate=None, ignore_collisions=False, validate_ip_addrs=True, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a new container\n\n    image\n        Image from which to create the container\n\n    name\n        Name for the new container. If not provided, Docker will randomly\n        generate one for you (it will be included in the return data).\n\n    start : False\n        If ``True``, start container after creating it\n\n        .. versionadded:: 2018.3.0\n\n    skip_translate\n        This function translates Salt CLI or SLS input into the format which\n        docker-py expects. However, in the event that Salt\'s translation logic\n        fails (due to potential changes in the Docker Remote API, or to bugs in\n        the translation code), this argument can be used to exert granular\n        control over which arguments are translated and which are not.\n\n        Pass this argument as a comma-separated list (or Python list) of\n        arguments, and translation for each passed argument name will be\n        skipped. Alternatively, pass ``True`` and *all* translation will be\n        skipped.\n\n        Skipping tranlsation allows for arguments to be formatted directly in\n        the format which docker-py expects. This allows for API changes and\n        other issues to be more easily worked around. An example of using this\n        option to skip translation would be:\n\n        .. code-block:: bash\n\n            salt myminion docker.create image=centos:7.3.1611 skip_translate=environment environment="{\'FOO\': \'bar\'}"\n\n        See the following links for more information:\n\n        - `docker-py Low-level API`_\n        - `Docker Engine API`_\n\n    ignore_collisions : False\n        Since many of docker-py\'s arguments differ in name from their CLI\n        counterparts (with which most Docker users are more familiar), Salt\n        detects usage of these and aliases them to the docker-py version of\n        that argument. However, if both the alias and the docker-py version of\n        the same argument (e.g. ``env`` and ``environment``) are used, an error\n        will be raised. Set this argument to ``True`` to suppress these errors\n        and keep the docker-py version of the argument.\n\n    validate_ip_addrs : True\n        For parameters which accept IP addresses as input, IP address\n        validation will be performed. To disable, set this to ``False``\n\n    client_timeout : 60\n        Timeout in seconds for the Docker client. This is not a timeout for\n        this function, but for receiving a response from the API.\n\n        .. note::\n\n            This is only used if Salt needs to pull the requested image.\n\n    **CONTAINER CONFIGURATION ARGUMENTS**\n\n    auto_remove (or *rm*) : False\n        Enable auto-removal of the container on daemon side when the\n        container’s process exits (analogous to running a docker container with\n        ``--rm`` on the CLI).\n\n        Examples:\n\n        - ``auto_remove=True``\n        - ``rm=True``\n\n    binds\n        Files/directories to bind mount. Each bind mount should be passed in\n        one of the following formats:\n\n        - ``<host_path>:<container_path>`` - ``host_path`` is mounted within\n          the container as ``container_path`` with read-write access.\n        - ``<host_path>:<container_path>:<selinux_context>`` - ``host_path`` is\n          mounted within the container as ``container_path`` with read-write\n          access. Additionally, the specified selinux context will be set\n          within the container.\n        - ``<host_path>:<container_path>:<read_only>`` - ``host_path`` is\n          mounted within the container as ``container_path``, with the\n          read-only or read-write setting explicitly defined.\n        - ``<host_path>:<container_path>:<read_only>,<selinux_context>`` -\n          ``host_path`` is mounted within the container as ``container_path``,\n          with the read-only or read-write setting explicitly defined.\n          Additionally, the specified selinux context will be set within the\n          container.\n\n        ``<read_only>`` can be either ``ro`` for read-write access, or ``ro``\n        for read-only access. When omitted, it is assumed to be read-write.\n\n        ``<selinux_context>`` can be ``z`` if the volume is shared between\n        multiple containers, or ``Z`` if the volume should be private.\n\n        .. note::\n            When both ``<read_only>`` and ``<selinux_context>`` are specified,\n            there must be a comma before ``<selinux_context>``.\n\n        Binds can be expressed as a comma-separated list or a Python list,\n        however in cases where both ro/rw and an selinux context are specified,\n        the binds *must* be specified as a Python list.\n\n        Examples:\n\n        - ``binds=/srv/www:/var/www:ro``\n        - ``binds=/srv/www:/var/www:rw``\n        - ``binds=/srv/www:/var/www``\n        - ``binds="[\'/srv/www:/var/www:ro,Z\']"``\n        - ``binds="[\'/srv/www:/var/www:rw,Z\']"``\n        - ``binds=/srv/www:/var/www:Z``\n\n        .. note::\n            The second and third examples above are equivalent to each other,\n            as are the last two examples.\n\n    blkio_weight\n        Block IO weight (relative weight), accepts a weight value between 10\n        and 1000.\n\n        Example: ``blkio_weight=100``\n\n    blkio_weight_device\n        Block IO weight (relative device weight), specified as a list of\n        expressions in the format ``PATH:WEIGHT``\n\n        Example: ``blkio_weight_device=/dev/sda:100``\n\n    cap_add\n        List of capabilities to add within the container. Can be passed as a\n        comma-separated list or a Python list. Requires Docker 1.2.0 or\n        newer.\n\n        Examples:\n\n        - ``cap_add=SYS_ADMIN,MKNOD``\n        - ``cap_add="[SYS_ADMIN, MKNOD]"``\n\n    cap_drop\n        List of capabilities to drop within the container. Can be passed as a\n        comma-separated string or a Python list. Requires Docker 1.2.0 or\n        newer.\n\n        Examples:\n\n        - ``cap_drop=SYS_ADMIN,MKNOD``,\n        - ``cap_drop="[SYS_ADMIN, MKNOD]"``\n\n    command (or *cmd*)\n        Command to run in the container\n\n        Example: ``command=bash`` or ``cmd=bash``\n\n        .. versionchanged:: 2015.8.1\n            ``cmd`` is now also accepted\n\n    cpuset_cpus (or *cpuset*)\n        CPUs on which which to allow execution, specified as a string\n        containing a range (e.g. ``0-3``) or a comma-separated list of CPUs\n        (e.g. ``0,1``).\n\n        Examples:\n\n        - ``cpuset_cpus="0-3"``\n        - ``cpuset="0,1"``\n\n    cpuset_mems\n        Memory nodes on which which to allow execution, specified as a string\n        containing a range (e.g. ``0-3``) or a comma-separated list of MEMs\n        (e.g. ``0,1``). Only effective on NUMA systems.\n\n        Examples:\n\n        - ``cpuset_mems="0-3"``\n        - ``cpuset_mems="0,1"``\n\n    cpu_group\n        The length of a CPU period in microseconds\n\n        Example: ``cpu_group=100000``\n\n    cpu_period\n        Microseconds of CPU time that the container can get in a CPU period\n\n        Example: ``cpu_period=50000``\n\n    cpu_shares\n        CPU shares (relative weight), specified as an integer between 2 and 1024.\n\n        Example: ``cpu_shares=512``\n\n    detach : False\n        If ``True``, run the container\'s command in the background (daemon\n        mode)\n\n        Example: ``detach=True``\n\n    devices\n        List of host devices to expose within the container\n\n        Examples:\n\n        - ``devices="/dev/net/tun,/dev/xvda1:/dev/xvda1,/dev/xvdb1:/dev/xvdb1:r"``\n        - ``devices="[\'/dev/net/tun\', \'/dev/xvda1:/dev/xvda1\', \'/dev/xvdb1:/dev/xvdb1:r\']"``\n\n    device_read_bps\n        Limit read rate (bytes per second) from a device, specified as a list\n        of expressions in the format ``PATH:RATE``, where ``RATE`` is either an\n        integer number of bytes, or a string ending in ``kb``, ``mb``, or\n        ``gb``.\n\n        Examples:\n\n        - ``device_read_bps="/dev/sda:1mb,/dev/sdb:5mb"``\n        - ``device_read_bps="[\'/dev/sda:100mb\', \'/dev/sdb:5mb\']"``\n\n    device_read_iops\n        Limit read rate (I/O per second) from a device, specified as a list\n        of expressions in the format ``PATH:RATE``, where ``RATE`` is a number\n        of I/O operations.\n\n        Examples:\n\n        - ``device_read_iops="/dev/sda:1000,/dev/sdb:500"``\n        - ``device_read_iops="[\'/dev/sda:1000\', \'/dev/sdb:500\']"``\n\n    device_write_bps\n        Limit write rate (bytes per second) from a device, specified as a list\n        of expressions in the format ``PATH:RATE``, where ``RATE`` is either an\n        integer number of bytes, or a string ending in ``kb``, ``mb`` or\n        ``gb``.\n\n\n        Examples:\n\n        - ``device_write_bps="/dev/sda:100mb,/dev/sdb:50mb"``\n        - ``device_write_bps="[\'/dev/sda:100mb\', \'/dev/sdb:50mb\']"``\n\n    device_write_iops\n        Limit write rate (I/O per second) from a device, specified as a list\n        of expressions in the format ``PATH:RATE``, where ``RATE`` is a number\n        of I/O operations.\n\n        Examples:\n\n        - ``device_write_iops="/dev/sda:1000,/dev/sdb:500"``\n        - ``device_write_iops="[\'/dev/sda:1000\', \'/dev/sdb:500\']"``\n\n    dns\n        List of DNS nameservers. Can be passed as a comma-separated list or a\n        Python list.\n\n        Examples:\n\n        - ``dns=8.8.8.8,8.8.4.4``\n        - ``dns="[\'8.8.8.8\', \'8.8.4.4\']"``\n\n        .. note::\n\n            To skip IP address validation, use ``validate_ip_addrs=False``\n\n    dns_opt\n        Additional options to be added to the container’s ``resolv.conf`` file\n\n        Example: ``dns_opt=ndots:9``\n\n    dns_search\n        List of DNS search domains. Can be passed as a comma-separated list\n        or a Python list.\n\n        Examples:\n\n        - ``dns_search=foo1.domain.tld,foo2.domain.tld``\n        - ``dns_search="[foo1.domain.tld, foo2.domain.tld]"``\n\n    domainname\n        The domain name to use for the container\n\n        Example: ``domainname=domain.tld``\n\n    entrypoint\n        Entrypoint for the container. Either a string (e.g. ``"mycmd --arg1\n        --arg2"``) or a Python list (e.g.  ``"[\'mycmd\', \'--arg1\', \'--arg2\']"``)\n\n        Examples:\n\n        - ``entrypoint="cat access.log"``\n        - ``entrypoint="[\'cat\', \'access.log\']"``\n\n    environment (or *env*)\n        Either a dictionary of environment variable names and their values, or\n        a Python list of strings in the format ``VARNAME=value``.\n\n        Examples:\n\n        - ``environment=\'VAR1=value,VAR2=value\'``\n        - ``environment="[\'VAR1=value\', \'VAR2=value\']"``\n        - ``environment="{\'VAR1\': \'value\', \'VAR2\': \'value\'}"``\n\n    extra_hosts\n        Additional hosts to add to the container\'s /etc/hosts file. Can be\n        passed as a comma-separated list or a Python list. Requires Docker\n        1.3.0 or newer.\n\n        Examples:\n\n        - ``extra_hosts=web1:10.9.8.7,web2:10.9.8.8``\n        - ``extra_hosts="[\'web1:10.9.8.7\', \'web2:10.9.8.8\']"``\n        - ``extra_hosts="{\'web1\': \'10.9.8.7\', \'web2\': \'10.9.8.8\'}"``\n\n        .. note::\n\n            To skip IP address validation, use ``validate_ip_addrs=False``\n\n    group_add\n        List of additional group names and/or IDs that the container process\n        will run as\n\n        Examples:\n\n        - ``group_add=web,network``\n        - ``group_add="[\'web\', \'network\']"``\n\n    hostname\n        Hostname of the container. If not provided, and if a ``name`` has been\n        provided, the ``hostname`` will default to the ``name`` that was\n        passed.\n\n        Example: ``hostname=web1``\n\n        .. warning::\n\n            If the container is started with ``network_mode=host``, the\n            hostname will be overridden by the hostname of the Minion.\n\n    interactive (or *stdin_open*): False\n        Leave stdin open, even if not attached\n\n        Examples:\n\n        - ``interactive=True``\n        - ``stdin_open=True``\n\n    ipc_mode (or *ipc*)\n        Set the IPC mode for the container. The default behavior is to create a\n        private IPC namespace for the container, but this option can be\n        used to change that behavior:\n\n        - ``container:<container_name_or_id>`` reuses another container shared\n          memory, semaphores and message queues\n        - ``host``: use the host\'s shared memory, semaphores and message queues\n\n        Examples:\n\n        - ``ipc_mode=container:foo``\n        - ``ipc=host``\n\n        .. warning::\n            Using ``host`` gives the container full access to local shared\n            memory and is therefore considered insecure.\n\n    isolation\n        Specifies the type of isolation technology used by containers\n\n        Example: ``isolation=hyperv``\n\n        .. note::\n            The default value on Windows server is ``process``, while the\n            default value on Windows client is ``hyperv``. On Linux, only\n            ``default`` is supported.\n\n    labels (or *label*)\n        Add metadata to the container. Labels can be set both with and without\n        values:\n\n        Examples:\n\n        - ``labels=foo,bar=baz``\n        - ``labels="[\'foo\', \'bar=baz\']"``\n\n        .. versionchanged:: 2018.3.0\n            Labels both with and without values can now be mixed. Earlier\n            releases only permitted one method or the other.\n\n    links\n        Link this container to another. Links should be specified in the format\n        ``<container_name_or_id>:<link_alias>``. Multiple links can be passed,\n        ether as a comma separated list or a Python list.\n\n        Examples:\n\n        - ``links=web1:link1,web2:link2``,\n        - ``links="[\'web1:link1\', \'web2:link2\']"``\n        - ``links="{\'web1\': \'link1\', \'web2\': \'link2\'}"``\n\n    log_driver\n        Set container\'s logging driver. Requires Docker 1.6 or newer.\n\n        Example:\n\n        - ``log_driver=syslog``\n\n        .. note::\n            The logging driver feature was improved in Docker 1.13 introducing\n            option name changes. Please see Docker\'s `Configure logging\n            drivers`_ documentation for more information.\n\n        .. _`Configure logging drivers`: https://docs.docker.com/engine/admin/logging/overview/\n\n    log_opt\n        Config options for the ``log_driver`` config option. Requires Docker\n        1.6 or newer.\n\n        Example:\n\n        - ``log_opt="syslog-address=tcp://192.168.0.42,syslog-facility=daemon"``\n        - ``log_opt="[\'syslog-address=tcp://192.168.0.42\', \'syslog-facility=daemon\']"``\n        - ``log_opt="{\'syslog-address\': \'tcp://192.168.0.42\', \'syslog-facility\': \'daemon\'}"``\n\n    lxc_conf\n        Additional LXC configuration parameters to set before starting the\n        container.\n\n        Examples:\n\n        - ``lxc_conf="lxc.utsname=docker,lxc.arch=x86_64"``\n        - ``lxc_conf="[\'lxc.utsname=docker\', \'lxc.arch=x86_64\']"``\n        - ``lxc_conf="{\'lxc.utsname\': \'docker\', \'lxc.arch\': \'x86_64\'}"``\n\n        .. note::\n\n            These LXC configuration parameters will only have the desired\n            effect if the container is using the LXC execution driver, which\n            has been deprecated for some time.\n\n    mac_address\n        MAC address to use for the container. If not specified, a random MAC\n        address will be used.\n\n        Example: ``mac_address=01:23:45:67:89:0a``\n\n    mem_limit (or *memory*) : 0\n        Memory limit. Can be specified in bytes or using single-letter units\n        (i.e. ``512M``, ``2G``, etc.). A value of ``0`` (the default) means no\n        memory limit.\n\n        Examples:\n\n        - ``mem_limit=512M``\n        - ``memory=1073741824``\n\n    mem_swappiness\n        Tune a container\'s memory swappiness behavior. Accepts an integer\n        between 0 and 100.\n\n        Example: ``mem_swappiness=60``\n\n    memswap_limit (or *memory_swap*) : -1\n        Total memory limit (memory plus swap). Set to ``-1`` to disable swap. A\n        value of ``0`` means no swap limit.\n\n        Examples:\n\n        - ``memswap_limit=1G``\n        - ``memory_swap=2147483648``\n\n    network_disabled : False\n        If ``True``, networking will be disabled within the container\n\n        Example: ``network_disabled=True``\n\n    network_mode : bridge\n        One of the following:\n\n        - ``bridge`` - Creates a new network stack for the container on the\n          docker bridge\n        - ``none`` - No networking (equivalent of the Docker CLI argument\n          ``--net=none``). Not to be confused with Python\'s ``None``.\n        - ``container:<name_or_id>`` - Reuses another container\'s network stack\n        - ``host`` - Use the host\'s network stack inside the container\n\n          .. warning::\n              Using ``host`` mode gives the container full access to the hosts\n              system\'s services (such as D-Bus), and is therefore considered\n              insecure.\n\n        Examples:\n\n        - ``network_mode=null``\n        - ``network_mode=container:web1``\n\n    oom_kill_disable\n        Whether to disable OOM killer\n\n        Example: ``oom_kill_disable=False``\n\n    oom_score_adj\n        An integer value containing the score given to the container in order\n        to tune OOM killer preferences\n\n        Example: ``oom_score_adj=500``\n\n    pid_mode\n        Set to ``host`` to use the host container\'s PID namespace within the\n        container. Requires Docker 1.5.0 or newer.\n\n        Example: ``pid_mode=host``\n\n    pids_limit\n        Set the container\'s PID limit. Set to ``-1`` for unlimited.\n\n        Example: ``pids_limit=2000``\n\n    port_bindings (or *publish*)\n        Bind exposed ports which were exposed using the ``ports`` argument to\n        :py:func:`docker.create <salt.modules.dockermod.create>`. These\n        should be passed in the same way as the ``--publish`` argument to the\n        ``docker run`` CLI command:\n\n        - ``ip:hostPort:containerPort`` - Bind a specific IP and port on the\n          host to a specific port within the container.\n        - ``ip::containerPort`` - Bind a specific IP and an ephemeral port to a\n          specific port within the container.\n        - ``hostPort:containerPort`` - Bind a specific port on all of the\n          host\'s interfaces to a specific port within the container.\n        - ``containerPort`` - Bind an ephemeral port on all of the host\'s\n          interfaces to a specific port within the container.\n\n        Multiple bindings can be separated by commas, or passed as a Python\n        list. The below two examples are equivalent:\n\n        - ``port_bindings="5000:5000,2123:2123/udp,8080"``\n        - ``port_bindings="[\'5000:5000\', \'2123:2123/udp\', 8080]"``\n\n        Port bindings can also include ranges:\n\n        - ``port_bindings="14505-14506:4505-4506"``\n\n        .. note::\n            When specifying a protocol, it must be passed in the\n            ``containerPort`` value, as seen in the examples above.\n\n    ports\n        A list of ports to expose on the container. Can be passed as\n        comma-separated list or a Python list. If the protocol is omitted, the\n        port will be assumed to be a TCP port.\n\n        Examples:\n\n        - ``ports=1111,2222/udp``\n        - ``ports="[1111, \'2222/udp\']"``\n\n    privileged : False\n        If ``True``, runs the exec process with extended privileges\n\n        Example: ``privileged=True``\n\n    publish_all_ports (or *publish_all*): False\n        Publish all ports to the host\n\n        Example: ``publish_all_ports=True``\n\n    read_only : False\n        If ``True``, mount the container’s root filesystem as read only\n\n        Example: ``read_only=True``\n\n    restart_policy (or *restart*)\n        Set a restart policy for the container. Must be passed as a string in\n        the format ``policy[:retry_count]`` where ``policy`` is one of\n        ``always``, ``unless-stopped``, or ``on-failure``, and ``retry_count``\n        is an optional limit to the number of retries. The retry count is ignored\n        when using the ``always`` or ``unless-stopped`` restart policy.\n\n        Examples:\n\n        - ``restart_policy=on-failure:5``\n        - ``restart_policy=always``\n\n    security_opt\n        Security configuration for MLS systems such as SELinux and AppArmor.\n        Can be passed as a comma-separated list or a Python list.\n\n        Examples:\n\n        - ``security_opt=apparmor:unconfined,param2:value2``\n        - ``security_opt=\'["apparmor:unconfined", "param2:value2"]\'``\n\n        .. important::\n            Some security options can contain commas. In these cases, this\n            argument *must* be passed as a Python list, as splitting by comma\n            will result in an invalid configuration.\n\n        .. note::\n            See the documentation for security_opt at\n            https://docs.docker.com/engine/reference/run/#security-configuration\n\n    shm_size\n        Size of /dev/shm\n\n        Example: ``shm_size=128M``\n\n    stop_signal\n        The signal used to stop the container. The default is ``SIGTERM``.\n\n        Example: ``stop_signal=SIGRTMIN+3``\n\n    stop_timeout\n        Timeout to stop the container, in seconds\n\n        Example: ``stop_timeout=5``\n\n    storage_opt\n        Storage driver options for the container\n\n        Examples:\n\n        - ``storage_opt=\'dm.basesize=40G\'``\n        - ``storage_opt="[\'dm.basesize=40G\']"``\n        - ``storage_opt="{\'dm.basesize\': \'40G\'}"``\n\n    sysctls (or *sysctl*)\n        Set sysctl options for the container\n\n        Examples:\n\n        - ``sysctl=\'fs.nr_open=1048576,kernel.pid_max=32768\'``\n        - ``sysctls="[\'fs.nr_open=1048576\', \'kernel.pid_max=32768\']"``\n        - ``sysctls="{\'fs.nr_open\': \'1048576\', \'kernel.pid_max\': \'32768\'}"``\n\n    tmpfs\n        A map of container directories which should be replaced by tmpfs\n        mounts, and their corresponding mount options. Can be passed as Python\n        list of PATH:VALUE mappings, or a Python dictionary. However, since\n        commas usually appear in the values, this option *cannot* be passed as\n        a comma-separated list.\n\n        Examples:\n\n        - ``tmpfs="[\'/run:rw,noexec,nosuid,size=65536k\', \'/var/lib/mysql:rw,noexec,nosuid,size=600m\']"``\n        - ``tmpfs="{\'/run\': \'rw,noexec,nosuid,size=65536k\', \'/var/lib/mysql\': \'rw,noexec,nosuid,size=600m\'}"``\n\n    tty : False\n        Attach TTYs\n\n        Example: ``tty=True``\n\n    ulimits (or *ulimit*)\n        List of ulimits. These limits should be passed in the format\n        ``<ulimit_name>:<soft_limit>:<hard_limit>``, with the hard limit being\n        optional. Can be passed as a comma-separated list or a Python list.\n\n        Examples:\n\n        - ``ulimits="nofile=1024:1024,nproc=60"``\n        - ``ulimits="[\'nofile=1024:1024\', \'nproc=60\']"``\n\n    user\n        User under which to run exec process\n\n        Example: ``user=foo``\n\n    userns_mode (or *user_ns_mode*)\n        Sets the user namsepace mode, when the user namespace remapping option\n        is enabled.\n\n        Example: ``userns_mode=host``\n\n    volumes (or *volume*)\n        List of directories to expose as volumes. Can be passed as a\n        comma-separated list or a Python list.\n\n        Examples:\n\n        - ``volumes=/mnt/vol1,/mnt/vol2``\n        - ``volume="[\'/mnt/vol1\', \'/mnt/vol2\']"``\n\n    volumes_from\n        Container names or IDs from which the container will get volumes. Can\n        be passed as a comma-separated list or a Python list.\n\n        Example: ``volumes_from=foo``, ``volumes_from=foo,bar``,\n        ``volumes_from="[foo, bar]"``\n\n    volume_driver\n        Sets the container\'s volume driver\n\n        Example: ``volume_driver=foobar``\n\n    working_dir (or *workdir*)\n        Working directory inside the container\n\n        Examples:\n\n        - ``working_dir=/var/log/nginx``\n        - ``workdir=/var/www/myapp``\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Id`` - ID of the newly-created container\n    - ``Name`` - Name of the newly-created container\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Create a data-only container\n        salt myminion docker.create myuser/mycontainer volumes="/mnt/vol1,/mnt/vol2"\n        # Create a CentOS 7 container that will stay running once started\n        salt myminion docker.create centos:7 name=mycent7 interactive=True tty=True command=bash\n    '
    if kwargs.pop('inspect', True) and (not resolve_image_id(image)):
        pull(image, client_timeout=client_timeout)
    (kwargs, unused_kwargs) = _get_create_kwargs(skip_translate=skip_translate, ignore_collisions=ignore_collisions, validate_ip_addrs=validate_ip_addrs, **kwargs)
    if unused_kwargs:
        log.warning('The following arguments were ignored because they are not recognized by docker-py: %s', sorted(unused_kwargs))
    log.debug('docker.create: creating container %susing the following arguments: %s', f"with name '{name}' " if name is not None else '', kwargs)
    time_started = time.time()
    response = _client_wrapper('create_container', image, name=name, **kwargs)
    response['Time_Elapsed'] = time.time() - time_started
    _clear_context()
    if name is None:
        name = inspect_container(response['Id'])['Name'].lstrip('/')
    response['Name'] = name
    if start:
        try:
            start_(name)
        except CommandExecutionError as exc:
            raise CommandExecutionError('Failed to start container after creation', info={'response': response, 'error': exc.__str__()})
        else:
            response['Started'] = True
    return response

@_refresh_mine_cache
def run_container(image, name=None, skip_translate=None, ignore_collisions=False, validate_ip_addrs=True, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT, bg=False, replace=False, force=False, networks=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2018.3.0\n\n    Equivalent to ``docker run`` on the Docker CLI. Runs the container, waits\n    for it to exit, and returns the container\'s logs when complete.\n\n    .. note::\n        Not to be confused with :py:func:`docker.run\n        <salt.modules.dockermod.run>`, which provides a :py:func:`cmd.run\n        <salt.modules.cmdmod.run>`-like interface for executing commands in a\n        running container.\n\n    This function accepts the same arguments as :py:func:`docker.create\n    <salt.modules.dockermod.create>`, with the exception of ``start``. In\n    addition, it accepts the arguments from :py:func:`docker.logs\n    <salt.modules.dockermod.logs>`, with the exception of ``follow``, to\n    control how logs are returned. Finally, the ``bg`` argument described below\n    can be used to optionally run the container in the background (the default\n    behavior is to block until the container exits).\n\n    bg : False\n        If ``True``, this function will not wait for the container to exit and\n        will not return its logs. It will however return the container\'s name\n        and ID, allowing for :py:func:`docker.logs\n        <salt.modules.dockermod.logs>` to be used to view the logs.\n\n        .. note::\n            The logs will be inaccessible once the container exits if\n            ``auto_remove`` is set to ``True``, so keep this in mind.\n\n    replace : False\n        If ``True``, and if the named container already exists, this will\n        remove the existing container. The default behavior is to return a\n        ``False`` result when the container already exists.\n\n    force : False\n        If ``True``, and the named container already exists, *and* ``replace``\n        is also set to ``True``, then the container will be forcibly removed.\n        Otherwise, the state will not proceed and will return a ``False``\n        result.\n\n    networks\n        Networks to which the container should be connected. If automatic IP\n        configuration is being used, the networks can be a simple list of\n        network names. If custom IP configuration is being used, then this\n        argument must be passed as a dictionary.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.run_container myuser/myimage command=/usr/local/bin/myscript.sh\n        # Run container in the background\n        salt myminion docker.run_container myuser/myimage command=/usr/local/bin/myscript.sh bg=True\n        # Connecting to two networks using automatic IP configuration\n        salt myminion docker.run_container myuser/myimage command=\'perl /scripts/sync.py\' networks=net1,net2\n        # net1 using automatic IP, net2 using static IPv4 address\n        salt myminion docker.run_container myuser/myimage command=\'perl /scripts/sync.py\' networks=\'{"net1": {}, "net2": {"ipv4_address": "192.168.27.12"}}\'\n    '
    if kwargs.pop('inspect', True) and (not resolve_image_id(image)):
        pull(image, client_timeout=client_timeout)
    removed_ids = None
    if name is not None:
        try:
            pre_state = __salt__['docker.state'](name)
        except CommandExecutionError:
            pass
        else:
            if pre_state == 'running' and (not (replace and force)):
                raise CommandExecutionError("Container '{}' exists and is running. Run with replace=True and force=True to force removal of the existing container.".format(name))
            elif not replace:
                raise CommandExecutionError("Container '{}' exists. Run with replace=True to remove the existing container".format(name))
            else:
                removed_ids = rm_(name, force=force)
    log_kwargs = {}
    for argname in get_client_args('logs')['logs']:
        try:
            log_kwargs[argname] = kwargs.pop(argname)
        except KeyError:
            pass
    log_kwargs.pop('stream', None)
    (kwargs, unused_kwargs) = _get_create_kwargs(skip_translate=skip_translate, ignore_collisions=ignore_collisions, validate_ip_addrs=validate_ip_addrs, **kwargs)
    auto_remove = kwargs.get('host_config', {}).get('AutoRemove', False)
    if unused_kwargs:
        log.warning('The following arguments were ignored because they are not recognized by docker-py: %s', sorted(unused_kwargs))
    if networks:
        if isinstance(networks, str):
            networks = {x: {} for x in networks.split(',')}
        if not isinstance(networks, dict) or not all((isinstance(x, dict) for x in networks.values())):
            raise SaltInvocationError('Invalid format for networks argument')
    log.debug('docker.create: creating container %susing the following arguments: %s', f"with name '{name}' " if name is not None else '', kwargs)
    time_started = time.time()
    ret = _client_wrapper('create_container', image, name=name, **kwargs)
    if removed_ids:
        ret['Replaces'] = removed_ids
    if name is None:
        name = inspect_container(ret['Id'])['Name'].lstrip('/')
    ret['Name'] = name

    def _append_warning(ret, msg):
        if False:
            i = 10
            return i + 15
        warnings = ret.pop('Warnings', None)
        if warnings is None:
            warnings = [msg]
        elif isinstance(ret, list):
            warnings.append(msg)
        else:
            warnings = [warnings, msg]
        ret['Warnings'] = warnings
    exc_info = {'return': ret}
    try:
        if networks:
            try:
                for (net_name, net_conf) in networks.items():
                    __salt__['docker.connect_container_to_network'](ret['Id'], net_name, **net_conf)
            except CommandExecutionError as exc:
                if auto_remove:
                    try:
                        rm_(name)
                    except CommandExecutionError as rm_exc:
                        exc_info.setdefault('other_errors', []).append(f'Failed to auto_remove container: {rm_exc}')
                raise CommandExecutionError(exc.__str__(), info=exc_info)
        output = []
        start_(ret['Id'])
        if not bg:
            try:
                for line in _client_wrapper('logs', ret['Id'], stream=True, timestamps=False):
                    output.append(salt.utils.stringutils.to_unicode(line))
            except CommandExecutionError:
                msg = 'Failed to get logs from container. This may be because the container exited before Salt was able to attach to it to retrieve the logs. Consider setting auto_remove to False.'
                _append_warning(ret, msg)
        ret['Time_Elapsed'] = time.time() - time_started
        _clear_context()
        if not bg:
            ret['Logs'] = ''.join(output)
            if not auto_remove:
                try:
                    cinfo = inspect_container(ret['Id'])
                except CommandExecutionError:
                    _append_warning(ret, 'Failed to inspect container after running')
                else:
                    cstate = cinfo.get('State', {})
                    cstatus = cstate.get('Status')
                    if cstatus != 'exited':
                        _append_warning(ret, "Container state is not 'exited'")
                    ret['ExitCode'] = cstate.get('ExitCode')
    except CommandExecutionError as exc:
        try:
            exc_info.update(exc.info)
        except (TypeError, ValueError):
            exc_info.setdefault('other_errors', []).append(exc.info)
        raise CommandExecutionError(exc.__str__(), info=exc_info)
    return ret

def copy_from(name, source, dest, overwrite=False, makedirs=False):
    if False:
        return 10
    "\n    Copy a file from inside a container to the Minion\n\n    name\n        Container name\n\n    source\n        Path of the file on the container's filesystem\n\n    dest\n        Destination on the Minion. Must be an absolute path. If the destination\n        is a directory, the file will be copied into that directory.\n\n    overwrite : False\n        Unless this option is set to ``True``, then if a file exists at the\n        location specified by the ``dest`` argument, an error will be raised.\n\n    makedirs : False\n        Create the parent directory on the container if it does not already\n        exist.\n\n\n    **RETURN DATA**\n\n    A boolean (``True`` if successful, otherwise ``False``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.copy_from mycontainer /var/log/nginx/access.log /home/myuser\n    "
    c_state = state(name)
    if c_state != 'running':
        raise CommandExecutionError(f"Container '{name}' is not running")
    if not os.path.isabs(dest):
        raise SaltInvocationError('Destination path must be absolute')
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(source))
        dest_dir = dest
    else:
        dest_dir = os.path.split(dest)[0]
        if not os.path.isdir(dest_dir):
            if makedirs:
                try:
                    os.makedirs(dest_dir)
                except OSError as exc:
                    raise CommandExecutionError('Unable to make destination directory {}: {}'.format(dest_dir, exc))
            else:
                raise SaltInvocationError(f'Directory {dest_dir} does not exist')
    if not overwrite and os.path.exists(dest):
        raise CommandExecutionError('Destination path {} already exists. Use overwrite=True to overwrite it'.format(dest))
    if not os.path.isabs(source):
        raise SaltInvocationError('Source path must be absolute')
    elif retcode(name, f'test -e {shlex.quote(source)}', ignore_retcode=True) == 0:
        if retcode(name, f'test -f {shlex.quote(source)}', ignore_retcode=True) != 0:
            raise SaltInvocationError('Source must be a regular file')
    else:
        raise SaltInvocationError(f'Source file {source} does not exist')
    source_md5 = _get_md5(name, source)
    if source_md5 == __salt__['file.get_sum'](dest, 'md5'):
        log.debug('%s:%s and %s are the same file, skipping copy', name, source, dest)
        return True
    log.debug("Copying %s from container '%s' to local path %s", source, name, dest)
    try:
        src_path = ':'.join((name, source))
    except TypeError:
        src_path = f'{name}:{source}'
    cmd = ['docker', 'cp', src_path, dest_dir]
    __salt__['cmd.run'](cmd, python_shell=False)
    return source_md5 == __salt__['file.get_sum'](dest, 'md5')
cp = salt.utils.functools.alias_function(copy_from, 'cp')

def copy_to(name, source, dest, exec_driver=None, overwrite=False, makedirs=False):
    if False:
        return 10
    '\n    Copy a file from the host into a container\n\n    name\n        Container name\n\n    source\n        File to be copied to the container. Can be a local path on the Minion\n        or a remote file from the Salt fileserver.\n\n    dest\n        Destination on the container. Must be an absolute path. If the\n        destination is a directory, the file will be copied into that\n        directory.\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    overwrite : False\n        Unless this option is set to ``True``, then if a file exists at the\n        location specified by the ``dest`` argument, an error will be raised.\n\n    makedirs : False\n        Create the parent directory on the container if it does not already\n        exist.\n\n\n    **RETURN DATA**\n\n    A boolean (``True`` if successful, otherwise ``False``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.copy_to mycontainer /tmp/foo /root/foo\n    '
    if exec_driver is None:
        exec_driver = _get_exec_driver()
    return __salt__['container_resource.copy_to'](name, __salt__['container_resource.cache_file'](source), dest, container_type=__virtualname__, exec_driver=exec_driver, overwrite=overwrite, makedirs=makedirs)

def export(name, path, overwrite=False, makedirs=False, compression=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Exports a container to a tar archive. It can also optionally compress that\n    tar archive, and push it up to the Master.\n\n    name\n        Container name or ID\n\n    path\n        Absolute path on the Minion where the container will be exported\n\n    overwrite : False\n        Unless this option is set to ``True``, then if a file exists at the\n        location specified by the ``path`` argument, an error will be raised.\n\n    makedirs : False\n        If ``True``, then if the parent directory of the file specified by the\n        ``path`` argument does not exist, Salt will attempt to create it.\n\n    compression : None\n        Can be set to any of the following:\n\n        - ``gzip`` or ``gz`` for gzip compression\n        - ``bzip2`` or ``bz2`` for bzip2 compression\n        - ``xz`` or ``lzma`` for XZ compression (requires `xz-utils`_, as well\n          as the ``lzma`` module from Python 3.3, available in Python 2 and\n          Python 3.0-3.2 as `backports.lzma`_)\n\n        This parameter can be omitted and Salt will attempt to determine the\n        compression type by examining the filename passed in the ``path``\n        parameter.\n\n        .. _`xz-utils`: http://tukaani.org/xz/\n        .. _`backports.lzma`: https://pypi.python.org/pypi/backports.lzma\n\n    push : False\n        If ``True``, the container will be pushed to the master using\n        :py:func:`cp.push <salt.modules.cp.push>`.\n\n        .. note::\n\n            This requires :conf_master:`file_recv` to be set to ``True`` on the\n            Master.\n\n\n    **RETURN DATA**\n\n    A dictionary will containing the following keys:\n\n    - ``Path`` - Path of the file that was exported\n    - ``Push`` - Reports whether or not the file was successfully pushed to the\n      Master\n\n      *(Only present if push=True)*\n    - ``Size`` - Size of the file, in bytes\n    - ``Size_Human`` - Size of the file, in human-readable units\n    - ``Time_Elapsed`` - Time in seconds taken to perform the export\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.export mycontainer /tmp/mycontainer.tar\n        salt myminion docker.export mycontainer /tmp/mycontainer.tar.xz push=True\n    '
    err = f"Path '{path}' is not absolute"
    try:
        if not os.path.isabs(path):
            raise SaltInvocationError(err)
    except AttributeError:
        raise SaltInvocationError(err)
    if os.path.exists(path) and (not overwrite):
        raise CommandExecutionError(f'{path} already exists')
    if compression is None:
        if path.endswith('.tar.gz') or path.endswith('.tgz'):
            compression = 'gzip'
        elif path.endswith('.tar.bz2') or path.endswith('.tbz2'):
            compression = 'bzip2'
        elif path.endswith('.tar.xz') or path.endswith('.txz'):
            if HAS_LZMA:
                compression = 'xz'
            else:
                raise CommandExecutionError('XZ compression unavailable. Install the backports.lzma module and xz-utils to enable XZ compression.')
    elif compression == 'gz':
        compression = 'gzip'
    elif compression == 'bz2':
        compression = 'bzip2'
    elif compression == 'lzma':
        compression = 'xz'
    if compression and compression not in ('gzip', 'bzip2', 'xz'):
        raise SaltInvocationError(f"Invalid compression type '{compression}'")
    parent_dir = os.path.dirname(path)
    if not os.path.isdir(parent_dir):
        if not makedirs:
            raise CommandExecutionError('Parent dir {} of destination path does not exist. Use makedirs=True to create it.'.format(parent_dir))
        try:
            os.makedirs(parent_dir)
        except OSError as exc:
            raise CommandExecutionError(f'Unable to make parent dir {parent_dir}: {exc}')
    if compression == 'gzip':
        try:
            out = gzip.open(path, 'wb')
        except OSError as exc:
            raise CommandExecutionError(f'Unable to open {path} for writing: {exc}')
    elif compression == 'bzip2':
        compressor = bz2.BZ2Compressor()
    elif compression == 'xz':
        compressor = lzma.LZMACompressor()
    time_started = time.time()
    try:
        if compression != 'gzip':
            out = __utils__['files.fopen'](path, 'wb')
        response = _client_wrapper('export', name)
        buf = None
        while buf != '':
            buf = response.read(4096)
            if buf:
                if compression in ('bzip2', 'xz'):
                    data = compressor.compress(buf)
                    if data:
                        out.write(data)
                else:
                    out.write(buf)
        if compression in ('bzip2', 'xz'):
            data = compressor.flush()
            if data:
                out.write(data)
        out.flush()
    except Exception as exc:
        try:
            os.remove(path)
        except OSError:
            pass
        raise CommandExecutionError(f'Error occurred during container export: {exc}')
    finally:
        out.close()
    ret = {'Time_Elapsed': time.time() - time_started}
    ret['Path'] = path
    ret['Size'] = os.stat(path).st_size
    ret['Size_Human'] = _size_fmt(ret['Size'])
    if kwargs.get(push, False):
        ret['Push'] = __salt__['cp.push'](path)
    return ret

@_refresh_mine_cache
def rm_(name, force=False, volumes=False, **kwargs):
    if False:
        return 10
    '\n    Removes a container\n\n    name\n        Container name or ID\n\n    force : False\n        If ``True``, the container will be killed first before removal, as the\n        Docker API will not permit a running container to be removed. This\n        option is set to ``False`` by default to prevent accidental removal of\n        a running container.\n\n    stop : False\n        If ``True``, the container will be stopped first before removal, as the\n        Docker API will not permit a running container to be removed. This\n        option is set to ``False`` by default to prevent accidental removal of\n        a running container.\n\n        .. versionadded:: 2017.7.0\n\n    timeout\n        Optional timeout to be passed to :py:func:`docker.stop\n        <salt.modules.dockermod.stop>` if stopping the container.\n\n        .. versionadded:: 2018.3.0\n\n    volumes : False\n        Also remove volumes associated with container\n\n\n    **RETURN DATA**\n\n    A list of the IDs of containers which were removed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.rm mycontainer\n        salt myminion docker.rm mycontainer force=True\n    '
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    stop_ = kwargs.pop('stop', False)
    timeout = kwargs.pop('timeout', None)
    auto_remove = False
    if kwargs:
        __utils__['args.invalid_kwargs'](kwargs)
    if state(name) == 'running' and (not (force or stop_)):
        raise CommandExecutionError("Container '{}' is running, use force=True to forcibly remove this container".format(name))
    if stop_ and (not force):
        inspect_results = inspect_container(name)
        try:
            auto_remove = inspect_results['HostConfig']['AutoRemove']
        except KeyError:
            log.error('Failed to find AutoRemove in inspect results, Docker API may have changed. Full results: %s', inspect_results)
        stop(name, timeout=timeout)
    pre = ps_(all=True)
    if not auto_remove:
        _client_wrapper('remove_container', name, v=volumes, force=force)
    _clear_context()
    return [x for x in pre if x not in ps_(all=True)]

def rename(name, new_name):
    if False:
        return 10
    '\n    .. versionadded:: 2017.7.0\n\n    Renames a container. Returns ``True`` if successful, and raises an error if\n    the API returns one. If unsuccessful and the API returns no error (should\n    not happen), then ``False`` will be returned.\n\n    name\n        Name or ID of existing container\n\n    new_name\n        New name to assign to container\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.rename foo bar\n    '
    id_ = inspect_container(name)['Id']
    log.debug("Renaming container '%s' (ID: %s) to '%s'", name, id_, new_name)
    _client_wrapper('rename', id_, new_name)
    return inspect_container(new_name)['Id'] == id_

def build(path=None, repository=None, tag=None, cache=True, rm=True, api_response=False, fileobj=None, dockerfile=None, buildargs=None):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2018.3.0\n        If the built image should be tagged, then the repository and tag must\n        now be passed separately using the ``repository`` and ``tag``\n        arguments, rather than together in the (now deprecated) ``image``\n        argument.\n\n    Builds a docker image from a Dockerfile or a URL\n\n    path\n        Path to directory on the Minion containing a Dockerfile\n\n    repository\n        Optional repository name for the image being built\n\n        .. versionadded:: 2018.3.0\n\n    tag : latest\n        Tag name for the image (required if ``repository`` is passed)\n\n        .. versionadded:: 2018.3.0\n\n    image\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n    cache : True\n        Set to ``False`` to force the build process not to use the Docker image\n        cache, and pull all required intermediate image layers\n\n    rm : True\n        Remove intermediate containers created during build\n\n    api_response : False\n        If ``True``: an ``API_Response`` key will be present in the return\n        data, containing the raw output from the Docker API.\n\n    fileobj\n        Allows for a file-like object containing the contents of the Dockerfile\n        to be passed in place of a file ``path`` argument. This argument should\n        not be used from the CLI, only from other Salt code.\n\n    dockerfile\n        Allows for an alternative Dockerfile to be specified. Path to\n        alternative Dockefile is relative to the build path for the Docker\n        container.\n\n        .. versionadded:: 2016.11.0\n\n    buildargs\n        A dictionary of build arguments provided to the docker build process.\n\n\n    **RETURN DATA**\n\n    A dictionary containing one or more of the following keys:\n\n    - ``Id`` - ID of the newly-built image\n    - ``Time_Elapsed`` - Time in seconds taken to perform the build\n    - ``Intermediate_Containers`` - IDs of containers created during the course\n      of the build process\n\n      *(Only present if rm=False)*\n    - ``Images`` - A dictionary containing one or more of the following keys:\n        - ``Already_Pulled`` - Layers that that were already present on the\n          Minion\n        - ``Pulled`` - Layers that that were pulled\n\n      *(Only present if the image specified by the "repository" and "tag"\n      arguments was not present on the Minion, or if cache=False)*\n    - ``Status`` - A string containing a summary of the pull action (usually a\n      message saying that an image was downloaded, or that it was up to date).\n\n      *(Only present if the image specified by the "repository" and "tag"\n      arguments was not present on the Minion, or if cache=False)*\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.build /path/to/docker/build/dir\n        salt myminion docker.build https://github.com/myuser/myrepo.git repository=myimage tag=latest\n        salt myminion docker.build /path/to/docker/build/dir dockerfile=Dockefile.different repository=myimage tag=dev\n    '
    _prep_pull()
    if repository or tag:
        if not repository and tag:
            raise SaltInvocationError('If tagging, both a repository and tag are required')
        else:
            if not isinstance(repository, str):
                repository = str(repository)
            if not isinstance(tag, str):
                tag = str(tag)
    image_tag = f'{repository}:{tag}' if repository and tag else None
    time_started = time.time()
    response = _client_wrapper('build', path=path, tag=image_tag, quiet=False, fileobj=fileobj, rm=rm, nocache=not cache, dockerfile=dockerfile, buildargs=buildargs)
    ret = {'Time_Elapsed': time.time() - time_started}
    _clear_context()
    if not response:
        raise CommandExecutionError(f'Build failed for {path}, no response returned from Docker API')
    stream_data = []
    for line in response:
        stream_data.extend(salt.utils.json.loads(line, cls=DockerJSONDecoder))
    errors = []
    for item in stream_data:
        try:
            item_type = next(iter(item))
        except StopIteration:
            continue
        if item_type == 'status':
            _pull_status(ret, item)
        if item_type == 'stream':
            _build_status(ret, item)
        elif item_type == 'errorDetail':
            _error_detail(errors, item)
    if 'Id' not in ret:
        msg = f'Build failed for {path}'
        log.error(msg)
        log.error(stream_data)
        if errors:
            msg += '. Error(s) follow:\n\n{}'.format('\n\n'.join(errors))
        raise CommandExecutionError(msg)
    resolved_tag = resolve_tag(ret['Id'], all=True)
    if resolved_tag:
        ret['Image'] = resolved_tag
    else:
        ret['Warning'] = f'Failed to tag image as {image_tag}'
    if api_response:
        ret['API_Response'] = stream_data
    if rm:
        ret.pop('Intermediate_Containers', None)
    return ret

def commit(name, repository, tag='latest', message=None, author=None):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2018.3.0\n        The repository and tag must now be passed separately using the\n        ``repository`` and ``tag`` arguments, rather than together in the (now\n        deprecated) ``image`` argument.\n\n    Commits a container, thereby promoting it to an image. Equivalent to\n    running the ``docker commit`` Docker CLI command.\n\n    name\n        Container name or ID to commit\n\n    repository\n        Repository name for the image being committed\n\n        .. versionadded:: 2018.3.0\n\n    tag : latest\n        Tag name for the image\n\n        .. versionadded:: 2018.3.0\n\n    image\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n    message\n        Commit message (Optional)\n\n    author\n        Author name (Optional)\n\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Id`` - ID of the newly-created image\n    - ``Image`` - Name of the newly-created image\n    - ``Time_Elapsed`` - Time in seconds taken to perform the commit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.commit mycontainer myuser/myimage mytag\n    '
    if not isinstance(repository, str):
        repository = str(repository)
    if not isinstance(tag, str):
        tag = str(tag)
    time_started = time.time()
    response = _client_wrapper('commit', name, repository=repository, tag=tag, message=message, author=author)
    ret = {'Time_Elapsed': time.time() - time_started}
    _clear_context()
    image_id = None
    for id_ in ('Id', 'id', 'ID'):
        if id_ in response:
            image_id = response[id_]
            break
    if image_id is None:
        raise CommandExecutionError('No image ID was returned in API response')
    ret['Id'] = image_id
    return ret

def dangling(prune=False, force=False):
    if False:
        i = 10
        return i + 15
    '\n    Return top-level images (those on which no other images depend) which do\n    not have a tag assigned to them. These include:\n\n    - Images which were once tagged but were later untagged, such as those\n      which were superseded by committing a new copy of an existing tagged\n      image.\n    - Images which were loaded using :py:func:`docker.load\n      <salt.modules.dockermod.load>` (or the ``docker load`` Docker CLI\n      command), but not tagged.\n\n    prune : False\n        Remove these images\n\n    force : False\n        If ``True``, and if ``prune=True``, then forcibly remove these images.\n\n    **RETURN DATA**\n\n    If ``prune=False``, the return data will be a list of dangling image IDs.\n\n    If ``prune=True``, the return data will be a dictionary with each key being\n    the ID of the dangling image, and the following information for each image:\n\n    - ``Comment`` - Any error encountered when trying to prune a dangling image\n\n      *(Only present if prune failed)*\n    - ``Removed`` - A boolean (``True`` if prune was successful, ``False`` if\n      not)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.dangling\n        salt myminion docker.dangling prune=True\n    '
    all_images = images(all=True)
    dangling_images = [x[:12] for x in _get_top_level_images(all_images) if all_images[x]['RepoTags'] is None]
    if not prune:
        return dangling_images
    ret = {}
    for image in dangling_images:
        try:
            ret.setdefault(image, {})['Removed'] = rmi(image, force=force)
        except Exception as exc:
            err = exc.__str__()
            log.error(err)
            ret.setdefault(image, {})['Comment'] = err
            ret[image]['Removed'] = False
    return ret

def import_(source, repository, tag='latest', api_response=False):
    if False:
        return 10
    '\n    .. versionchanged:: 2018.3.0\n        The repository and tag must now be passed separately using the\n        ``repository`` and ``tag`` arguments, rather than together in the (now\n        deprecated) ``image`` argument.\n\n    Imports content from a local tarball or a URL as a new docker image\n\n    source\n        Content to import (URL or absolute path to a tarball).  URL can be a\n        file on the Salt fileserver (i.e.\n        ``salt://path/to/rootfs/tarball.tar.xz``. To import a file from a\n        saltenv other than ``base`` (e.g. ``dev``), pass it at the end of the\n        URL (ex. ``salt://path/to/rootfs/tarball.tar.xz?saltenv=dev``).\n\n    repository\n        Repository name for the image being imported\n\n        .. versionadded:: 2018.3.0\n\n    tag : latest\n        Tag name for the image\n\n        .. versionadded:: 2018.3.0\n\n    image\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n    api_response : False\n        If ``True`` an ``api_response`` key will be present in the return data,\n        containing the raw output from the Docker API.\n\n\n    **RETURN DATA**\n\n    A dictionary containing the following keys:\n\n    - ``Id`` - ID of the newly-created image\n    - ``Image`` - Name of the newly-created image\n    - ``Time_Elapsed`` - Time in seconds taken to perform the commit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.import /tmp/cent7-minimal.tar.xz myuser/centos\n        salt myminion docker.import /tmp/cent7-minimal.tar.xz myuser/centos:7\n        salt myminion docker.import salt://dockerimages/cent7-minimal.tar.xz myuser/centos:7\n    '
    if not isinstance(repository, str):
        repository = str(repository)
    if not isinstance(tag, str):
        tag = str(tag)
    path = __salt__['container_resource.cache_file'](source)
    time_started = time.time()
    response = _client_wrapper('import_image', path, repository=repository, tag=tag)
    ret = {'Time_Elapsed': time.time() - time_started}
    _clear_context()
    if not response:
        raise CommandExecutionError(f'Import failed for {source}, no response returned from Docker API')
    elif api_response:
        ret['API_Response'] = response
    errors = []
    for item in response:
        try:
            item_type = next(iter(item))
        except StopIteration:
            continue
        if item_type == 'status':
            _import_status(ret, item, repository, tag)
        elif item_type == 'errorDetail':
            _error_detail(errors, item)
    if 'Id' not in ret:
        msg = f'Import failed for {source}'
        if errors:
            msg += '. Error(s) follow:\n\n{}'.format('\n\n'.join(errors))
        raise CommandExecutionError(msg)
    return ret

def load(path, repository=None, tag=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2018.3.0\n        If the loaded image should be tagged, then the repository and tag must\n        now be passed separately using the ``repository`` and ``tag``\n        arguments, rather than together in the (now deprecated) ``image``\n        argument.\n\n    Load a tar archive that was created using :py:func:`docker.save\n    <salt.modules.dockermod.save>` (or via the Docker CLI using ``docker save``).\n\n    path\n        Path to docker tar archive. Path can be a file on the Minion, or the\n        URL of a file on the Salt fileserver (i.e.\n        ``salt://path/to/docker/saved/image.tar``). To load a file from a\n        saltenv other than ``base`` (e.g. ``dev``), pass it at the end of the\n        URL (ex. ``salt://path/to/rootfs/tarball.tar.xz?saltenv=dev``).\n\n    repository\n        If specified, the topmost layer of the newly-loaded image will be\n        tagged with the specified repo using :py:func:`docker.tag\n        <salt.modules.dockermod.tag_>`. If a repository name is provided, then\n        the ``tag`` argument is also required.\n\n        .. versionadded:: 2018.3.0\n\n    tag\n        Tag name to go along with the repository name, if the loaded image is\n        to be tagged.\n\n        .. versionadded:: 2018.3.0\n\n    image\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``Path`` - Path of the file that was saved\n    - ``Layers`` - A list containing the IDs of the layers which were loaded.\n      Any layers in the file that was loaded, which were already present on the\n      Minion, will not be included.\n    - ``Image`` - Name of tag applied to topmost layer\n\n      *(Only present if tag was specified and tagging was successful)*\n    - ``Time_Elapsed`` - Time in seconds taken to load the file\n    - ``Warning`` - Message describing any problems encountered in attempt to\n      tag the topmost layer\n\n      *(Only present if tag was specified and tagging failed)*\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.load /path/to/image.tar\n        salt myminion docker.load salt://path/to/docker/saved/image.tar repository=myuser/myimage tag=mytag\n    '
    if (repository or tag) and (not (repository and tag)):
        raise SaltInvocationError('If tagging, both a repository and tag are required')
    local_path = __salt__['container_resource.cache_file'](path)
    if not os.path.isfile(local_path):
        raise CommandExecutionError(f'Source file {path} does not exist')
    pre = images(all=True)
    cmd = ['docker', 'load', '-i', local_path]
    time_started = time.time()
    result = __salt__['cmd.run_all'](cmd)
    ret = {'Time_Elapsed': time.time() - time_started}
    _clear_context()
    post = images(all=True)
    if result['retcode'] != 0:
        msg = f'Failed to load image(s) from {path}'
        if result['stderr']:
            msg += ': {}'.format(result['stderr'])
        raise CommandExecutionError(msg)
    ret['Path'] = path
    new_layers = [x for x in post if x not in pre]
    ret['Layers'] = [x[:12] for x in new_layers]
    top_level_images = _get_top_level_images(post, subset=new_layers)
    if repository or tag:
        if len(top_level_images) > 1:
            ret['Warning'] = 'More than one top-level image layer was loaded ({}), no image was tagged'.format(', '.join(top_level_images))
        else:
            tagged_image = f'{repository}:{tag}'
            try:
                result = tag_(top_level_images[0], repository=repository, tag=tag)
                ret['Image'] = tagged_image
            except IndexError:
                ret['Warning'] = 'No top-level image layers were loaded, no image was tagged'
            except Exception as exc:
                ret['Warning'] = 'Failed to tag {} as {}: {}'.format(top_level_images[0], tagged_image, exc)
    return ret

def layers(name):
    if False:
        while True:
            i = 10
    '\n    Returns a list of the IDs of layers belonging to the specified image, with\n    the top-most layer (the one correspnding to the passed name) appearing\n    last.\n\n    name\n        Image name or ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.layers centos:7\n    '
    ret = []
    cmd = ['docker', 'history', '-q', name]
    for line in reversed(__salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines()):
        ret.append(line)
    if not ret:
        raise CommandExecutionError(f"Image '{name}' not found")
    return ret

def pull(image, insecure_registry=False, api_response=False, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2018.3.0\n        If no tag is specified in the ``image`` argument, all tags for the\n        image will be pulled. For this reason is it recommended to pass\n        ``image`` using the ``repo:tag`` notation.\n\n    Pulls an image from a Docker registry\n\n    image\n        Image to be pulled\n\n    insecure_registry : False\n        If ``True``, the Docker client will permit the use of insecure\n        (non-HTTPS) registries.\n\n    api_response : False\n        If ``True``, an ``API_Response`` key will be present in the return\n        data, containing the raw output from the Docker API.\n\n        .. note::\n\n            This may result in a **lot** of additional return data, especially\n            for larger images.\n\n    client_timeout\n        Timeout in seconds for the Docker client. This is not a timeout for\n        this function, but for receiving a response from the API.\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``Layers`` - A dictionary containing one or more of the following keys:\n        - ``Already_Pulled`` - Layers that that were already present on the\n          Minion\n        - ``Pulled`` - Layers that that were pulled\n    - ``Status`` - A string containing a summary of the pull action (usually a\n      message saying that an image was downloaded, or that it was up to date).\n    - ``Time_Elapsed`` - Time in seconds taken to perform the pull\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.pull centos\n        salt myminion docker.pull centos:6\n    '
    _prep_pull()
    kwargs = {'stream': True, 'client_timeout': client_timeout}
    if insecure_registry:
        kwargs['insecure_registry'] = insecure_registry
    time_started = time.time()
    response = _client_wrapper('pull', image, **kwargs)
    ret = {'Time_Elapsed': time.time() - time_started, 'retcode': 0}
    _clear_context()
    if not response:
        raise CommandExecutionError(f'Pull failed for {image}, no response returned from Docker API')
    elif api_response:
        ret['API_Response'] = response
    errors = []
    for event in response:
        log.debug('pull event: %s', event)
        try:
            event = salt.utils.json.loads(event)
        except Exception as exc:
            raise CommandExecutionError(f"Unable to interpret API event: '{event}'", info={'Error': exc.__str__()})
        try:
            event_type = next(iter(event))
        except StopIteration:
            continue
        if event_type == 'status':
            _pull_status(ret, event)
        elif event_type == 'errorDetail':
            _error_detail(errors, event)
    if errors:
        ret['Errors'] = errors
        ret['retcode'] = 1
    return ret

def push(image, insecure_registry=False, api_response=False, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2015.8.4\n        The ``Id`` and ``Image`` keys are no longer present in the return data.\n        This is due to changes in the Docker Remote API.\n\n    Pushes an image to a Docker registry. See the documentation at top of this\n    page to configure authentication credentials.\n\n    image\n        Image to be pushed. If just the repository name is passed, then all\n        tagged images for the specified repo will be pushed. If the image name\n        is passed in ``repo:tag`` notation, only the specified image will be\n        pushed.\n\n    insecure_registry : False\n        If ``True``, the Docker client will permit the use of insecure\n        (non-HTTPS) registries.\n\n    api_response : False\n        If ``True``, an ``API_Response`` key will be present in the return\n        data, containing the raw output from the Docker API.\n\n    client_timeout\n        Timeout in seconds for the Docker client. This is not a timeout for\n        this function, but for receiving a response from the API.\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``Layers`` - A dictionary containing one or more of the following keys:\n        - ``Already_Pushed`` - Layers that that were already present on the\n          Minion\n        - ``Pushed`` - Layers that that were pushed\n    - ``Time_Elapsed`` - Time in seconds taken to perform the push\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.push myuser/mycontainer\n        salt myminion docker.push myuser/mycontainer:mytag\n    '
    if not isinstance(image, str):
        image = str(image)
    kwargs = {'stream': True, 'client_timeout': client_timeout}
    if insecure_registry:
        kwargs['insecure_registry'] = insecure_registry
    time_started = time.time()
    response = _client_wrapper('push', image, **kwargs)
    ret = {'Time_Elapsed': time.time() - time_started, 'retcode': 0}
    _clear_context()
    if not response:
        raise CommandExecutionError(f'Push failed for {image}, no response returned from Docker API')
    elif api_response:
        ret['API_Response'] = response
    errors = []
    for event in response:
        try:
            event = salt.utils.json.loads(event)
        except Exception as exc:
            raise CommandExecutionError(f"Unable to interpret API event: '{event}'", info={'Error': exc.__str__()})
        try:
            event_type = next(iter(event))
        except StopIteration:
            continue
        if event_type == 'status':
            _push_status(ret, event)
        elif event_type == 'errorDetail':
            _error_detail(errors, event)
    if errors:
        ret['Errors'] = errors
        ret['retcode'] = 1
    return ret

def rmi(*names, **kwargs):
    if False:
        print('Hello World!')
    '\n    Removes an image\n\n    name\n        Name (in ``repo:tag`` notation) or ID of image.\n\n    force : False\n        If ``True``, the image will be removed even if the Minion has\n        containers created from that image\n\n    prune : True\n        If ``True``, untagged parent image layers will be removed as well, set\n        this to ``False`` to keep them.\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following two keys:\n\n    - ``Layers`` - A list of the IDs of image layers that were removed\n    - ``Tags`` - A list of the tags that were removed\n    - ``Errors`` - A list of any errors that were encountered\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.rmi busybox\n        salt myminion docker.rmi busybox force=True\n        salt myminion docker.rmi foo bar baz\n    '
    pre_images = images(all=True)
    pre_tags = list_tags()
    force = kwargs.get('force', False)
    noprune = not kwargs.get('prune', True)
    errors = []
    for name in names:
        image_id = inspect_image(name)['Id']
        try:
            _client_wrapper('remove_image', image_id, force=force, noprune=noprune, catch_api_errors=False)
        except docker.errors.APIError as exc:
            if exc.response.status_code == 409:
                errors.append(exc.explanation)
                deps = depends(name)
                if deps['Containers'] or deps['Images']:
                    err = 'Image is in use by '
                    if deps['Containers']:
                        err += 'container(s): {}'.format(', '.join(deps['Containers']))
                    if deps['Images']:
                        if deps['Containers']:
                            err += ' and '
                        err += 'image(s): {}'.format(', '.join(deps['Images']))
                    errors.append(err)
            else:
                errors.append(f'Error {exc.response.status_code}: {exc.explanation}')
    _clear_context()
    ret = {'Layers': [x for x in pre_images if x not in images(all=True)], 'Tags': [x for x in pre_tags if x not in list_tags()], 'retcode': 0}
    if errors:
        ret['Errors'] = errors
        ret['retcode'] = 1
    return ret

def save(name, path, overwrite=False, makedirs=False, compression=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Saves an image and to a file on the minion. Equivalent to running the\n    ``docker save`` Docker CLI command, but unlike ``docker save`` this will\n    also work on named images instead of just images IDs.\n\n    name\n        Name or ID of image. Specify a specific tag by using the ``repo:tag``\n        notation.\n\n    path\n        Absolute path on the Minion where the image will be exported\n\n    overwrite : False\n        Unless this option is set to ``True``, then if the destination file\n        exists an error will be raised.\n\n    makedirs : False\n        If ``True``, then if the parent directory of the file specified by the\n        ``path`` argument does not exist, Salt will attempt to create it.\n\n    compression : None\n        Can be set to any of the following:\n\n        - ``gzip`` or ``gz`` for gzip compression\n        - ``bzip2`` or ``bz2`` for bzip2 compression\n        - ``xz`` or ``lzma`` for XZ compression (requires `xz-utils`_, as well\n          as the ``lzma`` module from Python 3.3, available in Python 2 and\n          Python 3.0-3.2 as `backports.lzma`_)\n\n        This parameter can be omitted and Salt will attempt to determine the\n        compression type by examining the filename passed in the ``path``\n        parameter.\n\n        .. note::\n            Since the Docker API does not support ``docker save``, compression\n            will be a bit slower with this function than with\n            :py:func:`docker.export <salt.modules.dockermod.export>` since the\n            image(s) will first be saved and then the compression done\n            afterwards.\n\n        .. _`xz-utils`: http://tukaani.org/xz/\n        .. _`backports.lzma`: https://pypi.python.org/pypi/backports.lzma\n\n    push : False\n        If ``True``, the container will be pushed to the master using\n        :py:func:`cp.push <salt.modules.cp.push>`.\n\n        .. note::\n\n            This requires :conf_master:`file_recv` to be set to ``True`` on the\n            Master.\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``Path`` - Path of the file that was saved\n    - ``Push`` - Reports whether or not the file was successfully pushed to the\n      Master\n\n      *(Only present if push=True)*\n    - ``Size`` - Size of the file, in bytes\n    - ``Size_Human`` - Size of the file, in human-readable units\n    - ``Time_Elapsed`` - Time in seconds taken to perform the save\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.save centos:7 /tmp/cent7.tar\n        salt myminion docker.save 0123456789ab cdef01234567 /tmp/saved.tar\n    '
    err = f"Path '{path}' is not absolute"
    try:
        if not os.path.isabs(path):
            raise SaltInvocationError(err)
    except AttributeError:
        raise SaltInvocationError(err)
    if os.path.exists(path) and (not overwrite):
        raise CommandExecutionError(f'{path} already exists')
    if compression is None:
        if path.endswith('.tar.gz') or path.endswith('.tgz'):
            compression = 'gzip'
        elif path.endswith('.tar.bz2') or path.endswith('.tbz2'):
            compression = 'bzip2'
        elif path.endswith('.tar.xz') or path.endswith('.txz'):
            if HAS_LZMA:
                compression = 'xz'
            else:
                raise CommandExecutionError('XZ compression unavailable. Install the backports.lzma module and xz-utils to enable XZ compression.')
    elif compression == 'gz':
        compression = 'gzip'
    elif compression == 'bz2':
        compression = 'bzip2'
    elif compression == 'lzma':
        compression = 'xz'
    if compression and compression not in ('gzip', 'bzip2', 'xz'):
        raise SaltInvocationError(f"Invalid compression type '{compression}'")
    parent_dir = os.path.dirname(path)
    if not os.path.isdir(parent_dir):
        if not makedirs:
            raise CommandExecutionError("Parent dir '{}' of destination path does not exist. Use makedirs=True to create it.".format(parent_dir))
    if compression:
        saved_path = __utils__['files.mkstemp']()
    else:
        saved_path = path
    image_to_save = name if name in inspect_image(name)['RepoTags'] else inspect_image(name)['Id']
    cmd = ['docker', 'save', '-o', saved_path, image_to_save]
    time_started = time.time()
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = f'Failed to save image(s) to {path}'
        if result['stderr']:
            err += ': {}'.format(result['stderr'])
        raise CommandExecutionError(err)
    if compression:
        if compression == 'gzip':
            try:
                out = gzip.open(path, 'wb')
            except OSError as exc:
                raise CommandExecutionError(f'Unable to open {path} for writing: {exc}')
        elif compression == 'bzip2':
            compressor = bz2.BZ2Compressor()
        elif compression == 'xz':
            compressor = lzma.LZMACompressor()
        try:
            with __utils__['files.fopen'](saved_path, 'rb') as uncompressed:
                if compression != 'gzip':
                    out = __utils__['files.fopen'](path, 'wb')
                buf = None
                while buf != '':
                    buf = uncompressed.read(4096)
                    if buf:
                        if compression in ('bzip2', 'xz'):
                            data = compressor.compress(buf)
                            if data:
                                out.write(data)
                        else:
                            out.write(buf)
                if compression in ('bzip2', 'xz'):
                    data = compressor.flush()
                    if data:
                        out.write(data)
                out.flush()
        except Exception as exc:
            try:
                os.remove(path)
            except OSError:
                pass
            raise CommandExecutionError(f'Error occurred during image save: {exc}')
        finally:
            try:
                os.remove(saved_path)
            except OSError:
                pass
            out.close()
    ret = {'Time_Elapsed': time.time() - time_started}
    ret['Path'] = path
    ret['Size'] = os.stat(path).st_size
    ret['Size_Human'] = _size_fmt(ret['Size'])
    if kwargs.get('push', False):
        ret['Push'] = __salt__['cp.push'](path)
    return ret

def tag_(name, repository, tag='latest', force=False):
    if False:
        while True:
            i = 10
    '\n    .. versionchanged:: 2018.3.0\n        The repository and tag must now be passed separately using the\n        ``repository`` and ``tag`` arguments, rather than together in the (now\n        deprecated) ``image`` argument.\n\n    Tag an image into a repository and return ``True``. If the tag was\n    unsuccessful, an error will be raised.\n\n    name\n        ID of image\n\n    repository\n        Repository name for the image to be built\n\n        .. versionadded:: 2018.3.0\n\n    tag : latest\n        Tag name for the image to be built\n\n        .. versionadded:: 2018.3.0\n\n    image\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n    force : False\n        Force apply tag\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.tag 0123456789ab myrepo/mycontainer mytag\n    '
    if not isinstance(repository, str):
        repository = str(repository)
    if not isinstance(tag, str):
        tag = str(tag)
    image_id = inspect_image(name)['Id']
    response = _client_wrapper('tag', image_id, repository=repository, tag=tag, force=force)
    _clear_context()
    return response

def networks(names=None, ids=None):
    if False:
        while True:
            i = 10
    '\n    .. versionchanged:: 2017.7.0\n        The ``names`` and ``ids`` can be passed as a comma-separated list now,\n        as well as a Python list.\n    .. versionchanged:: 2018.3.0\n        The ``Containers`` key for each network is no longer always empty.\n\n    List existing networks\n\n    names\n        Filter by name\n\n    ids\n        Filter by id\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.networks names=network-web\n        salt myminion docker.networks ids=1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    if names is not None:
        names = __utils__['args.split_input'](names)
    if ids is not None:
        ids = __utils__['args.split_input'](ids)
    response = _client_wrapper('networks', names=names, ids=ids)
    for (idx, netinfo) in enumerate(response):
        try:
            containers = inspect_network(netinfo['Id'])['Containers']
        except Exception:
            continue
        else:
            if containers:
                response[idx]['Containers'] = containers
    return response

def create_network(name, skip_translate=None, ignore_collisions=False, validate_ip_addrs=True, client_timeout=salt.utils.dockermod.CLIENT_TIMEOUT, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2018.3.0\n        Support added for network configuration options other than ``driver``\n        and ``driver_opts``, as well as IPAM configuration.\n\n    Create a new network\n\n    .. note::\n        This function supports all arguments for network and IPAM pool\n        configuration which are available for the release of docker-py\n        installed on the minion. For that reason, the arguments described below\n        in the :ref:`NETWORK CONFIGURATION ARGUMENTS\n        <salt-modules-dockermod-create-network-netconf>` and :ref:`IP ADDRESS\n        MANAGEMENT (IPAM) <salt-modules-dockermod-create-network-ipam>`\n        sections may not accurately reflect what is available on the minion.\n        The :py:func:`docker.get_client_args\n        <salt.modules.dockermod.get_client_args>` function can be used to check\n        the available arguments for the installed version of docker-py (they\n        are found in the ``network_config`` and ``ipam_config`` sections of the\n        return data), but Salt will not prevent a user from attempting to use\n        an argument which is unsupported in the release of Docker which is\n        installed. In those cases, network creation be attempted but will fail.\n\n    name\n        Network name\n\n    skip_translate\n        This function translates Salt CLI or SLS input into the format which\n        docker-py expects. However, in the event that Salt\'s translation logic\n        fails (due to potential changes in the Docker Remote API, or to bugs in\n        the translation code), this argument can be used to exert granular\n        control over which arguments are translated and which are not.\n\n        Pass this argument as a comma-separated list (or Python list) of\n        arguments, and translation for each passed argument name will be\n        skipped. Alternatively, pass ``True`` and *all* translation will be\n        skipped.\n\n        Skipping tranlsation allows for arguments to be formatted directly in\n        the format which docker-py expects. This allows for API changes and\n        other issues to be more easily worked around. See the following links\n        for more information:\n\n        - `docker-py Low-level API`_\n        - `Docker Engine API`_\n\n        .. versionadded:: 2018.3.0\n\n    ignore_collisions : False\n        Since many of docker-py\'s arguments differ in name from their CLI\n        counterparts (with which most Docker users are more familiar), Salt\n        detects usage of these and aliases them to the docker-py version of\n        that argument. However, if both the alias and the docker-py version of\n        the same argument (e.g. ``options`` and ``driver_opts``) are used, an error\n        will be raised. Set this argument to ``True`` to suppress these errors\n        and keep the docker-py version of the argument.\n\n        .. versionadded:: 2018.3.0\n\n    validate_ip_addrs : True\n        For parameters which accept IP addresses as input, IP address\n        validation will be performed. To disable, set this to ``False``\n\n        .. note::\n            When validating subnets, whether or not the IP portion of the\n            subnet is a valid subnet boundary will not be checked. The IP will\n            portion will be validated, and the subnet size will be checked to\n            confirm it is a valid number (1-32 for IPv4, 1-128 for IPv6).\n\n        .. versionadded:: 2018.3.0\n\n    .. _salt-modules-dockermod-create-network-netconf:\n\n    **NETWORK CONFIGURATION ARGUMENTS**\n\n    driver\n        Network driver\n\n        Example: ``driver=macvlan``\n\n    driver_opts (or *driver_opt*, or *options*)\n        Options for the network driver. Either a dictionary of option names and\n        values or a Python list of strings in the format ``varname=value``.\n\n        Examples:\n\n        - ``driver_opts=\'macvlan_mode=bridge,parent=eth0\'``\n        - ``driver_opts="[\'macvlan_mode=bridge\', \'parent=eth0\']"``\n        - ``driver_opts="{\'macvlan_mode\': \'bridge\', \'parent\': \'eth0\'}"``\n\n    check_duplicate : True\n        If ``True``, checks for networks with duplicate names. Since networks\n        are primarily keyed based on a random ID and not on the name, and\n        network name is strictly a user-friendly alias to the network which is\n        uniquely identified using ID, there is no guaranteed way to check for\n        duplicates. This option providess a best effort, checking for any\n        networks which have the same name, but it is not guaranteed to catch\n        all name collisions.\n\n        Example: ``check_duplicate=False``\n\n    internal : False\n        If ``True``, restricts external access to the network\n\n        Example: ``internal=True``\n\n    labels\n        Add metadata to the network. Labels can be set both with and without\n        values:\n\n        Examples (*with* values):\n\n        - ``labels="label1=value1,label2=value2"``\n        - ``labels="[\'label1=value1\', \'label2=value2\']"``\n        - ``labels="{\'label1\': \'value1\', \'label2\': \'value2\'}"``\n\n        Examples (*without* values):\n\n        - ``labels=label1,label2``\n        - ``labels="[\'label1\', \'label2\']"``\n\n    enable_ipv6 (or *ipv6*) : False\n        Enable IPv6 on the network\n\n        Example: ``enable_ipv6=True``\n\n        .. note::\n            While it should go without saying, this argument must be set to\n            ``True`` to :ref:`configure an IPv6 subnet\n            <salt-states-docker-network-present-ipam>`. Also, if this option is\n            turned on without an IPv6 subnet explicitly configured, you will\n            get an error unless you have set up a fixed IPv6 subnet. Consult\n            the `Docker IPv6 docs`_ for information on how to do this.\n\n            .. _`Docker IPv6 docs`: https://docs.docker.com/v17.09/engine/userguide/networking/default_network/ipv6/\n\n    attachable : False\n        If ``True``, and the network is in the global scope, non-service\n        containers on worker nodes will be able to connect to the network.\n\n        Example: ``attachable=True``\n\n        .. note::\n            While support for this option was added in API version 1.24, its\n            value was not added to the inpsect results until API version 1.26.\n            The version of Docker which is available for CentOS 7 runs API\n            version 1.24, meaning that while Salt can pass this argument to the\n            API, it has no way of knowing the value of this config option in an\n            existing Docker network.\n\n    scope\n        Specify the network\'s scope (``local``, ``global`` or ``swarm``)\n\n        Example: ``scope=local``\n\n    ingress : False\n        If ``True``, create an ingress network which provides the routing-mesh in\n        swarm mode\n\n        Example: ``ingress=True``\n\n    .. _salt-modules-dockermod-create-network-ipam:\n\n    **IP ADDRESS MANAGEMENT (IPAM)**\n\n    This function supports networks with either IPv4, or both IPv4 and IPv6. If\n    configuring IPv4, then you can pass the IPAM arguments as shown below, as\n    individual arguments on the Salt CLI. However, if configuring IPv4 and\n    IPv6, the arguments must be passed as a list of dictionaries, in the\n    ``ipam_pools`` argument. See the **CLI Examples** below. `These docs`_ also\n    have more information on these arguments.\n\n    .. _`These docs`: http://docker-py.readthedocs.io/en/stable/api.html#docker.types.IPAMPool\n\n    *IPAM ARGUMENTS*\n\n    ipam_driver\n        IPAM driver to use, if different from the default one\n\n        Example: ``ipam_driver=foo``\n\n    ipam_opts\n        Options for the IPAM driver. Either a dictionary of option names\n        and values or a Python list of strings in the format\n        ``varname=value``.\n\n        Examples:\n\n        - ``ipam_opts=\'foo=bar,baz=qux\'``\n        - ``ipam_opts="[\'foo=bar\', \'baz=quz\']"``\n        - ``ipam_opts="{\'foo\': \'bar\', \'baz\': \'qux\'}"``\n\n    *IPAM POOL ARGUMENTS*\n\n    subnet\n        Subnet in CIDR format that represents a network segment\n\n        Example: ``subnet=192.168.50.0/25``\n\n    iprange (or *ip_range*)\n        Allocate container IP from a sub-range within the subnet\n\n        Subnet in CIDR format that represents a network segment\n\n        Example: ``iprange=192.168.50.64/26``\n\n    gateway\n        IPv4 gateway for the master subnet\n\n        Example: ``gateway=192.168.50.1``\n\n    aux_addresses (or *aux_address*)\n        A dictionary of mapping container names to IP addresses which should be\n        allocated for them should they connect to the network. Either a\n        dictionary of option names and values or a Python list of strings in\n        the format ``host=ipaddr``.\n\n        Examples:\n\n        - ``aux_addresses=\'foo.bar.tld=192.168.50.10,hello.world.tld=192.168.50.11\'``\n        - ``aux_addresses="[\'foo.bar.tld=192.168.50.10\', \'hello.world.tld=192.168.50.11\']"``\n        - ``aux_addresses="{\'foo.bar.tld\': \'192.168.50.10\', \'hello.world.tld\': \'192.168.50.11\'}"``\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.create_network web_network driver=bridge\n        # IPv4\n        salt myminion docker.create_network macvlan_network driver=macvlan driver_opts="{\'parent\':\'eth0\'}" gateway=172.20.0.1 subnet=172.20.0.0/24\n        # IPv4 and IPv6\n        salt myminion docker.create_network mynet ipam_pools=\'[{"subnet": "10.0.0.0/24", "gateway": "10.0.0.1"}, {"subnet": "fe3f:2180:26:1::60/123", "gateway": "fe3f:2180:26:1::61"}]\'\n    '
    kwargs = __utils__['docker.translate_input'](salt.utils.dockermod.translate.network, skip_translate=skip_translate, ignore_collisions=ignore_collisions, validate_ip_addrs=validate_ip_addrs, **__utils__['args.clean_kwargs'](**kwargs))
    if 'ipam' not in kwargs:
        ipam_kwargs = {}
        for key in [x for x in ['ipam_driver', 'ipam_opts'] + get_client_args('ipam_config')['ipam_config'] if x in kwargs]:
            ipam_kwargs[key] = kwargs.pop(key)
        ipam_pools = kwargs.pop('ipam_pools', ())
        if ipam_pools or ipam_kwargs:
            kwargs['ipam'] = __utils__['docker.create_ipam_config'](*ipam_pools, **ipam_kwargs)
    response = _client_wrapper('create_network', name, **kwargs)
    _clear_context()
    return response

def remove_network(network_id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove a network\n\n    network_id\n        Network name or ID\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.remove_network mynet\n        salt myminion docker.remove_network 1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    response = _client_wrapper('remove_network', network_id)
    _clear_context()
    return True

def inspect_network(network_id):
    if False:
        i = 10
        return i + 15
    '\n    Inspect Network\n\n    network_id\n        ID of network\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.inspect_network 1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    response = _client_wrapper('inspect_network', network_id)
    _clear_context()
    return response

def connect_container_to_network(container, net_id, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2015.8.3\n    .. versionchanged:: 2017.7.0\n        Support for ``ipv4_address`` argument added\n    .. versionchanged:: 2018.3.0\n        All arguments are now passed through to\n        `connect_container_to_network()`_, allowing for any new arguments added\n        to this function to be supported automagically.\n\n    Connect container to network. See the `connect_container_to_network()`_\n    docs for information on supported arguments.\n\n    container\n        Container name or ID\n\n    net_id\n        Network name or ID\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.connect_container_to_network web-1 mynet\n        salt myminion docker.connect_container_to_network web-1 mynet ipv4_address=10.20.0.10\n        salt myminion docker.connect_container_to_network web-1 1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    log.debug("Connecting container '%s' to network '%s' with the following configuration: %s", container, net_id, kwargs)
    response = _client_wrapper('connect_container_to_network', container, net_id, **kwargs)
    log.debug("Successfully connected container '%s' to network '%s'", container, net_id)
    _clear_context()
    return True

def disconnect_container_from_network(container, network_id):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2015.8.3\n\n    Disconnect container from network\n\n    container\n        Container name or ID\n\n    network_id\n        Network name or ID\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.disconnect_container_from_network web-1 mynet\n        salt myminion docker.disconnect_container_from_network web-1 1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    log.debug("Disconnecting container '%s' from network '%s'", container, network_id)
    response = _client_wrapper('disconnect_container_from_network', container, network_id)
    log.debug("Successfully disconnected container '%s' from network '%s'", container, network_id)
    _clear_context()
    return True

def disconnect_all_containers_from_network(network_id):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2018.3.0\n\n    Runs :py:func:`docker.disconnect_container_from_network\n    <salt.modules.dockermod.disconnect_container_from_network>` on all\n    containers connected to the specified network, and returns the names of all\n    containers that were disconnected.\n\n    network_id\n        Network name or ID\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.disconnect_all_containers_from_network mynet\n        salt myminion docker.disconnect_all_containers_from_network 1f9d2454d0872b68dd9e8744c6e7a4c66b86f10abaccc21e14f7f014f729b2bc\n    '
    connected_containers = connected(network_id)
    ret = []
    failed = []
    for cname in connected_containers:
        try:
            disconnect_container_from_network(cname, network_id)
            ret.append(cname)
        except CommandExecutionError as exc:
            msg = exc.__str__()
            if '404' not in msg:
                failed.append(msg)
    if failed:
        raise CommandExecutionError('One or more containers failed to be removed', info={'removed': ret, 'errors': failed})
    return ret

def volumes(filters=None):
    if False:
        while True:
            i = 10
    '\n    List existing volumes\n\n    .. versionadded:: 2015.8.4\n\n    filters\n      There is one available filter: dangling=true\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.volumes filters="{\'dangling\': True}"\n    '
    response = _client_wrapper('volumes', filters=filters)
    return response

def create_volume(name, driver=None, driver_opts=None):
    if False:
        while True:
            i = 10
    '\n    Create a new volume\n\n    .. versionadded:: 2015.8.4\n\n    name\n        name of volume\n\n    driver\n        Driver of the volume\n\n    driver_opts\n        Options for the driver volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.create_volume my_volume driver=local\n    '
    response = _client_wrapper('create_volume', name, driver=driver, driver_opts=driver_opts)
    _clear_context()
    return response

def remove_volume(name):
    if False:
        return 10
    '\n    Remove a volume\n\n    .. versionadded:: 2015.8.4\n\n    name\n        Name of volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.remove_volume my_volume\n    '
    response = _client_wrapper('remove_volume', name)
    _clear_context()
    return True

def inspect_volume(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inspect Volume\n\n    .. versionadded:: 2015.8.4\n\n    name\n      Name of volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.inspect_volume my_volume\n    '
    response = _client_wrapper('inspect_volume', name)
    _clear_context()
    return response

@_refresh_mine_cache
def kill(name):
    if False:
        while True:
            i = 10
    '\n    Kill all processes in a running container instead of performing a graceful\n    shutdown\n\n    name\n        Container name or ID\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``comment`` - Only present if the container cannot be killed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.kill mycontainer\n    '
    return _change_state(name, 'kill', 'stopped')

@_refresh_mine_cache
def pause(name):
    if False:
        print('Hello World!')
    '\n    Pauses a container\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``comment`` - Only present if the container cannot be paused\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.pause mycontainer\n    '
    orig_state = state(name)
    if orig_state == 'stopped':
        return {'result': False, 'state': {'old': orig_state, 'new': orig_state}, 'comment': f"Container '{name}' is stopped, cannot pause"}
    return _change_state(name, 'pause', 'paused')
freeze = salt.utils.functools.alias_function(pause, 'freeze')

def restart(name, timeout=10):
    if False:
        i = 10
        return i + 15
    '\n    Restarts a container\n\n    name\n        Container name or ID\n\n    timeout : 10\n        Timeout in seconds after which the container will be killed (if it has\n        not yet gracefully shut down)\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``restarted`` - If restart was successful, this key will be present and\n      will be set to ``True``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.restart mycontainer\n        salt myminion docker.restart mycontainer timeout=20\n    '
    ret = _change_state(name, 'restart', 'running', timeout=timeout)
    if ret['result']:
        ret['restarted'] = True
    return ret

@_refresh_mine_cache
def signal_(name, signal):
    if False:
        print('Hello World!')
    '\n    Send a signal to a container. Signals can be either strings or numbers, and\n    are defined in the **Standard Signals** section of the ``signal(7)``\n    manpage. Run ``man 7 signal`` on a Linux host to browse this manpage.\n\n    name\n        Container name or ID\n\n    signal\n        Signal to send to container\n\n    **RETURN DATA**\n\n    If the signal was successfully sent, ``True`` will be returned. Otherwise,\n    an error will be raised.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.signal mycontainer SIGHUP\n    '
    _client_wrapper('kill', name, signal=signal)
    return True

@_refresh_mine_cache
def start_(name):
    if False:
        return 10
    '\n    Start a container\n\n    name\n        Container name or ID\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``comment`` - Only present if the container cannot be started\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.start mycontainer\n    '
    orig_state = state(name)
    if orig_state == 'paused':
        return {'result': False, 'state': {'old': orig_state, 'new': orig_state}, 'comment': f"Container '{name}' is paused, cannot start"}
    return _change_state(name, 'start', 'running')

@_refresh_mine_cache
def stop(name, timeout=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Stops a running container\n\n    name\n        Container name or ID\n\n    unpause : False\n        If ``True`` and the container is paused, it will be unpaused before\n        attempting to stop the container.\n\n    timeout\n        Timeout in seconds after which the container will be killed (if it has\n        not yet gracefully shut down)\n\n        .. versionchanged:: 2017.7.0\n            If this argument is not passed, then the container's configuration\n            will be checked. If the container was created using the\n            ``stop_timeout`` argument, then the configured timeout will be\n            used, otherwise the timeout will be 10 seconds.\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``comment`` - Only present if the container can not be stopped\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.stop mycontainer\n        salt myminion docker.stop mycontainer unpause=True\n        salt myminion docker.stop mycontainer timeout=20\n    "
    if timeout is None:
        try:
            timeout = inspect_container(name)['Config']['StopTimeout']
        except KeyError:
            timeout = salt.utils.dockermod.SHUTDOWN_TIMEOUT
    orig_state = state(name)
    if orig_state == 'paused':
        if kwargs.get('unpause', False):
            unpause_result = _change_state(name, 'unpause', 'running')
            if unpause_result['result'] is False:
                unpause_result['comment'] = "Failed to unpause container '{}'".format(name)
                return unpause_result
        else:
            return {'result': False, 'state': {'old': orig_state, 'new': orig_state}, 'comment': "Container '{}' is paused, run with unpause=True to unpause before stopping".format(name)}
    ret = _change_state(name, 'stop', 'stopped', timeout=timeout)
    ret['state']['old'] = orig_state
    return ret

@_refresh_mine_cache
def unpause(name):
    if False:
        while True:
            i = 10
    '\n    Unpauses a container\n\n    name\n        Container name or ID\n\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``comment`` - Only present if the container can not be unpaused\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.pause mycontainer\n    '
    orig_state = state(name)
    if orig_state == 'stopped':
        return {'result': False, 'state': {'old': orig_state, 'new': orig_state}, 'comment': f"Container '{name}' is stopped, cannot unpause"}
    return _change_state(name, 'unpause', 'running')
unfreeze = salt.utils.functools.alias_function(unpause, 'unfreeze')

def wait(name, ignore_already_stopped=False, fail_on_exit_status=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wait for the container to exit gracefully, and return its exit code\n\n    .. note::\n\n        This function will block until the container is stopped.\n\n    name\n        Container name or ID\n\n    ignore_already_stopped\n        Boolean flag that prevents execution to fail, if a container\n        is already stopped.\n\n    fail_on_exit_status\n        Boolean flag to report execution as failure if ``exit_status``\n        is different than 0.\n\n    **RETURN DATA**\n\n    A dictionary will be returned, containing the following keys:\n\n    - ``status`` - A dictionary showing the prior state of the container as\n      well as the new state\n    - ``result`` - A boolean noting whether or not the action was successful\n    - ``exit_status`` - Exit status for the container\n    - ``comment`` - Only present if the container is already stopped\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.wait mycontainer\n    '
    try:
        pre = state(name)
    except CommandExecutionError:
        return {'result': ignore_already_stopped, 'comment': f"Container '{name}' absent"}
    already_stopped = pre == 'stopped'
    response = _client_wrapper('wait', name)
    _clear_context()
    try:
        post = state(name)
    except CommandExecutionError:
        post = None
    if already_stopped:
        success = ignore_already_stopped
    elif post == 'stopped':
        success = True
    else:
        success = False
    result = {'result': success, 'state': {'old': pre, 'new': post}, 'exit_status': response}
    if already_stopped:
        result['comment'] = f"Container '{name}' already stopped"
    if fail_on_exit_status and result['result']:
        result['result'] = result['exit_status'] == 0
    return result

def prune(containers=False, networks=False, images=False, build=False, volumes=False, system=None, **filters):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Prune Docker's various subsystems\n\n    .. note::\n        This requires docker-py version 2.1.0 or later.\n\n    containers : False\n        If ``True``, prunes stopped containers (documentation__)\n\n        .. __: https://docs.docker.com/engine/reference/commandline/container_prune/#filtering\n\n    images : False\n        If ``True``, prunes unused images (documentation__)\n\n        .. __: https://docs.docker.com/engine/reference/commandline/image_prune/#filtering\n\n    networks : False\n        If ``False``, prunes unreferenced networks (documentation__)\n\n        .. __: https://docs.docker.com/engine/reference/commandline/network_prune/#filtering)\n\n    build : False\n        If ``True``, clears the builder cache\n\n        .. note::\n            Only supported in Docker 17.07.x and newer. Additionally, filters\n            do not apply to this argument.\n\n    volumes : False\n        If ``True``, prunes unreferenced volumes (documentation__)\n\n        .. __: https://docs.docker.com/engine/reference/commandline/volume_prune/\n\n    system\n        If ``True``, prunes containers, images, networks, and builder cache.\n        Assumed to be ``True`` if none of ``containers``, ``images``,\n        ``networks``, or ``build`` are set to ``True``.\n\n        .. note::\n            ``volumes=True`` must still be used to prune volumes\n\n    filters\n        - ``dangling=True`` (images only) - remove only dangling images\n\n        - ``until=<timestamp>`` - only remove objects created before given\n          timestamp. Not applicable to volumes. See the documentation links\n          above for examples of valid time expressions.\n\n        - ``label`` - only remove objects matching the label expression. Valid\n          expressions include ``labelname`` or ``labelname=value``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion docker.prune system=True\n        salt myminion docker.prune system=True until=12h\n        salt myminion docker.prune images=True dangling=True\n        salt myminion docker.prune images=True label=foo,bar=baz\n    "
    if system is None and (not any((containers, images, networks, build))):
        system = True
    filters = __utils__['args.clean_kwargs'](**filters)
    for fname in list(filters):
        if not isinstance(filters[fname], bool):
            filters[fname] = salt.utils.args.split_input(filters[fname])
    ret = {}
    if system or containers:
        ret['containers'] = _client_wrapper('prune_containers', filters=filters)
    if system or images:
        ret['images'] = _client_wrapper('prune_images', filters=filters)
    if system or networks:
        ret['networks'] = _client_wrapper('prune_networks', filters=filters)
    if system or build:
        try:
            ret['build'] = _client_wrapper('prune_build', filters=filters)
        except SaltInvocationError:
            ret['build'] = _client_wrapper('_result', _client_wrapper('_post', _client_wrapper('_url', '/build/prune')), True)
    if volumes:
        ret['volumes'] = _client_wrapper('prune_volumes', filters=filters)
    return ret

@_refresh_mine_cache
def _run(name, cmd, exec_driver=None, output=None, stdin=None, python_shell=True, output_loglevel='debug', ignore_retcode=False, use_vt=False, keep_env=None):
    if False:
        while True:
            i = 10
    '\n    Common logic for docker.run functions\n    '
    if exec_driver is None:
        exec_driver = _get_exec_driver()
    ret = __salt__['container_resource.run'](name, cmd, container_type=__virtualname__, exec_driver=exec_driver, output=output, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt, keep_env=keep_env)
    if output in (None, 'all'):
        return ret
    else:
        return ret[output]

@_refresh_mine_cache
def _script(name, source, saltenv='base', args=None, template=None, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', ignore_retcode=False, use_vt=False, keep_env=None):
    if False:
        return 10
    '\n    Common logic to run a script on a container\n    '

    def _cleanup_tempfile(path):
        if False:
            i = 10
            return i + 15
        '\n        Remove the tempfile allocated for the script\n        '
        try:
            os.remove(path)
        except OSError as exc:
            log.error("cmd.script: Unable to clean tempfile '%s': %s", path, exc)
    path = __utils__['files.mkstemp'](dir='/tmp', prefix='salt', suffix=os.path.splitext(source)[1])
    if template:
        fn_ = __salt__['cp.get_template'](source, path, template, saltenv)
        if not fn_:
            _cleanup_tempfile(path)
            return {'pid': 0, 'retcode': 1, 'stdout': '', 'stderr': '', 'cache_error': True}
    else:
        fn_ = __salt__['cp.cache_file'](source, saltenv)
        if not fn_:
            _cleanup_tempfile(path)
            return {'pid': 0, 'retcode': 1, 'stdout': '', 'stderr': '', 'cache_error': True}
        shutil.copyfile(fn_, path)
    if exec_driver is None:
        exec_driver = _get_exec_driver()
    copy_to(name, path, path, exec_driver=exec_driver)
    run(name, 'chmod 700 ' + path)
    ret = run_all(name, path + ' ' + str(args) if args else path, exec_driver=exec_driver, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt, keep_env=keep_env)
    _cleanup_tempfile(path)
    run(name, 'rm ' + path)
    return ret

def retcode(name, cmd, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        while True:
            i = 10
    "\n    Run :py:func:`cmd.retcode <salt.modules.cmdmod.retcode>` within a container\n\n    name\n        Container name or ID in which to run the command\n\n    cmd\n        Command to run\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.retcode mycontainer 'ls -l /etc'\n    "
    return _run(name, cmd, exec_driver=exec_driver, output='retcode', stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run(name, cmd, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        while True:
            i = 10
    "\n    Run :py:func:`cmd.run <salt.modules.cmdmod.run>` within a container\n\n    name\n        Container name or ID in which to run the command\n\n    cmd\n        Command to run\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.run mycontainer 'ls -l /etc'\n    "
    return _run(name, cmd, exec_driver=exec_driver, output=None, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_all(name, cmd, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        while True:
            i = 10
    "\n    Run :py:func:`cmd.run_all <salt.modules.cmdmod.run_all>` within a container\n\n    .. note::\n\n        While the command is run within the container, it is initiated from the\n        host. Therefore, the PID in the return dict is from the host, not from\n        the container.\n\n    name\n        Container name or ID in which to run the command\n\n    cmd\n        Command to run\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.run_all mycontainer 'ls -l /etc'\n    "
    return _run(name, cmd, exec_driver=exec_driver, output='all', stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_stderr(name, cmd, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run :py:func:`cmd.run_stderr <salt.modules.cmdmod.run_stderr>` within a\n    container\n\n    name\n        Container name or ID in which to run the command\n\n    cmd\n        Command to run\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.run_stderr mycontainer 'ls -l /etc'\n    "
    return _run(name, cmd, exec_driver=exec_driver, output='stderr', stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def run_stdout(name, cmd, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', use_vt=False, ignore_retcode=False, keep_env=None):
    if False:
        return 10
    "\n    Run :py:func:`cmd.run_stdout <salt.modules.cmdmod.run_stdout>` within a\n    container\n\n    name\n        Container name or ID in which to run the command\n\n    cmd\n        Command to run\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the command\n\n    output_loglevel : debug\n        Level at which to log the output from the command. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack's utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container's host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.run_stdout mycontainer 'ls -l /etc'\n    "
    return _run(name, cmd, exec_driver=exec_driver, output='stdout', stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, use_vt=use_vt, ignore_retcode=ignore_retcode, keep_env=keep_env)

def script(name, source, saltenv='base', args=None, template=None, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', ignore_retcode=False, use_vt=False, keep_env=None):
    if False:
        while True:
            i = 10
    '\n    Run :py:func:`cmd.script <salt.modules.cmdmod.script>` within a container\n\n    .. note::\n\n        While the command is run within the container, it is initiated from the\n        host. Therefore, the PID in the return dict is from the host, not from\n        the container.\n\n    name\n        Container name or ID\n\n    source\n        Path to the script. Can be a local path on the Minion or a remote file\n        from the Salt fileserver.\n\n    args\n        A string containing additional command-line options to pass to the\n        script.\n\n    template : None\n        Templating engine to use on the script before running.\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the script\n\n    output_loglevel : debug\n        Level at which to log the output from the script. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack\'s utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container\'s host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.script mycontainer salt://docker_script.py\n        salt myminion docker.script mycontainer salt://scripts/runme.sh \'arg1 arg2 "arg 3"\'\n        salt myminion docker.script mycontainer salt://scripts/runme.sh stdin=\'one\\ntwo\\nthree\\nfour\\nfive\\n\' output_loglevel=quiet\n    '
    return _script(name, source, saltenv=saltenv, args=args, template=template, exec_driver=exec_driver, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt, keep_env=keep_env)

def script_retcode(name, source, saltenv='base', args=None, template=None, exec_driver=None, stdin=None, python_shell=True, output_loglevel='debug', ignore_retcode=False, use_vt=False, keep_env=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run :py:func:`cmd.script_retcode <salt.modules.cmdmod.script_retcode>`\n    within a container\n\n    name\n        Container name or ID\n\n    source\n        Path to the script. Can be a local path on the Minion or a remote file\n        from the Salt fileserver.\n\n    args\n        A string containing additional command-line options to pass to the\n        script.\n\n    template : None\n        Templating engine to use on the script before running.\n\n    exec_driver : None\n        If not passed, the execution driver will be detected as described\n        :ref:`above <docker-execution-driver>`.\n\n    stdin : None\n        Standard input to be used for the script\n\n    output_loglevel : debug\n        Level at which to log the output from the script. Set to ``quiet`` to\n        suppress logging.\n\n    use_vt : False\n        Use SaltStack\'s utils.vt to stream output to console.\n\n    keep_env : None\n        If not passed, only a sane default PATH environment variable will be\n        set. If ``True``, all environment variables from the container\'s host\n        will be kept. Otherwise, a comma-separated list (or Python list) of\n        environment variable names can be passed, and those environment\n        variables will be kept.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.script_retcode mycontainer salt://docker_script.py\n        salt myminion docker.script_retcode mycontainer salt://scripts/runme.sh \'arg1 arg2 "arg 3"\'\n        salt myminion docker.script_retcode mycontainer salt://scripts/runme.sh stdin=\'one\\ntwo\\nthree\\nfour\\nfive\\n\' output_loglevel=quiet\n    '
    return _script(name, source, saltenv=saltenv, args=args, template=template, exec_driver=exec_driver, stdin=stdin, python_shell=python_shell, output_loglevel=output_loglevel, ignore_retcode=ignore_retcode, use_vt=use_vt, keep_env=keep_env)['retcode']

def _generate_tmp_path():
    if False:
        return 10
    return os.path.join('/tmp', f'salt.docker.{uuid.uuid4().hex[:6]}')

def _prepare_trans_tar(name, sls_opts, mods=None, pillar=None, extra_filerefs=''):
    if False:
        print('Hello World!')
    '\n    Prepares a self contained tarball that has the state\n    to be applied in the container\n    '
    chunks = _compile_state(sls_opts, mods)
    refs = salt.client.ssh.state.lowstate_file_refs(chunks, extra_filerefs)
    with salt.fileclient.get_file_client(__opts__) as fileclient:
        return salt.client.ssh.state.prep_trans_tar(fileclient, chunks, refs, pillar, name)

def _compile_state(sls_opts, mods=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates the chunks of lowdata from the list of modules\n    '
    with HighState(sls_opts) as st_:
        if not mods:
            return st_.compile_low_chunks()
        (high_data, errors) = st_.render_highstate({sls_opts['saltenv']: mods})
        (high_data, ext_errors) = st_.state.reconcile_extend(high_data)
        errors += ext_errors
        errors += st_.state.verify_high(high_data)
        if errors:
            return errors
        (high_data, req_in_errors) = st_.state.requisite_in(high_data)
        errors += req_in_errors
        high_data = st_.state.apply_exclude(high_data)
        if errors:
            return errors
        return st_.state.compile_high_data(high_data)

def call(name, function, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Executes a Salt function inside a running container\n\n    .. versionadded:: 2016.11.0\n\n    The container does not need to have Salt installed, but Python is required.\n\n    name\n        Container name or ID\n\n    function\n        Salt execution module function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.call test.ping\n        salt myminion test.arg arg1 arg2 key1=val1\n        salt myminion dockerng.call compassionate_mirzakhani test.arg arg1 arg2 key1=val1\n\n    '
    thin_dest_path = _generate_tmp_path()
    mkdirp_thin_argv = ['mkdir', '-p', thin_dest_path]
    ret = run_all(name, subprocess.list2cmdline(mkdirp_thin_argv))
    if ret['retcode'] != 0:
        return {'result': False, 'comment': ret['stderr']}
    if function is None:
        raise CommandExecutionError('Missing function parameter')
    thin_path = __utils__['thin.gen_thin'](__opts__['cachedir'], extra_mods=__salt__['config.option']('thin_extra_mods', ''), so_mods=__salt__['config.option']('thin_so_mods', ''))
    ret = copy_to(name, thin_path, os.path.join(thin_dest_path, os.path.basename(thin_path)))
    pycmds = ('python3', '/usr/libexec/platform-python')
    container_python_bin = None
    for py_cmd in pycmds:
        cmd = [py_cmd] + ['--version']
        ret = run_all(name, subprocess.list2cmdline(cmd))
        if ret['retcode'] == 0:
            container_python_bin = py_cmd
            break
    if not container_python_bin:
        raise CommandExecutionError('Python interpreter cannot be found inside the container. Make sure Python is installed in the container')
    untar_cmd = [container_python_bin, '-c', 'import tarfile; tarfile.open("{0}/{1}").extractall(path="{0}")'.format(thin_dest_path, os.path.basename(thin_path))]
    ret = run_all(name, subprocess.list2cmdline(untar_cmd))
    if ret['retcode'] != 0:
        return {'result': False, 'comment': ret['stderr']}
    try:
        salt_argv = [container_python_bin, os.path.join(thin_dest_path, 'salt-call'), '--metadata', '--local', '--log-file', os.path.join(thin_dest_path, 'log'), '--cachedir', os.path.join(thin_dest_path, 'cache'), '--out', 'json', '-l', 'quiet', '--', function] + list(args) + [f'{key}={value}' for (key, value) in kwargs.items() if not key.startswith('__')]
        ret = run_all(name, subprocess.list2cmdline(map(str, salt_argv)))
        if ret['retcode'] != 0:
            raise CommandExecutionError(ret['stderr'])
        try:
            data = __utils__['json.find_json'](ret['stdout'])
            local = data.get('local', data)
            if isinstance(local, dict):
                if 'retcode' in local:
                    __context__['retcode'] = local['retcode']
            return local.get('return', data)
        except ValueError:
            return {'result': False, 'comment': "Can't parse container command output"}
    finally:
        rm_thin_argv = ['rm', '-rf', thin_dest_path]
        run_all(name, subprocess.list2cmdline(rm_thin_argv))

def apply_(name, mods=None, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Apply states! This function will call highstate or state.sls based on the\n    arguments passed in, ``apply`` is intended to be the main gateway for\n    all state executions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'docker' docker.apply web01\n        salt 'docker' docker.apply web01 test\n        salt 'docker' docker.apply web01 test,pkgs\n    "
    if mods:
        return sls(name, mods, **kwargs)
    return highstate(name, **kwargs)

def sls(name, mods=None, **kwargs):
    if False:
        return 10
    '\n    Apply the states defined by the specified SLS modules to the running\n    container\n\n    .. versionadded:: 2016.11.0\n\n    The container does not need to have Salt installed, but Python is required.\n\n    name\n        Container name or ID\n\n    mods : None\n        A string containing comma-separated list of SLS with defined states to\n        apply to the container.\n\n    saltenv : base\n        Specify the environment from which to retrieve the SLS indicated by the\n        `mods` parameter.\n\n    pillarenv\n        Specify a Pillar environment to be used when applying states. This\n        can also be set in the minion config file using the\n        :conf_minion:`pillarenv` option. When neither the\n        :conf_minion:`pillarenv` minion config option nor this CLI argument is\n        used, all Pillar environments will be merged together.\n\n        .. versionadded:: 2018.3.0\n\n    pillar\n        Custom Pillar values, passed as a dictionary of key-value pairs\n\n        .. note::\n            Values passed this way will override Pillar values set via\n            ``pillar_roots`` or an external Pillar source.\n\n        .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.sls compassionate_mirzakhani mods=rails,web\n\n    '
    mods = [item.strip() for item in mods.split(',')] if mods else []
    pillar_override = kwargs.pop('pillar', None)
    if 'saltenv' not in kwargs:
        kwargs['saltenv'] = 'base'
    sls_opts = __utils__['state.get_sls_opts'](__opts__, **kwargs)
    grains = call(name, 'grains.items')
    pillar = salt.pillar.get_pillar(__opts__, grains, __opts__['id'], pillar_override=pillar_override, pillarenv=sls_opts['pillarenv']).compile_pillar()
    if pillar_override and isinstance(pillar_override, dict):
        pillar.update(pillar_override)
    sls_opts['grains'].update(grains)
    sls_opts['pillar'].update(pillar)
    trans_tar = _prepare_trans_tar(name, sls_opts, mods=mods, pillar=pillar, extra_filerefs=kwargs.get('extra_filerefs', ''))
    trans_dest_path = _generate_tmp_path()
    mkdirp_trans_argv = ['mkdir', '-p', trans_dest_path]
    ret = run_all(name, subprocess.list2cmdline(mkdirp_trans_argv))
    if ret['retcode'] != 0:
        return {'result': False, 'comment': ret['stderr']}
    ret = None
    try:
        trans_tar_sha256 = __utils__['hashutils.get_hash'](trans_tar, 'sha256')
        copy_to(name, trans_tar, os.path.join(trans_dest_path, 'salt_state.tgz'), exec_driver=_get_exec_driver(), overwrite=True)
        ret = call(name, 'state.pkg', os.path.join(trans_dest_path, 'salt_state.tgz'), trans_tar_sha256, 'sha256')
    finally:
        rm_trans_argv = ['rm', '-rf', trans_dest_path]
        run_all(name, subprocess.list2cmdline(rm_trans_argv))
        try:
            os.remove(trans_tar)
        except OSError as exc:
            log.error("docker.sls: Unable to remove state tarball '%s': %s", trans_tar, exc)
    if not isinstance(ret, dict):
        __context__['retcode'] = 1
    elif not __utils__['state.check_result'](ret):
        __context__['retcode'] = 2
    else:
        __context__['retcode'] = 0
    return ret

def highstate(name, saltenv='base', **kwargs):
    if False:
        print('Hello World!')
    '\n    Apply a highstate to the running container\n\n    .. versionadded:: 2019.2.0\n\n    The container does not need to have Salt installed, but Python is required.\n\n    name\n        Container name or ID\n\n    saltenv : base\n        Specify the environment from which to retrieve the SLS indicated by the\n        `mods` parameter.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.highstate compassionate_mirzakhani\n\n    '
    return sls(name, saltenv='base', **kwargs)

def sls_build(repository, tag='latest', base='opensuse/python', mods=None, dryrun=False, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionchanged:: 2018.3.0\n        The repository and tag must now be passed separately using the\n        ``repository`` and ``tag`` arguments, rather than together in the (now\n        deprecated) ``image`` argument.\n\n    Build a Docker image using the specified SLS modules on top of base image\n\n    .. versionadded:: 2016.11.0\n\n    The base image does not need to have Salt installed, but Python is required.\n\n    repository\n        Repository name for the image to be built\n\n        .. versionadded:: 2018.3.0\n\n    tag : latest\n        Tag name for the image to be built\n\n        .. versionadded:: 2018.3.0\n\n    name\n        .. deprecated:: 2018.3.0\n            Use both ``repository`` and ``tag`` instead\n\n    base : opensuse/python\n        Name or ID of the base image\n\n    mods\n        A string containing comma-separated list of SLS with defined states to\n        apply to the base image.\n\n    saltenv : base\n        Specify the environment from which to retrieve the SLS indicated by the\n        `mods` parameter.\n\n    pillarenv\n        Specify a Pillar environment to be used when applying states. This\n        can also be set in the minion config file using the\n        :conf_minion:`pillarenv` option. When neither the\n        :conf_minion:`pillarenv` minion config option nor this CLI argument is\n        used, all Pillar environments will be merged together.\n\n        .. versionadded:: 2018.3.0\n\n    pillar\n        Custom Pillar values, passed as a dictionary of key-value pairs\n\n        .. note::\n            Values passed this way will override Pillar values set via\n            ``pillar_roots`` or an external Pillar source.\n\n        .. versionadded:: 2018.3.0\n\n    dryrun: False\n        when set to True the container will not be committed at the end of\n        the build. The dryrun succeed also when the state contains errors.\n\n    **RETURN DATA**\n\n    A dictionary with the ID of the new container. In case of a dryrun,\n    the state result is returned and the container gets removed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion docker.sls_build imgname base=mybase mods=rails,web\n\n    '
    create_kwargs = __utils__['args.clean_kwargs'](**copy.deepcopy(kwargs))
    for key in ('image', 'name', 'cmd', 'interactive', 'tty', 'extra_filerefs'):
        try:
            del create_kwargs[key]
        except KeyError:
            pass
    ret = create(image=base, cmd='sleep infinity', interactive=True, tty=True, **create_kwargs)
    id_ = ret['Id']
    try:
        start_(id_)
        ret = sls(id_, mods, **kwargs)
        if not dryrun and (not __utils__['state.check_result'](ret)):
            raise CommandExecutionError(ret)
        if dryrun is False:
            ret = commit(id_, repository, tag=tag)
    finally:
        stop(id_)
        rm_(id_)
    return ret