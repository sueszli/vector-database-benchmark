"""
Create virtualenv environments.

.. versionadded:: 0.17.0
"""
import glob
import logging
import os
import re
import shutil
import sys
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.verify
from salt.exceptions import CommandExecutionError, SaltInvocationError
KNOWN_BINARY_NAMES = frozenset(['virtualenv-{}.{}'.format(*sys.version_info[:2]), 'virtualenv{}'.format(sys.version_info[0]), 'virtualenv'])
log = logging.getLogger(__name__)
__opts__ = {'venv_bin': salt.utils.path.which_bin(KNOWN_BINARY_NAMES) or 'virtualenv'}
__pillar__ = {}
__virtualname__ = 'virtualenv'

def __virtual__():
    if False:
        while True:
            i = 10
    return __virtualname__

def virtualenv_ver(venv_bin, user=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    return virtualenv version if exists\n    '
    try:
        import virtualenv
        version = getattr(virtualenv, '__version__', None)
        if not version:
            version = virtualenv.virtualenv_version
    except ImportError:
        version_cmd = [venv_bin, '--version']
        ret = __salt__['cmd.run_all'](version_cmd, runas=user, python_shell=False, redirect_stderr=True, **kwargs)
        if ret['retcode'] > 0 or not ret['stdout'].strip():
            raise CommandExecutionError("Unable to get the virtualenv version output using '{}'. Returned data: {}".format(version_cmd, ret))
        version = ''.join([x for x in ret['stdout'].strip().split() if re.search('^\\d.\\d*', x)])
    virtualenv_version_info = tuple((int(i) for i in re.sub('(rc|\\+ds).*$', '', version).split('.')))
    return virtualenv_version_info

def create(path, venv_bin=None, system_site_packages=False, distribute=False, clear=False, python=None, extra_search_dir=None, never_download=None, prompt=None, pip=False, symlinks=None, upgrade=None, user=None, use_vt=False, saltenv='base', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a virtualenv\n\n    path\n        The path to the virtualenv to be created\n\n    venv_bin\n        The name (and optionally path) of the virtualenv command. This can also\n        be set globally in the minion config file as ``virtualenv.venv_bin``.\n        Defaults to ``virtualenv``.\n\n    system_site_packages : False\n        Passthrough argument given to virtualenv or pyvenv\n\n    distribute : False\n        Passthrough argument given to virtualenv\n\n    pip : False\n        Install pip after creating a virtual environment. Implies\n        ``distribute=True``\n\n    clear : False\n        Passthrough argument given to virtualenv or pyvenv\n\n    python : None (default)\n        Passthrough argument given to virtualenv\n\n    extra_search_dir : None (default)\n        Passthrough argument given to virtualenv\n\n    never_download : None (default)\n        Passthrough argument given to virtualenv if True\n\n    prompt : None (default)\n        Passthrough argument given to virtualenv if not None\n\n    symlinks : None\n        Passthrough argument given to pyvenv if True\n\n    upgrade : None\n        Passthrough argument given to pyvenv if True\n\n    user : None\n        Set ownership for the virtualenv\n\n        .. note::\n            On Windows you must also pass a ``password`` parameter. Additionally,\n            the user must have permissions to the location where the virtual\n            environment is being created\n\n    runas : None\n        Set ownership for the virtualenv\n\n        .. deprecated:: 2014.1.0\n            ``user`` should be used instead\n\n    use_vt : False\n        Use VT terminal emulation (see output while installing)\n\n        .. versionadded:: 2015.5.0\n\n    saltenv : 'base'\n        Specify a different environment. The default environment is ``base``.\n\n        .. versionadded:: 2014.1.0\n\n    .. note::\n        The ``runas`` argument is deprecated as of 2014.1.0. ``user`` should be\n        used instead.\n\n    CLI Example:\n\n    .. code-block:: console\n\n        salt '*' virtualenv.create /path/to/new/virtualenv\n\n     Example of using --always-copy environment variable (in case your fs doesn't support symlinks).\n     This will copy files into the virtualenv instead of symlinking them.\n\n     .. code-block:: yaml\n\n         - env:\n           - VIRTUALENV_ALWAYS_COPY: 1\n    "
    if venv_bin is None:
        venv_bin = __opts__.get('venv_bin') or __pillar__.get('venv_bin')
    cmd = [venv_bin]
    if 'pyvenv' not in venv_bin:
        if upgrade is not None:
            raise CommandExecutionError("The `upgrade`(`--upgrade`) option is not supported by '{}'".format(venv_bin))
        elif symlinks is not None:
            raise CommandExecutionError("The `symlinks`(`--symlinks`) option is not supported by '{}'".format(venv_bin))
        virtualenv_version_info = virtualenv_ver(venv_bin, user=user, **kwargs)
        if distribute:
            if virtualenv_version_info >= (1, 10):
                log.info("The virtualenv '--distribute' option has been deprecated in virtualenv(>=1.10), as such, the 'distribute' option to `virtualenv.create()` has also been deprecated and it's not necessary anymore.")
            else:
                cmd.append('--distribute')
        if python is not None and python.strip() != '':
            if not salt.utils.path.which(python):
                raise CommandExecutionError('Cannot find requested python ({}).'.format(python))
            cmd.append('--python={}'.format(python))
        if extra_search_dir is not None:
            if isinstance(extra_search_dir, str) and extra_search_dir.strip() != '':
                extra_search_dir = [e.strip() for e in extra_search_dir.split(',')]
            for entry in extra_search_dir:
                cmd.append('--extra-search-dir={}'.format(entry))
        if never_download is True:
            if (1, 10) <= virtualenv_version_info < (14, 0, 0):
                log.info('--never-download was deprecated in 1.10.0, but reimplemented in 14.0.0. If this feature is needed, please install a supported virtualenv version.')
            else:
                cmd.append('--never-download')
        if prompt is not None and prompt.strip() != '':
            cmd.append("--prompt='{}'".format(prompt))
    else:
        if python is not None and python.strip() != '':
            raise CommandExecutionError("The `python`(`--python`) option is not supported by '{}'".format(venv_bin))
        elif extra_search_dir is not None and extra_search_dir.strip() != '':
            raise CommandExecutionError("The `extra_search_dir`(`--extra-search-dir`) option is not supported by '{}'".format(venv_bin))
        elif never_download is not None:
            raise CommandExecutionError("The `never_download`(`--never-download`) option is not supported by '{}'".format(venv_bin))
        elif prompt is not None and prompt.strip() != '':
            raise CommandExecutionError("The `prompt`(`--prompt`) option is not supported by '{}'".format(venv_bin))
        if upgrade is True:
            cmd.append('--upgrade')
        if symlinks is True:
            cmd.append('--symlinks')
    if clear is True:
        cmd.append('--clear')
    if system_site_packages is True:
        cmd.append('--system-site-packages')
    cmd.append(path)
    ret = __salt__['cmd.run_all'](cmd, runas=user, python_shell=False, **kwargs)
    if ret['retcode'] != 0:
        return ret
    if salt.utils.platform.is_windows():
        venv_python = os.path.join(path, 'Scripts', 'python.exe')
        venv_pip = os.path.join(path, 'Scripts', 'pip.exe')
        venv_setuptools = os.path.join(path, 'Scripts', 'easy_install.exe')
    else:
        venv_python = os.path.join(path, 'bin', 'python')
        venv_pip = os.path.join(path, 'bin', 'pip')
        venv_setuptools = os.path.join(path, 'bin', 'easy_install')
    if (pip or distribute) and (not os.path.exists(venv_setuptools)):
        _install_script('https://bootstrap.pypa.io/ez_setup.py', path, venv_python, user, saltenv=saltenv, use_vt=use_vt)
        for fpath in glob.glob(os.path.join(path, 'distribute-*.tar.gz*')):
            os.unlink(fpath)
    if ret['retcode'] != 0:
        return ret
    if pip and (not os.path.exists(venv_pip)):
        _ret = _install_script('https://bootstrap.pypa.io/get-pip.py', path, venv_python, user, saltenv=saltenv, use_vt=use_vt)
        ret.update(retcode=_ret['retcode'], stdout='{}\n{}'.format(ret['stdout'], _ret['stdout']).strip(), stderr='{}\n{}'.format(ret['stderr'], _ret['stderr']).strip())
    return ret

def get_site_packages(venv):
    if False:
        print('Hello World!')
    "\n    Return the path to the site-packages directory of a virtualenv\n\n    venv\n        Path to the virtualenv.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virtualenv.get_site_packages /path/to/my/venv\n    "
    bin_path = _verify_virtualenv(venv)
    ret = __salt__['cmd.exec_code_all'](bin_path, 'import sysconfig; print(sysconfig.get_path("purelib"))')
    if ret['retcode'] != 0:
        raise CommandExecutionError('{stdout}\n{stderr}'.format(**ret))
    return ret['stdout']

def get_distribution_path(venv, distribution):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the path to a distribution installed inside a virtualenv\n\n    .. versionadded:: 2016.3.0\n\n    venv\n        Path to the virtualenv.\n    distribution\n        Name of the distribution. Note, all non-alphanumeric characters\n        will be converted to dashes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virtualenv.get_distribution_path /path/to/my/venv my_distribution\n    "
    _verify_safe_py_code(distribution)
    bin_path = _verify_virtualenv(venv)
    ret = __salt__['cmd.exec_code_all'](bin_path, "import pkg_resources; print(pkg_resources.get_distribution('{}').location)".format(distribution))
    if ret['retcode'] != 0:
        raise CommandExecutionError('{stdout}\n{stderr}'.format(**ret))
    return ret['stdout']

def get_resource_path(venv, package=None, resource=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the path to a package resource installed inside a virtualenv\n\n    .. versionadded:: 2015.5.0\n\n    venv\n        Path to the virtualenv\n\n    package\n        Name of the package in which the resource resides\n\n        .. versionadded:: 2016.3.0\n\n    resource\n        Name of the resource of which the path is to be returned\n\n        .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virtualenv.get_resource_path /path/to/my/venv my_package my/resource.xml\n    "
    _verify_safe_py_code(package, resource)
    bin_path = _verify_virtualenv(venv)
    ret = __salt__['cmd.exec_code_all'](bin_path, "import pkg_resources; print(pkg_resources.resource_filename('{}', '{}'))".format(package, resource))
    if ret['retcode'] != 0:
        raise CommandExecutionError('{stdout}\n{stderr}'.format(**ret))
    return ret['stdout']

def get_resource_content(venv, package=None, resource=None):
    if False:
        while True:
            i = 10
    "\n    Return the content of a package resource installed inside a virtualenv\n\n    .. versionadded:: 2015.5.0\n\n    venv\n        Path to the virtualenv\n\n    package\n        Name of the package in which the resource resides\n\n        .. versionadded:: 2016.3.0\n\n    resource\n        Name of the resource of which the content is to be returned\n\n        .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virtualenv.get_resource_content /path/to/my/venv my_package my/resource.xml\n    "
    _verify_safe_py_code(package, resource)
    bin_path = _verify_virtualenv(venv)
    ret = __salt__['cmd.exec_code_all'](bin_path, "import pkg_resources; print(pkg_resources.resource_string('{}', '{}'))".format(package, resource))
    if ret['retcode'] != 0:
        raise CommandExecutionError('{stdout}\n{stderr}'.format(**ret))
    return ret['stdout']

def _install_script(source, cwd, python, user, saltenv='base', use_vt=False):
    if False:
        while True:
            i = 10
    if not salt.utils.platform.is_windows():
        tmppath = salt.utils.files.mkstemp(dir=cwd)
    else:
        tmppath = __salt__['cp.cache_file'](source, saltenv)
    if not salt.utils.platform.is_windows():
        fn_ = __salt__['cp.cache_file'](source, saltenv)
        shutil.copyfile(fn_, tmppath)
        os.chmod(tmppath, 320)
        os.chown(tmppath, __salt__['file.user_to_uid'](user), -1)
    try:
        return __salt__['cmd.run_all']([python, tmppath], runas=user, cwd=cwd, env={'VIRTUAL_ENV': cwd}, use_vt=use_vt, python_shell=False)
    finally:
        os.remove(tmppath)

def _verify_safe_py_code(*args):
    if False:
        i = 10
        return i + 15
    for arg in args:
        if not salt.utils.verify.safe_py_code(arg):
            raise SaltInvocationError("Unsafe python code detected in '{}'".format(arg))

def _verify_virtualenv(venv_path):
    if False:
        for i in range(10):
            print('nop')
    bin_path = os.path.join(venv_path, 'bin/python')
    if not os.path.exists(bin_path):
        raise CommandExecutionError("Path '{}' does not appear to be a virtualenv: bin/python not found.".format(venv_path))
    return bin_path