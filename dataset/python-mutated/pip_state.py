"""
Installation of Python Packages Using pip
=========================================

These states manage system installed python packages. Note that pip must be
installed for these states to be available, so pip states should include a
requisite to a pkg.installed state for the package which provides pip
(``python-pip`` in most cases). Example:

.. code-block:: yaml

    python-pip:
      pkg.installed

    virtualenvwrapper:
      pip.installed:
        - require:
          - pkg: python-pip
"""
import logging
import re
import sys
import types
import salt.utils.data
import salt.utils.versions
from salt.exceptions import CommandExecutionError, CommandNotFoundError
try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

def purge_pip():
    if False:
        for i in range(10):
            print('nop')
    '\n    Purge pip and its sub-modules\n    '
    if 'pip' not in sys.modules:
        return
    pip_related_entries = [(k, v) for (k, v) in sys.modules.items() if getattr(v, '__module__', '').startswith('pip.') or (isinstance(v, types.ModuleType) and v.__name__.startswith('pip.'))]
    for (name, entry) in pip_related_entries:
        sys.modules.pop(name)
        del entry
    if 'pip' in globals():
        del globals()['pip']
    if 'pip' in locals():
        del locals()['pip']
    sys_modules_pip = sys.modules.pop('pip', None)
    if sys_modules_pip is not None:
        del sys_modules_pip

def pip_has_internal_exceptions_mod(ver):
    if False:
        i = 10
        return i + 15
    '\n    True when the pip version has the `pip._internal.exceptions` module\n    '
    return salt.utils.versions.compare(ver1=ver, oper='>=', ver2='10.0')

def pip_has_exceptions_mod(ver):
    if False:
        print('Hello World!')
    '\n    True when the pip version has the `pip.exceptions` module\n    '
    if pip_has_internal_exceptions_mod(ver):
        return False
    return salt.utils.versions.compare(ver1=ver, oper='>=', ver2='1.0')
try:
    import pip
    HAS_PIP = True
except ImportError:
    HAS_PIP = False
    purge_pip()
if HAS_PIP is True:
    if not hasattr(purge_pip, '__pip_ver__'):
        purge_pip.__pip_ver__ = pip.__version__
    elif purge_pip.__pip_ver__ != pip.__version__:
        purge_pip()
        import pip
        purge_pip.__pip_ver__ = pip.__version__
    if salt.utils.versions.compare(ver1=pip.__version__, oper='>=', ver2='10.0'):
        from pip._internal.exceptions import InstallationError
    elif salt.utils.versions.compare(ver1=pip.__version__, oper='>=', ver2='1.0'):
        from pip.exceptions import InstallationError
    else:
        InstallationError = ValueError
logger = logging.getLogger(__name__)
__virtualname__ = 'pip'

def _from_line(*args, **kwargs):
    if False:
        return 10
    import pip
    if salt.utils.versions.compare(ver1=pip.__version__, oper='>=', ver2='18.1'):
        import pip._internal.req.constructors
        return pip._internal.req.constructors.install_req_from_line(*args, **kwargs)
    elif salt.utils.versions.compare(ver1=pip.__version__, oper='>=', ver2='10.0'):
        import pip._internal.req
        return pip._internal.req.InstallRequirement.from_line(*args, **kwargs)
    else:
        import pip.req
        return pip.req.InstallRequirement.from_line(*args, **kwargs)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the pip module is available in __salt__\n    '
    if HAS_PKG_RESOURCES is False:
        return (False, 'The pkg_resources python library is not installed')
    if 'pip.list' in __salt__:
        return __virtualname__
    return False

def _fulfills_version_spec(version, version_spec):
    if False:
        return 10
    '\n    Check version number against version specification info and return a\n    boolean value based on whether or not the version number meets the\n    specified version.\n    '
    for (oper, spec) in version_spec:
        if oper is None:
            continue
        if not salt.utils.versions.compare(ver1=version, oper=oper, ver2=spec, cmp_func=_pep440_version_cmp):
            return False
    return True

def _check_pkg_version_format(pkg):
    if False:
        while True:
            i = 10
    '\n    Takes a package name and version specification (if any) and checks it using\n    the pip library.\n    '
    ret = {'result': False, 'comment': None, 'prefix': None, 'version_spec': None}
    if not HAS_PIP:
        ret['comment'] = "An importable Python pip module is required but could not be found on your system. This usually means that the system's pip package is not installed properly."
        return ret
    from_vcs = False
    try:
        try:
            logger.debug('Installed pip version: %s', pip.__version__)
            install_req = _from_line(pkg)
        except AttributeError:
            logger.debug('Installed pip version is lower than 1.2')
            supported_vcs = ('git', 'svn', 'hg', 'bzr')
            if pkg.startswith(supported_vcs):
                for vcs in supported_vcs:
                    if pkg.startswith(vcs):
                        from_vcs = True
                        install_req = _from_line(pkg.split(f'{vcs}+')[-1])
                        break
            else:
                install_req = _from_line(pkg)
    except (ValueError, InstallationError) as exc:
        ret['result'] = False
        if not from_vcs and '=' in pkg and ('==' not in pkg):
            ret['comment'] = "Invalid version specification in package {}. '=' is not supported, use '==' instead.".format(pkg)
            return ret
        ret['comment'] = "pip raised an exception while parsing '{}': {}".format(pkg, exc)
        return ret
    if install_req.req is None:
        ret['result'] = True
        ret['prefix'] = ''
        ret['version_spec'] = []
    else:
        ret['result'] = True
        try:
            ret['prefix'] = install_req.req.project_name
            ret['version_spec'] = install_req.req.specs
        except Exception:
            ret['prefix'] = re.sub('[^A-Za-z0-9.]+', '-', install_req.name)
            if hasattr(install_req, 'specifier'):
                specifier = install_req.specifier
            else:
                specifier = install_req.req.specifier
            ret['version_spec'] = [(spec.operator, spec.version) for spec in specifier]
    return ret

def _check_if_installed(prefix, state_pkg_name, version_spec, ignore_installed, force_reinstall, upgrade, user, cwd, bin_env, env_vars, index_url, extra_index_url, pip_list=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Takes a package name and version specification (if any) and checks it is\n    installed\n\n    Keyword arguments include:\n        pip_list: optional dict of installed pip packages, and their versions,\n            to search through to check if the package is installed. If not\n            provided, one will be generated in this function by querying the\n            system.\n\n    Returns:\n     result: None means the command failed to run\n     result: True means the package is installed\n     result: False means the package is not installed\n    '
    ret = {'result': False, 'comment': None}
    pip_list = salt.utils.data.CaseInsensitiveDict(pip_list or __salt__['pip.list'](prefix, bin_env=bin_env, user=user, cwd=cwd, env_vars=env_vars, **kwargs))
    if ignore_installed is False and prefix in pip_list:
        if force_reinstall is False and (not upgrade):
            if any(version_spec) and _fulfills_version_spec(pip_list[prefix], version_spec) or not any(version_spec):
                ret['result'] = True
                ret['comment'] = 'Python package {} was already installed'.format(state_pkg_name)
                return ret
        if force_reinstall is False and upgrade:
            include_alpha = False
            include_beta = False
            include_rc = False
            if any(version_spec):
                for spec in version_spec:
                    if 'a' in spec[1]:
                        include_alpha = True
                    if 'b' in spec[1]:
                        include_beta = True
                    if 'rc' in spec[1]:
                        include_rc = True
            available_versions = __salt__['pip.list_all_versions'](prefix, bin_env=bin_env, include_alpha=include_alpha, include_beta=include_beta, include_rc=include_rc, user=user, cwd=cwd, index_url=index_url, extra_index_url=extra_index_url)
            desired_version = ''
            if any(version_spec) and available_versions:
                for version in reversed(available_versions):
                    if _fulfills_version_spec(version, version_spec):
                        desired_version = version
                        break
            elif available_versions:
                desired_version = available_versions[-1]
            if not desired_version:
                ret['result'] = True
                ret['comment'] = "Python package {} was already installed and\nthe available upgrade doesn't fulfills the version requirements".format(prefix)
                return ret
            if _pep440_version_cmp(pip_list[prefix], desired_version) == 0:
                ret['result'] = True
                ret['comment'] = 'Python package {} was already installed'.format(state_pkg_name)
                return ret
    return ret

def _pep440_version_cmp(pkg1, pkg2, ignore_epoch=False):
    if False:
        return 10
    '\n    Compares two version strings using pkg_resources.parse_version.\n    Return -1 if version1 < version2, 0 if version1 ==version2,\n    and 1 if version1 > version2. Return None if there was a problem\n    making the comparison.\n    '
    if HAS_PKG_RESOURCES is False:
        logger.warning('The pkg_resources packages was not loaded. Please install setuptools.')
        return None
    normalize = lambda x: str(x).split('!', 1)[-1] if ignore_epoch else str(x)
    pkg1 = normalize(pkg1)
    pkg2 = normalize(pkg2)
    try:
        if pkg_resources.parse_version(pkg1) < pkg_resources.parse_version(pkg2):
            return -1
        if pkg_resources.parse_version(pkg1) == pkg_resources.parse_version(pkg2):
            return 0
        if pkg_resources.parse_version(pkg1) > pkg_resources.parse_version(pkg2):
            return 1
    except Exception as exc:
        logger.exception(f'Comparison of package versions "{pkg1}" and "{pkg2}" failed: {exc}')
    return None

def installed(name, pkgs=None, pip_bin=None, requirements=None, bin_env=None, use_wheel=False, no_use_wheel=False, log=None, proxy=None, timeout=None, repo=None, editable=None, find_links=None, index_url=None, extra_index_url=None, no_index=False, mirrors=None, build=None, target=None, download=None, download_cache=None, source=None, upgrade=False, force_reinstall=False, ignore_installed=False, exists_action=None, no_deps=False, no_install=False, no_download=False, install_options=None, global_options=None, user=None, cwd=None, pre_releases=False, cert=None, allow_all_external=False, allow_external=None, allow_unverified=None, process_dependency_links=False, env_vars=None, use_vt=False, trusted_host=None, no_cache_dir=False, cache_dir=None, no_binary=None, extra_args=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Make sure the package is installed\n\n    name\n        The name of the python package to install. You can also specify version\n        numbers here using the standard operators ``==, >=, <=``. If\n        ``requirements`` or ``pkgs`` is given, this parameter will be ignored.\n\n        Example:\n\n        .. code-block:: yaml\n\n            django:\n              pip.installed:\n                - name: django >= 1.6, <= 1.7\n                - require:\n                  - pkg: python-pip\n\n        Installs the latest Django version greater than 1.6 but less\n        than 1.7.\n\n    pkgs\n        A list of python packages to install. This let you install multiple\n        packages at the same time.\n\n        Example:\n\n        .. code-block:: yaml\n\n            django-and-psycopg2:\n              pip.installed:\n                - pkgs:\n                  - django >= 1.6, <= 1.7\n                  - psycopg2 >= 2.8.4\n                - require:\n                  - pkg: python-pip\n\n        Installs the latest Django version greater than 1.6 but less than 1.7\n        and the latest psycopg2 greater than 2.8.4 at the same time.\n\n    requirements\n        Path to a pip requirements file. If the path begins with salt://\n        the file will be transferred from the master file server.\n\n    user\n        The user under which to run pip\n\n    use_wheel : False\n        Prefer wheel archives (requires pip>=1.4)\n\n    no_use_wheel : False\n        Force to not use wheel archives (requires pip>=1.4)\n\n    no_binary\n        Force to not use binary packages (requires pip >= 7.0.0)\n        Accepts either :all: to disable all binary packages, :none: to empty the set,\n        or a list of one or more packages\n\n    Example:\n\n    .. code-block:: yaml\n\n        django:\n          pip.installed:\n            - no_binary: ':all:'\n\n        flask:\n          pip.installed:\n            - no_binary:\n              - itsdangerous\n              - click\n\n    log\n        Log file where a complete (maximum verbosity) record will be kept\n\n    proxy\n        Specify a proxy in the form\n        user:passwd@proxy.server:port. Note that the\n        user:password@ is optional and required only if you\n        are behind an authenticated proxy.  If you provide\n        user@proxy.server:port then you will be prompted for a\n        password.\n\n    timeout\n        Set the socket timeout (default 15 seconds)\n\n    editable\n        install something editable (i.e.\n        git+https://github.com/worldcompany/djangoembed.git#egg=djangoembed)\n\n    find_links\n        URL to look for packages at\n\n    index_url\n        Base URL of Python Package Index\n\n    extra_index_url\n        Extra URLs of package indexes to use in addition to ``index_url``\n\n    no_index\n        Ignore package index\n\n    mirrors\n        Specific mirror URL(s) to query (automatically adds --use-mirrors)\n\n    build\n        Unpack packages into ``build`` dir\n\n    target\n        Install packages into ``target`` dir\n\n    download\n        Download packages into ``download`` instead of installing them\n\n    download_cache\n        Cache downloaded packages in ``download_cache`` dir\n\n    source\n        Check out ``editable`` packages into ``source`` dir\n\n    upgrade\n        Upgrade all packages to the newest available version\n\n    force_reinstall\n        When upgrading, reinstall all packages even if they are already\n        up-to-date.\n\n    ignore_installed\n        Ignore the installed packages (reinstalling instead)\n\n    exists_action\n        Default action when a path already exists: (s)witch, (i)gnore, (w)ipe,\n        (b)ackup\n\n    no_deps\n        Ignore package dependencies\n\n    no_install\n        Download and unpack all packages, but don't actually install them\n\n    no_cache_dir:\n        Disable the cache.\n\n    cwd\n        Current working directory to run pip from\n\n    pre_releases\n        Include pre-releases in the available versions\n\n    cert\n        Provide a path to an alternate CA bundle\n\n    allow_all_external\n        Allow the installation of all externally hosted files\n\n    allow_external\n        Allow the installation of externally hosted files (comma separated list)\n\n    allow_unverified\n        Allow the installation of insecure and unverifiable files (comma separated list)\n\n    process_dependency_links\n        Enable the processing of dependency links\n\n    env_vars\n        Add or modify environment variables. Useful for tweaking build steps,\n        such as specifying INCLUDE or LIBRARY paths in Makefiles, build scripts or\n        compiler calls.  This must be in the form of a dictionary or a mapping.\n\n        Example:\n\n        .. code-block:: yaml\n\n            django:\n              pip.installed:\n                - name: django_app\n                - env_vars:\n                    CUSTOM_PATH: /opt/django_app\n                    VERBOSE: True\n\n    use_vt\n        Use VT terminal emulation (see output while installing)\n\n    trusted_host\n        Mark this host as trusted, even though it does not have valid or any\n        HTTPS.\n\n    bin_env : None\n        Absolute path to a virtual environment directory or absolute path to\n        a pip executable. The example below assumes a virtual environment\n        has been created at ``/foo/.virtualenvs/bar``.\n\n        Example:\n\n        .. code-block:: yaml\n\n            django:\n            pip.installed:\n                - name: django >= 1.6, <= 1.7\n                - bin_env: /foo/.virtualenvs/bar\n                - require:\n                - pkg: python-pip\n\n        Or\n\n        Example:\n\n        .. code-block:: yaml\n\n            django:\n            pip.installed:\n                - name: django >= 1.6, <= 1.7\n                - bin_env: /foo/.virtualenvs/bar/bin/pip\n                - require:\n                - pkg: python-pip\n\n    .. admonition:: Attention\n\n        The following arguments are deprecated, do not use.\n\n    pip_bin : None\n        Deprecated, use ``bin_env``\n\n    .. versionchanged:: 0.17.0\n        ``use_wheel`` option added.\n\n    install_options\n\n        Extra arguments to be supplied to the setup.py install command.\n        If you are using an option with a directory path, be sure to use\n        absolute path.\n\n        Example:\n\n        .. code-block:: yaml\n\n            django:\n              pip.installed:\n                - name: django\n                - install_options:\n                  - --prefix=/blah\n                - require:\n                  - pkg: python-pip\n\n    global_options\n        Extra global options to be supplied to the setup.py call before the\n        install command.\n\n        .. versionadded:: 2014.1.3\n\n    .. admonition:: Attention\n\n        As of Salt 0.17.0 the pip state **needs** an importable pip module.\n        This usually means having the system's pip package installed or running\n        Salt from an active `virtualenv`_.\n\n        The reason for this requirement is because ``pip`` already does a\n        pretty good job parsing its own requirements. It makes no sense for\n        Salt to do ``pip`` requirements parsing and validation before passing\n        them to the ``pip`` library. It's functionality duplication and it's\n        more error prone.\n\n\n    .. admonition:: Attention\n\n        Please set ``reload_modules: True`` to have the salt minion\n        import this module after installation.\n\n\n    Example:\n\n    .. code-block:: yaml\n\n        pyopenssl:\n            pip.installed:\n                - name: pyOpenSSL\n                - reload_modules: True\n                - exists_action: i\n\n    extra_args\n        pip keyword and positional arguments not yet implemented in salt\n\n        .. code-block:: yaml\n\n            pandas:\n              pip.installed:\n                - name: pandas\n                - extra_args:\n                  - --latest-pip-kwarg: param\n                  - --latest-pip-arg\n\n        .. warning::\n\n            If unsupported options are passed here that are not supported in a\n            minion's version of pip, a `No such option error` will be thrown.\n\n\n    .. _`virtualenv`: http://www.virtualenv.org/en/latest/\n\n    If you are using onedir packages and you need to install python packages into\n    the system python environment, you must provide the pip_bin or\n    bin_env to the pip state module.\n\n\n    .. code-block:: yaml\n\n        lib-foo:\n          pip.installed:\n            - pip_bin: /usr/bin/pip3\n        lib-bar:\n          pip.installed:\n            - bin_env: /usr/bin/python3\n    "
    if pip_bin and (not bin_env):
        bin_env = pip_bin
    if pkgs:
        if not isinstance(pkgs, list):
            return {'name': name, 'result': False, 'changes': {}, 'comment': 'pkgs argument must be formatted as a list'}
    else:
        pkgs = [name]
    prepro = lambda pkg: pkg if isinstance(pkg, str) else ' '.join((pkg.items()[0][0], pkg.items()[0][1]))
    pkgs = [prepro(pkg) for pkg in pkgs]
    ret = {'name': ';'.join(pkgs), 'result': None, 'comment': '', 'changes': {}}
    try:
        cur_version = __salt__['pip.version'](bin_env)
    except (CommandNotFoundError, CommandExecutionError) as err:
        ret['result'] = False
        ret['comment'] = f"Error installing '{name}': {err}"
        return ret
    if use_wheel:
        min_version = '1.4'
        max_version = '9.0.3'
        too_low = salt.utils.versions.compare(ver1=cur_version, oper='<', ver2=min_version)
        too_high = salt.utils.versions.compare(ver1=cur_version, oper='>', ver2=max_version)
        if too_low or too_high:
            ret['result'] = False
            ret['comment'] = "The 'use_wheel' option is only supported in pip between {} and {}. The version of pip detected was {}.".format(min_version, max_version, cur_version)
            return ret
    if no_use_wheel:
        min_version = '1.4'
        max_version = '9.0.3'
        too_low = salt.utils.versions.compare(ver1=cur_version, oper='<', ver2=min_version)
        too_high = salt.utils.versions.compare(ver1=cur_version, oper='>', ver2=max_version)
        if too_low or too_high:
            ret['result'] = False
            ret['comment'] = "The 'no_use_wheel' option is only supported in pip between {} and {}. The version of pip detected was {}.".format(min_version, max_version, cur_version)
            return ret
    if no_binary:
        min_version = '7.0.0'
        too_low = salt.utils.versions.compare(ver1=cur_version, oper='<', ver2=min_version)
        if too_low:
            ret['result'] = False
            ret['comment'] = "The 'no_binary' option is only supported in pip {} and newer. The version of pip detected was {}.".format(min_version, cur_version)
            return ret
    pkgs_details = []
    if pkgs and (not (requirements or editable)):
        comments = []
        for pkg in iter(pkgs):
            out = _check_pkg_version_format(pkg)
            if out['result'] is False:
                ret['result'] = False
                comments.append(out['comment'])
            elif out['result'] is True:
                pkgs_details.append((out['prefix'], pkg, out['version_spec']))
        if ret['result'] is False:
            ret['comment'] = '\n'.join(comments)
            return ret
    target_pkgs = []
    already_installed_comments = []
    if requirements or editable:
        comments = []
        if __opts__['test']:
            ret['result'] = None
            if requirements:
                comments.append(f"Requirements file '{requirements}' will be processed.")
            if editable:
                comments.append('Package will be installed in editable mode (i.e. setuptools "develop mode") from {}.'.format(editable))
            ret['comment'] = ' '.join(comments)
            return ret
    else:
        try:
            pip_list = __salt__['pip.list'](bin_env=bin_env, user=user, cwd=cwd, env_vars=env_vars)
        except Exception as exc:
            logger.exception(f'Pre-caching of PIP packages during states.pip.installed failed by exception from pip.list: {exc}')
            pip_list = False
        for (prefix, state_pkg_name, version_spec) in pkgs_details:
            if prefix:
                out = _check_if_installed(prefix, state_pkg_name, version_spec, ignore_installed, force_reinstall, upgrade, user, cwd, bin_env, env_vars, index_url, extra_index_url, pip_list, **kwargs)
                if out['result'] is None:
                    ret['result'] = False
                    ret['comment'] = out['comment']
                    return ret
            else:
                out = {'result': False, 'comment': None}
            result = out['result']
            if result is False:
                target_pkgs.append((prefix, state_pkg_name.replace(',', ';')))
                if __opts__['test']:
                    if len(pkgs_details) > 1:
                        msg = 'Python package(s) set to be installed:'
                        for pkg in pkgs_details:
                            msg += '\n'
                            msg += pkg[1]
                            ret['comment'] = msg
                    else:
                        msg = 'Python package {0} is set to be installed'
                        ret['comment'] = msg.format(state_pkg_name)
                    ret['result'] = None
                    return ret
            elif result is True:
                already_installed_comments.append(out['comment'])
            elif result is None:
                ret['result'] = None
                ret['comment'] = out['comment']
                return ret
        if not target_pkgs:
            ret['result'] = True
            aicomms = '\n'.join(already_installed_comments)
            last_line = 'All specified packages are already installed' + (' and up-to-date' if upgrade else '')
            ret['comment'] = aicomms + ('\n' if aicomms else '') + last_line
            return ret
    pkgs_str = ','.join([state_name for (_, state_name) in target_pkgs])
    pip_install_call = __salt__['pip.install'](pkgs=f'{pkgs_str}' if pkgs_str else '', requirements=requirements, bin_env=bin_env, use_wheel=use_wheel, no_use_wheel=no_use_wheel, no_binary=no_binary, log=log, proxy=proxy, timeout=timeout, editable=editable, find_links=find_links, index_url=index_url, extra_index_url=extra_index_url, no_index=no_index, mirrors=mirrors, build=build, target=target, download=download, download_cache=download_cache, source=source, upgrade=upgrade, force_reinstall=force_reinstall, ignore_installed=ignore_installed, exists_action=exists_action, no_deps=no_deps, no_install=no_install, no_download=no_download, install_options=install_options, global_options=global_options, user=user, cwd=cwd, pre_releases=pre_releases, cert=cert, allow_all_external=allow_all_external, allow_external=allow_external, allow_unverified=allow_unverified, process_dependency_links=process_dependency_links, saltenv=__env__, env_vars=env_vars, use_vt=use_vt, trusted_host=trusted_host, no_cache_dir=no_cache_dir, extra_args=extra_args, disable_version_check=True, **kwargs)
    if pip_install_call and pip_install_call.get('retcode', 1) == 0:
        ret['result'] = True
        if requirements or editable:
            comments = []
            if requirements:
                PIP_REQUIREMENTS_NOCHANGE = ['Requirement already satisfied', 'Requirement already up-to-date', 'Requirement not upgraded', 'Collecting', 'Cloning', 'Cleaning up...', 'Looking in indexes']
                for line in pip_install_call.get('stdout', '').split('\n'):
                    if not any([line.strip().startswith(x) for x in PIP_REQUIREMENTS_NOCHANGE]):
                        ret['changes']['requirements'] = True
                if ret['changes'].get('requirements'):
                    comments.append('Successfully processed requirements file {}.'.format(requirements))
                else:
                    comments.append('Requirements were already installed.')
            if editable:
                comments.append('Package successfully installed from VCS checkout {}.'.format(editable))
                ret['changes']['editable'] = True
            ret['comment'] = ' '.join(comments)
        else:
            pkg_404_comms = []
            already_installed_packages = set()
            for line in pip_install_call.get('stdout', '').split('\n'):
                if line.startswith('Requirement already up-to-date: '):
                    package = line.split(':', 1)[1].split()[0]
                    already_installed_packages.add(package.lower())
            for (prefix, state_name) in target_pkgs:
                if prefix:
                    pipsearch = salt.utils.data.CaseInsensitiveDict(__salt__['pip.list'](prefix, bin_env, user=user, cwd=cwd, env_vars=env_vars, **kwargs))
                    if not pipsearch:
                        pkg_404_comms.append("There was no error installing package '{}' although it does not show when calling 'pip.freeze'.".format(pkg))
                    elif prefix in pipsearch and prefix.lower() not in already_installed_packages:
                        ver = pipsearch[prefix]
                        ret['changes'][f'{prefix}=={ver}'] = 'Installed'
                else:
                    ret['changes'][f'{state_name}==???'] = 'Installed'
            aicomms = '\n'.join(already_installed_comments)
            succ_comm = 'All packages were successfully installed' if not pkg_404_comms else '\n'.join(pkg_404_comms)
            ret['comment'] = aicomms + ('\n' if aicomms else '') + succ_comm
            return ret
    elif pip_install_call:
        ret['result'] = False
        if 'stdout' in pip_install_call:
            error = 'Error: {} {}'.format(pip_install_call['stdout'], pip_install_call['stderr'])
        else:
            error = 'Error: {}'.format(pip_install_call['comment'])
        if requirements or editable:
            comments = []
            if requirements:
                comments.append(f'Unable to process requirements file "{requirements}"')
            if editable:
                comments.append(f'Unable to install from VCS checkout {editable}.')
            comments.append(error)
            ret['comment'] = ' '.join(comments)
        else:
            pkgs_str = ', '.join([state_name for (_, state_name) in target_pkgs])
            aicomms = '\n'.join(already_installed_comments)
            error_comm = f'Failed to install packages: {pkgs_str}. {error}'
            ret['comment'] = aicomms + ('\n' if aicomms else '') + error_comm
    else:
        ret['result'] = False
        ret['comment'] = 'Could not install package'
    return ret

def removed(name, requirements=None, bin_env=None, log=None, proxy=None, timeout=None, user=None, cwd=None, use_vt=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make sure that a package is not installed.\n\n    name\n        The name of the package to uninstall\n    user\n        The user under which to run pip\n    bin_env : None\n        the pip executable or virtualenenv to use\n    use_vt\n        Use VT terminal emulation (see output while installing)\n    '
    ret = {'name': name, 'result': None, 'comment': '', 'changes': {}}
    try:
        pip_list = __salt__['pip.list'](bin_env=bin_env, user=user, cwd=cwd)
    except (CommandExecutionError, CommandNotFoundError) as err:
        ret['result'] = False
        ret['comment'] = f"Error uninstalling '{name}': {err}"
        return ret
    if name not in pip_list:
        ret['result'] = True
        ret['comment'] = 'Package is not installed.'
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = f'Package {name} is set to be removed'
        return ret
    if __salt__['pip.uninstall'](pkgs=name, requirements=requirements, bin_env=bin_env, log=log, proxy=proxy, timeout=timeout, user=user, cwd=cwd, use_vt=use_vt):
        ret['result'] = True
        ret['changes'][name] = 'Removed'
        ret['comment'] = 'Package was successfully removed.'
    else:
        ret['result'] = False
        ret['comment'] = 'Could not remove package.'
    return ret

def uptodate(name, bin_env=None, user=None, cwd=None, use_vt=False):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2015.5.0\n\n    Verify that the system is completely up to date.\n\n    name\n        The name has no functional value and is only used as a tracking\n        reference\n    user\n        The user under which to run pip\n    bin_env\n        the pip executable or virtualenenv to use\n    use_vt\n        Use VT terminal emulation (see output while installing)\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': 'Failed to update.'}
    try:
        packages = __salt__['pip.list_upgrades'](bin_env=bin_env, user=user, cwd=cwd)
    except Exception as e:
        ret['comment'] = str(e)
        return ret
    if not packages:
        ret['comment'] = 'System is already up-to-date.'
        ret['result'] = True
        return ret
    elif __opts__['test']:
        ret['comment'] = 'System update will be performed'
        ret['result'] = None
        return ret
    updated = __salt__['pip.upgrade'](bin_env=bin_env, user=user, cwd=cwd, use_vt=use_vt)
    if updated.get('result') is False:
        ret.update(updated)
    elif updated:
        ret['changes'] = updated
        ret['comment'] = 'Upgrade successful.'
        ret['result'] = True
    else:
        ret['comment'] = 'Upgrade failed.'
    return ret