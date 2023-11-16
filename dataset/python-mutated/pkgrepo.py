"""
Management of APT/DNF/YUM/Zypper package repos
==============================================

States for managing software package repositories on Linux distros. Supported
package managers are APT, DNF, YUM and Zypper. Here is some example SLS:

.. code-block:: yaml

    base:
      pkgrepo.managed:
        - humanname: CentOS-$releasever - Base
        - mirrorlist: http://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo=os
        - comments:
            - 'http://mirror.centos.org/centos/$releasever/os/$basearch/'
        - gpgcheck: 1
        - gpgkey: file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-6

.. code-block:: yaml

    base:
      pkgrepo.managed:
        - humanname: Logstash PPA
        - name: deb http://ppa.launchpad.net/wolfnet/logstash/ubuntu precise main
        - dist: precise
        - file: /etc/apt/sources.list.d/logstash.list
        - keyid: 28B04E4A
        - keyserver: keyserver.ubuntu.com
        - require_in:
          - pkg: logstash

      pkg.latest:
        - name: logstash
        - refresh: True

.. code-block:: yaml

    base:
      pkgrepo.managed:
        - humanname: deb-multimedia
        - name: deb http://www.deb-multimedia.org stable main
        - file: /etc/apt/sources.list.d/deb-multimedia.list
        - key_url: salt://deb-multimedia/files/marillat.pub

.. code-block:: yaml

    base:
      pkgrepo.managed:
        - humanname: Google Chrome
        - name: deb http://dl.google.com/linux/chrome/deb/ stable main
        - dist: stable
        - file: /etc/apt/sources.list.d/chrome-browser.list
        - require_in:
          - pkg: google-chrome-stable
        - gpgcheck: 1
        - key_url: https://dl-ssl.google.com/linux/linux_signing_key.pub

.. code-block:: yaml

    base:
      pkgrepo.managed:
        - ppa: wolfnet/logstash
      pkg.latest:
        - name: logstash
        - refresh: True

.. note::

    On Ubuntu systems, the ``python-software-properties`` package should be
    installed for better support of PPA repositories. To check if this package
    is installed, run ``dpkg -l python-software-properties``.

    On Ubuntu & Debian systems, the ``python-apt`` package is required to be
    installed. To check if this package is installed, run ``dpkg -l python-apt``.
    ``python-apt`` will need to be manually installed if it is not present.

.. code-block:: yaml

    hello-copr:
        pkgrepo.managed:
            - copr: mymindstorm/hello
        pkg.installed:
            - name: hello


apt-key deprecated
------------------
``apt-key`` is deprecated and will be last available in Debian 11 and
Ubuntu 22.04. The recommended way to manage repo keys going forward
is to download the keys into /etc/apt/keyrings and use ``signed-by``
in your repo file pointing to the key. This module was updated
in version 3005 to implement the recommended approach. You need to add
``- aptkey: False`` to your state and set ``signed-by`` in your repo
name, to use this recommended approach.  If the cli command ``apt-key``
is not available it will automatically set ``aptkey`` to False.


Using ``aptkey: False`` with ``key_url`` example:

.. code-block:: yaml

    deb [signed-by=/etc/apt/keyrings/salt-archive-keyring.gpg arch=amd64] https://repo.saltproject.io/py3/ubuntu/18.04/amd64/latest bionic main:
      pkgrepo.managed:
        - file: /etc/apt/sources.list.d/salt.list
        - key_url: https://repo.saltproject.io/py3/ubuntu/18.04/amd64/latest/salt-archive-keyring.gpg
        - aptkey: False

Using ``aptkey: False`` with ``keyserver`` and ``keyid``:

.. code-block:: yaml

    deb [signed-by=/etc/apt/keyrings/salt-archive-keyring.gpg arch=amd64] https://repo.saltproject.io/py3/ubuntu/18.04/amd64/latest bionic main:
      pkgrepo.managed:
        - file: /etc/apt/sources.list.d/salt.list
        - keyserver: keyserver.ubuntu.com
        - keyid: 0E08A149DE57BFBE
        - aptkey: False
"""
import sys
import salt.utils.data
import salt.utils.files
import salt.utils.pkg.deb
import salt.utils.pkg.rpm
import salt.utils.versions
from salt.exceptions import CommandExecutionError, SaltInvocationError
from salt.state import STATE_INTERNAL_KEYWORDS as _STATE_INTERNAL_KEYWORDS

def __virtual__():
    if False:
        return 10
    '\n    Only load if modifying repos is available for this package type\n    '
    return 'pkg.mod_repo' in __salt__

def managed(name, ppa=None, copr=None, aptkey=True, **kwargs):
    if False:
        return 10
    "\n    This state manages software package repositories. Currently, :mod:`yum\n    <salt.modules.yumpkg>`, :mod:`apt <salt.modules.aptpkg>`, and :mod:`zypper\n    <salt.modules.zypperpkg>` repositories are supported.\n\n    **YUM/DNF/ZYPPER-BASED SYSTEMS**\n\n    .. note::\n        One of ``baseurl`` or ``mirrorlist`` below is required. Additionally,\n        note that this state is not presently capable of managing more than one\n        repo in a single repo file, so each instance of this state will manage\n        a single repo file containing the configuration for a single repo.\n\n    name\n        This value will be used in two ways: Firstly, it will be the repo ID,\n        as seen in the entry in square brackets (e.g. ``[foo]``) for a given\n        repo. Secondly, it will be the name of the file as stored in\n        /etc/yum.repos.d (e.g. ``/etc/yum.repos.d/foo.conf``).\n\n    enabled : True\n        Whether the repo is enabled or not. Can be specified as ``True``/``False`` or\n        ``1``/``0``.\n\n    disabled : False\n        Included to reduce confusion due to APT's use of the ``disabled``\n        argument. If this is passed for a YUM/DNF/Zypper-based distro, then the\n        reverse will be passed as ``enabled``. For example passing\n        ``disabled=True`` will assume ``enabled=False``.\n\n    copr\n        Fedora and RedHat based distributions only. Use community packages\n        outside of the main package repository.\n\n        .. versionadded:: 3002\n\n    humanname\n        This is used as the ``name`` value in the repo file in\n        ``/etc/yum.repos.d/`` (or ``/etc/zypp/repos.d`` for SUSE distros).\n\n    baseurl\n        The URL to a yum repository\n\n    mirrorlist\n        A URL which points to a file containing a collection of baseurls\n\n    comments\n        Sometimes you want to supply additional information, but not as\n        enabled configuration. Anything supplied for this list will be saved\n        in the repo configuration with a comment marker (#) in front.\n\n    gpgautoimport\n        Only valid for Zypper package manager. If set to ``True``, automatically\n        trust and import the new repository signing key. The key should be\n        specified with ``gpgkey`` parameter. See details below.\n\n    Additional configuration values seen in YUM/DNF/Zypper repo files, such as\n    ``gpgkey`` or ``gpgcheck``, will be used directly as key-value pairs.\n    For example:\n\n    .. code-block:: yaml\n\n        foo:\n          pkgrepo.managed:\n            - humanname: Personal repo for foo\n            - baseurl: https://mydomain.tld/repo/foo/$releasever/$basearch\n            - gpgkey: file:///etc/pki/rpm-gpg/foo-signing-key\n            - gpgcheck: 1\n\n\n    **APT-BASED SYSTEMS**\n\n    ppa\n        On Ubuntu, you can take advantage of Personal Package Archives on\n        Launchpad simply by specifying the user and archive name. The keyid\n        will be queried from launchpad and everything else is set\n        automatically. You can override any of the below settings by simply\n        setting them as you would normally. For example:\n\n        .. code-block:: yaml\n\n            logstash-ppa:\n              pkgrepo.managed:\n                - ppa: wolfnet/logstash\n\n    ppa_auth\n        For Ubuntu PPAs there can be private PPAs that require authentication\n        to access. For these PPAs the username/password can be passed as an\n        HTTP Basic style username/password combination.\n\n        .. code-block:: yaml\n\n            logstash-ppa:\n              pkgrepo.managed:\n                - ppa: wolfnet/logstash\n                - ppa_auth: username:password\n\n    name\n        On apt-based systems this must be the complete entry as it would be\n        seen in the ``sources.list`` file. This can have a limited subset of\n        components (e.g. ``main``) which can be added/modified with the\n        ``comps`` option.\n\n        .. code-block:: yaml\n\n            precise-repo:\n              pkgrepo.managed:\n                - name: deb http://us.archive.ubuntu.com/ubuntu precise main\n\n        .. note::\n\n            The above example is intended as a more readable way of configuring\n            the SLS, it is equivalent to the following:\n\n            .. code-block:: yaml\n\n                'deb http://us.archive.ubuntu.com/ubuntu precise main':\n                  pkgrepo.managed\n\n    disabled : False\n        Toggles whether or not the repo is used for resolving dependencies\n        and/or installing packages.\n\n    enabled : True\n        Included to reduce confusion due to YUM/DNF/Zypper's use of the\n        ``enabled`` argument. If this is passed for an APT-based distro, then\n        the reverse will be passed as ``disabled``. For example, passing\n        ``enabled=False`` will assume ``disabled=False``.\n\n    architectures\n        On apt-based systems, ``architectures`` can restrict the available\n        architectures that the repository provides (e.g. only ``amd64``).\n        ``architectures`` should be a comma-separated list.\n\n    comps\n        On apt-based systems, comps dictate the types of packages to be\n        installed from the repository (e.g. ``main``, ``nonfree``, ...).  For\n        purposes of this, ``comps`` should be a comma-separated list.\n\n    file\n        The filename for the ``*.list`` that the repository is configured in.\n        It is important to include the full-path AND make sure it is in\n        a directory that APT will look in when handling packages\n\n    dist\n        This dictates the release of the distro the packages should be built\n        for.  (e.g. ``unstable``). This option is rarely needed.\n\n    keyid\n        The KeyID or a list of KeyIDs of the GPG key to install.\n        This option also requires the ``keyserver`` option to be set.\n\n    keyserver\n        This is the name of the keyserver to retrieve GPG keys from. The\n        ``keyid`` option must also be set for this option to work.\n\n    key_url\n        URL to retrieve a GPG key from. Allows the usage of\n        ``https://`` as well as ``salt://``.  If ``allow_insecure_key`` is True,\n        this also allows ``http://``.\n\n        .. note::\n\n            Use either ``keyid``/``keyserver`` or ``key_url``, but not both.\n\n    key_text\n        The string representation of the GPG key to install.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n\n            Use either ``keyid``/``keyserver``, ``key_url``, or ``key_text`` but\n            not more than one method.\n\n    consolidate : False\n        If set to ``True``, this will consolidate all sources definitions to the\n        ``sources.list`` file, cleanup the now unused files, consolidate components\n        (e.g. ``main``) for the same URI, type, and architecture to a single line,\n        and finally remove comments from the ``sources.list`` file.  The consolidation\n        will run every time the state is processed. The option only needs to be\n        set on one repo managed by Salt to take effect.\n\n    clean_file : False\n        If set to ``True``, empty the file before configuring the defined repository\n\n        .. note::\n            Use with care. This can be dangerous if multiple sources are\n            configured in the same file.\n\n        .. versionadded:: 2015.8.0\n\n    refresh : True\n        If set to ``False`` this will skip refreshing the apt package database\n        on Debian based systems.\n\n    refresh_db : True\n        .. deprecated:: 2018.3.0\n            Use ``refresh`` instead.\n\n    require_in\n        Set this to a list of :mod:`pkg.installed <salt.states.pkg.installed>` or\n        :mod:`pkg.latest <salt.states.pkg.latest>` to trigger the\n        running of ``apt-get update`` prior to attempting to install these\n        packages. Setting a require in the pkg state will not work for this.\n\n    aptkey:\n        Use the binary apt-key. If the command ``apt-key`` is not found\n        in the path, aptkey will be False, regardless of what is passed into\n        this argument.\n\n\n    allow_insecure_key : True\n        Whether to allow an insecure (e.g. http vs. https) key_url.\n\n        .. versionadded:: 3006.0\n    "
    if not salt.utils.path.which('apt-key'):
        aptkey = False
    ret = {'name': name, 'changes': {}, 'result': None, 'comment': ''}
    if 'pkg.get_repo' not in __salt__:
        ret['result'] = False
        ret['comment'] = 'Repo management not implemented on this platform'
        return ret
    if 'key_url' in kwargs and ('keyid' in kwargs or 'keyserver' in kwargs):
        ret['result'] = False
        ret['comment'] = 'You may not use both "keyid"/"keyserver" and "key_url" argument.'
    if 'key_text' in kwargs and ('keyid' in kwargs or 'keyserver' in kwargs):
        ret['result'] = False
        ret['comment'] = 'You may not use both "keyid"/"keyserver" and "key_text" argument.'
    if 'key_text' in kwargs and 'key_url' in kwargs:
        ret['result'] = False
        ret['comment'] = 'You may not use both "key_url" and "key_text" argument.'
    if 'repo' in kwargs:
        ret['result'] = False
        ret['comment'] = "'repo' is not a supported argument for this state. The 'name' argument is probably what was intended."
        return ret
    enabled = kwargs.pop('enabled', None)
    disabled = kwargs.pop('disabled', None)
    if enabled is not None and disabled is not None:
        ret['result'] = False
        ret['comment'] = 'Only one of enabled/disabled is allowed'
        return ret
    elif enabled is None and disabled is None:
        enabled = True
    allow_insecure_key = kwargs.pop('allow_insecure_key', True)
    key_is_insecure = kwargs.get('key_url', '').strip().startswith('http:')
    if key_is_insecure:
        if allow_insecure_key:
            salt.utils.versions.warn_until(3008, 'allow_insecure_key will default to False starting in salt 3008.')
        else:
            ret['result'] = False
            ret['comment'] = "Cannot have 'key_url' using http with 'allow_insecure_key' set to True"
            return ret
    repo = name
    if __grains__['os'] in ('Ubuntu', 'Mint'):
        if ppa is not None:
            try:
                repo = ':'.join(('ppa', ppa))
            except TypeError:
                repo = ':'.join(('ppa', str(ppa)))
        kwargs['disabled'] = not salt.utils.data.is_true(enabled) if enabled is not None else salt.utils.data.is_true(disabled)
    elif __grains__['os_family'] in ('RedHat', 'Suse'):
        if __grains__['os_family'] in 'RedHat':
            if copr is not None:
                repo = ':'.join(('copr', copr))
                kwargs['name'] = name
        if 'humanname' in kwargs:
            kwargs['name'] = kwargs.pop('humanname')
        if 'name' not in kwargs:
            kwargs['name'] = repo
        kwargs['enabled'] = not salt.utils.data.is_true(disabled) if disabled is not None else salt.utils.data.is_true(enabled)
    elif __grains__['os_family'] in ('NILinuxRT', 'Poky'):
        kwargs['enabled'] = not salt.utils.data.is_true(disabled) if disabled is not None else salt.utils.data.is_true(enabled)
    for kwarg in _STATE_INTERNAL_KEYWORDS:
        kwargs.pop(kwarg, None)
    try:
        pre = __salt__['pkg.get_repo'](repo=repo, **kwargs)
    except CommandExecutionError as exc:
        ret['result'] = False
        ret['comment'] = f"Failed to examine repo '{name}': {exc}"
        return ret
    if __grains__.get('os_family') == 'Debian':
        from salt.modules.aptpkg import _expand_repo_def
        os_name = __grains__['os']
        os_codename = __grains__['oscodename']
        sanitizedkwargs = _expand_repo_def(os_name=os_name, os_codename=os_codename, repo=repo, **kwargs)
    else:
        sanitizedkwargs = kwargs
    if pre:
        pre.pop('file', None)
        sanitizedkwargs.pop('file', None)
        for kwarg in sanitizedkwargs:
            if kwarg not in pre:
                if kwarg == 'enabled':
                    if __grains__['os_family'] == 'RedHat':
                        if not salt.utils.data.is_true(sanitizedkwargs[kwarg]):
                            break
                    else:
                        break
                else:
                    break
            elif kwarg in ('comps', 'key_url'):
                if sorted(sanitizedkwargs[kwarg]) != sorted(pre[kwarg]):
                    break
            elif kwarg == 'line' and __grains__['os_family'] == 'Debian':
                if not sanitizedkwargs['disabled']:
                    sanitizedsplit = sanitizedkwargs[kwarg].split()
                    sanitizedsplit[3:] = sorted(sanitizedsplit[3:])
                    (reposplit, _, pre_comments) = (x.strip() for x in pre[kwarg].partition('#'))
                    reposplit = reposplit.split()
                    reposplit[3:] = sorted(reposplit[3:])
                    if sanitizedsplit != reposplit:
                        break
                    if 'comments' in kwargs:
                        post_comments = salt.utils.pkg.deb.combine_comments(kwargs['comments'])
                        if pre_comments != post_comments:
                            break
            elif kwarg == 'comments' and __grains__['os_family'] == 'RedHat':
                precomments = salt.utils.pkg.rpm.combine_comments(pre[kwarg])
                kwargcomments = salt.utils.pkg.rpm.combine_comments(sanitizedkwargs[kwarg])
                if precomments != kwargcomments:
                    break
            elif kwarg == 'architectures' and sanitizedkwargs[kwarg]:
                if set(sanitizedkwargs[kwarg]) != set(pre[kwarg]):
                    break
            elif __grains__['os_family'] in ('RedHat', 'Suse') and any((isinstance(x, bool) for x in (sanitizedkwargs[kwarg], pre[kwarg]))):
                if salt.utils.data.is_true(sanitizedkwargs[kwarg]) != salt.utils.data.is_true(pre[kwarg]):
                    break
            elif str(sanitizedkwargs[kwarg]) != str(pre[kwarg]):
                break
        else:
            ret['result'] = True
            ret['comment'] = f"Package repo '{name}' already configured"
            return ret
    if __opts__['test']:
        ret['comment'] = "Package repo '{}' would be configured. This may cause pkg states to behave differently than stated if this action is repeated without test=True, due to the differences in the configured repositories.".format(name)
        if pre:
            for kwarg in sanitizedkwargs:
                if sanitizedkwargs.get(kwarg) != pre.get(kwarg):
                    ret['changes'][kwarg] = {'new': sanitizedkwargs.get(kwarg), 'old': pre.get(kwarg)}
        else:
            ret['changes']['repo'] = name
        return ret
    if kwargs.get('clean_file', False):
        with salt.utils.files.fopen(kwargs['file'], 'w'):
            pass
    try:
        if __grains__['os_family'] == 'Debian':
            __salt__['pkg.mod_repo'](repo, saltenv=__env__, aptkey=aptkey, **kwargs)
        else:
            __salt__['pkg.mod_repo'](repo, **kwargs)
    except Exception as exc:
        ret['result'] = False
        ret['comment'] = f"Failed to configure repo '{name}': {exc}"
        return ret
    try:
        post = __salt__['pkg.get_repo'](repo=repo, **kwargs)
        if pre:
            for kwarg in sanitizedkwargs:
                if post.get(kwarg) != pre.get(kwarg):
                    ret['changes'][kwarg] = {'new': post.get(kwarg), 'old': pre.get(kwarg)}
        else:
            ret['changes'] = {'repo': repo}
        ret['result'] = True
        ret['comment'] = f"Configured package repo '{name}'"
    except Exception as exc:
        ret['result'] = False
        ret['comment'] = f"Failed to confirm config of repo '{name}': {exc}"
    if ret['changes']:
        sys.modules[__salt__['test.ping'].__module__].__context__.pop('pkg._avail', None)
    return ret

def absent(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    This function deletes the specified repo on the system, if it exists. It\n    is essentially a wrapper around :mod:`pkg.del_repo <salt.modules.pkg.del_repo>`.\n\n    name\n        The name of the package repo, as it would be referred to when running\n        the regular package manager commands.\n\n    .. note::\n        On apt-based systems this must be the complete source entry. For\n        example, if you include ``[arch=amd64]``, and a repo matching the\n        specified URI, dist, etc. exists _without_ an architecture, then no\n        changes will be made and the state will report a ``True`` result.\n\n    **FEDORA/REDHAT-SPECIFIC OPTIONS**\n\n    copr\n        Use community packages outside of the main package repository.\n\n        .. versionadded:: 3002\n\n        .. code-block:: yaml\n\n            hello-copr:\n                pkgrepo.absent:\n                  - copr: mymindstorm/hello\n\n    **UBUNTU-SPECIFIC OPTIONS**\n\n    ppa\n        On Ubuntu, you can take advantage of Personal Package Archives on\n        Launchpad simply by specifying the user and archive name.\n\n        .. code-block:: yaml\n\n            logstash-ppa:\n              pkgrepo.absent:\n                - ppa: wolfnet/logstash\n\n    ppa_auth\n        For Ubuntu PPAs there can be private PPAs that require authentication\n        to access. For these PPAs the username/password can be specified.  This\n        is required for matching if the name format uses the ``ppa:`` specifier\n        and is private (requires username/password to access, which is encoded\n        in the URI).\n\n        .. code-block:: yaml\n\n            logstash-ppa:\n              pkgrepo.absent:\n                - ppa: wolfnet/logstash\n                - ppa_auth: username:password\n\n    keyid\n        If passed, then the GPG key corresponding to the passed KeyID will also\n        be removed.\n\n    keyid_ppa : False\n        If set to ``True``, the GPG key's ID will be looked up from\n        ppa.launchpad.net and removed, and the ``keyid`` argument will be\n        ignored.\n\n        .. note::\n            This option will be disregarded unless the ``ppa`` argument is\n            present.\n    "
    ret = {'name': name, 'changes': {}, 'result': None, 'comment': ''}
    if 'ppa' in kwargs and __grains__['os'] in ('Ubuntu', 'Mint'):
        name = kwargs.pop('ppa')
        if not name.startswith('ppa:'):
            name = 'ppa:' + name
    if 'copr' in kwargs and __grains__['os_family'] in 'RedHat':
        name = kwargs.pop('copr')
        if not name.startswith('copr:'):
            name = 'copr:' + name
    remove_key = any((kwargs.get(x) is not None for x in ('keyid', 'keyid_ppa')))
    if remove_key and 'pkg.del_repo_key' not in __salt__:
        ret['result'] = False
        ret['comment'] = 'Repo key management is not implemented for this platform'
        return ret
    if __grains__['os_family'] == 'Debian':
        stripname = salt.utils.pkg.deb.strip_uri(name)
    else:
        stripname = name
    try:
        repo = __salt__['pkg.get_repo'](stripname, **kwargs)
    except CommandExecutionError as exc:
        ret['result'] = False
        ret['comment'] = f"Failed to configure repo '{name}': {exc}"
        return ret
    if repo and (__grains__['os_family'].lower() == 'debian' or __opts__.get('providers', {}).get('pkg') == 'aptpkg'):
        from salt.modules.aptpkg import _split_repo_str
        if set(_split_repo_str(stripname)['architectures']) != set(repo['architectures']):
            repo = {}
    if not repo:
        ret['comment'] = f'Package repo {name} is absent'
        ret['result'] = True
        return ret
    if __opts__['test']:
        ret['comment'] = "Package repo '{}' will be removed. This may cause pkg states to behave differently than stated if this action is repeated without test=True, due to the differences in the configured repositories.".format(name)
        return ret
    try:
        __salt__['pkg.del_repo'](repo=stripname, **kwargs)
    except (CommandExecutionError, SaltInvocationError) as exc:
        ret['result'] = False
        ret['comment'] = exc.strerror
        return ret
    repos = __salt__['pkg.list_repos']()
    if stripname not in repos:
        ret['changes']['repo'] = name
        ret['comment'] = f'Removed repo {name}'
        if not remove_key:
            ret['result'] = True
        else:
            try:
                removed_keyid = __salt__['pkg.del_repo_key'](stripname, **kwargs)
            except (CommandExecutionError, SaltInvocationError) as exc:
                ret['result'] = False
                ret['comment'] += f', but failed to remove key: {exc}'
            else:
                ret['result'] = True
                ret['changes']['keyid'] = removed_keyid
                ret['comment'] += f', and keyid {removed_keyid}'
    else:
        ret['result'] = False
        ret['comment'] = f'Failed to remove repo {name}'
    return ret