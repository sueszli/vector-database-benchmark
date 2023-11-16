"""
The pkgbuild state is the front of Salt package building backend. It
automatically builds DEB and RPM packages from specified sources

.. versionadded:: 2015.8.0

.. code-block:: yaml

    salt_2015.5.2:
      pkgbuild.built:
        - runas: thatch
        - results:
          - salt-2015.5.2-2.el7.centos.noarch.rpm
          - salt-api-2015.5.2-2.el7.centos.noarch.rpm
          - salt-cloud-2015.5.2-2.el7.centos.noarch.rpm
          - salt-master-2015.5.2-2.el7.centos.noarch.rpm
          - salt-minion-2015.5.2-2.el7.centos.noarch.rpm
          - salt-ssh-2015.5.2-2.el7.centos.noarch.rpm
          - salt-syndic-2015.5.2-2.el7.centos.noarch.rpm
        - dest_dir: /tmp/pkg
        - spec: salt://pkg/salt/spec/salt.spec
        - template: jinja
        - deps:
          - salt://pkg/salt/sources/required_dependency.rpm
        - tgt: epel-7-x86_64
        - sources:
          - salt://pkg/salt/sources/logrotate.salt
          - salt://pkg/salt/sources/README.fedora
          - salt://pkg/salt/sources/salt-2015.5.2.tar.gz
          - salt://pkg/salt/sources/salt-2015.5.2-tests.patch
          - salt://pkg/salt/sources/salt-api
          - salt://pkg/salt/sources/salt-api.service
          - salt://pkg/salt/sources/salt-master
          - salt://pkg/salt/sources/salt-master.service
          - salt://pkg/salt/sources/salt-minion
          - salt://pkg/salt/sources/salt-minion.service
          - salt://pkg/salt/sources/saltpkg.sls
          - salt://pkg/salt/sources/salt-syndic
          - salt://pkg/salt/sources/salt-syndic.service
          - salt://pkg/salt/sources/SaltTesting-2015.5.8.tar.gz
    /tmp/pkg:
      pkgbuild.repo
"""
import errno
import logging
import os
log = logging.getLogger(__name__)

def _get_missing_results(results, dest_dir):
    if False:
        while True:
            i = 10
    '\n    Return a list of the filenames specified in the ``results`` argument, which\n    are not present in the dest_dir.\n    '
    try:
        present = set(os.listdir(dest_dir))
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            log.debug("pkgbuild.built: dest_dir '%s' does not exist", dest_dir)
        elif exc.errno == errno.EACCES:
            log.error("pkgbuilt.built: cannot access dest_dir '%s'", dest_dir)
        present = set()
    return sorted(set(results).difference(present))

def built(name, runas, dest_dir, spec, sources, tgt, template=None, deps=None, env=None, results=None, force=False, saltenv='base', log_dir='/var/log/salt/pkgbuild'):
    if False:
        while True:
            i = 10
    "\n    Ensure that the named package is built and exists in the named directory\n\n    name\n        The name to track the build, the name value is otherwise unused\n\n    runas\n        The user to run the build process as\n\n    dest_dir\n        The directory on the minion to place the built package(s)\n\n    spec\n        The location of the spec file (used for rpms)\n\n    sources\n        The list of package sources\n\n    tgt\n        The target platform to run the build on\n\n    template\n        Run the spec file through a templating engine\n\n        .. versionchanged:: 2015.8.2\n\n            This argument is now optional, allowing for no templating engine to\n            be used if none is desired.\n\n    deps\n        Packages required to ensure that the named package is built\n        can be hosted on either the salt master server or on an HTTP\n        or FTP server.  Both HTTPS and HTTP are supported as well as\n        downloading directly from Amazon S3 compatible URLs with both\n        pre-configured and automatic IAM credentials\n\n    env\n        A dictionary of environment variables to be set prior to execution.\n        Example:\n\n        .. code-block:: yaml\n\n            - env:\n                DEB_BUILD_OPTIONS: 'nocheck'\n\n        .. warning::\n\n            The above illustrates a common PyYAML pitfall, that **yes**,\n            **no**, **on**, **off**, **true**, and **false** are all loaded as\n            boolean ``True`` and ``False`` values, and must be enclosed in\n            quotes to be used as strings. More info on this (and other) PyYAML\n            idiosyncrasies can be found :ref:`here <yaml-idiosyncrasies>`.\n\n    results\n        The names of the expected rpms that will be built\n\n    force : False\n        If ``True``, packages will be built even if they already exist in the\n        ``dest_dir``. This is useful when building a package for continuous or\n        nightly package builds.\n\n        .. versionadded:: 2015.8.2\n\n    saltenv\n        The saltenv to use for files downloaded from the salt filesever\n\n    log_dir : /var/log/salt/rpmbuild\n        Root directory for log files created from the build. Logs will be\n        organized by package name, version, OS release, and CPU architecture\n        under this directory.\n\n        .. versionadded:: 2015.8.2\n    "
    ret = {'name': name, 'changes': {}, 'comment': '', 'result': True}
    if not results:
        ret['comment'] = "'results' argument is required"
        ret['result'] = False
        return ret
    if isinstance(results, str):
        results = results.split(',')
    needed = _get_missing_results(results, dest_dir)
    if not force and (not needed):
        ret['comment'] = 'All needed packages exist'
        return ret
    if __opts__['test']:
        ret['result'] = None
        if force:
            ret['comment'] = 'Packages will be force-built'
        else:
            ret['comment'] = 'The following packages need to be built: '
            ret['comment'] += ', '.join(needed)
        return ret
    if env is not None and (not isinstance(env, dict)):
        ret['comment'] = "Invalidly-formatted 'env' parameter. See documentation."
        ret['result'] = False
        return ret
    func = 'pkgbuild.build'
    if __grains__.get('os_family', False) not in ('RedHat', 'Suse'):
        for res in results:
            if res.endswith('.rpm'):
                func = 'rpmbuild.build'
                break
    ret['changes'] = __salt__[func](runas, tgt, dest_dir, spec, sources, deps, env, template, saltenv, log_dir)
    needed = _get_missing_results(results, dest_dir)
    if needed:
        ret['comment'] = 'The following packages were not built: '
        ret['comment'] += ', '.join(needed)
        ret['result'] = False
    else:
        ret['comment'] = 'All needed packages were built'
    return ret

def repo(name, keyid=None, env=None, use_passphrase=False, gnupghome='/etc/salt/gpgkeys', runas='builder', timeout=15.0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make a package repository and optionally sign it and packages present\n\n    The name is directory to turn into a repo. This state is best used\n    with onchanges linked to your package building states.\n\n    name\n        The directory to find packages that will be in the repository\n\n    keyid\n        .. versionchanged:: 2016.3.0\n\n        Optional Key ID to use in signing packages and repository.\n        Utilizes Public and Private keys associated with keyid which have\n        been loaded into the minion\'s Pillar data.\n\n        For example, contents from a Pillar data file with named Public\n        and Private keys as follows:\n\n        .. code-block:: yaml\n\n            gpg_pkg_priv_key: |\n              -----BEGIN PGP PRIVATE KEY BLOCK-----\n              Version: GnuPG v1\n\n              lQO+BFciIfQBCADAPCtzx7I5Rl32escCMZsPzaEKWe7bIX1em4KCKkBoX47IG54b\n              w82PCE8Y1jF/9Uk2m3RKVWp3YcLlc7Ap3gj6VO4ysvVz28UbnhPxsIkOlf2cq8qc\n              .\n              .\n              Ebe+8JCQTwqSXPRTzXmy/b5WXDeM79CkLWvuGpXFor76D+ECMRPv/rawukEcNptn\n              R5OmgHqvydEnO4pWbn8JzQO9YX/Us0SMHBVzLC8eIi5ZIopzalvX\n              =JvW8\n              -----END PGP PRIVATE KEY BLOCK-----\n\n            gpg_pkg_priv_keyname: gpg_pkg_key.pem\n\n            gpg_pkg_pub_key: |\n              -----BEGIN PGP PUBLIC KEY BLOCK-----\n              Version: GnuPG v1\n\n              mQENBFciIfQBCADAPCtzx7I5Rl32escCMZsPzaEKWe7bIX1em4KCKkBoX47IG54b\n              w82PCE8Y1jF/9Uk2m3RKVWp3YcLlc7Ap3gj6VO4ysvVz28UbnhPxsIkOlf2cq8qc\n              .\n              .\n              bYP7t5iwJmQzRMyFInYRt77wkJBPCpJc9FPNebL9vlZcN4zv0KQta+4alcWivvoP\n              4QIxE+/+trC6QRw2m2dHk6aAeq/J0Sc7ilZufwnNA71hf9SzRIwcFXMsLx4iLlki\n              inNqW9c=\n              =s1CX\n              -----END PGP PUBLIC KEY BLOCK-----\n\n            gpg_pkg_pub_keyname: gpg_pkg_key.pub\n\n    env\n        .. versionchanged:: 2016.3.0\n\n        A dictionary of environment variables to be utilized in creating the\n        repository. Example:\n\n        .. code-block:: yaml\n\n            - env:\n                OPTIONS: \'ask-passphrase\'\n\n        .. warning::\n\n            The above illustrates a common ``PyYAML`` pitfall, that **yes**,\n            **no**, **on**, **off**, **true**, and **false** are all loaded as\n            boolean ``True`` and ``False`` values, and must be enclosed in\n            quotes to be used as strings. More info on this (and other)\n            ``PyYAML`` idiosyncrasies can be found :ref:`here\n            <yaml-idiosyncrasies>`.\n\n            Use of ``OPTIONS`` on some platforms, for example:\n            ``ask-passphrase``, will require ``gpg-agent`` or similar to cache\n            passphrases.\n\n        .. note::\n\n            This parameter is not used for making ``yum`` repositories.\n\n    use_passphrase : False\n        .. versionadded:: 2016.3.0\n\n        Use a passphrase with the signing key presented in ``keyid``.\n        Passphrase is received from Pillar data which could be passed on the\n        command line with ``pillar`` parameter. For example:\n\n        .. code-block:: bash\n\n            pillar=\'{ "gpg_passphrase" : "my_passphrase" }\'\n\n    gnupghome : /etc/salt/gpgkeys\n        .. versionadded:: 2016.3.0\n\n        Location where GPG related files are stored, used with \'keyid\'\n\n    runas : builder\n        .. versionadded:: 2016.3.0\n\n        User to create the repository as, and optionally sign packages.\n\n        .. note::\n\n            Ensure the user has correct permissions to any files and\n            directories which are to be utilized.\n\n    timeout : 15.0\n        .. versionadded:: 2016.3.4\n\n        Timeout in seconds to wait for the prompt for inputting the passphrase.\n\n    '
    ret = {'name': name, 'changes': {}, 'comment': '', 'result': True}
    if __opts__['test'] is True:
        ret['result'] = None
        ret['comment'] = 'Package repo metadata at {} will be refreshed'.format(name)
        return ret
    if env is not None and (not isinstance(env, dict)):
        ret['comment'] = "Invalidly-formatted 'env' parameter. See documentation."
        return ret
    func = 'pkgbuild.make_repo'
    if __grains__.get('os_family', False) not in ('RedHat', 'Suse'):
        for file in os.listdir(name):
            if file.endswith('.rpm'):
                func = 'rpmbuild.make_repo'
                break
    res = __salt__[func](name, keyid, env, use_passphrase, gnupghome, runas, timeout)
    if res['retcode'] > 0:
        ret['result'] = False
    else:
        ret['changes'] = {'refresh': True}
    if res['stdout'] and res['stderr']:
        ret['comment'] = '{}\n{}'.format(res['stdout'], res['stderr'])
    elif res['stdout']:
        ret['comment'] = res['stdout']
    elif res['stderr']:
        ret['comment'] = res['stderr']
    return ret