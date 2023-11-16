"""
Pass Renderer for Salt
======================

pass_ is an encrypted on-disk password store.

.. _pass: https://www.passwordstore.org/

.. versionadded:: 2017.7.0

Setup
-----

*Note*: ``<user>`` needs to be replaced with the user salt-master will be
running as.

Have private gpg loaded into ``user``'s gpg keyring

.. code-block:: yaml

    load_private_gpg_key:
      cmd.run:
        - name: gpg --import <location_of_private_gpg_key>
        - unless: gpg --list-keys '<gpg_name>'

Said private key's public key should have been used when encrypting pass entries
that are of interest for pillar data.

Fetch and keep local pass git repo up-to-date

.. code-block:: yaml

        update_pass:
          git.latest:
            - force_reset: True
            - name: <git_repo>
            - target: /<user>/.password-store
            - identity: <location_of_ssh_private_key>
            - require:
              - cmd: load_private_gpg_key

Install pass binary

.. code-block:: yaml

        pass:
          pkg.installed

Salt master configuration options

.. code-block:: yaml

        # If the prefix is *not* set (default behavior), all template variables are
        # considered for fetching secrets from Pass. Those that cannot be resolved
        # to a secret are passed through.
        #
        # If the prefix is set, only the template variables with matching prefix are
        # considered for fetching the secrets, other variables are passed through.
        #
        # For ease of use it is recommended to set the following options as well:
        #       renderer: 'jinja|yaml|pass'
        #       pass_strict_fetch: true
        #
        pass_variable_prefix: 'pass:'

        # If set to 'true', error out when unable to fetch a secret for a template variable.
        pass_strict_fetch: true

        # Set GNUPGHOME env for Pass.
        # Defaults to: ~/.gnupg
        pass_gnupghome: <path>

        # Set PASSWORD_STORE_DIR env for Pass.
        # Defaults to: ~/.password-store
        pass_dir: <path>
"""
import logging
import os
from os.path import expanduser
from subprocess import PIPE, Popen
import salt.utils.path
from salt.exceptions import SaltConfigurationError, SaltRenderError
log = logging.getLogger(__name__)

def _get_pass_exec():
    if False:
        while True:
            i = 10
    '\n    Return the pass executable or raise an error\n    '
    pass_exec = salt.utils.path.which('pass')
    if pass_exec:
        return pass_exec
    else:
        raise SaltRenderError('pass unavailable')

def _fetch_secret(pass_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetch secret from pass based on pass_path. If there is\n    any error, return back the original pass_path value\n    '
    pass_exec = _get_pass_exec()
    original_pass_path = pass_path
    pass_prefix = __opts__['pass_variable_prefix']
    if pass_prefix:
        if not pass_path.startswith(pass_prefix):
            return pass_path
        pass_path = pass_path[len(pass_prefix):]
    pass_strict_fetch = __opts__['pass_strict_fetch']
    if pass_strict_fetch and (not pass_prefix):
        msg = "The 'pass_strict_fetch' option requires 'pass_variable_prefix' option enabled"
        raise SaltConfigurationError(msg)
    pass_path = pass_path.strip()
    cmd = [pass_exec, 'show', pass_path]
    log.debug('Fetching secret: %s', ' '.join(cmd))
    env = os.environ.copy()
    env['HOME'] = expanduser('~')
    pass_dir = __opts__['pass_dir']
    if pass_dir:
        env['PASSWORD_STORE_DIR'] = pass_dir
    pass_gnupghome = __opts__['pass_gnupghome']
    if pass_gnupghome:
        env['GNUPGHOME'] = pass_gnupghome
    try:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, env=env, encoding='utf-8')
        (pass_data, pass_error) = proc.communicate()
        pass_returncode = proc.returncode
    except (OSError, UnicodeDecodeError) as e:
        (pass_data, pass_error) = ('', str(e))
        pass_returncode = 1
    if pass_returncode or not pass_data:
        msg = f"Could not fetch secret '{pass_path}' from the password store: {pass_error}"
        if pass_strict_fetch:
            raise SaltRenderError(msg)
        else:
            log.warning(msg)
            return original_pass_path
    return pass_data.rstrip('\r\n')

def _decrypt_object(obj):
    if False:
        print('Hello World!')
    '\n    Recursively try to find a pass path (string) that can be handed off to pass\n    '
    if isinstance(obj, str):
        return _fetch_secret(obj)
    elif isinstance(obj, dict):
        for (pass_key, pass_path) in obj.items():
            obj[pass_key] = _decrypt_object(pass_path)
    elif isinstance(obj, list):
        for (pass_key, pass_path) in enumerate(obj):
            obj[pass_key] = _decrypt_object(pass_path)
    return obj

def render(pass_info, saltenv='base', sls='', argline='', **kwargs):
    if False:
        while True:
            i = 10
    '\n    Fetch secret from pass based on pass_path\n    '
    return _decrypt_object(pass_info)