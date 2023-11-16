"""
Support for htpasswd command. Requires the apache2-utils package for Debian-based distros.

.. versionadded:: 2014.1.0

The functions here will load inside the webutil module. This allows other
functions that don't use htpasswd to use the webutil module name.
"""
import logging
import os
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'webutil'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load the module if htpasswd is installed\n    '
    if salt.utils.path.which('htpasswd'):
        return __virtualname__
    return (False, 'The htpasswd execution mdule cannot be loaded: htpasswd binary not in path.')

def useradd(pwfile, user, password, opts='', runas=None):
    if False:
        while True:
            i = 10
    "\n    Add a user to htpasswd file using the htpasswd command. If the htpasswd\n    file does not exist, it will be created.\n\n    pwfile\n        Path to htpasswd file\n\n    user\n        User name\n\n    password\n        User password\n\n    opts\n        Valid options that can be passed are:\n\n            - `n`  Don't update file; display results on stdout.\n            - `m`  Force MD5 encryption of the password (default).\n            - `d`  Force CRYPT encryption of the password.\n            - `p`  Do not encrypt the password (plaintext).\n            - `s`  Force SHA encryption of the password.\n\n    runas\n        The system user to run htpasswd command with\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' webutil.useradd /etc/httpd/htpasswd larry badpassword\n        salt '*' webutil.useradd /etc/httpd/htpasswd larry badpass opts=ns\n    "
    if not os.path.exists(pwfile):
        opts += 'c'
    cmd = ['htpasswd', '-b{}'.format(opts), pwfile, user, password]
    return __salt__['cmd.run_all'](cmd, runas=runas, python_shell=False)

def userdel(pwfile, user, runas=None, all_results=False):
    if False:
        i = 10
        return i + 15
    "\n    Delete a user from the specified htpasswd file.\n\n    pwfile\n        Path to htpasswd file\n\n    user\n        User name\n\n    runas\n        The system user to run htpasswd command with\n\n    all_results\n        Return stdout, stderr, and retcode, not just stdout\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' webutil.userdel /etc/httpd/htpasswd larry\n    "
    if not os.path.exists(pwfile):
        return 'Error: The specified htpasswd file does not exist'
    cmd = ['htpasswd', '-D', pwfile, user]
    if all_results:
        out = __salt__['cmd.run_all'](cmd, runas=runas, python_shell=False)
    else:
        out = __salt__['cmd.run'](cmd, runas=runas, python_shell=False).splitlines()
    return out

def verify(pwfile, user, password, opts='', runas=None):
    if False:
        print('Hello World!')
    "\n    Return True if the htpasswd file exists, the user has an entry, and their\n    password matches.\n\n    pwfile\n        Fully qualified path to htpasswd file\n\n    user\n        User name\n\n    password\n        User password\n\n    opts\n        Valid options that can be passed are:\n\n            - `m`  Force MD5 encryption of the password (default).\n            - `d`  Force CRYPT encryption of the password.\n            - `p`  Do not encrypt the password (plaintext).\n            - `s`  Force SHA encryption of the password.\n\n    runas\n        The system user to run htpasswd command with\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' webutil.verify /etc/httpd/htpasswd larry maybepassword\n        salt '*' webutil.verify /etc/httpd/htpasswd larry maybepassword opts=ns\n    "
    if not os.path.exists(pwfile):
        return False
    cmd = ['htpasswd', '-bv{}'.format(opts), pwfile, user, password]
    ret = __salt__['cmd.run_all'](cmd, runas=runas, python_shell=False)
    log.debug('Result of verifying htpasswd for user %s: %s', user, ret)
    return ret['retcode'] == 0