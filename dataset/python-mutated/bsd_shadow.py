"""
Manage the password database on BSD systems

.. important::
    If you feel that Salt should be using this module to manage passwords on a
    minion, and it is using a different module (or gives an error similar to
    *'shadow.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import salt.utils.files
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, SaltInvocationError
try:
    import pwd
except ImportError:
    pass
try:
    import salt.utils.pycrypto
    HAS_CRYPT = True
except ImportError:
    HAS_CRYPT = False
__virtualname__ = 'shadow'

def __virtual__():
    if False:
        return 10
    if 'BSD' in __grains__.get('os', ''):
        return __virtualname__
    return (False, 'The bsd_shadow execution module cannot be loaded: only available on BSD family systems.')

def default_hash():
    if False:
        i = 10
        return i + 15
    "\n    Returns the default hash used for unset passwords\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.default_hash\n    "
    return '*' if __grains__['os'].lower() == 'freebsd' else '*************'

def gen_password(password, crypt_salt=None, algorithm='sha512'):
    if False:
        return 10
    "\n    Generate hashed password\n\n    .. note::\n\n        When called this function is called directly via remote-execution,\n        the password argument may be displayed in the system's process list.\n        This may be a security risk on certain systems.\n\n    password\n        Plaintext password to be hashed.\n\n    crypt_salt\n        Crpytographic salt. If not given, a random 8-character salt will be\n        generated.\n\n    algorithm\n        The following hash algorithms are supported:\n\n        * md5\n        * blowfish (not in mainline glibc, only available in distros that add it)\n        * sha256\n        * sha512 (default)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.gen_password 'I_am_password'\n        salt '*' shadow.gen_password 'I_am_password' crypt_salt='I_am_salt' algorithm=sha256\n    "
    if not HAS_CRYPT:
        raise CommandExecutionError('gen_password is not available on this operating system because the "crypt" python module is not available.')
    return salt.utils.pycrypto.gen_hash(crypt_salt, password, algorithm)

def info(name):
    if False:
        print('Hello World!')
    "\n    Return information for the specified user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.info someuser\n    "
    try:
        data = pwd.getpwnam(name)
        ret = {'name': data.pw_name, 'passwd': data.pw_passwd}
    except KeyError:
        return {'name': '', 'passwd': ''}
    if not isinstance(name, str):
        name = str(name)
    if ':' in name:
        raise SaltInvocationError("Invalid username '{}'".format(name))
    if __salt__['cmd.has_exec']('pw'):
        (change, expire) = __salt__['cmd.run_stdout'](['pw', 'user', 'show', name], python_shell=False).split(':')[5:7]
    elif __grains__['kernel'] in ('NetBSD', 'OpenBSD'):
        try:
            with salt.utils.files.fopen('/etc/master.passwd', 'r') as fp_:
                for line in fp_:
                    line = salt.utils.stringutils.to_unicode(line)
                    if line.startswith('{}:'.format(name)):
                        key = line.split(':')
                        (change, expire) = key[5:7]
                        ret['passwd'] = str(key[1])
                        break
        except OSError:
            change = expire = None
    else:
        change = expire = None
    try:
        ret['change'] = int(change)
    except ValueError:
        pass
    try:
        ret['expire'] = int(expire)
    except ValueError:
        pass
    return ret

def set_change(name, change):
    if False:
        i = 10
        return i + 15
    "\n    Sets the time at which the password expires (in seconds since the UNIX\n    epoch). See ``man 8 usermod`` on NetBSD and OpenBSD or ``man 8 pw`` on\n    FreeBSD.\n\n    A value of ``0`` sets the password to never expire.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_change username 1419980400\n    "
    pre_info = info(name)
    if change == pre_info['change']:
        return True
    if __grains__['kernel'] == 'FreeBSD':
        cmd = ['pw', 'user', 'mod', name, '-f', change]
    else:
        cmd = ['usermod', '-f', change, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    if post_info['change'] != pre_info['change']:
        return post_info['change'] == change

def set_expire(name, expire):
    if False:
        while True:
            i = 10
    "\n    Sets the time at which the account expires (in seconds since the UNIX\n    epoch). See ``man 8 usermod`` on NetBSD and OpenBSD or ``man 8 pw`` on\n    FreeBSD.\n\n    A value of ``0`` sets the account to never expire.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_expire username 1419980400\n    "
    pre_info = info(name)
    if expire == pre_info['expire']:
        return True
    if __grains__['kernel'] == 'FreeBSD':
        cmd = ['pw', 'user', 'mod', name, '-e', expire]
    else:
        cmd = ['usermod', '-e', expire, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    if post_info['expire'] != pre_info['expire']:
        return post_info['expire'] == expire

def del_password(name):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2015.8.2\n\n    Delete the password from name user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.del_password username\n    "
    cmd = 'pw user mod {} -w none'.format(name)
    __salt__['cmd.run'](cmd, python_shell=False, output_loglevel='quiet')
    uinfo = info(name)
    return not uinfo['passwd']

def set_password(name, password):
    if False:
        i = 10
        return i + 15
    "\n    Set the password for a named user. The password must be a properly defined\n    hash. A password hash can be generated with :py:func:`gen_password`.\n\n    It is important to make sure that a supported cipher is used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_password someuser '$1$UYCIxa628.9qXjpQCjM4a..'\n    "
    if __grains__.get('os', '') == 'FreeBSD':
        cmd = ['pw', 'user', 'mod', name, '-H', '0']
        stdin = password
    else:
        cmd = ['usermod', '-p', password, name]
        stdin = None
    __salt__['cmd.run'](cmd, stdin=stdin, output_loglevel='quiet', python_shell=False)
    return info(name)['passwd'] == password