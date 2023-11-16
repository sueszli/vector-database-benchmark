"""
Manage the password database on Solaris systems

.. important::
    If you feel that Salt should be using this module to manage passwords on a
    minion, and it is using a different module (or gives an error similar to
    *'shadow.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import os
import salt.utils.files
from salt.exceptions import CommandExecutionError
try:
    import spwd
    HAS_SPWD = True
except ImportError:
    HAS_SPWD = False
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
    '\n    Only work on POSIX-like systems\n    '
    if __grains__.get('kernel', '') == 'SunOS':
        return __virtualname__
    return (False, 'The solaris_shadow execution module failed to load: only available on Solaris systems.')

def default_hash():
    if False:
        print('Hello World!')
    "\n    Returns the default hash used for unset passwords\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.default_hash\n    "
    return '!'

def info(name):
    if False:
        return 10
    "\n    Return information for the specified user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.info root\n    "
    if HAS_SPWD:
        try:
            data = spwd.getspnam(name)
            ret = {'name': data.sp_nam, 'passwd': data.sp_pwd, 'lstchg': data.sp_lstchg, 'min': data.sp_min, 'max': data.sp_max, 'warn': data.sp_warn, 'inact': data.sp_inact, 'expire': data.sp_expire}
        except KeyError:
            ret = {'name': '', 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
        return ret
    ret = {'name': '', 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
    try:
        data = pwd.getpwnam(name)
        ret.update({'name': name})
    except KeyError:
        return ret
    s_file = '/etc/shadow'
    if not os.path.isfile(s_file):
        return ret
    with salt.utils.files.fopen(s_file, 'r') as ifile:
        for line in ifile:
            comps = line.strip().split(':')
            if comps[0] == name:
                ret.update({'passwd': comps[1]})
    output = __salt__['cmd.run_all']('passwd -s {}'.format(name), python_shell=False)
    if output['retcode'] != 0:
        return ret
    fields = output['stdout'].split()
    if len(fields) == 2:
        return ret
    ret.update({'name': data.pw_name, 'lstchg': fields[2], 'min': int(fields[3]), 'max': int(fields[4]), 'warn': int(fields[5]), 'inact': '', 'expire': ''})
    return ret

def set_maxdays(name, maxdays):
    if False:
        i = 10
        return i + 15
    "\n    Set the maximum number of days during which a password is valid. See man\n    passwd.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_maxdays username 90\n    "
    pre_info = info(name)
    if maxdays == pre_info['max']:
        return True
    cmd = 'passwd -x {} {}'.format(maxdays, name)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    if post_info['max'] != pre_info['max']:
        return post_info['max'] == maxdays

def set_mindays(name, mindays):
    if False:
        i = 10
        return i + 15
    "\n    Set the minimum number of days between password changes. See man passwd.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_mindays username 7\n    "
    pre_info = info(name)
    if mindays == pre_info['min']:
        return True
    cmd = 'passwd -n {} {}'.format(mindays, name)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    if post_info['min'] != pre_info['min']:
        return post_info['min'] == mindays
    return False

def gen_password(password, crypt_salt=None, algorithm='sha512'):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2015.8.8\n\n    Generate hashed password\n\n    .. note::\n\n        When called this function is called directly via remote-execution,\n        the password argument may be displayed in the system's process list.\n        This may be a security risk on certain systems.\n\n    password\n        Plaintext password to be hashed.\n\n    crypt_salt\n        Crpytographic salt. If not given, a random 8-character salt will be\n        generated.\n\n    algorithm\n        The following hash algorithms are supported:\n\n        * md5\n        * blowfish (not in mainline glibc, only available in distros that add it)\n        * sha256\n        * sha512 (default)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.gen_password 'I_am_password'\n        salt '*' shadow.gen_password 'I_am_password' crypt_salt='I_am_salt' algorithm=sha256\n    "
    if not HAS_CRYPT:
        raise CommandExecutionError('gen_password is not available on this operating system because the "crypt" python module is not available.')
    return salt.utils.pycrypto.gen_hash(crypt_salt, password, algorithm)

def del_password(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2015.8.8\n\n    Delete the password from name user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.del_password username\n    "
    cmd = 'passwd -d {}'.format(name)
    __salt__['cmd.run'](cmd, python_shell=False, output_loglevel='quiet')
    uinfo = info(name)
    return not uinfo['passwd']

def set_password(name, password):
    if False:
        print('Hello World!')
    "\n    Set the password for a named user. The password must be a properly defined\n    hash, the password hash can be generated with this command:\n    ``openssl passwd -1 <plaintext password>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_password root $1$UYCIxa628.9qXjpQCjM4a..\n    "
    s_file = '/etc/shadow'
    ret = {}
    if not os.path.isfile(s_file):
        return ret
    lines = []
    with salt.utils.files.fopen(s_file, 'r') as ifile:
        for line in ifile:
            comps = line.strip().split(':')
            if comps[0] != name:
                lines.append(line)
                continue
            comps[1] = password
            line = ':'.join(comps)
            lines.append('{}\n'.format(line))
    with salt.utils.files.fopen(s_file, 'w+') as ofile:
        lines = [salt.utils.stringutils.to_str(_l) for _l in lines]
        ofile.writelines(lines)
    uinfo = info(name)
    return uinfo['passwd'] == password

def set_warndays(name, warndays):
    if False:
        while True:
            i = 10
    "\n    Set the number of days of warning before a password change is required.\n    See man passwd.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_warndays username 7\n    "
    pre_info = info(name)
    if warndays == pre_info['warn']:
        return True
    cmd = 'passwd -w {} {}'.format(warndays, name)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    if post_info['warn'] != pre_info['warn']:
        return post_info['warn'] == warndays
    return False