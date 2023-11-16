"""
Manage accounts in Samba's passdb using pdbedit

:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      posix

.. versionadded:: 2017.7.0
"""
import binascii
import hashlib
import logging
import re
import shlex
import salt.modules.cmdmod
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'pdbedit'
__func_alias__ = {'list_users': 'list', 'get_user': 'get'}

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Provides pdbedit if available\n    '
    if not salt.utils.path.which('pdbedit'):
        return (False, 'pdbedit command is not available')
    ver = salt.modules.cmdmod.run('pdbedit -V')
    ver_regex = re.compile('^Version\\s(\\d+)\\.(\\d+)\\.(\\d+).*$')
    ver_match = ver_regex.match(ver)
    if not ver_match:
        return (False, 'pdbedit -V returned an unknown version format')
    if not (int(ver_match.group(1)) >= 4 and int(ver_match.group(2)) >= 5):
        return (False, 'pdbedit is to old, 4.5.0 or newer is required')
    try:
        hashlib.new('md4', ''.encode('utf-16le'))
    except ValueError:
        return (False, 'Hash type md4 unsupported')
    return __virtualname__

def generate_nt_hash(password):
    if False:
        return 10
    "\n    Generate a NT HASH\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.generate_nt_hash my_passwd\n    "
    return binascii.hexlify(hashlib.new('md4', password.encode('utf-16le')).digest()).upper()

def list_users(verbose=True, hashes=False):
    if False:
        print('Hello World!')
    "\n    List user accounts\n\n    verbose : boolean\n        return all information\n    hashes : boolean\n        include NT HASH and LM HASH in verbose output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.list\n    "
    users = {} if verbose else []
    if verbose:
        res = __salt__['cmd.run_all']('pdbedit --list --verbose {hashes}'.format(hashes='--smbpasswd-style' if hashes else ''))
        if res['retcode'] > 0:
            log.error(res['stderr'] if 'stderr' in res else res['stdout'])
            return users
        for batch in re.split('\n-+|-+\n', res['stdout']):
            user_data = {}
            last_label = None
            for line in batch.splitlines():
                if not line.strip():
                    continue
                (label, sep, data) = line.partition(':')
                label = label.strip().lower()
                data = data.strip()
                if not sep:
                    user_data[last_label] += line.strip()
                else:
                    last_label = label
                    user_data[label] = data
            if user_data:
                users[user_data['unix username']] = user_data
    else:
        res = __salt__['cmd.run_all']('pdbedit --list')
        if res['retcode'] > 0:
            return {'Error': res['stderr'] if 'stderr' in res else res['stdout']}
        for user in res['stdout'].splitlines():
            if ':' not in user:
                continue
            user_data = user.split(':')
            if len(user_data) >= 3:
                users.append(user_data[0])
    return users

def get_user(login, hashes=False):
    if False:
        while True:
            i = 10
    "\n    Get user account details\n\n    login : string\n        login name\n    hashes : boolean\n        include NTHASH and LMHASH in verbose output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.get kaylee\n    "
    users = list_users(verbose=True, hashes=hashes)
    return users[login] if login in users else {}

def delete(login):
    if False:
        i = 10
        return i + 15
    "\n    Delete user account\n\n    login : string\n        login name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.delete wash\n    "
    if login in list_users(False):
        res = __salt__['cmd.run_all'](f'pdbedit --delete {shlex.quote(login)}')
        if res['retcode'] > 0:
            return {login: res['stderr'] if 'stderr' in res else res['stdout']}
        return {login: 'deleted'}
    return {login: 'absent'}

def create(login, password, password_hashed=False, machine_account=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create user account\n\n    login : string\n        login name\n    password : string\n        password\n    password_hashed : boolean\n        set if password is a nt hash instead of plain text\n    machine_account : boolean\n        set to create a machine trust account instead\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.create zoe 9764951149F84E770889011E1DC4A927 nthash\n        salt '*' pdbedit.create river  1sw4ll0w3d4bug\n    "
    ret = 'unchanged'
    if password_hashed:
        password_hash = password.upper()
        password = ''
    else:
        password_hash = generate_nt_hash(password).decode('ascii')
    if login not in list_users(False):
        res = __salt__['cmd.run_all'](cmd='pdbedit --create --user {login} -t {machine}'.format(login=shlex.quote(login), machine='--machine' if machine_account else ''), stdin='{password}\n{password}\n'.format(password=password))
        if res['retcode'] > 0:
            return {login: res['stderr'] if 'stderr' in res else res['stdout']}
        ret = 'created'
    user = get_user(login, True)
    if user['nt hash'] != password_hash:
        res = __salt__['cmd.run_all']('pdbedit --modify --user {login} --set-nt-hash={nthash}'.format(login=shlex.quote(login), nthash=shlex.quote(password_hash)))
        if res['retcode'] > 0:
            return {login: res['stderr'] if 'stderr' in res else res['stdout']}
        if ret != 'created':
            ret = 'updated'
    return {login: ret}

def modify(login, password=None, password_hashed=False, domain=None, profile=None, script=None, drive=None, homedir=None, fullname=None, account_desc=None, account_control=None, machine_sid=None, user_sid=None, reset_login_hours=False, reset_bad_password_count=False):
    if False:
        i = 10
        return i + 15
    "\n    Modify user account\n\n    login : string\n        login name\n    password : string\n        password\n    password_hashed : boolean\n        set if password is a nt hash instead of plain text\n    domain : string\n        users domain\n    profile : string\n        profile path\n    script : string\n        logon script\n    drive : string\n        home drive\n    homedir : string\n        home directory\n    fullname : string\n        full name\n    account_desc : string\n        account description\n    machine_sid : string\n        specify the machines new primary group SID or rid\n    user_sid : string\n        specify the users new primary group SID or rid\n    account_control : string\n        specify user account control properties\n\n        .. note::\n            Only the following can be set:\n            - N: No password required\n            - D: Account disabled\n            - H: Home directory required\n            - L: Automatic Locking\n            - X: Password does not expire\n    reset_login_hours : boolean\n        reset the users allowed logon hours\n    reset_bad_password_count : boolean\n        reset the stored bad login counter\n\n    .. note::\n        if user is absent and password is provided, the user will be created\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pdbedit.modify inara fullname='Inara Serra'\n        salt '*' pdbedit.modify simon password=r1v3r\n        salt '*' pdbedit.modify jane drive='V:' homedir='\\\\serenity\\jane\\profile'\n        salt '*' pdbedit.modify mal account_control=NX\n    "
    ret = 'unchanged'
    flags = {'domain': '--domain=', 'full name': '--fullname=', 'account desc': '--account-desc=', 'home directory': '--homedir=', 'homedir drive': '--drive=', 'profile path': '--profile=', 'logon script': '--script=', 'account flags': '--account-control=', 'user sid': '-U ', 'machine sid': '-M '}
    provided = {'domain': domain, 'full name': fullname, 'account desc': account_desc, 'home directory': homedir, 'homedir drive': drive, 'profile path': profile, 'logon script': script, 'account flags': account_control, 'user sid': user_sid, 'machine sid': machine_sid}
    if password:
        ret = create(login, password, password_hashed)[login]
        if ret not in ['updated', 'created', 'unchanged']:
            return {login: ret}
    elif login not in list_users(False):
        return {login: 'absent'}
    current = get_user(login, hashes=True)
    changes = {}
    for (key, val) in provided.items():
        if key in ['user sid', 'machine sid']:
            if val is not None and key in current and (not current[key].endswith(str(val))):
                changes[key] = str(val)
        elif key in ['account flags']:
            if val is not None:
                if val.startswith('['):
                    val = val[1:-1]
                new = []
                for f in val.upper():
                    if f not in ['N', 'D', 'H', 'L', 'X']:
                        log.warning('pdbedit.modify - unknown %s flag for account_control, ignored', f)
                    else:
                        new.append(f)
                changes[key] = '[{flags}]'.format(flags=''.join(new))
        elif val is not None and key in current and (current[key] != val):
            changes[key] = val
    if len(changes) > 0 or reset_login_hours or reset_bad_password_count:
        cmds = []
        for change in changes:
            cmds.append('{flag}{value}'.format(flag=flags[change], value=shlex.quote(changes[change])))
        if reset_login_hours:
            cmds.append('--logon-hours-reset')
        if reset_bad_password_count:
            cmds.append('--bad-password-count-reset')
        res = __salt__['cmd.run_all']('pdbedit --modify --user {login} {changes}'.format(login=shlex.quote(login), changes=' '.join(cmds)))
        if res['retcode'] > 0:
            return {login: res['stderr'] if 'stderr' in res else res['stdout']}
        if ret != 'created':
            ret = 'updated'
    return {login: ret}