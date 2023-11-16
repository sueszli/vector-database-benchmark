"""
Module for Solaris' Role-Based Access Control
"""
import logging
import salt.utils.files
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'rbac'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Provides rbac if we are running on a solaris like system\n    '
    if __grains__['kernel'] == 'SunOS' and salt.utils.path.which('profiles'):
        return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded on a solaris like system')

def profile_list(default_only=False):
    if False:
        while True:
            i = 10
    "\n    List all available profiles\n\n    default_only : boolean\n        return only default profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.profile_list\n    "
    profiles = {}
    default_profiles = ['All']
    with salt.utils.files.fopen('/etc/security/policy.conf', 'r') as policy_conf:
        for policy in policy_conf:
            policy = salt.utils.stringutils.to_unicode(policy)
            policy = policy.split('=')
            if policy[0].strip() == 'PROFS_GRANTED':
                default_profiles.extend(policy[1].strip().split(','))
    with salt.utils.files.fopen('/etc/security/prof_attr', 'r') as prof_attr:
        for profile in prof_attr:
            profile = salt.utils.stringutils.to_unicode(profile)
            profile = profile.split(':')
            if len(profile) != 5:
                continue
            profiles[profile[0]] = profile[3]
    if default_only:
        for p in [p for p in profiles if p not in default_profiles]:
            del profiles[p]
    return profiles

def profile_get(user, default_hidden=True):
    if False:
        while True:
            i = 10
    "\n    List profiles for user\n\n    user : string\n        username\n    default_hidden : boolean\n        hide default profiles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.profile_get leo\n        salt '*' rbac.profile_get leo default_hidden=False\n    "
    user_profiles = []
    with salt.utils.files.fopen('/etc/user_attr', 'r') as user_attr:
        for profile in user_attr:
            profile = salt.utils.stringutils.to_unicode(profile)
            profile = profile.strip().split(':')
            if len(profile) != 5:
                continue
            if profile[0] != user:
                continue
            attrs = {}
            for attr in profile[4].strip().split(';'):
                (attr_key, attr_val) = attr.strip().split('=')
                if attr_key in ['auths', 'profiles', 'roles']:
                    attrs[attr_key] = attr_val.strip().split(',')
                else:
                    attrs[attr_key] = attr_val
            if 'profiles' in attrs:
                user_profiles.extend(attrs['profiles'])
    if default_hidden:
        for profile in profile_list(default_only=True):
            if profile in user_profiles:
                user_profiles.remove(profile)
    return list(set(user_profiles))

def profile_add(user, profile):
    if False:
        print('Hello World!')
    "\n    Add profile to user\n\n    user : string\n        username\n    profile : string\n        profile name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.profile_add martine 'Primary Administrator'\n        salt '*' rbac.profile_add martine 'User Management,User Security'\n    "
    ret = {}
    profiles = profile.split(',')
    known_profiles = profile_list().keys()
    valid_profiles = [p for p in profiles if p in known_profiles]
    log.debug('rbac.profile_add - profiles=%s, known_profiles=%s, valid_profiles=%s', profiles, known_profiles, valid_profiles)
    if valid_profiles:
        res = __salt__['cmd.run_all']('usermod -P "{profiles}" {login}'.format(login=user, profiles=','.join(set(profile_get(user) + valid_profiles))))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_profiles = profile_get(user, False)
    for p in profiles:
        if p not in valid_profiles:
            ret[p] = 'Unknown'
        elif p in active_profiles:
            ret[p] = 'Added'
        else:
            ret[p] = 'Failed'
    return ret

def profile_rm(user, profile):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove profile from user\n\n    user : string\n        username\n    profile : string\n        profile name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.profile_rm jorge 'Primary Administrator'\n        salt '*' rbac.profile_rm jorge 'User Management,User Security'\n    "
    ret = {}
    profiles = profile.split(',')
    known_profiles = profile_list().keys()
    valid_profiles = [p for p in profiles if p in known_profiles]
    log.debug('rbac.profile_rm - profiles=%s, known_profiles=%s, valid_profiles=%s', profiles, known_profiles, valid_profiles)
    if valid_profiles:
        res = __salt__['cmd.run_all']('usermod -P "{profiles}" {login}'.format(login=user, profiles=','.join([p for p in profile_get(user) if p not in valid_profiles])))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_profiles = profile_get(user, False)
    for p in profiles:
        if p not in valid_profiles:
            ret[p] = 'Unknown'
        elif p in active_profiles:
            ret[p] = 'Failed'
        else:
            ret[p] = 'Remove'
    return ret

def role_list():
    if False:
        i = 10
        return i + 15
    "\n    List all available roles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.role_list\n    "
    roles = {}
    with salt.utils.files.fopen('/etc/user_attr', 'r') as user_attr:
        for role in user_attr:
            role = salt.utils.stringutils.to_unicode(role)
            role = role.split(':')
            if len(role) != 5:
                continue
            attrs = {}
            for attr in role[4].split(';'):
                (attr_key, attr_val) = attr.split('=')
                if attr_key in ['auths', 'profiles', 'roles']:
                    attrs[attr_key] = attr_val.split(',')
                else:
                    attrs[attr_key] = attr_val
            role[4] = attrs
            if 'type' in role[4] and role[4]['type'] == 'role':
                del role[4]['type']
                roles[role[0]] = role[4]
    return roles

def role_get(user):
    if False:
        print('Hello World!')
    "\n    List roles for user\n\n    user : string\n        username\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.role_get leo\n    "
    user_roles = []
    with salt.utils.files.fopen('/etc/user_attr', 'r') as user_attr:
        for role in user_attr:
            role = salt.utils.stringutils.to_unicode(role)
            role = role.strip().strip().split(':')
            if len(role) != 5:
                continue
            if role[0] != user:
                continue
            attrs = {}
            for attr in role[4].strip().split(';'):
                (attr_key, attr_val) = attr.strip().split('=')
                if attr_key in ['auths', 'profiles', 'roles']:
                    attrs[attr_key] = attr_val.strip().split(',')
                else:
                    attrs[attr_key] = attr_val
            if 'roles' in attrs:
                user_roles.extend(attrs['roles'])
    return list(set(user_roles))

def role_add(user, role):
    if False:
        while True:
            i = 10
    "\n    Add role to user\n\n    user : string\n        username\n    role : string\n        role name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.role_add martine netcfg\n        salt '*' rbac.role_add martine netcfg,zfssnap\n    "
    ret = {}
    roles = role.split(',')
    known_roles = role_list().keys()
    valid_roles = [r for r in roles if r in known_roles]
    log.debug('rbac.role_add - roles=%s, known_roles=%s, valid_roles=%s', roles, known_roles, valid_roles)
    if valid_roles:
        res = __salt__['cmd.run_all']('usermod -R "{roles}" {login}'.format(login=user, roles=','.join(set(role_get(user) + valid_roles))))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_roles = role_get(user)
    for r in roles:
        if r not in valid_roles:
            ret[r] = 'Unknown'
        elif r in active_roles:
            ret[r] = 'Added'
        else:
            ret[r] = 'Failed'
    return ret

def role_rm(user, role):
    if False:
        print('Hello World!')
    "\n    Remove role from user\n\n    user : string\n        username\n    role : string\n        role name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.role_rm jorge netcfg\n        salt '*' rbac.role_rm jorge netcfg,zfssnap\n    "
    ret = {}
    roles = role.split(',')
    known_roles = role_list().keys()
    valid_roles = [r for r in roles if r in known_roles]
    log.debug('rbac.role_rm - roles=%s, known_roles=%s, valid_roles=%s', roles, known_roles, valid_roles)
    if valid_roles:
        res = __salt__['cmd.run_all']('usermod -R "{roles}" {login}'.format(login=user, roles=','.join([r for r in role_get(user) if r not in valid_roles])))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_roles = role_get(user)
    for r in roles:
        if r not in valid_roles:
            ret[r] = 'Unknown'
        elif r in active_roles:
            ret[r] = 'Failed'
        else:
            ret[r] = 'Remove'
    return ret

def auth_list():
    if False:
        while True:
            i = 10
    "\n    List all available authorization\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.auth_list\n    "
    auths = {}
    with salt.utils.files.fopen('/etc/security/auth_attr', 'r') as auth_attr:
        for auth in auth_attr:
            auth = salt.utils.stringutils.to_unicode(auth)
            auth = auth.split(':')
            if len(auth) != 6:
                continue
            if auth[0][-1:] == '.':
                auth[0] = f'{auth[0]}*'
            auths[auth[0]] = auth[3]
    return auths

def auth_get(user, computed=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    List authorization for user\n\n    user : string\n        username\n    computed : boolean\n        merge results from `auths` command into data from user_attr\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.auth_get leo\n    "
    user_auths = []
    with salt.utils.files.fopen('/etc/user_attr', 'r') as user_attr:
        for auth in user_attr:
            auth = salt.utils.stringutils.to_unicode(auth)
            auth = auth.strip().split(':')
            if len(auth) != 5:
                continue
            if auth[0] != user:
                continue
            attrs = {}
            for attr in auth[4].strip().split(';'):
                (attr_key, attr_val) = attr.strip().split('=')
                if attr_key in ['auths', 'profiles', 'roles']:
                    attrs[attr_key] = attr_val.strip().split(',')
                else:
                    attrs[attr_key] = attr_val
            if 'auths' in attrs:
                user_auths.extend(attrs['auths'])
    if computed:
        res = __salt__['cmd.run_all'](f'auths {user}')
        if res['retcode'] == 0:
            for auth in res['stdout'].splitlines():
                if ',' in auth:
                    user_auths.extend(auth.strip().split(','))
                else:
                    user_auths.append(auth.strip())
    return list(set(user_auths))

def auth_add(user, auth):
    if False:
        print('Hello World!')
    "\n    Add authorization to user\n\n    user : string\n        username\n    auth : string\n        authorization name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.auth_add martine solaris.zone.manage\n        salt '*' rbac.auth_add martine solaris.zone.manage,solaris.mail.mailq\n    "
    ret = {}
    auths = auth.split(',')
    known_auths = auth_list().keys()
    valid_auths = [r for r in auths if r in known_auths]
    log.debug('rbac.auth_add - auths=%s, known_auths=%s, valid_auths=%s', auths, known_auths, valid_auths)
    if valid_auths:
        res = __salt__['cmd.run_all']('usermod -A "{auths}" {login}'.format(login=user, auths=','.join(set(auth_get(user, False) + valid_auths))))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_auths = auth_get(user, False)
    for a in auths:
        if a not in valid_auths:
            ret[a] = 'Unknown'
        elif a in active_auths:
            ret[a] = 'Added'
        else:
            ret[a] = 'Failed'
    return ret

def auth_rm(user, auth):
    if False:
        print('Hello World!')
    "\n    Remove authorization from user\n\n    user : string\n        username\n    auth : string\n        authorization name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbac.auth_rm jorge solaris.zone.manage\n        salt '*' rbac.auth_rm jorge solaris.zone.manage,solaris.mail.mailq\n    "
    ret = {}
    auths = auth.split(',')
    known_auths = auth_list().keys()
    valid_auths = [a for a in auths if a in known_auths]
    log.debug('rbac.auth_rm - auths=%s, known_auths=%s, valid_auths=%s', auths, known_auths, valid_auths)
    if valid_auths:
        res = __salt__['cmd.run_all']('usermod -A "{auths}" {login}'.format(login=user, auths=','.join([a for a in auth_get(user, False) if a not in valid_auths])))
        if res['retcode'] > 0:
            ret['Error'] = {'retcode': res['retcode'], 'message': res['stderr'] if 'stderr' in res else res['stdout']}
            return ret
    active_auths = auth_get(user, False)
    for a in auths:
        if a not in valid_auths:
            ret[a] = 'Unknown'
        elif a in active_auths:
            ret[a] = 'Failed'
        else:
            ret[a] = 'Remove'
    return ret