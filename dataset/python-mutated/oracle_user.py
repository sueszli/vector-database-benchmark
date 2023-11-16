from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: oracle_user\nshort_description: Adds or removes a user from a Oracle database\ndescription:\n    - Adds or removes a user from a Oracle database.\n\noptions:\n  authentication_type:\n    description:\n        - Authentication type of the user(default password)\n    required: false\n    type: str\n    choices: [\'external\', \'global\', \'no_authentication\', \'password\']\n  default_tablespace:\n    description:\n        - The default tablespace for the user\n        - If not provided, the default is used\n    required: false\n    type: str\n  oracle_home:\n    description:\n        - Define the directory into which all Oracle software is installed.\n        - Define ORACLE_HOME environment variable if set.\n    type: str\n  state:\n    description:\n      - The database user state.\n    default: present\n    choices: [absent, present]\n    type: str\n  update_password:\n    default: always\n    choices: [always, on_create]\n    description:\n      - C(always) will always update passwords and cause the module to return changed.\n      - C(on_create) will only set the password for newly created users.\n    type: str\n  temporary_tablespace:\n    description:\n        - The default temporary tablespace for the user\n        - If not provided, the default is used\n    required: false\n    type: str\n  name:\n    description:\n      - The name of the user to add or remove.\n    required: true\n    aliases: [user]\n    type: str\n  password:\n    description:\n      - The password to use for the user.\n    type: str\n    aliases: [pass]\n    \nrequirements:\n  - "oracledb"\n'
EXAMPLES = '\n- name: Create default tablespace user with name \'jms\' and password \'123456\'.\n  oracle_user:\n    hostname: "remote server"\n    login_database: "helowin"\n    login_user: "system"\n    login_password: "123456"\n    name: "jms"\n    password: "123456"\n\n- name: Delete user with name \'jms\'.\n  oracle_user:\n    hostname: "remote server"\n    login_database: "helowin"\n    login_user: "system"\n    login_password: "123456"\n    name: "jms"\n    state: "absent"\n'
RETURN = '\nname:\n    description: The name of the user to add or remove.\n    returned: success\n    type: str\n'
from ansible.module_utils.basic import AnsibleModule
from ops.ansible.modules_utils.oracle_common import OracleClient, oracle_common_argument_spec

def user_find(oracle_client, username):
    if False:
        while True:
            i = 10
    user = None
    username = username.upper()
    user_find_sql = "select username,        authentication_type,        default_tablespace,        temporary_tablespace from dba_users where username='%s'" % username
    (rtn, err) = oracle_client.execute(user_find_sql)
    if isinstance(rtn, dict):
        user = rtn
    return user

def user_add(module, oracle_client, username, password, auth_type, default_tablespace, temporary_tablespace):
    if False:
        while True:
            i = 10
    username = username.upper()
    extend_sql = None
    user = user_find(oracle_client, username)
    auth_type = auth_type.lower()
    identified_suffix_map = {'external': 'identified externally ', 'global': 'identified globally ', 'password': 'identified by "%s" '}
    if user:
        user_sql = 'alter user %s ' % username
        user_sql += identified_suffix_map.get(auth_type, 'no authentication ') % password
        if default_tablespace and default_tablespace.lower() != user['default_tablespace'].lower():
            user_sql += 'default tablespace %s quota unlimited on %s ' % (default_tablespace, default_tablespace)
        if temporary_tablespace and temporary_tablespace.lower() != user['temporary_tablespace'].lower():
            user_sql += 'temporary tablespace %s ' % temporary_tablespace
    else:
        user_sql = 'create user %s ' % username
        user_sql += identified_suffix_map.get(auth_type, 'no authentication ') % password
        if default_tablespace:
            user_sql += 'default tablespace %s quota unlimited on %s ' % (default_tablespace, default_tablespace)
        if temporary_tablespace:
            user_sql += 'temporary tablespace %s ' % temporary_tablespace
        extend_sql = 'grant connect to %s' % username
    (rtn, err) = oracle_client.execute(user_sql)
    if err:
        module.fail_json(msg='Cannot add/edit user %s: %s' % (username, err), changed=False)
    else:
        if extend_sql:
            oracle_client.execute(extend_sql)
        module.exit_json(msg='User %s has been created.' % username, changed=True, name=username)

def user_remove(module, oracle_client, username):
    if False:
        i = 10
        return i + 15
    user = user_find(oracle_client, username)
    if user:
        (rtn, err) = oracle_client.execute('drop user %s cascade' % username)
        if err:
            module.fail_json(msg='Cannot drop user %s: %s' % (username, err), changed=False)
        else:
            module.exit_json(msg='User %s dropped.' % username, changed=True, name=username)
    else:
        module.exit_json(msg="User %s doesn't exist." % username, changed=False, name=username)

def main():
    if False:
        i = 10
        return i + 15
    argument_spec = oracle_common_argument_spec()
    argument_spec.update(authentication_type=dict(type='str', required=False, choices=['external', 'global', 'no_authentication', 'password']), default_tablespace=dict(required=False, aliases=['db']), name=dict(required=True, aliases=['user']), password=dict(aliases=['pass'], no_log=True), state=dict(type='str', default='present', choices=['absent', 'present']), update_password=dict(default='always', choices=['always', 'on_create'], no_log=False), temporary_tablespace=dict(type='str', default=None))
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    authentication_type = module.params['authentication_type'] or 'password'
    default_tablespace = module.params['default_tablespace']
    user = module.params['name']
    password = module.params['password']
    state = module.params['state']
    update_password = module.params['update_password']
    temporary_tablespace = module.params['temporary_tablespace']
    oracle_client = OracleClient(module)
    if state == 'present':
        if password is None and update_password == 'always':
            module.fail_json(msg='password parameter required when adding a user unless update_password is set to on_create')
        user_add(module, oracle_client, username=user, password=password, auth_type=authentication_type, default_tablespace=default_tablespace, temporary_tablespace=temporary_tablespace)
    elif state == 'absent':
        user_remove(oracle_client)
    module.exit_json(changed=True, user=user)
if __name__ == '__main__':
    main()