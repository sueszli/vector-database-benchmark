from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: mongodb_user\nshort_description: Adds or removes a user from a MongoDB database\ndescription:\n    - Adds or removes a user from a MongoDB database.\nversion_added: "1.0.0"\n\nextends_documentation_fragment:\n  - community.mongodb.login_options\n  - community.mongodb.ssl_options\n\noptions:\n  replica_set:\n    description:\n      - Replica set to connect to (automatically connects to primary for writes).\n    type: str\n  database:\n    description:\n      - The name of the database to add/remove the user from.\n    required: true\n    type: str\n    aliases: [db]\n  name:\n    description:\n      - The name of the user to add or remove.\n    required: true\n    aliases: [user]\n    type: str\n  password:\n    description:\n      - The password to use for the user.\n    type: str\n    aliases: [pass]\n  roles:\n    type: list\n    elements: raw\n    description:\n      - >\n          The database user roles valid values could either be one or more of the following strings:\n          \'read\', \'readWrite\', \'dbAdmin\', \'userAdmin\', \'clusterAdmin\', \'readAnyDatabase\', \'readWriteAnyDatabase\', \'userAdminAnyDatabase\',\n          \'dbAdminAnyDatabase\'\n      - "Or the following dictionary \'{ db: DATABASE_NAME, role: ROLE_NAME }\'."\n      - "This param requires pymongo 2.5+. If it is a string, mongodb 2.4+ is also required. If it is a dictionary, mongo 2.6+ is required."\n  state:\n    description:\n      - The database user state.\n    default: present\n    choices: [absent, present]\n    type: str\n  update_password:\n    default: always\n    choices: [always, on_create]\n    description:\n      - C(always) will always update passwords and cause the module to return changed.\n      - C(on_create) will only set the password for newly created users.\n      - This must be C(always) to use the localhost exception when adding the first admin user.\n      - This option is effectively ignored when using x.509 certs. It is defaulted to \'on_create\' to maintain a           a specific module behaviour when the login_database is \'$external\'.\n    type: str\n  create_for_localhost_exception:\n    type: path\n    description:\n      - This is parmeter is only useful for handling special treatment around the localhost exception.\n      - If C(login_user) is defined, then the localhost exception is not active and this parameter has no effect.\n      - If this file is NOT present (and C(login_user) is not defined), then touch this file after successfully adding the user.\n      - If this file is present (and C(login_user) is not defined), then skip this task.\n\nnotes:\n    - Requires the pymongo Python package on the remote host, version 2.4.2+. This\n      can be installed using pip or the OS package manager. Newer mongo server versions require newer\n      pymongo versions. @see http://api.mongodb.org/python/current/installation.html\nrequirements:\n  - "pymongo"\nauthor:\n    - "Elliott Foster (@elliotttf)"\n    - "Julien Thebault (@Lujeni)"\n'
EXAMPLES = '\n- name: Create \'burgers\' database user with name \'bob\' and password \'12345\'.\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: bob\n    password: 12345\n    state: present\n\n- name: Create a database user via SSL (MongoDB must be compiled with the SSL option and configured properly)\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: bob\n    password: 12345\n    state: present\n    ssl: True\n\n- name: Delete \'burgers\' database user with name \'bob\'.\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: bob\n    state: absent\n\n- name: Define more users with various specific roles (if not defined, no roles is assigned, and the user will be added via pre mongo 2.2 style)\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: ben\n    password: 12345\n    roles: read\n    state: present\n\n- name: Define roles\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: jim\n    password: 12345\n    roles: readWrite,dbAdmin,userAdmin\n    state: present\n\n- name: Define roles\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: joe\n    password: 12345\n    roles: readWriteAnyDatabase\n    state: present\n\n- name: Add a user to database in a replica set, the primary server is automatically discovered and written to\n  community.mongodb.mongodb_user:\n    database: burgers\n    name: bob\n    replica_set: belcher\n    password: 12345\n    roles: readWriteAnyDatabase\n    state: present\n\n# add a user \'oplog_reader\' with read only access to the \'local\' database on the replica_set \'belcher\'. This is useful for oplog access (MONGO_OPLOG_URL).\n# please notice the credentials must be added to the \'admin\' database because the \'local\' database is not synchronized and can\'t receive user credentials\n# To login with such user, the connection string should be MONGO_OPLOG_URL="mongodb://oplog_reader:oplog_reader_password@server1,server2/local?authSource=admin"\n# This syntax requires mongodb 2.6+ and pymongo 2.5+\n- name: Roles as a dictionary\n  community.mongodb.mongodb_user:\n    login_user: root\n    login_password: root_password\n    database: admin\n    user: oplog_reader\n    password: oplog_reader_password\n    state: present\n    replica_set: belcher\n    roles:\n      - db: local\n        role: read\n\n- name: Adding a user with X.509 Member Authentication\n  community.mongodb.mongodb_user:\n    login_host: "mongodb-host.test"\n    login_port: 27001\n    login_database: "$external"\n    database: "admin"\n    name: "admin"\n    password: "test"\n    roles:\n    - dbAdminAnyDatabase\n    ssl: true\n    ssl_ca_certs: "/tmp/ca.crt"\n    ssl_certfile: "/tmp/tls.key" #cert and key in one file\n    state: present\n    auth_mechanism: "MONGODB-X509"\n    connection_options:\n     - "tlsAllowInvalidHostnames=true"\n'
RETURN = '\nuser:\n    description: The name of the user to add or remove.\n    returned: success\n    type: str\n'
import os
import traceback
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native, to_bytes
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import missing_required_lib, mongodb_common_argument_spec, mongo_auth, PYMONGO_IMP_ERR, pymongo_found, get_mongodb_client

def user_find(client, user, db_name):
    if False:
        return 10
    "Check if the user exists.\n\n    Args:\n        client (cursor): Mongodb cursor on admin database.\n        user (str): User to check.\n        db_name (str): User's database.\n\n    Returns:\n        dict: when user exists, False otherwise.\n    "
    try:
        for mongo_user in client[db_name].command('usersInfo')['users']:
            if mongo_user['user'] == user:
                if 'db' not in mongo_user:
                    return mongo_user
                if mongo_user['db'] in [db_name, 'admin']:
                    return mongo_user
    except Exception as excep:
        if hasattr(excep, 'code') and excep.code == 11:
            pass
        else:
            raise
    return False

def user_add(module, client, db_name, user, password, roles):
    if False:
        print('Hello World!')
    db = client[db_name]
    try:
        exists = user_find(client, user, db_name)
    except Exception as excep:
        if hasattr(excep, 'code') and excep.code == 13:
            exists = False
        else:
            raise
    if exists:
        user_add_db_command = 'updateUser'
        if not roles:
            roles = None
    else:
        user_add_db_command = 'createUser'
    user_dict = {}
    if password is not None:
        user_dict['pwd'] = password
    if roles is not None:
        user_dict['roles'] = roles
    db.command(user_add_db_command, user, **user_dict)

def user_remove(module, client, db_name, user):
    if False:
        print('Hello World!')
    exists = user_find(client, user, db_name)
    if exists:
        if module.check_mode:
            module.exit_json(changed=True, user=user)
        db = client[db_name]
        db.command('dropUser', user)
    else:
        module.exit_json(changed=False, user=user)

def check_if_roles_changed(uinfo, roles, db_name):
    if False:
        print('Hello World!')

    def make_sure_roles_are_a_list_of_dict(roles, db_name):
        if False:
            print('Hello World!')
        output = list()
        for role in roles:
            if isinstance(role, (binary_type, text_type)):
                new_role = {'role': role, 'db': db_name}
                output.append(new_role)
            else:
                output.append(role)
        return output
    roles_as_list_of_dict = make_sure_roles_are_a_list_of_dict(roles, db_name)
    uinfo_roles = uinfo.get('roles', [])
    if sorted(roles_as_list_of_dict, key=itemgetter('db')) == sorted(uinfo_roles, key=itemgetter('db')):
        return False
    return True

def main():
    if False:
        while True:
            i = 10
    argument_spec = mongodb_common_argument_spec()
    argument_spec.update(database=dict(required=True, aliases=['db']), name=dict(required=True, aliases=['user']), password=dict(aliases=['pass'], no_log=True), replica_set=dict(default=None), roles=dict(default=None, type='list', elements='raw'), state=dict(default='present', choices=['absent', 'present']), update_password=dict(default='always', choices=['always', 'on_create'], no_log=False), create_for_localhost_exception=dict(default=None, type='path'))
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    login_user = module.params['login_user']
    if module.params['login_database'] == '$external':
        module.params['update_password'] = 'on_create'
    if not pymongo_found:
        module.fail_json(msg=missing_required_lib('pymongo'), exception=PYMONGO_IMP_ERR)
    create_for_localhost_exception = module.params['create_for_localhost_exception']
    b_create_for_localhost_exception = to_bytes(create_for_localhost_exception, errors='surrogate_or_strict') if create_for_localhost_exception is not None else None
    db_name = module.params['database']
    user = module.params['name']
    password = module.params['password']
    roles = module.params['roles'] or []
    state = module.params['state']
    update_password = module.params['update_password']
    try:
        directConnection = False
        if module.params['replica_set'] is None:
            directConnection = True
        client = get_mongodb_client(module, directConnection=directConnection)
        client = mongo_auth(module, client, directConnection=directConnection)
    except Exception as e:
        module.fail_json(msg='Unable to connect to database: %s' % to_native(e))
    if state == 'present':
        if password is None and update_password == 'always':
            module.fail_json(msg='password parameter required when adding a user unless update_password is set to on_create')
        if login_user is None and create_for_localhost_exception is not None:
            if os.path.exists(b_create_for_localhost_exception):
                try:
                    client.close()
                except Exception:
                    pass
                module.exit_json(changed=False, user=user, skipped=True, msg='The path in create_for_localhost_exception exists.')
        try:
            if update_password != 'always':
                uinfo = user_find(client, user, db_name)
                if uinfo:
                    password = None
                    if not check_if_roles_changed(uinfo, roles, db_name):
                        module.exit_json(changed=False, user=user)
            if module.check_mode:
                module.exit_json(changed=True, user=user)
            user_add(module, client, db_name, user, password, roles)
        except Exception as e:
            module.fail_json(msg='Unable to add or update user: %s' % to_native(e), exception=traceback.format_exc())
        finally:
            try:
                client.close()
            except Exception:
                pass
        if login_user is None and create_for_localhost_exception is not None:
            try:
                open(b_create_for_localhost_exception, 'wb').close()
            except Exception as e:
                module.fail_json(changed=True, msg='Added user but unable to touch create_for_localhost_exception file %s: %s' % (create_for_localhost_exception, to_native(e)), exception=traceback.format_exc())
    elif state == 'absent':
        try:
            user_remove(module, client, db_name, user)
        except Exception as e:
            module.fail_json(msg='Unable to remove user: %s' % to_native(e), exception=traceback.format_exc())
        finally:
            try:
                client.close()
            except Exception:
                pass
    module.exit_json(changed=True, user=user)
if __name__ == '__main__':
    main()