from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: oracle_info\nshort_description: Gather information about Oracle servers\ndescription:\n- Gathers information about Oracle servers.\n\noptions:\n  filter:\n    description:\n    - Limit the collected information by comma separated string or YAML list.\n    - Allowable values are C(version), C(databases), C(settings), C(users).\n    - By default, collects all subsets.\n    - You can use \'!\' before value (for example, C(!users)) to exclude it from the information.\n    - If you pass including and excluding values to the filter, for example, I(filter=!settings,version),\n      the excluding values, C(!settings) in this case, will be ignored.\n    type: list\n    elements: str\n  login_db:\n    description:\n    - Database name to connect to.\n    - It makes sense if I(login_user) is allowed to connect to a specific database only.\n    type: str\n  exclude_fields:\n    description:\n    - List of fields which are not needed to collect.\n    - "Supports elements: C(db_size). Unsupported elements will be ignored."\n    type: list\n    elements: str\n'
EXAMPLES = '\n- name: Get Oracle version with non-default credentials\n  oracle_info:\n    login_user: mysuperuser\n    login_password: mysuperpass\n    login_database: service_name\n    filter: version\n\n- name: Collect all info except settings and users by sys\n  oracle_info:\n    login_user: sys\n    login_password: sys_pass\n    login_database: service_name\n    filter: "!settings,!users"\n    exclude_fields: db_size\n'
RETURN = '\nversion:\n  description: Database server version.\n  returned: if not excluded by filter\n  type: dict\n  sample: { "version": {"full": "11.2.0.1.0"} }\n  contains:\n    full:\n      description: Full server version.\n      returned: if not excluded by filter\n      type: str\n      sample: "11.2.0.1.0"\ndatabases:\n  description: Information about databases.\n  returned: if not excluded by filter\n  type: dict\n  sample:\n  - { "USERS": { "size": 5242880 }, "EXAMPLE": { "size": 104857600 } }\n  contains:\n    size:\n      description: Database size in bytes.\n      returned: if not excluded by filter\n      type: dict\n      sample: { \'size\': 656594 }\nsettings:\n  description: Global settings (variables) information.\n  returned: if not excluded by filter\n  type: dict\n  sample:\n  - { "result_cache_mode": "MANUAL", "instance_type": "RDBMS" }\nusers:\n  description: Users information.\n  returned: if not excluded by filter\n  type: dict\n  sample:\n  - { "USERS": { "TEST": { "USERNAME": "TEST", "ACCOUNT_STATUS": "OPEN" } } }\n'
from ansible.module_utils.basic import AnsibleModule
from ops.ansible.modules_utils.oracle_common import OracleClient, oracle_common_argument_spec

class OracleInfo(object):

    def __init__(self, module, oracle_client):
        if False:
            while True:
                i = 10
        self.module = module
        self.oracle_client = oracle_client
        self.info = {'version': {}, 'databases': {}, 'settings': {}, 'users': {}}

    def get_info(self, filter_, exclude_fields):
        if False:
            for i in range(10):
                print('nop')
        include_list = []
        exclude_list = []
        if filter_:
            partial_info = {}
            for fi in filter_:
                if fi.lstrip('!') not in self.info:
                    self.module.warn('filter element: %s is not allowable, ignored' % fi)
                    continue
                if fi[0] == '!':
                    exclude_list.append(fi.lstrip('!'))
                else:
                    include_list.append(fi)
            if include_list:
                self.__collect(exclude_fields, set(include_list))
                for i in self.info:
                    if i in include_list:
                        partial_info[i] = self.info[i]
            else:
                not_in_exclude_list = list(set(self.info) - set(exclude_list))
                self.__collect(exclude_fields, set(not_in_exclude_list))
                for i in self.info:
                    if i not in exclude_list:
                        partial_info[i] = self.info[i]
            return partial_info
        else:
            self.__collect(exclude_fields, set(self.info))
            return self.info

    def __collect(self, exclude_fields, wanted):
        if False:
            while True:
                i = 10
        'Collect all possible subsets.'
        if 'version' in wanted:
            self.__get_version()
        if 'settings' in wanted:
            self.__get_settings()
        if 'databases' in wanted:
            self.__get_databases(exclude_fields)
        if 'users' in wanted:
            self.__get_users()

    def __get_version(self):
        if False:
            print('Hello World!')
        version_sql = 'SELECT VERSION FROM PRODUCT_COMPONENT_VERSION where ROWNUM=1'
        (rtn, err) = self.oracle_client.execute(version_sql, exception_to_fail=True)
        self.info['version'] = {'full': rtn.get('version')}

    def __get_settings(self):
        if False:
            print('Hello World!')
        'Get global variables (instance settings).'

        def _set_settings_value(item_dict):
            if False:
                print('Hello World!')
            try:
                self.info['settings'][item_dict['name']] = item_dict['value']
            except KeyError:
                pass
        settings_sql = 'SELECT name, value FROM V$PARAMETER'
        (rtn, err) = self.oracle_client.execute(settings_sql, exception_to_fail=True)
        if isinstance(rtn, dict):
            _set_settings_value(rtn)
        elif isinstance(rtn, list):
            for i in rtn:
                _set_settings_value(i)

    def __get_users(self):
        if False:
            i = 10
            return i + 15
        'Get user info.'

        def _set_users_value(item_dict):
            if False:
                return 10
            try:
                tablespace = item_dict.pop('default_tablespace')
                username = item_dict.pop('username')
                partial_users = self.info['users'].get(tablespace, {})
                partial_users[username] = item_dict
                self.info['users'][tablespace] = partial_users
            except KeyError:
                pass
        users_sql = 'SELECT * FROM dba_users'
        (rtn, err) = self.oracle_client.execute(users_sql, exception_to_fail=True)
        if isinstance(rtn, dict):
            _set_users_value(rtn)
        elif isinstance(rtn, list):
            for i in rtn:
                _set_users_value(i)

    def __get_databases(self, exclude_fields):
        if False:
            i = 10
            return i + 15
        'Get info about databases.'

        def _set_databases_value(item_dict):
            if False:
                while True:
                    i = 10
            try:
                tablespace_name = item_dict.pop('tablespace_name')
                size = item_dict.get('size')
                partial_params = {}
                if size:
                    partial_params['size'] = size
                self.info['databases'][tablespace_name] = partial_params
            except KeyError:
                pass
        database_sql = 'SELECT       tablespace_name, sum(bytes) as "size"FROM dba_data_files GROUP BY tablespace_name'
        if exclude_fields and 'db_size' in exclude_fields:
            database_sql = 'SELECT       tablespace_name FROM dba_data_files GROUP BY tablespace_name'
        (rtn, err) = self.oracle_client.execute(database_sql, exception_to_fail=True)
        if isinstance(rtn, dict):
            _set_databases_value(rtn)
        elif isinstance(rtn, list):
            for i in rtn:
                _set_databases_value(i)

def main():
    if False:
        i = 10
        return i + 15
    argument_spec = oracle_common_argument_spec()
    argument_spec.update(filter=dict(type='list'), exclude_fields=dict(type='list'))
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    filter_ = module.params['filter']
    exclude_fields = module.params['exclude_fields']
    if filter_:
        filter_ = [f.strip() for f in filter_]
    if exclude_fields:
        exclude_fields = set([f.strip() for f in exclude_fields])
    oracle_client = OracleClient(module)
    oracle = OracleInfo(module, oracle_client)
    module.exit_json(changed=False, **oracle.get_info(filter_, exclude_fields))
if __name__ == '__main__':
    main()