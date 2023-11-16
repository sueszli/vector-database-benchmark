from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: oracle_ping\nshort_description: Check remote Oracle server availability\ndescription:\n- Simple module to check remote Oracle server availability.\n\nrequirements:\n  - "oracledb"\n'
EXAMPLES = '\n- name: >\n    Ping Oracle server using non-default credentials and SSL\n    registering the return values into the result variable for future use\n  oracle_ping:\n    login_host: jumpserver\n    login_port: 1521\n    login_user: jms\n    login_password: secret_pass\n    login_database: test_db\n'
RETURN = "\nis_available:\n  description: Oracle server availability.\n  returned: always\n  type: bool\n  sample: true\nserver_version:\n  description: Oracle server version.\n  returned: always\n  type: str\n  sample: '4.0.0'\nconn_err_msg:\n  description: Connection error message.\n  returned: always\n  type: str\n  sample: ''\n"
from ansible.module_utils.basic import AnsibleModule
from ops.ansible.modules_utils.oracle_common import OracleClient, oracle_common_argument_spec

class OracleDBPing(object):

    def __init__(self, module, oracle_client):
        if False:
            while True:
                i = 10
        self.module = module
        self.oracle_client = oracle_client
        self.is_available = False
        self.conn_err_msg = ''
        self.version = ''

    def do(self):
        if False:
            print('Hello World!')
        self.get_oracle_version()
        return (self.is_available, self.version)

    def get_err(self):
        if False:
            print('Hello World!')
        return self.conn_err_msg

    def get_oracle_version(self):
        if False:
            print('Hello World!')
        version_sql = 'SELECT VERSION FROM PRODUCT_COMPONENT_VERSION where ROWNUM=1'
        (rtn, err) = self.oracle_client.execute(version_sql)
        if err:
            self.conn_err_msg = err
        else:
            self.version = rtn.get('version')
            self.is_available = True

def main():
    if False:
        while True:
            i = 10
    argument_spec = oracle_common_argument_spec()
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    result = {'changed': False, 'is_available': False, 'server_version': ''}
    oracle_client = OracleClient(module)
    oracle_ping = OracleDBPing(module, oracle_client)
    (result['is_available'], result['server_version']) = oracle_ping.do()
    conn_err_msg = oracle_ping.get_err()
    oracle_client.close()
    if conn_err_msg:
        module.fail_json(msg='Unable to connect to database: %s' % conn_err_msg)
    return module.exit_json(**result)
if __name__ == '__main__':
    main()