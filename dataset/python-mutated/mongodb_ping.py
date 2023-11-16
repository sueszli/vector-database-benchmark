from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: mongodb_ping\nshort_description: Check remote MongoDB server availability\ndescription:\n- Simple module to check remote MongoDB server availability.\n\nrequirements:\n  - "pymongo"\n'
EXAMPLES = '\n- name: >\n    Ping MongoDB server using non-default credentials and SSL\n    registering the return values into the result variable for future use\n  mongodb_ping:\n    login_db: test_db\n    login_host: jumpserver\n    login_user: jms\n    login_password: secret_pass\n    ssl: True\n    ssl_ca_certs: "/tmp/ca.crt"\n    ssl_certfile: "/tmp/tls.key" #cert and key in one file\n    connection_options:\n     - "tlsAllowInvalidHostnames=true"\n'
RETURN = "\nis_available:\n  description: MongoDB server availability.\n  returned: always\n  type: bool\n  sample: true\nserver_version:\n  description: MongoDB server version.\n  returned: always\n  type: str\n  sample: '4.0.0'\nconn_err_msg:\n  description: Connection error message.\n  returned: always\n  type: str\n  sample: ''\n"
from pymongo.errors import PyMongoError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import mongodb_common_argument_spec, mongo_auth, get_mongodb_client

class MongoDBPing(object):

    def __init__(self, module, client):
        if False:
            for i in range(10):
                print('nop')
        self.module = module
        self.client = client
        self.is_available = False
        self.conn_err_msg = ''
        self.version = ''

    def do(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_mongodb_version()
        return (self.is_available, self.version)

    def get_err(self):
        if False:
            return 10
        return self.conn_err_msg

    def get_mongodb_version(self):
        if False:
            print('Hello World!')
        try:
            server_info = self.client.server_info()
            self.is_available = True
            self.version = server_info.get('version', '')
        except PyMongoError as err:
            self.is_available = False
            self.version = ''
            self.conn_err_msg = err

def main():
    if False:
        return 10
    argument_spec = mongodb_common_argument_spec()
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    client = None
    result = {'changed': False, 'is_available': False, 'server_version': ''}
    try:
        client = get_mongodb_client(module, directConnection=True)
        client = mongo_auth(module, client, directConnection=True)
    except Exception as e:
        module.fail_json(msg='Unable to connect to database: %s' % to_native(e))
    mongodb_ping = MongoDBPing(module, client)
    (result['is_available'], result['server_version']) = mongodb_ping.do()
    conn_err_msg = mongodb_ping.get_err()
    if conn_err_msg:
        module.fail_json(msg='Unable to connect to database: %s' % conn_err_msg)
    try:
        client.close()
    except Exception:
        pass
    return module.exit_json(**result)
if __name__ == '__main__':
    main()