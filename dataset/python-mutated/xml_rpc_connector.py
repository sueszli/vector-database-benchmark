import xmlrpclib
from kams_erp.models.kamserp_config import ODOO_DATABASE_CONNECTOR as DATABASE_CONNECTOR, ODOO_DATABASE_USER as DATABASE_USER, ODOO_DATABASE_PASSWORD as DATABASE_PASSWORD, ODOO_DATABASE_ADDRESS as DATABASE_ADDRESS, ODOO_DATABASE_DATABASE_NAME as DATABASE_DATABASE_NAME, ODOO_PORT as PORT

class XmlRpcConnector(object):
    """
    Class responsible for connection with odoo database.
    """

    def __init__(self, connector=DATABASE_CONNECTOR, user=DATABASE_USER, password=DATABASE_PASSWORD, address=DATABASE_ADDRESS, dbname=DATABASE_DATABASE_NAME, port=PORT):
        if False:
            print('Hello World!')
        '\n        OpenERP Common login Service proxy object.\n\n        :param connector: connector for database.\n        :param user:  the database user name.\n        :param password: the password of the Odoo user.\n        :param dbname: the odoo database name.\n        '
        self.connector = connector
        self.user = user
        self.password = password
        self.address = address
        self.dbname = dbname
        self.port = port
        address_server = self.connector + '://' + self.address + ':' + port
        sock_common = xmlrpclib.ServerProxy('{}/xmlrpc/2/common'.format(address_server))
        self.uid = sock_common.authenticate(self.dbname, self.user, self.password, {})
        self.sock = xmlrpclib.ServerProxy('{}/xmlrpc/2/object'.format(address_server))

    def get_sock(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets socket of RPC connector.\n\n        :return: the socket.\n        '
        return self.sock

    def create(self, table, data_record):
        if False:
            print('Hello World!')
        '\n        Executes data via xml rpc connector.\n\n        :param table: the table name to execute data record.\n        :param data_record: data to execute.\n        :return: return result of execution.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'create', data_record)

    def delete(self, table, data_record):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove data via xml rpc connector.\n\n        :param table: the table name to remove data record.\n        :param data_record: data to remove.\n        :return: return result of remove operation.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'unlink', data_record)

    def search(self, table, data_record):
        if False:
            i = 10
            return i + 15
        '\n        Search data via xml rpc connector.\n\n        :param table: the table name to search data record.\n        :param data_record: data to search.\n        :return: return result of search operation.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'search', data_record)

    def read(self, table, data_record):
        if False:
            i = 10
            return i + 15
        '\n        Read data via xml rpc connector.\n\n        :param table: the table name to read data record.\n        :param data_record: data to read.\n        :return: return result of read operation.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'read', data_record)

    def update(self, table, data_to_update):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates data via xml rpc connector.\n\n        :param table: the table name to read data record.\n        :param data_to_update: data to update.\n        :return: return result of update operation.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'write', data_to_update)

    def custom(self, table, method, data):
        if False:
            while True:
                i = 10
        '\n        Custom template method to create own xml rpc executions.\n        :param table: the table name to execute data record.\n        :param method: method name of model/table\n        :param data: data to execute\n        :return: return result of update execution.\n        '
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, method, data)