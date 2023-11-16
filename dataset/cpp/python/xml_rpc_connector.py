# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

import xmlrpclib

from kams_erp.models.kamserp_config import ODOO_DATABASE_CONNECTOR as DATABASE_CONNECTOR, \
    ODOO_DATABASE_USER as DATABASE_USER, ODOO_DATABASE_PASSWORD as DATABASE_PASSWORD, \
    ODOO_DATABASE_ADDRESS as DATABASE_ADDRESS, ODOO_DATABASE_DATABASE_NAME as DATABASE_DATABASE_NAME, \
    ODOO_PORT as PORT


class XmlRpcConnector(object):
    """
    Class responsible for connection with odoo database.
    """

    def __init__(self, connector=DATABASE_CONNECTOR, user=DATABASE_USER,
                 password=DATABASE_PASSWORD, address=DATABASE_ADDRESS,
                 dbname=DATABASE_DATABASE_NAME, port=PORT):
        """
        OpenERP Common login Service proxy object.

        :param connector: connector for database.
        :param user:  the database user name.
        :param password: the password of the Odoo user.
        :param dbname: the odoo database name.
        """
        self.connector = connector
        self.user = user
        self.password = password
        self.address = address
        self.dbname = dbname
        self.port = port

        address_server = self.connector + '://' + self.address + ':' + port
        sock_common = xmlrpclib.ServerProxy('{}/xmlrpc/2/common'.format(address_server))
        self.uid = sock_common.authenticate(self.dbname, self.user, self.password, {})

        # replace localhost with the address of the server if it is not on the same server
        self.sock = xmlrpclib.ServerProxy('{}/xmlrpc/2/object'.format(address_server))

    def get_sock(self):
        """
        Gets socket of RPC connector.

        :return: the socket.
        """
        return self.sock

    def create(self, table, data_record):
        """
        Executes data via xml rpc connector.

        :param table: the table name to execute data record.
        :param data_record: data to execute.
        :return: return result of execution.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'create', data_record)

    def delete(self, table, data_record):
        """
        Remove data via xml rpc connector.

        :param table: the table name to remove data record.
        :param data_record: data to remove.
        :return: return result of remove operation.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'unlink', data_record)

    def search(self, table, data_record):
        """
        Search data via xml rpc connector.

        :param table: the table name to search data record.
        :param data_record: data to search.
        :return: return result of search operation.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'search', data_record)

    def read(self, table, data_record):
        """
        Read data via xml rpc connector.

        :param table: the table name to read data record.
        :param data_record: data to read.
        :return: return result of read operation.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'read', data_record)

    def update(self, table, data_to_update):
        """
        Updates data via xml rpc connector.

        :param table: the table name to read data record.
        :param data_to_update: data to update.
        :return: return result of update operation.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, 'write', data_to_update)

    def custom(self, table, method, data):
        """
        Custom template method to create own xml rpc executions.
        :param table: the table name to execute data record.
        :param method: method name of model/table
        :param data: data to execute
        :return: return result of update execution.
        """
        return self.sock.execute_kw(self.dbname, self.uid, self.password, table, method, data)
