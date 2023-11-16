# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2017] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
from sqlalchemy.orm import create_session
from kams_erp.models.kamserp_config import SUBIEKT_DATABASE_CONNECTOR, SUBIEKT_DATABASE_USER, SUBIEKT_DATABASE_PASSWORD, \
    SUBIEKT_DATABASE_ADDRESS, SUBIEKT_DATABASE_DATABASE_NAME, SUBIEKT_DATABASE_PORT, SUBIEKT_DATABASE_DRIVER, \
    ODOO_DATABASE_ADDRESS, ODOO_PORT, ODOO_DATABASE_CONNECTOR
from kams_erp.utils.database_connector import DatabaseConnector
from kams_erp.utils.xml_rpc_connector import XmlRpcConnector
from kams_erp.utils.xml_rpc_operations import XmlRpcOperations


class InstallKamsERP_Configuration(object):
    """
    Class responsible for install configuration.
    """

    def __init__(self):
        self.connector = XmlRpcConnector()
        self.xml_operand = XmlRpcOperations()
        self.session = create_session(bind=DatabaseConnector(dbname="kamsbhp_sklep2").get_engine())
        dbobject = DatabaseConnector(connector=SUBIEKT_DATABASE_CONNECTOR,
                                     user=SUBIEKT_DATABASE_USER,
                                     password=SUBIEKT_DATABASE_PASSWORD,
                                     host=SUBIEKT_DATABASE_ADDRESS + ':' + SUBIEKT_DATABASE_PORT,
                                     dbname=SUBIEKT_DATABASE_DATABASE_NAME,
                                     driver=SUBIEKT_DATABASE_DRIVER,
                                     charset=None)
        self.subiekt_session = create_session(bind=dbobject.get_engine())
        self.url = ODOO_DATABASE_CONNECTOR + '://' + ODOO_DATABASE_ADDRESS + ':' + ODOO_PORT
