# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import create_session

from kams_erp.models.kamserp_config import DATABASE_CONNECTOR, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, \
    DATABASE_DATABASE_NAME, DATABASE_CHARSET


class DatabaseConnector(object):
    """
    Class responsible for connection with database, provide also cron task with fetching data.
    """

    def __init__(self, connector=DATABASE_CONNECTOR, user=DATABASE_USER,
                 password=DATABASE_PASSWORD, host=DATABASE_HOST,
                 dbname=DATABASE_DATABASE_NAME, charset=DATABASE_CHARSET, driver=None):
        """
        Create and engine and get the metadata.

        :param connector: connector for database: 'mysql', 'sqlite' etc...
        :param user: database user name
        :param password: password for user
        :param host: host name
        :param dbname: database name
        """
        if driver is None:
            self.engine = create_engine(
                connector + '://' + user + ':' + password + '@' + host + '/' + dbname + charset, echo=True)
        else:
            self.engine = create_engine(
                connector + '://' + user + ':' + password + '@' + host + '/' + dbname + driver, echo=True)
        self.metadata = MetaData(bind=self.engine)
        self.session = create_session(bind=self.engine)

    def get_engine(self):
        """
        Gets engine of sqlalchemy, which provide connection with database.

        :return: the engine.
        """
        return self.engine
