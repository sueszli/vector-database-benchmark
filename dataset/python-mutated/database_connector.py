from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import create_session
from kams_erp.models.kamserp_config import DATABASE_CONNECTOR, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_DATABASE_NAME, DATABASE_CHARSET

class DatabaseConnector(object):
    """
    Class responsible for connection with database, provide also cron task with fetching data.
    """

    def __init__(self, connector=DATABASE_CONNECTOR, user=DATABASE_USER, password=DATABASE_PASSWORD, host=DATABASE_HOST, dbname=DATABASE_DATABASE_NAME, charset=DATABASE_CHARSET, driver=None):
        if False:
            print('Hello World!')
        "\n        Create and engine and get the metadata.\n\n        :param connector: connector for database: 'mysql', 'sqlite' etc...\n        :param user: database user name\n        :param password: password for user\n        :param host: host name\n        :param dbname: database name\n        "
        if driver is None:
            self.engine = create_engine(connector + '://' + user + ':' + password + '@' + host + '/' + dbname + charset, echo=True)
        else:
            self.engine = create_engine(connector + '://' + user + ':' + password + '@' + host + '/' + dbname + driver, echo=True)
        self.metadata = MetaData(bind=self.engine)
        self.session = create_session(bind=self.engine)

    def get_engine(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets engine of sqlalchemy, which provide connection with database.\n\n        :return: the engine.\n        '
        return self.engine