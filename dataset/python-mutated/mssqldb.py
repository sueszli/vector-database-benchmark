import logging
import luigi
logger = logging.getLogger('luigi-interface')
try:
    from pymssql import _mssql
except ImportError:
    logger.warning('Loading MSSQL module without the python package pymssql.         This will crash at runtime if SQL Server functionality is used.')

class MSSqlTarget(luigi.Target):
    """
    Target for a resource in Microsoft SQL Server.
    This module is primarily derived from mysqldb.py.  Much of MSSqlTarget,
    MySqlTarget and PostgresTarget are similar enough to potentially add a
    RDBMSTarget abstract base class to rdbms.py that these classes could be
    derived from.
    """
    marker_table = luigi.configuration.get_config().get('mssql', 'marker-table', 'table_updates')

    def __init__(self, host, database, user, password, table, update_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a MsSqlTarget instance.\n\n        :param host: MsSql server address. Possibly a host:port string.\n        :type host: str\n        :param database: database name.\n        :type database: str\n        :param user: database user\n        :type user: str\n        :param password: password for specified user.\n        :type password: str\n        :param update_id: an identifier for this data set.\n        :type update_id: str\n        '
        if ':' in host:
            (self.host, self.port) = host.split(':')
            self.port = int(self.port)
        else:
            self.host = host
            self.port = 1433
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.update_id = update_id

    def __str__(self):
        if False:
            print('Hello World!')
        return self.table

    def touch(self, connection=None):
        if False:
            while True:
                i = 10
        "\n        Mark this update as complete.\n\n        IMPORTANT, If the marker table doesn't exist,\n        the connection transaction will be aborted and the connection reset.\n        Then the marker table will be created.\n        "
        self.create_marker_table()
        if connection is None:
            connection = self.connect()
        connection.execute_non_query('IF NOT EXISTS(SELECT 1\n                            FROM {marker_table}\n                            WHERE update_id = %(update_id)s)\n                    INSERT INTO {marker_table} (update_id, target_table)\n                        VALUES (%(update_id)s, %(table)s)\n                ELSE\n                    UPDATE t\n                    SET target_table = %(table)s\n                        , inserted = GETDATE()\n                    FROM {marker_table} t\n                    WHERE update_id = %(update_id)s\n              '.format(marker_table=self.marker_table), {'update_id': self.update_id, 'table': self.table})
        assert self.exists(connection)

    def exists(self, connection=None):
        if False:
            print('Hello World!')
        if connection is None:
            connection = self.connect()
        try:
            row = connection.execute_row('SELECT 1 FROM {marker_table}\n                                            WHERE update_id = %s\n                                    '.format(marker_table=self.marker_table), (self.update_id,))
        except _mssql.MssqlDatabaseException as e:
            if e.number == 208:
                row = None
            else:
                raise
        return row is not None

    def connect(self):
        if False:
            print('Hello World!')
        '\n        Create a SQL Server connection and return a connection object\n        '
        connection = _mssql.connect(user=self.user, password=self.password, server=self.host, port=self.port, database=self.database)
        return connection

    def create_marker_table(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create marker table if it doesn't exist.\n        Use a separate connection since the transaction might have to be reset.\n        "
        connection = self.connect()
        try:
            connection.execute_non_query(' CREATE TABLE {marker_table} (\n                        id            BIGINT    NOT NULL IDENTITY(1,1),\n                        update_id     VARCHAR(128)  NOT NULL,\n                        target_table  VARCHAR(128),\n                        inserted      DATETIME DEFAULT(GETDATE()),\n                        PRIMARY KEY (update_id)\n                    )\n                '.format(marker_table=self.marker_table))
        except _mssql.MssqlDatabaseException as e:
            if e.number == 2714:
                pass
            else:
                raise
        connection.close()