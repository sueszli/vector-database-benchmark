import mysql.connector
from mysql.connector.constants import SQLMode
from urllib.parse import urlparse
from lib.core.exceptions import InvalidURLException
from lib.reports.base import SQLBaseReport

class MySQLReport(SQLBaseReport):

    def connect(self, url):
        if False:
            return 10
        parsed = urlparse(url)
        if not parsed.scheme == 'mysql':
            raise InvalidURLException('Provided MySQL URL does not start with mysql://')
        self.conn = mysql.connector.connect(host=parsed.hostname, port=parsed.port or 3306, user=parsed.username, password=parsed.password, database=parsed.path.lstrip('/'))
        self.conn.sql_mode = [SQLMode.ANSI_QUOTES]
        self.cursor = self.conn.cursor()