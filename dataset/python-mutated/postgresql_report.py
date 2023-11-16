import psycopg
from lib.core.exceptions import InvalidURLException
from lib.reports.base import SQLBaseReport

class PostgreSQLReport(SQLBaseReport):

    def connect(self, url):
        if False:
            while True:
                i = 10
        if not url.startswith('postgresql://'):
            raise InvalidURLException('Provided PostgreSQL URL does not start with postgresql://')
        self.conn = psycopg.connect(url)
        self.cursor = self.conn.cursor()