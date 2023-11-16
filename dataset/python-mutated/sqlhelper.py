from configure.settings import DBSelector
import loguru
logger = loguru.logger

class SQLHelper:

    def __init__(self, host, db_name):
        if False:
            print('Hello World!')
        self.conn = DBSelector().get_engine(db_name, host)
        self.db = DBSelector().get_mysql_conn(db_name, host)
        self.cursor = self.db.cursor()

    def query(self, sql_str, args):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.cursor.execute(sql_str, args=args)
        except Exception as e:
            logger.error(e)
            self.db.rollback()
            return None
        else:
            ret = self.cursor.fetchall()
            return ret

    def update(self, sql_str, args=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.cursor.execute(sql_str, args=args)
        except Exception as e:
            logger.error(e)
            self.db.rollback()
            return False
        else:
            self.db.commit()
            return True