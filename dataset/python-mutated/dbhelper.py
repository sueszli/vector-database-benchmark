import pymysql
from scrapy.utils.project import get_project_settings

class DBHelper:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.settings = get_project_settings()
        self.host = self.settings['MYSQL_HOST']
        self.port = self.settings['MYSQL_PORT']
        self.user = self.settings['MYSQL_USER']
        self.passwd = self.settings['MYSQL_PASSWD']
        self.db = self.settings['MYSQL_DBNAME']

    def connectMysql(self):
        if False:
            return 10
        conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, charset='utf8')
        return conn

    def connectDatabase(self):
        if False:
            while True:
                i = 10
        conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db, charset='utf8')
        return conn

    def createDatabase(self):
        if False:
            return 10
        conn = self.connectMysql()
        sql = 'create database if not exists ' + self.db
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def createTable(self, sql):
        if False:
            for i in range(10):
                print('nop')
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def insert(self, sql, *params):
        if False:
            i = 10
            return i + 15
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def update(self, sql, *params):
        if False:
            i = 10
            return i + 15
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def delete(self, sql, *params):
        if False:
            while True:
                i = 10
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

class TestDBHelper:

    def __init__(self):
        if False:
            print('Hello World!')
        self.dbHelper = DBHelper()

    def testCreateDatebase(self):
        if False:
            print('Hello World!')
        self.dbHelper.createDatabase()

    def testCreateTable(self):
        if False:
            return 10
        sql = 'create table testtable(id int primary key auto_increment,name varchar(50),url varchar(200))'
        self.dbHelper.createTable(sql)

    def testInsert(self):
        if False:
            for i in range(10):
                print('nop')
        sql = 'insert into testtable(name,url) values(%s,%s)'
        params = ('test', 'test')
        self.dbHelper.insert(sql, *params)

    def testUpdate(self):
        if False:
            i = 10
            return i + 15
        sql = 'update testtable set name=%s,url=%s where id=%s'
        params = ('update', 'update', '1')
        self.dbHelper.update(sql, *params)

    def testDelete(self):
        if False:
            print('Hello World!')
        sql = 'delete from testtable where id=%s'
        params = '1'
        self.dbHelper.delete(sql, *params)
if __name__ == '__main__':
    testDBHelper = TestDBHelper()