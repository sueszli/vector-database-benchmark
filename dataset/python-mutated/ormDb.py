import re
import os
import sys

class ORM:
    __DB_PASS = None
    __DB_USER = 'root'
    __DB_PORT = 3306
    __DB_HOST = 'localhost'
    __DB_CONN = None
    __DB_CUR = None
    __DB_ERR = None
    __DB_CNF = '/etc/my.cnf'
    __DB_SOCKET = '/www/server/mysql/mysql.sock'
    __DB_CHARSET = 'utf8'

    def __Conn(self):
        if False:
            return 10
        '连接MYSQL数据库'
        try:
            try:
                import MySQLdb
            except Exception as ex:
                self.__DB_ERR = ex
                return False
            if os.path.exists(self.__DB_SOCKET):
                try:
                    self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=int(self.__DB_PORT), charset=self.__DB_CHARSET, connect_timeout=1, unix_socket=self.__DB_SOCKET)
                except Exception as e:
                    print(e)
                    self.__DB_HOST = '127.0.0.1'
                    self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=int(self.__DB_PORT), charset=self.__DB_CHARSET, connect_timeout=1, unix_socket=self.__DB_SOCKET)
            else:
                try:
                    self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=int(self.__DB_PORT), charset=self.__DB_CHARSET, connect_timeout=1)
                except Exception as e:
                    self.__DB_HOST = '127.0.0.1'
                    self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=int(self.__DB_PORT), charset=self.__DB_CHARSET, connect_timeout=1)
            self.__DB_CUR = self.__DB_CONN.cursor()
            return True
        except MySQLdb.Error as e:
            self.__DB_ERR = e
            return False

    def setDbConf(self, conf):
        if False:
            while True:
                i = 10
        self.__DB_CNF = conf

    def setSocket(self, sock):
        if False:
            print('Hello World!')
        self.__DB_SOCKET = sock

    def setCharset(self, charset):
        if False:
            for i in range(10):
                print('nop')
        self.__DB_CHARSET = charset

    def setPort(self, port):
        if False:
            for i in range(10):
                print('nop')
        self.__DB_PORT = port

    def setPwd(self, pwd):
        if False:
            while True:
                i = 10
        self.__DB_PASS = pwd

    def getPwd(self):
        if False:
            i = 10
            return i + 15
        return self.__DB_PASS

    def setDbName(self, name):
        if False:
            while True:
                i = 10
        self.__DB_NAME = name

    def setUser(self, user):
        if False:
            return 10
        self.__DB_USER = user

    def execute(self, sql):
        if False:
            while True:
                i = 10
        if not self.__Conn():
            return self.__DB_ERR
        try:
            result = self.__DB_CUR.execute(sql)
            self.__DB_CONN.commit()
            self.__Close()
            return result
        except Exception as ex:
            return ex

    def query(self, sql):
        if False:
            print('Hello World!')
        if not self.__Conn():
            return self.__DB_ERR
        try:
            self.__DB_CUR.execute(sql)
            result = self.__DB_CUR.fetchall()
            data = map(list, result)
            self.__Close()
            return data
        except Exception as ex:
            return ex

    def __Close(self):
        if False:
            while True:
                i = 10
        self.__DB_CUR.close()
        self.__DB_CONN.close()