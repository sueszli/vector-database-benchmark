import re, os, sys, public

class panelMysql:
    __DB_PASS = None
    __DB_USER = 'root'
    __DB_PORT = 3306
    __DB_HOST = 'localhost'
    __DB_CONN = None
    __DB_CUR = None
    __DB_ERR = None
    __DB_NET = None

    def __Conn(self):
        if False:
            i = 10
            return i + 15
        if self.__DB_NET:
            return True
        try:
            myconf = public.readFile('/etc/my.cnf')
            socket_re = re.search('socket\\s*=\\s*(.+)', myconf)
            if socket_re:
                socket = socket_re.groups()[0]
            else:
                socket = '/tmp/mysql.sock'
            try:
                if sys.version_info[0] != 2:
                    try:
                        import pymysql
                    except:
                        public.ExecShell('pip install pymysql')
                        import pymysql
                    pymysql.install_as_MySQLdb()
                import MySQLdb
                if sys.version_info[0] == 2:
                    reload(MySQLdb)
            except:
                try:
                    import pymysql
                    pymysql.install_as_MySQLdb()
                    import MySQLdb
                except Exception as e:
                    self.__DB_ERR = e
                    return False
            try:
                rep = 'port\\s*=\\s*([0-9]+)'
                self.__DB_PORT = int(re.search(rep, myconf).groups()[0])
            except:
                self.__DB_PORT = 3306
            self.__DB_PASS = public.M('config').where('id=?', (1,)).getField('mysql_root')
            try:
                self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=self.__DB_PORT, charset='utf8', connect_timeout=1, unix_socket=socket)
            except MySQLdb.Error as e:
                self.__DB_HOST = '127.0.0.1'
                self.__DB_CONN = MySQLdb.connect(host=self.__DB_HOST, user=self.__DB_USER, passwd=self.__DB_PASS, port=self.__DB_PORT, charset='utf8', connect_timeout=1, unix_socket=socket)
            self.__DB_CUR = self.__DB_CONN.cursor()
            return True
        except MySQLdb.Error as e:
            self.__DB_ERR = e
            return False

    def connect_network(self, host, port, username, password):
        if False:
            return 10
        self.__DB_NET = True
        try:
            try:
                if sys.version_info[0] != 2:
                    try:
                        import pymysql
                    except:
                        public.ExecShell('pip install pymysql')
                        import pymysql
                    pymysql.install_as_MySQLdb()
                import MySQLdb
                if sys.version_info[0] == 2:
                    reload(MySQLdb)
            except:
                try:
                    import pymysql
                    pymysql.install_as_MySQLdb()
                    import MySQLdb
                except Exception as e:
                    self.__DB_ERR = e
                    return False
            self.__DB_CONN = MySQLdb.connect(host=host, user=username, passwd=password, port=port, charset='utf8', connect_timeout=10)
            self.__DB_CUR = self.__DB_CONN.cursor()
            return True
        except MySQLdb.Error as e:
            self.__DB_ERR = e
            return False

    def execute(self, sql):
        if False:
            return 10
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
            while True:
                i = 10
        if not self.__Conn():
            return self.__DB_ERR
        try:
            self.__DB_CUR.execute(sql)
            result = self.__DB_CUR.fetchall()
            if sys.version_info[0] == 2:
                data = map(list, result)
            else:
                data = list(map(list, result))
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