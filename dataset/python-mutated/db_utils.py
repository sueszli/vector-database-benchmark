import pymysql

def create_db(hostname='localhost', port=3306, username=None, password=None, dbname=None):
    if False:
        for i in range(10):
            print('nop')
    'Create test database.\n\n    :param hostname: string\n    :param port: int\n    :param username: string\n    :param password: string\n    :param dbname: string\n    :return:\n\n    '
    cn = pymysql.connect(host=hostname, port=port, user=username, password=password, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with cn.cursor() as cr:
        cr.execute('drop database if exists ' + dbname)
        cr.execute('create database ' + dbname)
    cn.close()
    cn = create_cn(hostname, port, password, username, dbname)
    return cn

def create_cn(hostname, port, password, username, dbname):
    if False:
        return 10
    'Open connection to database.\n\n    :param hostname:\n    :param port:\n    :param password:\n    :param username:\n    :param dbname: string\n    :return: psycopg2.connection\n\n    '
    cn = pymysql.connect(host=hostname, port=port, user=username, password=password, db=dbname, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    return cn

def drop_db(hostname='localhost', port=3306, username=None, password=None, dbname=None):
    if False:
        print('Hello World!')
    'Drop database.\n\n    :param hostname: string\n    :param port: int\n    :param username: string\n    :param password: string\n    :param dbname: string\n\n    '
    cn = pymysql.connect(host=hostname, port=port, user=username, password=password, db=dbname, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with cn.cursor() as cr:
        cr.execute('drop database if exists ' + dbname)
    close_cn(cn)

def close_cn(cn=None):
    if False:
        print('Hello World!')
    'Close connection.\n\n    :param connection: pymysql.connection\n\n    '
    if cn:
        cn.close()