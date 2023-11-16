import copy
import logging
import sys
from .writer import Writer
logger = logging.getLogger('spider.sqlite_writer')

class SqliteWriter(Writer):

    def __init__(self, sqlite_config):
        if False:
            for i in range(10):
                print('nop')
        self.sqlite_config = sqlite_config

    def _sqlite_create(self, connection, sql):
        if False:
            return 10
        '创建sqlite数据库或表'
        try:
            cursor = connection.cursor()
            cursor.execute(sql)
        finally:
            connection.close()

    def _sqlite_create_table(self, sql):
        if False:
            i = 10
            return i + 15
        '创建sqlite表'
        import sqlite3
        connection = sqlite3.connect(self.sqlite_config)
        self._sqlite_create(connection, sql)

    def _sqlite_insert(self, table, data_list):
        if False:
            i = 10
            return i + 15
        '向sqlite表插入或更新数据'
        import sqlite3
        if len(data_list) > 0:
            data_list = [{k: v for (k, v) in data.items() if v is not None} for data in data_list]
            keys = ', '.join(data_list[0].keys())
            values = ', '.join(['?'] * len(data_list[0]))
            connection = sqlite3.connect(self.sqlite_config)
            cursor = connection.cursor()
            sql = 'INSERT OR REPLACE INTO {table}({keys}) VALUES ({values})'.format(table=table, keys=keys, values=values)
            try:
                cursor.executemany(sql, [tuple(data.values()) for data in data_list])
                connection.commit()
            except Exception as e:
                connection.rollback()
                logger.exception(e)
            finally:
                connection.close()

    def write_weibo(self, weibos):
        if False:
            for i in range(10):
                print('nop')
        '将爬取的微博信息写入sqlite数据库'
        create_table = '\n                CREATE TABLE IF NOT EXISTS weibo (\n                id varchar(10) NOT NULL,\n                user_id varchar(12),\n                content varchar(2000),\n                article_url varchar(200),\n                original_pictures varchar(3000),\n                retweet_pictures varchar(3000),\n                original BOOLEAN NOT NULL DEFAULT 1,\n                video_url varchar(300),\n                publish_place varchar(100),\n                publish_time DATETIME NOT NULL,\n                publish_tool varchar(30),\n                up_num INT NOT NULL,\n                retweet_num INT NOT NULL,\n                comment_num INT NOT NULL,\n                PRIMARY KEY (id)\n                )'
        self._sqlite_create_table(create_table)
        weibo_list = []
        info_list = copy.deepcopy(weibos)
        for weibo in info_list:
            weibo.user_id = self.user.id
            weibo_list.append(weibo.__dict__)
        self._sqlite_insert('weibo', weibo_list)
        logger.info(u'%d条微博写入sqlite数据库完毕', len(weibos))

    def write_user(self, user):
        if False:
            for i in range(10):
                print('nop')
        '将爬取的用户信息写入sqlite数据库'
        self.user = user
        create_table = '\n                CREATE TABLE IF NOT EXISTS user (\n                id varchar(20) NOT NULL,\n                nickname varchar(30),\n                gender varchar(10),\n                location varchar(200),\n                birthday varchar(40),\n                description varchar(400),\n                verified_reason varchar(140),\n                talent varchar(200),\n                education varchar(200),\n                work varchar(200),\n                weibo_num INT,\n                following INT,\n                followers INT,\n                PRIMARY KEY (id)\n                )'
        self._sqlite_create_table(create_table)
        self._sqlite_insert('user', [user.__dict__])
        logger.info(u'%s信息写入sqlite数据库完毕', user.nickname)