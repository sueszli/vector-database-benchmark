import copy
import logging
import sys
from .writer import Writer
logger = logging.getLogger('spider.mysql_writer')

class MySqlWriter(Writer):

    def __init__(self, mysql_config):
        if False:
            print('Hello World!')
        self.mysql_config = mysql_config
        create_database = 'CREATE DATABASE IF NOT EXISTS weibo DEFAULT\n                            CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci'
        self._mysql_create_database(create_database)
        self.mysql_config['db'] = 'weibo'

    def _mysql_create(self, connection, sql):
        if False:
            i = 10
            return i + 15
        '创建MySQL数据库或表'
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
        finally:
            connection.close()

    def _mysql_create_database(self, sql):
        if False:
            while True:
                i = 10
        '创建MySQL数据库'
        try:
            import pymysql
        except ImportError:
            logger.warning(u'系统中可能没有安装pymysql库，请先运行 pip install pymysql ，再运行程序')
            sys.exit()
        try:
            connection = pymysql.connect(**self.mysql_config)
            self._mysql_create(connection, sql)
        except pymysql.OperationalError:
            logger.warning(u'系统中可能没有安装或正确配置MySQL数据库，请先根据系统环境安装或配置MySQL，再运行程序')
            sys.exit()

    def _mysql_create_table(self, sql):
        if False:
            return 10
        '创建MySQL表'
        import pymysql
        connection = pymysql.connect(**self.mysql_config)
        self._mysql_create(connection, sql)

    def _mysql_insert(self, table, data_list):
        if False:
            return 10
        '向MySQL表插入或更新数据'
        import pymysql
        if len(data_list) > 0:
            data_list = [{k: v for (k, v) in data.items() if v is not None} for data in data_list]
            keys = ', '.join(data_list[0].keys())
            values = ', '.join(['%s'] * len(data_list[0]))
            connection = pymysql.connect(**self.mysql_config)
            cursor = connection.cursor()
            sql = 'INSERT INTO {table}({keys}) VALUES ({values}) ON\n                        DUPLICATE KEY UPDATE'.format(table=table, keys=keys, values=values)
            update = ','.join([' {key} = values({key})'.format(key=key) for key in data_list[0]])
            sql += update
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
        '将爬取的微博信息写入MySQL数据库'
        try:
            create_table = '\n                    CREATE TABLE IF NOT EXISTS weibo (\n                    id varchar(10) NOT NULL,\n                    user_id varchar(12),\n                    content varchar(5000),\n                    article_url varchar(200),\n                    original_pictures varchar(3000),\n                    retweet_pictures varchar(3000),\n                    original BOOLEAN NOT NULL DEFAULT 1,\n                    video_url varchar(300),\n                    publish_place varchar(100),\n                    publish_time DATETIME NOT NULL,\n                    publish_tool varchar(30),\n                    up_num INT NOT NULL,\n                    retweet_num INT NOT NULL,\n                    comment_num INT NOT NULL,\n                    PRIMARY KEY (id)\n                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'
            self._mysql_create_table(create_table)
            weibo_list = []
            info_list = copy.deepcopy(weibos)
            for weibo in info_list:
                weibo.user_id = self.user.id
                weibo_list.append(weibo.__dict__)
            self._mysql_insert('weibo', weibo_list)
            logger.info(u'%d条微博写入MySQL数据库完毕', len(weibos))
        except Exception as e:
            logger.exception(e)

    def write_user(self, user):
        if False:
            for i in range(10):
                print('nop')
        '将爬取的用户信息写入MySQL数据库'
        try:
            self.user = user
            create_table = '\n                    CREATE TABLE IF NOT EXISTS user (\n                    id varchar(20) NOT NULL,\n                    nickname varchar(30),\n                    gender varchar(10),\n                    location varchar(200),\n                    birthday varchar(40),\n                    description varchar(400),\n                    verified_reason varchar(140),\n                    talent varchar(200),\n                    education varchar(200),\n                    work varchar(200),\n                    weibo_num INT,\n                    following INT,\n                    followers INT,\n                    PRIMARY KEY (id)\n                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'
            self._mysql_create_table(create_table)
            self._mysql_insert('user', [user.__dict__])
            logger.info(u'%s信息写入MySQL数据库完毕', user.nickname)
        except Exception as e:
            logger.exception(e)