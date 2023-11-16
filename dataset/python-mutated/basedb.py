from __future__ import unicode_literals, division, absolute_import
import logging
logger = logging.getLogger('database.basedb')
from six import itervalues
from pyspider.libs import utils

class BaseDB:
    """
    BaseDB

    dbcur should be overwirte
    """
    __tablename__ = None
    placeholder = '%s'
    maxlimit = -1

    @staticmethod
    def escape(string):
        if False:
            for i in range(10):
                print('nop')
        return '`%s`' % string

    @property
    def dbcur(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _execute(self, sql_query, values=[]):
        if False:
            while True:
                i = 10
        dbcur = self.dbcur
        dbcur.execute(sql_query, values)
        return dbcur

    def _select(self, tablename=None, what='*', where='', where_values=[], offset=0, limit=None):
        if False:
            i = 10
            return i + 15
        tablename = self.escape(tablename or self.__tablename__)
        if isinstance(what, list) or isinstance(what, tuple) or what is None:
            what = ','.join((self.escape(f) for f in what)) if what else '*'
        sql_query = 'SELECT %s FROM %s' % (what, tablename)
        if where:
            sql_query += ' WHERE %s' % where
        if limit:
            sql_query += ' LIMIT %d, %d' % (offset, limit)
        elif offset:
            sql_query += ' LIMIT %d, %d' % (offset, self.maxlimit)
        logger.debug('<sql: %s>', sql_query)
        for row in self._execute(sql_query, where_values):
            yield row

    def _select2dic(self, tablename=None, what='*', where='', where_values=[], order=None, offset=0, limit=None):
        if False:
            return 10
        tablename = self.escape(tablename or self.__tablename__)
        if isinstance(what, list) or isinstance(what, tuple) or what is None:
            what = ','.join((self.escape(f) for f in what)) if what else '*'
        sql_query = 'SELECT %s FROM %s' % (what, tablename)
        if where:
            sql_query += ' WHERE %s' % where
        if order:
            sql_query += ' ORDER BY %s' % order
        if limit:
            sql_query += ' LIMIT %d, %d' % (offset, limit)
        elif offset:
            sql_query += ' LIMIT %d, %d' % (offset, self.maxlimit)
        logger.debug('<sql: %s>', sql_query)
        dbcur = self._execute(sql_query, where_values)
        fields = [utils.text(f[0]) for f in dbcur.description]
        for row in dbcur:
            yield dict(zip(fields, row))

    def _replace(self, tablename=None, **values):
        if False:
            print('Hello World!')
        tablename = self.escape(tablename or self.__tablename__)
        if values:
            _keys = ', '.join((self.escape(k) for k in values))
            _values = ', '.join([self.placeholder] * len(values))
            sql_query = 'REPLACE INTO %s (%s) VALUES (%s)' % (tablename, _keys, _values)
        else:
            sql_query = 'REPLACE INTO %s DEFAULT VALUES' % tablename
        logger.debug('<sql: %s>', sql_query)
        if values:
            dbcur = self._execute(sql_query, list(itervalues(values)))
        else:
            dbcur = self._execute(sql_query)
        return dbcur.lastrowid

    def _insert(self, tablename=None, **values):
        if False:
            i = 10
            return i + 15
        tablename = self.escape(tablename or self.__tablename__)
        if values:
            _keys = ', '.join((self.escape(k) for k in values))
            _values = ', '.join([self.placeholder] * len(values))
            sql_query = 'INSERT INTO %s (%s) VALUES (%s)' % (tablename, _keys, _values)
        else:
            sql_query = 'INSERT INTO %s DEFAULT VALUES' % tablename
        logger.debug('<sql: %s>', sql_query)
        if values:
            dbcur = self._execute(sql_query, list(itervalues(values)))
        else:
            dbcur = self._execute(sql_query)
        return dbcur.lastrowid

    def _update(self, tablename=None, where='1=0', where_values=[], **values):
        if False:
            while True:
                i = 10
        tablename = self.escape(tablename or self.__tablename__)
        _key_values = ', '.join(['%s = %s' % (self.escape(k), self.placeholder) for k in values])
        sql_query = 'UPDATE %s SET %s WHERE %s' % (tablename, _key_values, where)
        logger.debug('<sql: %s>', sql_query)
        return self._execute(sql_query, list(itervalues(values)) + list(where_values))

    def _delete(self, tablename=None, where='1=0', where_values=[]):
        if False:
            print('Hello World!')
        tablename = self.escape(tablename or self.__tablename__)
        sql_query = 'DELETE FROM %s' % tablename
        if where:
            sql_query += ' WHERE %s' % where
        logger.debug('<sql: %s>', sql_query)
        return self._execute(sql_query, where_values)
if __name__ == '__main__':
    import sqlite3

    class DB(BaseDB):
        __tablename__ = 'test'
        placeholder = '?'

        def __init__(self):
            if False:
                return 10
            self.conn = sqlite3.connect(':memory:')
            cursor = self.conn.cursor()
            cursor.execute('CREATE TABLE `%s` (id INTEGER PRIMARY KEY AUTOINCREMENT, name, age)' % self.__tablename__)

        @property
        def dbcur(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.conn.cursor()
    db = DB()
    assert db._insert(db.__tablename__, name='binux', age=23) == 1
    assert db._select(db.__tablename__, 'name, age').next() == ('binux', 23)
    assert db._select2dic(db.__tablename__, 'name, age').next()['name'] == 'binux'
    assert db._select2dic(db.__tablename__, 'name, age').next()['age'] == 23
    db._replace(db.__tablename__, id=1, age=24)
    assert db._select(db.__tablename__, 'name, age').next() == (None, 24)
    db._update(db.__tablename__, 'id = 1', age=16)
    assert db._select(db.__tablename__, 'name, age').next() == (None, 16)
    db._delete(db.__tablename__, 'id = 1')
    assert [row for row in db._select(db.__tablename__)] == []