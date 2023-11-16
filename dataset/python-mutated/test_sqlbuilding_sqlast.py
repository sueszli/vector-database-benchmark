from __future__ import absolute_import, print_function, division
import unittest
from pony.orm.core import Database, db_session
from pony.orm.sqlsymbols import *
from pony.orm.tests import setup_database, only_for

@only_for('sqlite')
class TestSQLAST(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.db = Database()
        setup_database(self.db)
        with db_session:
            conn = self.db.get_connection()
            conn.executescript("\n            create table if not exists T1(\n                a integer primary key,\n                b varchar(20) not null\n                );\n            insert or ignore into T1 values(1, 'abc');\n            ")

    def tearDown(self):
        if False:
            print('Hello World!')
        with db_session:
            conn = self.db.get_connection()
            conn.executescript('drop table T1\n            ')

    @db_session
    def test_alias(self):
        if False:
            i = 10
            return i + 15
        sql_ast = [SELECT, [ALL, [COLUMN, 'Group', 'a']], [FROM, ['Group', TABLE, 'T1']]]
        (sql, adapter) = self.db._ast2sql(sql_ast)
        cursor = self.db._exec_sql(sql)

    @db_session
    def test_alias2(self):
        if False:
            for i in range(10):
                print('nop')
        sql_ast = [SELECT, [ALL, [COLUMN, None, 'a']], [FROM, [None, TABLE, 'T1']]]
        (sql, adapter) = self.db._ast2sql(sql_ast)
        cursor = self.db._exec_sql(sql)
if __name__ == '__main__':
    unittest.main()