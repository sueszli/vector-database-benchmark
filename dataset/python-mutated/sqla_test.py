"""
This file implements unit test cases for luigi/contrib/sqla.py
Author: Gouthaman Balaraman
Date: 01/02/2015
"""
import os
import shutil
import tempfile
import unittest
import luigi
import sqlalchemy
from luigi.contrib import sqla
from luigi.mock import MockTarget
import pytest
from helpers import skipOnTravisAndGithubActions

class BaseTask(luigi.Task):
    TASK_LIST = ['item%d\tproperty%d\n' % (i, i) for i in range(10)]

    def output(self):
        if False:
            return 10
        return MockTarget('BaseTask', mirror_on_stderr=True)

    def run(self):
        if False:
            while True:
                i = 10
        out = self.output().open('w')
        for task in self.TASK_LIST:
            out.write(task)
        out.close()

@pytest.mark.contrib
class TestSQLA(unittest.TestCase):
    NUM_WORKERS = 1

    def _clear_tables(self):
        if False:
            while True:
                i = 10
        meta = sqlalchemy.MetaData()
        meta.reflect(bind=self.engine)
        for table in reversed(meta.sorted_tables):
            self.engine.execute(table.delete())

    def setUp(self):
        if False:
            return 10
        self.tempdir = tempfile.mkdtemp()
        self.connection_string = self.get_connection_string()
        self.connect_args = {'timeout': 5.0}
        self.engine = sqlalchemy.create_engine(self.connection_string, connect_args=self.connect_args)

        class SQLATask(sqla.CopyToTable):
            columns = [(['item', sqlalchemy.String(64)], {}), (['property', sqlalchemy.String(64)], {})]
            connection_string = self.connection_string
            connect_args = self.connect_args
            table = 'item_property'
            chunk_size = 1

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return BaseTask()
        self.SQLATask = SQLATask

    def tearDown(self):
        if False:
            while True:
                i = 10
        self._clear_tables()
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def get_connection_string(self, db='sqlatest.db'):
        if False:
            print('Hello World!')
        return 'sqlite:///{path}'.format(path=os.path.join(self.tempdir, db))

    def test_create_table(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that this method creates table that we require\n        :return:\n        '

        class TestSQLData(sqla.CopyToTable):
            connection_string = self.connection_string
            connect_args = self.connect_args
            table = 'test_table'
            columns = [(['id', sqlalchemy.Integer], dict(primary_key=True)), (['name', sqlalchemy.String(64)], {}), (['value', sqlalchemy.String(64)], {})]
            chunk_size = 1

            def output(self):
                if False:
                    while True:
                        i = 10
                pass
        sql_copy = TestSQLData()
        eng = sqlalchemy.create_engine(TestSQLData.connection_string)
        self.assertFalse(eng.dialect.has_table(eng.connect(), TestSQLData.table))
        sql_copy.create_table(eng)
        self.assertTrue(eng.dialect.has_table(eng.connect(), TestSQLData.table))
        sql_copy.create_table(eng)

    def test_create_table_raises_no_columns(self):
        if False:
            print('Hello World!')
        '\n        Check that the test fails when the columns are not set\n        :return:\n        '

        class TestSQLData(sqla.CopyToTable):
            connection_string = self.connection_string
            table = 'test_table'
            columns = []
            chunk_size = 1

        def output(self):
            if False:
                i = 10
                return i + 15
            pass
        sql_copy = TestSQLData()
        eng = sqlalchemy.create_engine(TestSQLData.connection_string)
        self.assertRaises(NotImplementedError, sql_copy.create_table, eng)

    def _check_entries(self, engine):
        if False:
            i = 10
            return i + 15
        with engine.begin() as conn:
            meta = sqlalchemy.MetaData()
            meta.reflect(bind=engine)
            self.assertEqual({'table_updates', 'item_property'}, set(meta.tables.keys()))
            table = meta.tables[self.SQLATask.table]
            s = sqlalchemy.select([sqlalchemy.func.count(table.c.item)])
            result = conn.execute(s).fetchone()
            self.assertEqual(len(BaseTask.TASK_LIST), result[0])
            s = sqlalchemy.select([table]).order_by(table.c.item)
            result = conn.execute(s).fetchall()
            for i in range(len(BaseTask.TASK_LIST)):
                given = BaseTask.TASK_LIST[i].strip('\n').split('\t')
                given = (str(given[0]), str(given[1]))
                self.assertEqual(given, tuple(result[i]))

    def test_rows(self):
        if False:
            i = 10
            return i + 15
        (task, task0) = (self.SQLATask(), BaseTask())
        luigi.build([task, task0], local_scheduler=True, workers=self.NUM_WORKERS)
        for (i, row) in enumerate(task.rows()):
            given = BaseTask.TASK_LIST[i].strip('\n').split('\t')
            self.assertEqual(row, given)

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Checking that the runs go as expected. Rerunning the same shouldn't end up\n        inserting more rows into the db.\n        :return:\n        "
        (task, task0) = (self.SQLATask(), BaseTask())
        self.engine = sqlalchemy.create_engine(task.connection_string)
        luigi.build([task0, task], local_scheduler=True)
        self._check_entries(self.engine)
        luigi.build([task0, task], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_run_with_chunk_size(self):
        if False:
            i = 10
            return i + 15
        '\n        The chunk_size can be specified in order to control the batch size for inserts.\n        :return:\n        '
        (task, task0) = (self.SQLATask(), BaseTask())
        self.engine = sqlalchemy.create_engine(task.connection_string)
        task.chunk_size = 2
        luigi.build([task, task0], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_reflect(self):
        if False:
            i = 10
            return i + 15
        '\n        If the table is setup already, then one can set reflect to True, and\n        completely skip the columns part. It is not even required at that point.\n        :return:\n        '
        SQLATask = self.SQLATask

        class AnotherSQLATask(sqla.CopyToTable):
            connection_string = self.connection_string
            table = 'item_property'
            reflect = True
            chunk_size = 1

            def requires(self):
                if False:
                    print('Hello World!')
                return SQLATask()

            def copy(self, conn, ins_rows, table_bound):
                if False:
                    i = 10
                    return i + 15
                ins = table_bound.update().where(table_bound.c.property == sqlalchemy.bindparam('_property')).values({table_bound.c.item: sqlalchemy.bindparam('_item')})
                conn.execute(ins, ins_rows)

            def rows(self):
                if False:
                    print('Hello World!')
                for line in BaseTask.TASK_LIST:
                    yield line.strip('\n').split('\t')
        (task0, task1, task2) = (AnotherSQLATask(), self.SQLATask(), BaseTask())
        luigi.build([task0, task1, task2], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_create_marker_table(self):
        if False:
            print('Hello World!')
        '\n        Is the marker table created as expected for the SQLAlchemyTarget\n        :return:\n        '
        target = sqla.SQLAlchemyTarget(self.connection_string, 'test_table', '12312123')
        target.create_marker_table()
        self.assertTrue(target.engine.dialect.has_table(target.engine.connect(), target.marker_table))

    def test_touch(self):
        if False:
            return 10
        '\n        Touch takes care of creating a checkpoint for task completion\n        :return:\n        '
        target = sqla.SQLAlchemyTarget(self.connection_string, 'test_table', '12312123')
        target.create_marker_table()
        self.assertFalse(target.exists())
        target.touch()
        self.assertTrue(target.exists())

    def test_row_overload(self):
        if False:
            for i in range(10):
                print('nop')
        'Overload the rows method and we should be able to insert data into database'

        class SQLARowOverloadTest(sqla.CopyToTable):
            columns = [(['item', sqlalchemy.String(64)], {}), (['property', sqlalchemy.String(64)], {})]
            connection_string = self.connection_string
            table = 'item_property'
            chunk_size = 1

            def rows(self):
                if False:
                    return 10
                tasks = [('item0', 'property0'), ('item1', 'property1'), ('item2', 'property2'), ('item3', 'property3'), ('item4', 'property4'), ('item5', 'property5'), ('item6', 'property6'), ('item7', 'property7'), ('item8', 'property8'), ('item9', 'property9')]
                for row in tasks:
                    yield row
        task = SQLARowOverloadTest()
        luigi.build([task], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_column_row_separator(self):
        if False:
            while True:
                i = 10
        '\n        Test alternate column row separator works\n        :return:\n        '

        class ModBaseTask(luigi.Task):

            def output(self):
                if False:
                    for i in range(10):
                        print('nop')
                return MockTarget('ModBaseTask', mirror_on_stderr=True)

            def run(self):
                if False:
                    i = 10
                    return i + 15
                out = self.output().open('w')
                tasks = ['item%d,property%d\n' % (i, i) for i in range(10)]
                for task in tasks:
                    out.write(task)
                out.close()

        class ModSQLATask(sqla.CopyToTable):
            columns = [(['item', sqlalchemy.String(64)], {}), (['property', sqlalchemy.String(64)], {})]
            connection_string = self.connection_string
            table = 'item_property'
            column_separator = ','
            chunk_size = 1

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return ModBaseTask()
        (task1, task2) = (ModBaseTask(), ModSQLATask())
        luigi.build([task1, task2], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_update_rows_test(self):
        if False:
            i = 10
            return i + 15
        '\n        Overload the copy() method and implement an update action.\n        :return:\n        '

        class ModBaseTask(luigi.Task):

            def output(self):
                if False:
                    for i in range(10):
                        print('nop')
                return MockTarget('BaseTask', mirror_on_stderr=True)

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                out = self.output().open('w')
                for task in self.TASK_LIST:
                    out.write('dummy_' + task)
                out.close()

        class ModSQLATask(sqla.CopyToTable):
            connection_string = self.connection_string
            table = 'item_property'
            columns = [(['item', sqlalchemy.String(64)], {}), (['property', sqlalchemy.String(64)], {})]
            chunk_size = 1

            def requires(self):
                if False:
                    while True:
                        i = 10
                return ModBaseTask()

        class UpdateSQLATask(sqla.CopyToTable):
            connection_string = self.connection_string
            table = 'item_property'
            reflect = True
            chunk_size = 1

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return ModSQLATask()

            def copy(self, conn, ins_rows, table_bound):
                if False:
                    i = 10
                    return i + 15
                ins = table_bound.update().where(table_bound.c.property == sqlalchemy.bindparam('_property')).values({table_bound.c.item: sqlalchemy.bindparam('_item')})
                conn.execute(ins, ins_rows)

            def rows(self):
                if False:
                    print('Hello World!')
                for task in self.TASK_LIST:
                    yield task.strip('\n').split('\t')
        (task1, task2, task3) = (ModBaseTask(), ModSQLATask(), UpdateSQLATask())
        luigi.build([task1, task2, task3], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    @skipOnTravisAndGithubActions('AssertionError: 10 != 7; https://travis-ci.org/spotify/luigi/jobs/156732446')
    def test_multiple_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a case where there are multiple tasks\n        :return:\n        '

        class SmallSQLATask(sqla.CopyToTable):
            item = luigi.Parameter()
            property = luigi.Parameter()
            columns = [(['item', sqlalchemy.String(64)], {}), (['property', sqlalchemy.String(64)], {})]
            connection_string = self.connection_string
            table = 'item_property'
            chunk_size = 1

            def rows(self):
                if False:
                    while True:
                        i = 10
                yield (self.item, self.property)

        class ManyBaseTask(luigi.Task):

            def requires(self):
                if False:
                    print('Hello World!')
                for t in BaseTask.TASK_LIST:
                    (item, property) = t.strip().split('\t')
                    yield SmallSQLATask(item=item, property=property)
        task2 = ManyBaseTask()
        luigi.build([task2], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(self.engine)

    def test_multiple_engines(self):
        if False:
            print('Hello World!')
        '\n        Test case where different tasks require different SQL engines.\n        '
        alt_db = self.get_connection_string('sqlatest2.db')

        class MultiEngineTask(self.SQLATask):
            connection_string = alt_db
        (task0, task1, task2) = (BaseTask(), self.SQLATask(), MultiEngineTask())
        self.assertTrue(task1.output().engine != task2.output().engine)
        luigi.build([task2, task1, task0], local_scheduler=True, workers=self.NUM_WORKERS)
        self._check_entries(task1.output().engine)
        self._check_entries(task2.output().engine)

@pytest.mark.contrib
class TestSQLA2(TestSQLA):
    """ 2 workers version
    """
    NUM_WORKERS = 2