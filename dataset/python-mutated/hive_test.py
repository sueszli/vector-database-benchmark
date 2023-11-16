from collections import OrderedDict
import os
import sys
import tempfile
from helpers import unittest
import luigi.contrib.hive
import mock
from luigi import LocalTarget
import pytest

@pytest.mark.apache
class HiveTest(unittest.TestCase):
    count = 0

    def mock_hive_cmd(self, args, check_return=True):
        if False:
            return 10
        self.last_hive_cmd = args
        self.count += 1
        return 'statement{}'.format(self.count)

    def setUp(self):
        if False:
            print('Hello World!')
        self.run_hive_cmd_saved = luigi.contrib.hive.run_hive
        luigi.contrib.hive.run_hive = self.mock_hive_cmd

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        luigi.contrib.hive.run_hive = self.run_hive_cmd_saved

    def test_run_hive_command(self):
        if False:
            print('Hello World!')
        pre_count = self.count
        res = luigi.contrib.hive.run_hive_cmd('foo')
        self.assertEqual(['-e', 'foo'], self.last_hive_cmd)
        self.assertEqual('statement{0}'.format(pre_count + 1), res)

    def test_run_hive_script_not_exists(self):
        if False:
            while True:
                i = 10

        def test():
            if False:
                for i in range(10):
                    print('nop')
            luigi.contrib.hive.run_hive_script('/tmp/some-non-existant-file______')
        self.assertRaises(RuntimeError, test)

    def test_run_hive_script_exists(self):
        if False:
            print('Hello World!')
        with tempfile.NamedTemporaryFile(delete=True) as f:
            pre_count = self.count
            res = luigi.contrib.hive.run_hive_script(f.name)
            self.assertEqual(['-f', f.name], self.last_hive_cmd)
            self.assertEqual('statement{0}'.format(pre_count + 1), res)

    def test_create_parent_dirs(self):
        if False:
            while True:
                i = 10
        dirname = '/tmp/hive_task_test_dir'

        class FooHiveTask:

            def output(self):
                if False:
                    for i in range(10):
                        print('nop')
                return LocalTarget(os.path.join(dirname, 'foo'))
        runner = luigi.contrib.hive.HiveQueryRunner()
        runner.prepare_outputs(FooHiveTask())
        self.assertTrue(os.path.exists(dirname))

@pytest.mark.apache
class HiveCommandClientTest(unittest.TestCase):
    """Note that some of these tests are really for the CDH releases of Hive, to which I do not currently have access.
    Hopefully there are no significant differences in the expected output"""

    def setUp(self):
        if False:
            return 10
        self.client = luigi.contrib.hive.HiveCommandClient()
        self.apacheclient = luigi.contrib.hive.ApacheHiveCommandClient()
        self.metastoreclient = luigi.contrib.hive.MetastoreClient()

    @mock.patch('luigi.contrib.hive.run_hive_cmd')
    def test_default_table_location(self, run_command):
        if False:
            for i in range(10):
                print('nop')
        run_command.return_value = 'Protect Mode:       \tNone                \t \nRetention:          \t0                   \t \nLocation:           \thdfs://localhost:9000/user/hive/warehouse/mytable\t \nTable Type:         \tMANAGED_TABLE       \t \n'
        returned = self.client.table_location('mytable')
        self.assertEqual('hdfs://localhost:9000/user/hive/warehouse/mytable', returned)

    @mock.patch('luigi.contrib.hive.run_hive_cmd')
    def test_table_exists(self, run_command):
        if False:
            return 10
        run_command.return_value = 'OK'
        returned = self.client.table_exists('mytable')
        self.assertFalse(returned)
        run_command.return_value = 'OK\nmytable'
        returned = self.client.table_exists('mytable')
        self.assertTrue(returned)
        returned = self.client.table_exists('MyTable')
        self.assertTrue(returned)
        run_command.return_value = 'day=2013-06-28/hour=3\nday=2013-06-28/hour=4\nday=2013-07-07/hour=2\n'
        self.client.partition_spec = mock.Mock(name='partition_spec')
        self.client.partition_spec.return_value = 'somepart'
        returned = self.client.table_exists('mytable', partition={'a': 'b'})
        self.assertTrue(returned)
        run_command.return_value = ''
        returned = self.client.table_exists('mytable', partition={'a': 'b'})
        self.assertFalse(returned)

    @mock.patch('luigi.contrib.hive.run_hive_cmd')
    def test_table_schema(self, run_command):
        if False:
            for i in range(10):
                print('nop')
        run_command.return_value = 'FAILED: SemanticException [Error 10001]: blah does not exist\nSome other stuff'
        returned = self.client.table_schema('mytable')
        self.assertFalse(returned)
        run_command.return_value = 'OK\ncol1       \tstring              \tNone                \ncol2            \tstring              \tNone                \ncol3         \tstring              \tNone                \nday                 \tstring              \tNone                \nhour                \tsmallint            \tNone                \n\n# Partition Information\t \t \n# col_name            \tdata_type           \tcomment             \n\nday                 \tstring              \tNone                \nhour                \tsmallint            \tNone                \nTime taken: 2.08 seconds, Fetched: 34 row(s)\n'
        expected = [('OK',), ('col1', 'string', 'None'), ('col2', 'string', 'None'), ('col3', 'string', 'None'), ('day', 'string', 'None'), ('hour', 'smallint', 'None'), ('',), ('# Partition Information',), ('# col_name', 'data_type', 'comment'), ('',), ('day', 'string', 'None'), ('hour', 'smallint', 'None'), ('Time taken: 2.08 seconds, Fetched: 34 row(s)',)]
        returned = self.client.table_schema('mytable')
        self.assertEqual(expected, returned)

    def test_partition_spec(self):
        if False:
            print('Hello World!')
        returned = self.client.partition_spec({'a': 'b', 'c': 'd'})
        self.assertEqual("`a`='b',`c`='d'", returned)

    @mock.patch('luigi.contrib.hive.run_hive_cmd')
    def test_apacheclient_table_exists(self, run_command):
        if False:
            print('Hello World!')
        run_command.return_value = 'OK'
        returned = self.apacheclient.table_exists('mytable')
        self.assertFalse(returned)
        run_command.return_value = 'OK\nmytable'
        returned = self.apacheclient.table_exists('mytable')
        self.assertTrue(returned)
        returned = self.apacheclient.table_exists('MyTable')
        self.assertTrue(returned)
        run_command.return_value = 'day=2013-06-28/hour=3\nday=2013-06-28/hour=4\nday=2013-07-07/hour=2\n'
        self.apacheclient.partition_spec = mock.Mock(name='partition_spec')
        self.apacheclient.partition_spec.return_value = 'somepart'
        returned = self.apacheclient.table_exists('mytable', partition={'a': 'b'})
        self.assertTrue(returned)
        run_command.return_value = ''
        returned = self.apacheclient.table_exists('mytable', partition={'a': 'b'})
        self.assertFalse(returned)

    @mock.patch('luigi.contrib.hive.run_hive_cmd')
    def test_apacheclient_table_schema(self, run_command):
        if False:
            return 10
        run_command.return_value = 'FAILED: SemanticException [Error 10001]: Table not found mytable\nSome other stuff'
        returned = self.apacheclient.table_schema('mytable')
        self.assertFalse(returned)
        run_command.return_value = 'OK\ncol1       \tstring              \tNone                \ncol2            \tstring              \tNone                \ncol3         \tstring              \tNone                \nday                 \tstring              \tNone                \nhour                \tsmallint            \tNone                \n\n# Partition Information\t \t \n# col_name            \tdata_type           \tcomment             \n\nday                 \tstring              \tNone                \nhour                \tsmallint            \tNone                \nTime taken: 2.08 seconds, Fetched: 34 row(s)\n'
        expected = [('OK',), ('col1', 'string', 'None'), ('col2', 'string', 'None'), ('col3', 'string', 'None'), ('day', 'string', 'None'), ('hour', 'smallint', 'None'), ('',), ('# Partition Information',), ('# col_name', 'data_type', 'comment'), ('',), ('day', 'string', 'None'), ('hour', 'smallint', 'None'), ('Time taken: 2.08 seconds, Fetched: 34 row(s)',)]
        returned = self.apacheclient.table_schema('mytable')
        self.assertEqual(expected, returned)

    @mock.patch('luigi.contrib.hive.HiveThriftContext')
    def test_metastoreclient_partition_existence_regardless_of_order(self, thrift_context):
        if False:
            for i in range(10):
                print('nop')
        thrift_context.return_value = thrift_context
        client_mock = mock.Mock(name='clientmock')
        client_mock.return_value = client_mock
        thrift_context.__enter__ = client_mock
        client_mock.get_partition_names = mock.Mock(return_value=['p1=x/p2=y', 'p1=a/p2=b'])
        partition_spec = OrderedDict([('p1', 'a'), ('p2', 'b')])
        self.assertTrue(self.metastoreclient.table_exists('table', 'default', partition_spec))
        partition_spec = OrderedDict([('p2', 'b'), ('p1', 'a')])
        self.assertTrue(self.metastoreclient.table_exists('table', 'default', partition_spec))

    def test_metastore_partition_spec_has_the_same_order(self):
        if False:
            for i in range(10):
                print('nop')
        partition_spec = OrderedDict([('p1', 'a'), ('p2', 'b')])
        spec_string = luigi.contrib.hive.MetastoreClient().partition_spec(partition_spec)
        self.assertEqual(spec_string, 'p1=a/p2=b')
        partition_spec = OrderedDict([('p2', 'b'), ('p1', 'a')])
        spec_string = luigi.contrib.hive.MetastoreClient().partition_spec(partition_spec)
        self.assertEqual(spec_string, 'p1=a/p2=b')

    @mock.patch('luigi.configuration')
    def test_client_def(self, hive_syntax):
        if False:
            return 10
        hive_syntax.get_config.return_value.get.return_value = 'cdh4'
        client = luigi.contrib.hive.get_default_client()
        self.assertEqual(luigi.contrib.hive.HiveCommandClient, type(client))
        hive_syntax.get_config.return_value.get.return_value = 'cdh3'
        client = luigi.contrib.hive.get_default_client()
        self.assertEqual(luigi.contrib.hive.HiveCommandClient, type(client))
        hive_syntax.get_config.return_value.get.return_value = 'apache'
        client = luigi.contrib.hive.get_default_client()
        self.assertEqual(luigi.contrib.hive.ApacheHiveCommandClient, type(client))
        hive_syntax.get_config.return_value.get.return_value = 'metastore'
        client = luigi.contrib.hive.get_default_client()
        self.assertEqual(luigi.contrib.hive.MetastoreClient, type(client))
        hive_syntax.get_config.return_value.get.return_value = 'warehouse'
        client = luigi.contrib.hive.get_default_client()
        self.assertEqual(luigi.contrib.hive.WarehouseHiveClient, type(client))

    @mock.patch('subprocess.Popen')
    def test_run_hive_command(self, popen):
        if False:
            print('Hello World!')
        comm = mock.Mock(name='communicate_mock')
        comm.return_value = (b'some return stuff', '')
        preturn = mock.Mock(name='open_mock')
        preturn.returncode = 0
        preturn.communicate = comm
        popen.return_value = preturn
        returned = luigi.contrib.hive.run_hive(['blah', 'blah'])
        self.assertEqual('some return stuff', returned)
        preturn.returncode = 17
        self.assertRaises(luigi.contrib.hive.HiveCommandError, luigi.contrib.hive.run_hive, ['blah', 'blah'])
        comm.return_value = (b'', 'some stderr stuff')
        returned = luigi.contrib.hive.run_hive(['blah', 'blah'], False)
        self.assertEqual('', returned)

class WarehouseHiveClientTest(unittest.TestCase):

    def test_table_exists_files_actually_exist(self):
        if False:
            return 10
        hdfs_client = mock.Mock(name='hdfs_client')
        hdfs_client.exists.return_value = True
        hdfs_client.listdir.return_value = ['00000_0', '00000_1', '00000_2', '.tmp/']
        warehouse_hive_client = luigi.contrib.hive.WarehouseHiveClient(hdfs_client=hdfs_client, warehouse_location='/apps/hive/warehouse')
        exists = warehouse_hive_client.table_exists(database='some_db', table='table_name', partition=OrderedDict(a=1, b=2))
        assert exists
        hdfs_client.exists.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/a=1/b=2')

    @mock.patch('luigi.configuration')
    def test_table_exists_without_partition_spec_files_actually_exist(self, warehouse_location):
        if False:
            print('Hello World!')
        warehouse_location.get_config.return_value.get.return_value = '/apps/hive/warehouse'
        hdfs_client = mock.Mock(name='hdfs_client')
        hdfs_client.exists.return_value = True
        hdfs_client.listdir.return_value = ['00000_0', '00000_1', '00000_2', '.tmp/']
        warehouse_hive_client = luigi.contrib.hive.WarehouseHiveClient(hdfs_client=hdfs_client)
        exists = warehouse_hive_client.table_exists(database='some_db', table='table_name')
        assert exists
        hdfs_client.exists.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/')
        hdfs_client.listdir.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/')

    @mock.patch('luigi.configuration')
    def test_table_exists_only_tmp_files_exist(self, ignored_file_masks):
        if False:
            for i in range(10):
                print('nop')
        ignored_file_masks.get_config.return_value.get.return_value = '(\\.tmp.*)'
        hdfs_client = mock.Mock(name='hdfs_client')
        hdfs_client.exists.return_value = True
        hdfs_client.listdir.return_value = ['.tmp/']
        warehouse_hive_client = luigi.contrib.hive.WarehouseHiveClient(hdfs_client=hdfs_client, warehouse_location='/apps/hive/warehouse')
        exists = warehouse_hive_client.table_exists(database='some_db', table='table_name', partition={'a': 1})
        assert not exists
        hdfs_client.exists.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/a=1')
        hdfs_client.listdir.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/a=1')

    @mock.patch('luigi.configuration')
    def test_table_exists_ambiguous_partition(self, ignored_file_masks):
        if False:
            while True:
                i = 10
        ignored_file_masks.get_config.return_value.get.return_value = '(\\.tmp.*)'
        hdfs_client = mock.Mock(name='hdfs_client')
        hdfs_client.exists.return_value = True
        hdfs_client.listdir.return_value = ['.tmp/']
        warehouse_hive_client = luigi.contrib.hive.WarehouseHiveClient(hdfs_client=hdfs_client, warehouse_location='/apps/hive/warehouse')

        def _call_exists():
            if False:
                i = 10
                return i + 15
            return warehouse_hive_client.table_exists(database='some_db', table='table_name', partition={'a': 1, 'b': 2})
        if sys.version_info >= (3, 7):
            exists = _call_exists()
            assert not exists
            hdfs_client.exists.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/a=1/b=2')
            hdfs_client.listdir.assert_called_once_with('/apps/hive/warehouse/some_db.db/table_name/a=1/b=2')
        else:
            self.assertRaises(ValueError, _call_exists)

class MyHiveTask(luigi.contrib.hive.HiveQueryTask):
    param = luigi.Parameter()

    def query(self):
        if False:
            return 10
        return 'banana banana %s' % self.param

@pytest.mark.apache
class TestHiveTask(unittest.TestCase):
    task_class = MyHiveTask

    @mock.patch('luigi.contrib.hadoop.run_and_track_hadoop_job')
    def test_run(self, run_and_track_hadoop_job):
        if False:
            print('Hello World!')
        success = luigi.run([self.task_class.__name__, '--param', 'foo', '--local-scheduler', '--no-lock'])
        self.assertTrue(success)
        self.assertEqual('hive', run_and_track_hadoop_job.call_args[0][0][0])

class MyHiveTaskArgs(MyHiveTask):

    def hivevars(self):
        if False:
            print('Hello World!')
        return {'my_variable1': 'value1', 'my_variable2': 'value2'}

    def hiveconfs(self):
        if False:
            while True:
                i = 10
        return {'hive.additional.conf': 'conf_value'}

class TestHiveTaskArgs(TestHiveTask):
    task_class = MyHiveTaskArgs

    def test_arglist(self):
        if False:
            print('Hello World!')
        task = self.task_class(param='foo')
        f_name = 'my_file'
        runner = luigi.contrib.hive.HiveQueryRunner()
        arglist = runner.get_arglist(f_name, task)
        f_idx = arglist.index('-f')
        self.assertEqual(arglist[f_idx + 1], f_name)
        hivevars = ['{}={}'.format(k, v) for (k, v) in task.hivevars().items()]
        for var in hivevars:
            idx = arglist.index(var)
            self.assertEqual(arglist[idx - 1], '--hivevar')
        hiveconfs = ['{}={}'.format(k, v) for (k, v) in task.hiveconfs().items()]
        for conf in hiveconfs:
            idx = arglist.index(conf)
            self.assertEqual(arglist[idx - 1], '--hiveconf')

@pytest.mark.apache
class TestHiveTarget(unittest.TestCase):

    def test_hive_table_target(self):
        if False:
            return 10
        client = mock.Mock()
        target = luigi.contrib.hive.HiveTableTarget(database='db', table='foo', client=client)
        target.exists()
        client.table_exists.assert_called_with('foo', 'db', None)

    def test_hive_partition_target(self):
        if False:
            while True:
                i = 10
        client = mock.Mock()
        target = luigi.contrib.hive.HivePartitionTarget(database='db', table='foo', partition='bar', client=client)
        target.exists()
        client.table_exists.assert_called_with('foo', 'db', 'bar')

class ExternalHiveTaskTest(unittest.TestCase):

    def test_table(self):
        if False:
            while True:
                i = 10

        class _Task(luigi.contrib.hive.ExternalHiveTask):
            database = 'schema1'
            table = 'table1'
        output = _Task().output()
        assert isinstance(output, luigi.contrib.hive.HivePartitionTarget)
        assert output.database == 'schema1'
        assert output.table == 'table1'
        assert output.partition == {}

    def test_partition_exists(self):
        if False:
            i = 10
            return i + 15

        class _Task(luigi.contrib.hive.ExternalHiveTask):
            database = 'schema2'
            table = 'table2'
            partition = {'a': 1}
        output = _Task().output()
        assert isinstance(output, luigi.contrib.hive.HivePartitionTarget)
        assert output.database == 'schema2'
        assert output.table == 'table2'
        assert output.partition == {'a': 1}