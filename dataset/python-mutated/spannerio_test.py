import datetime
import logging
import random
import string
import typing
import unittest
import mock
import apache_beam as beam
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    from google.cloud import spanner
    from apache_beam.io.gcp.experimental.spannerio import create_transaction
    from apache_beam.io.gcp.experimental.spannerio import ReadOperation
    from apache_beam.io.gcp.experimental.spannerio import ReadFromSpanner
    from apache_beam.io.gcp.experimental.spannerio import WriteMutation
    from apache_beam.io.gcp.experimental.spannerio import MutationGroup
    from apache_beam.io.gcp.experimental.spannerio import WriteToSpanner
    from apache_beam.io.gcp.experimental.spannerio import _BatchFn
    from apache_beam.io.gcp import resource_identifiers
    from apache_beam.metrics import monitoring_infos
    from apache_beam.metrics.execution import MetricsEnvironment
    from apache_beam.metrics.metricbase import MetricName
except ImportError:
    spanner = None
MAX_DB_NAME_LENGTH = 30
TEST_PROJECT_ID = 'apache-beam-testing'
TEST_INSTANCE_ID = 'beam-test'
TEST_DATABASE_PREFIX = 'spanner-testdb-'
FAKE_TRANSACTION_INFO = {'session_id': 'qwerty', 'transaction_id': 'qwerty'}
FAKE_ROWS = [[1, 'Alice'], [2, 'Bob'], [3, 'Carl'], [4, 'Dan'], [5, 'Evan'], [6, 'Floyd']]

def _generate_database_name():
    if False:
        i = 10
        return i + 15
    mask = string.ascii_lowercase + string.digits
    length = MAX_DB_NAME_LENGTH - 1 - len(TEST_DATABASE_PREFIX)
    return TEST_DATABASE_PREFIX + ''.join((random.choice(mask) for i in range(length)))

def _generate_test_data():
    if False:
        print('Hello World!')
    mask = string.ascii_lowercase + string.digits
    length = 100
    return [('users', ['Key', 'Value'], [(x, ''.join((random.choice(mask) for _ in range(length)))) for x in range(1, 5)])]

@unittest.skipIf(spanner is None, 'GCP dependencies are not installed.')
@mock.patch('apache_beam.io.gcp.experimental.spannerio.Client')
@mock.patch('apache_beam.io.gcp.experimental.spannerio.BatchSnapshot')
class SpannerReadTest(unittest.TestCase):

    def test_read_with_query_batch(self, mock_batch_snapshot_class, mock_client_class):
        if False:
            while True:
                i = 10
        mock_snapshot_instance = mock.MagicMock()
        mock_snapshot_instance.generate_query_batches.return_value = [{'query': {'sql': 'SELECT * FROM users'}, 'partition': 'test_partition'} for _ in range(3)]
        mock_snapshot_instance.to_dict.return_value = {}
        mock_batch_snapshot_instance = mock.MagicMock()
        mock_batch_snapshot_instance.process_query_batch.side_effect = [FAKE_ROWS[0:2], FAKE_ROWS[2:4], FAKE_ROWS[4:]] * 3
        mock_client_class.return_value.instance.return_value.database.return_value.batch_snapshot.return_value = mock_snapshot_instance
        mock_batch_snapshot_class.from_dict.return_value = mock_batch_snapshot_instance
        ro = [ReadOperation.query('Select * from users')]
        with TestPipeline() as pipeline:
            read = pipeline | 'read' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), sql='SELECT * FROM users')
            assert_that(read, equal_to(FAKE_ROWS), label='checkRead')
        with TestPipeline() as pipeline:
            readall = pipeline | 'read all' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), read_operations=ro)
            assert_that(readall, equal_to(FAKE_ROWS), label='checkReadAll')
        with TestPipeline() as pipeline:
            readpipeline = pipeline | 'create reads' >> beam.Create(ro) | 'reads' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name())
            assert_that(readpipeline, equal_to(FAKE_ROWS), label='checkReadPipeline')
        self.assertEqual(mock_snapshot_instance.generate_query_batches.call_count, 3)
        self.assertEqual(mock_batch_snapshot_instance.process_query_batch.call_count, 3 * 3)

    def test_read_with_table_batch(self, mock_batch_snapshot_class, mock_client_class):
        if False:
            return 10
        mock_snapshot_instance = mock.MagicMock()
        mock_snapshot_instance.generate_read_batches.return_value = [{'read': {'table': 'users', 'keyset': {'all': True}, 'columns': ['Key', 'Value'], 'index': ''}, 'partition': 'test_partition'} for _ in range(3)]
        mock_snapshot_instance.to_dict.return_value = {}
        mock_batch_snapshot_instance = mock.MagicMock()
        mock_batch_snapshot_instance.process_read_batch.side_effect = [FAKE_ROWS[0:2], FAKE_ROWS[2:4], FAKE_ROWS[4:]] * 3
        mock_client_class.return_value.instance.return_value.database.return_value.batch_snapshot.return_value = mock_snapshot_instance
        mock_batch_snapshot_class.from_dict.return_value = mock_batch_snapshot_instance
        ro = [ReadOperation.table('users', ['Key', 'Value'])]
        with TestPipeline() as pipeline:
            read = pipeline | 'read' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), table='users', columns=['Key', 'Value'])
            assert_that(read, equal_to(FAKE_ROWS), label='checkRead')
        with TestPipeline() as pipeline:
            readall = pipeline | 'read all' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), read_operations=ro)
            assert_that(readall, equal_to(FAKE_ROWS), label='checkReadAll')
        with TestPipeline() as pipeline:
            readpipeline = pipeline | 'create reads' >> beam.Create(ro) | 'reads' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name())
            assert_that(readpipeline, equal_to(FAKE_ROWS), label='checkReadPipeline')
        self.assertEqual(mock_snapshot_instance.generate_read_batches.call_count, 3)
        self.assertEqual(mock_batch_snapshot_instance.process_read_batch.call_count, 3 * 3)
        with TestPipeline() as pipeline, self.assertRaises(ValueError):
            _ = pipeline | 'reads error' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), table='users')

    def test_read_with_index(self, mock_batch_snapshot_class, mock_client_class):
        if False:
            return 10
        mock_snapshot_instance = mock.MagicMock()
        mock_snapshot_instance.generate_read_batches.return_value = [{'read': {'table': 'users', 'keyset': {'all': True}, 'columns': ['Key', 'Value'], 'index': ''}, 'partition': 'test_partition'} for _ in range(3)]
        mock_batch_snapshot_instance = mock.MagicMock()
        mock_batch_snapshot_instance.process_read_batch.side_effect = [FAKE_ROWS[0:2], FAKE_ROWS[2:4], FAKE_ROWS[4:]] * 3
        mock_snapshot_instance.to_dict.return_value = {}
        mock_client_class.return_value.instance.return_value.database.return_value.batch_snapshot.return_value = mock_snapshot_instance
        mock_batch_snapshot_class.from_dict.return_value = mock_batch_snapshot_instance
        ro = [ReadOperation.table('users', ['Key', 'Value'], index='Key')]
        with TestPipeline() as pipeline:
            read = pipeline | 'read' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), table='users', columns=['Key', 'Value'])
            assert_that(read, equal_to(FAKE_ROWS), label='checkRead')
        with TestPipeline() as pipeline:
            readall = pipeline | 'read all' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), read_operations=ro)
            assert_that(readall, equal_to(FAKE_ROWS), label='checkReadAll')
        with TestPipeline() as pipeline:
            readpipeline = pipeline | 'create reads' >> beam.Create(ro) | 'reads' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name())
            assert_that(readpipeline, equal_to(FAKE_ROWS), label='checkReadPipeline')
        self.assertEqual(mock_snapshot_instance.generate_read_batches.call_count, 3)
        self.assertEqual(mock_batch_snapshot_instance.process_read_batch.call_count, 3 * 3)
        with TestPipeline() as pipeline, self.assertRaises(ValueError):
            _ = pipeline | 'reads error' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), table='users')

    def test_read_with_transaction(self, mock_batch_snapshot_class, mock_client_class):
        if False:
            for i in range(10):
                print('nop')
        mock_snapshot_instance = mock.MagicMock()
        mock_snapshot_instance.to_dict.return_value = FAKE_TRANSACTION_INFO
        mock_transaction_instance = mock.MagicMock()
        mock_transaction_instance.execute_sql.return_value = FAKE_ROWS
        mock_transaction_instance.read.return_value = FAKE_ROWS
        mock_client_class.return_value.instance.return_value.database.return_value.batch_snapshot.return_value = mock_snapshot_instance
        mock_client_class.return_value.instance.return_value.database.return_value.session.return_value.transaction.return_value.__enter__.return_value = mock_transaction_instance
        ro = [ReadOperation.query('Select * from users')]
        with TestPipeline() as p:
            transaction = p | create_transaction(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), exact_staleness=datetime.timedelta(seconds=10))
            read_query = p | 'with query' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), transaction=transaction, sql='Select * from users')
            assert_that(read_query, equal_to(FAKE_ROWS), label='checkQuery')
            read_table = p | 'with table' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), transaction=transaction, table='users', columns=['Key', 'Value'])
            assert_that(read_table, equal_to(FAKE_ROWS), label='checkTable')
            read_indexed_table = p | 'with index' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), transaction=transaction, table='users', index='Key', columns=['Key', 'Value'])
            assert_that(read_indexed_table, equal_to(FAKE_ROWS), label='checkTableIndex')
            read = p | 'read all' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), transaction=transaction, read_operations=ro)
            assert_that(read, equal_to(FAKE_ROWS), label='checkReadAll')
            read_pipeline = p | 'create read operations' >> beam.Create(ro) | 'reads' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), transaction=transaction)
            assert_that(read_pipeline, equal_to(FAKE_ROWS), label='checkReadPipeline')
        self.assertEqual(mock_snapshot_instance.to_dict.call_count, 1)
        self.assertEqual(mock_transaction_instance.execute_sql.call_count, 3)
        self.assertEqual(mock_transaction_instance.read.call_count, 2)
        with TestPipeline() as p, self.assertRaises(ValueError):
            transaction = p | create_transaction(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), exact_staleness=datetime.timedelta(seconds=10))
            _ = p | 'create read operations2' >> beam.Create(ro) | 'reads with error' >> ReadFromSpanner(TEST_PROJECT_ID, TEST_INSTANCE_ID, _generate_database_name(), transaction=transaction, read_operations=ro)

    def test_invalid_transaction(self, mock_batch_snapshot_class, mock_client_class):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError), TestPipeline() as p:
            transaction = p | beam.Create([{'invalid': 'transaction'}]).with_output_types(typing.Any)
            _ = p | 'with query' >> ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), transaction=transaction, sql='Select * from users')

    def test_display_data(self, *args):
        if False:
            while True:
                i = 10
        dd_sql = ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), sql='Select * from users').display_data()
        dd_table = ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), table='users', columns=['id', 'name']).display_data()
        dd_transaction = ReadFromSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), table='users', columns=['id', 'name'], transaction={'transaction_id': 'test123', 'session_id': 'test456'}).display_data()
        self.assertTrue('sql' in dd_sql)
        self.assertTrue('table' in dd_table)
        self.assertTrue('table' in dd_transaction)
        self.assertTrue('transaction' in dd_transaction)

@unittest.skipIf(spanner is None, 'GCP dependencies are not installed.')
@mock.patch('apache_beam.io.gcp.experimental.spannerio.Client')
@mock.patch('google.cloud.spanner_v1.database.BatchCheckout')
class SpannerWriteTest(unittest.TestCase):

    def test_spanner_write(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            return 10
        ks = spanner.KeySet(keys=[[1233], [1234]])
        mutations = [WriteMutation.delete('roles', ks), WriteMutation.insert('roles', ('key', 'rolename'), [('1233', 'mutations-inset-1233')]), WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234')]), WriteMutation.update('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1233-updated')])]
        p = TestPipeline()
        _ = p | beam.Create(mutations) | WriteToSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), max_batch_size_bytes=1024)
        res = p.run()
        res.wait_until_finish()
        metric_results = res.metrics().query(MetricsFilter().with_name('SpannerBatches'))
        batches_counter = metric_results['counters'][0]
        self.assertEqual(batches_counter.committed, 2)
        self.assertEqual(batches_counter.attempted, 2)

    def test_spanner_bundles_size(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            return 10
        ks = spanner.KeySet(keys=[[1233], [1234]])
        mutations = [WriteMutation.delete('roles', ks), WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234')])] * 50
        p = TestPipeline()
        _ = p | beam.Create(mutations) | WriteToSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), max_batch_size_bytes=1024)
        res = p.run()
        res.wait_until_finish()
        metric_results = res.metrics().query(MetricsFilter().with_name('SpannerBatches'))
        batches_counter = metric_results['counters'][0]
        self.assertEqual(batches_counter.committed, 53)
        self.assertEqual(batches_counter.attempted, 53)

    def test_spanner_write_mutation_groups(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            print('Hello World!')
        ks = spanner.KeySet(keys=[[1233], [1234]])
        mutation_groups = [MutationGroup([WriteMutation.insert('roles', ('key', 'rolename'), [('9001233', 'mutations-inset-1233')]), WriteMutation.insert('roles', ('key', 'rolename'), [('9001234', 'mutations-inset-1234')])]), MutationGroup([WriteMutation.update('roles', ('key', 'rolename'), [('9001234', 'mutations-inset-9001233-updated')])]), MutationGroup([WriteMutation.delete('roles', ks)])]
        p = TestPipeline()
        _ = p | beam.Create(mutation_groups) | WriteToSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), max_batch_size_bytes=100)
        res = p.run()
        res.wait_until_finish()
        metric_results = res.metrics().query(MetricsFilter().with_name('SpannerBatches'))
        batches_counter = metric_results['counters'][0]
        self.assertEqual(batches_counter.committed, 3)
        self.assertEqual(batches_counter.attempted, 3)

    def test_batch_byte_size(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            for i in range(10):
                print('nop')
        mutation_group = [MutationGroup([WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234')])])] * 50
        with TestPipeline() as p:
            res = p | beam.Create(mutation_group) | beam.ParDo(_BatchFn(max_batch_size_bytes=1450, max_number_rows=50, max_number_cells=500)) | beam.Map(lambda x: len(x))
            assert_that(res, equal_to([25] * 2))

    def test_batch_disable(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            i = 10
            return i + 15
        mutation_group = [MutationGroup([WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234')])])] * 4
        with TestPipeline() as p:
            res = p | beam.Create(mutation_group) | beam.ParDo(_BatchFn(max_batch_size_bytes=1450, max_number_rows=0, max_number_cells=500)) | beam.Map(lambda x: len(x))
            assert_that(res, equal_to([1] * 4))

    def test_batch_max_rows(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            for i in range(10):
                print('nop')
        mutation_group = [MutationGroup([WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234'), ('1235', 'mutations-inset-1235')])])] * 50
        with TestPipeline() as p:
            res = p | beam.Create(mutation_group) | beam.ParDo(_BatchFn(max_batch_size_bytes=1048576, max_number_rows=10, max_number_cells=500)) | beam.Map(lambda x: len(x))
            assert_that(res, equal_to([5] * 10))

    def test_batch_max_cells(self, mock_batch_snapshot_class, mock_batch_checkout):
        if False:
            return 10
        mutation_group = [MutationGroup([WriteMutation.insert('roles', ('key', 'rolename'), [('1234', 'mutations-inset-1234'), ('1235', 'mutations-inset-1235')])])] * 50
        with TestPipeline() as p:
            res = p | beam.Create(mutation_group) | beam.ParDo(_BatchFn(max_batch_size_bytes=1048576, max_number_rows=500, max_number_cells=50)) | beam.Map(lambda x: len(x))
            assert_that(res, equal_to([12, 12, 12, 12, 2]))

    def test_write_mutation_error(self, *args):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            WriteMutation(insert='table-name', update='table-name')

    def test_display_data(self, *args):
        if False:
            for i in range(10):
                print('nop')
        data = WriteToSpanner(project_id=TEST_PROJECT_ID, instance_id=TEST_INSTANCE_ID, database_id=_generate_database_name(), max_batch_size_bytes=1024).display_data()
        self.assertTrue('project_id' in data)
        self.assertTrue('instance_id' in data)
        self.assertTrue('pool' in data)
        self.assertTrue('database' in data)
        self.assertTrue('batch_size' in data)
        self.assertTrue('max_number_rows' in data)
        self.assertTrue('max_number_cells' in data)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()