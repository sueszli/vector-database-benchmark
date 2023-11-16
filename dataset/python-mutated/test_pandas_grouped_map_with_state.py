import random
import shutil
import string
import sys
import tempfile
import unittest
from typing import cast
from pyspark import SparkConf
from pyspark.sql.streaming.state import GroupStateTimeout, GroupState
from pyspark.sql.types import LongType, StringType, StructType, StructField, Row
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message
from pyspark.testing.utils import eventually
if have_pandas:
    import pandas as pd
if have_pyarrow:
    import pyarrow as pa

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class GroupedApplyInPandasWithStateTestsMixin:

    @classmethod
    def conf(cls):
        if False:
            return 10
        cfg = SparkConf()
        cfg.set('spark.sql.shuffle.partitions', '5')
        return cfg

    def _test_apply_in_pandas_with_state_basic(self, func, check_results):
        if False:
            return 10
        input_path = tempfile.mkdtemp()

        def prepare_test_resource():
            if False:
                i = 10
                return i + 15
            with open(input_path + '/text-test.txt', 'w') as fw:
                fw.write('hello\n')
                fw.write('this\n')
        prepare_test_resource()
        df = self.spark.readStream.format('text').load(input_path)
        for q in self.spark.streams.active:
            q.stop()
        self.assertTrue(df.isStreaming)
        output_type = StructType([StructField('key', StringType()), StructField('countAsString', StringType())])
        state_type = StructType([StructField('c', LongType())])
        q = df.groupBy(df['value']).applyInPandasWithState(func, output_type, state_type, 'Update', GroupStateTimeout.NoTimeout).writeStream.queryName('this_query').foreachBatch(check_results).outputMode('update').start()
        self.assertEqual(q.name, 'this_query')
        self.assertTrue(q.isActive)
        q.processAllAvailable()

    def test_apply_in_pandas_with_state_basic(self):
        if False:
            i = 10
            return i + 15

        def func(key, pdf_iter, state):
            if False:
                return 10
            assert isinstance(state, GroupState)
            total_len = 0
            for pdf in pdf_iter:
                total_len += len(pdf)
            state.update((total_len,))
            assert state.get[0] == 1
            yield pd.DataFrame({'key': [key[0]], 'countAsString': [str(total_len)]})

        def check_results(batch_df, _):
            if False:
                while True:
                    i = 10
            self.assertEqual(set(batch_df.sort('key').collect()), {Row(key='hello', countAsString='1'), Row(key='this', countAsString='1')})
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_basic_no_state(self):
        if False:
            for i in range(10):
                print('nop')

        def func(key, pdf_iter, state):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(state, GroupState)
            yield pd.DataFrame({'key': [key[0], 'foo'], 'countAsString': ['100', '222']})

        def check_results(batch_df, _):
            if False:
                while True:
                    i = 10
            self.assertEqual(set(batch_df.sort('key').collect()), {Row(key='hello', countAsString='100'), Row(key='this', countAsString='100'), Row(key='foo', countAsString='222')})
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_basic_no_state_no_data(self):
        if False:
            while True:
                i = 10

        def func(key, pdf_iter, state):
            if False:
                i = 10
                return i + 15
            assert isinstance(state, GroupState)
            yield pd.DataFrame({'key': [], 'countAsString': []})

        def check_results(batch_df, _):
            if False:
                print('Hello World!')
            self.assertTrue(len(set(batch_df.sort('key').collect())) == 0)
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_basic_more_data(self):
        if False:
            i = 10
            return i + 15

        def func(key, pdf_iter, state):
            if False:
                for i in range(10):
                    print('nop')
            state.update((1,))
            assert isinstance(state, GroupState)
            yield pd.DataFrame({'key': [key[0], 'foo', key[0] + '_2'], 'countAsString': ['1', '666', '2']})

        def check_results(batch_df, _):
            if False:
                return 10
            self.assertEqual(set(batch_df.sort('key').collect()), {Row(key='hello', countAsString='1'), Row(key='foo', countAsString='666'), Row(key='hello_2', countAsString='2'), Row(key='this', countAsString='1'), Row(key='this_2', countAsString='2')})
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_basic_fewer_data(self):
        if False:
            return 10

        def func(key, pdf_iter, state):
            if False:
                i = 10
                return i + 15
            state.update((1,))
            assert isinstance(state, GroupState)
            yield pd.DataFrame({'key': [], 'countAsString': []})

        def check_results(batch_df, _):
            if False:
                while True:
                    i = 10
            self.assertTrue(len(set(batch_df.sort('key').collect())) == 0)
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_basic_with_null(self):
        if False:
            while True:
                i = 10

        def func(key, pdf_iter, state):
            if False:
                i = 10
                return i + 15
            assert isinstance(state, GroupState)
            total_len = 0
            for pdf in pdf_iter:
                total_len += len(pdf)
            state.update((total_len,))
            assert state.get[0] == 1
            yield pd.DataFrame({'key': [None], 'countAsString': [str(total_len)]})

        def check_results(batch_df, _):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(set(batch_df.sort('key').collect()), {Row(key=None, countAsString='1')})
        self._test_apply_in_pandas_with_state_basic(func, check_results)

    def test_apply_in_pandas_with_state_python_worker_random_failure(self):
        if False:
            while True:
                i = 10
        input_path = tempfile.mkdtemp()
        output_path = tempfile.mkdtemp()
        checkpoint_loc = tempfile.mkdtemp()
        shutil.rmtree(output_path)
        shutil.rmtree(checkpoint_loc)

        def prepare_test_resource():
            if False:
                for i in range(10):
                    print('nop')
            data_range = list(string.ascii_lowercase)
            for i in range(5):
                picked_data = [data_range[random.randrange(0, len(data_range) - 1)] for x in range(100)]
                with open(input_path + '/part-%i.txt' % i, 'w') as fw:
                    for data in picked_data:
                        fw.write(data + '\n')

        def run_query():
            if False:
                i = 10
                return i + 15
            df = self.spark.readStream.format('text').option('maxFilesPerTrigger', '1').load(input_path)
            for q in self.spark.streams.active:
                q.stop()
            self.assertTrue(df.isStreaming)
            output_type = StructType([StructField('value', StringType()), StructField('count', LongType())])
            state_type = StructType([StructField('cnt', LongType())])

            def func(key, pdf_iter, state):
                if False:
                    print('Hello World!')
                assert isinstance(state, GroupState)
                if random.randrange(30) == 1:
                    sys.exit(1)
                count = state.getOption
                if count is None:
                    count = 0
                else:
                    count = count[0]
                for pdf in pdf_iter:
                    count += len(pdf)
                state.update((count,))
                yield pd.DataFrame({'value': [key[0]], 'count': [count]})
            query = df.groupBy(df['value']).applyInPandasWithState(func, output_type, state_type, 'Append', GroupStateTimeout.NoTimeout).writeStream.queryName('this_query').format('json').outputMode('append').option('path', output_path).option('checkpointLocation', checkpoint_loc).start()
            return query
        prepare_test_resource()
        expected = self.spark.read.format('text').load(input_path).groupBy('value').count().sort('value').collect()
        q = run_query()
        self.assertEqual(q.name, 'this_query')
        self.assertTrue(q.isActive)

        def assert_test():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal q
            if not q.isActive:
                print('query has been terminated, rerunning query...')
                q = run_query()
                self.assertEqual(q.name, 'this_query')
                self.assertTrue(q.isActive)
            curr_status = q.status
            if not curr_status['isDataAvailable'] and (not curr_status['isTriggerActive']):
                result = self.spark.read.schema('value string, count int').format('json').load(output_path).groupBy('value').max('count').selectExpr('value', '`max(count)` AS count').sort('value').collect()
                return result == expected
            else:
                return False
        try:
            eventually(timeout=120)(assert_test)()
        finally:
            q.stop()

class GroupedApplyInPandasWithStateTests(GroupedApplyInPandasWithStateTestsMixin, ReusedSQLTestCase):
    pass
if __name__ == '__main__':
    from pyspark.sql.tests.pandas.test_pandas_grouped_map_with_state import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)