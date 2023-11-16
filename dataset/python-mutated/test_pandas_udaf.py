import uuid
from pyflink.table.expressions import col, call, lit, row_interval
from pyflink.table.types import DataTypes
from pyflink.table.udf import udaf, udf, AggregateFunction
from pyflink.testing import source_sink_utils
from pyflink.testing.test_case_utils import PyFlinkBatchTableTestCase, PyFlinkStreamTableTestCase

def generate_random_table_name():
    if False:
        return 10
    return 'Table{0}'.format(str(uuid.uuid1()).replace('-', '_'))

class BatchPandasUDAFITTests(PyFlinkBatchTableTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(BatchPandasUDAFITTests, cls).setUpClass()
        cls.t_env.create_temporary_system_function('max_add', udaf(MaxAdd(), result_type=DataTypes.INT(), func_type='pandas'))
        cls.t_env.create_temporary_system_function('mean_udaf', mean_udaf)

    def test_check_result_type(self):
        if False:
            print('Hello World!')

        def pandas_udaf():
            if False:
                print('Hello World!')
            pass
        with self.assertRaises(TypeError, msg="Invalid returnType: Pandas UDAF doesn't support DataType type MAP currently"):
            udaf(pandas_udaf, result_type=DataTypes.MAP(DataTypes.INT(), DataTypes.INT()), func_type='pandas')

    def test_group_aggregate_function(self):
        if False:
            print('Hello World!')
        t = self.t_env.from_elements([(1, 2, 3), (3, 2, 3), (2, 1, 3), (1, 5, 4), (1, 8, 6), (2, 3, 4)], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TINYINT()), DataTypes.FIELD('b', DataTypes.SMALLINT()), DataTypes.FIELD('c', DataTypes.INT())]))
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(\n                a TINYINT,\n                b FLOAT,\n                c ROW<a INT, b INT>,\n                d STRING\n            ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        add = udf(lambda a: a + 1, result_type=DataTypes.INT())
        substract = udf(lambda a: a - 1, result_type=DataTypes.INT(), func_type='pandas')
        max_udaf = udaf(lambda a: (a.max(), a.min()), result_type=DataTypes.ROW([DataTypes.FIELD('a', DataTypes.INT()), DataTypes.FIELD('b', DataTypes.INT())]), func_type='pandas')

        @udaf(result_type=DataTypes.STRING(), func_type='pandas')
        def multiply_udaf(a, b):
            if False:
                print('Hello World!')
            return len(a) * b[0]
        t.group_by(t.a).select(t.a, mean_udaf(add(t.b)), max_udaf(substract(t.c)), multiply_udaf(t.b, 'abc')).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 6.0, +I[5, 2], abcabcabc]', '+I[2, 3.0, +I[3, 2], abcabc]', '+I[3, 3.0, +I[2, 2], abc]'])

    def test_group_aggregate_without_keys(self):
        if False:
            i = 10
            return i + 15
        t = self.t_env.from_elements([(1, 2, 3), (3, 2, 3), (2, 1, 3), (1, 5, 4), (1, 8, 6), (2, 3, 4)], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TINYINT()), DataTypes.FIELD('b', DataTypes.SMALLINT()), DataTypes.FIELD('c', DataTypes.INT())]))
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(a INT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        min_add = udaf(lambda a, b, c: a.min() + b.min() + c.min(), result_type=DataTypes.INT(), func_type='pandas')
        t.select(min_add(t.a, t.b, t.c)).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[5]'])

    def test_group_aggregate_with_aux_group(self):
        if False:
            while True:
                i = 10
        t = self.t_env.from_elements([(1, 2, 3), (3, 2, 3), (2, 1, 3), (1, 5, 4), (1, 8, 6), (2, 3, 4)], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TINYINT()), DataTypes.FIELD('b', DataTypes.SMALLINT()), DataTypes.FIELD('c', DataTypes.INT())]))
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, b INT, c FLOAT, d INT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        self.t_env.get_config().get_configuration().set_string('python.metric.enabled', 'true')
        self.t_env.get_config().set('python.metric.enabled', 'true')
        t.group_by(t.a).select(t.a, (t.a + 1).alias('b'), (t.a + 2).alias('c')).group_by(t.a, t.b).select(t.a, t.b, mean_udaf(t.b), call('max_add', t.b, t.c, 1)).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2, 2.0, 6]', '+I[2, 3, 3.0, 8]', '+I[3, 4, 4.0, 10]'])

    def test_tumble_group_window_aggregate_function(self):
        if False:
            i = 10
            return i + 15
        from pyflink.table.window import Tumble
        data = ['1,2,3,2018-03-11 03:10:00', '3,2,4,2018-03-11 03:10:00', '2,1,2,2018-03-11 03:10:00', '1,3,1,2018-03-11 03:40:00', '1,8,5,2018-03-11 04:20:00', '2,3,6,2018-03-11 03:30:00']
        source_path = self.tempdir + '/test_tumble_group_window_aggregate_function.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c INT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(\n                a TIMESTAMP(3),\n                b TIMESTAMP(3),\n                c FLOAT\n            ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        tumble_window = Tumble.over(lit(1).hours).on(col('rowtime')).alias('w')
        t.window(tumble_window).group_by(col('w')).select(col('w').start, col('w').end, mean_udaf(t.b)).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[2018-03-11T03:00, 2018-03-11T04:00, 2.2]', '+I[2018-03-11T04:00, 2018-03-11T05:00, 8.0]'])

    def test_slide_group_window_aggregate_function(self):
        if False:
            return 10
        from pyflink.table.window import Slide
        data = ['1,2,3,2018-03-11 03:10:00', '3,2,4,2018-03-11 03:10:00', '2,1,2,2018-03-11 03:10:00', '1,3,1,2018-03-11 03:40:00', '1,8,5,2018-03-11 04:20:00', '2,3,6,2018-03-11 03:30:00']
        source_path = self.tempdir + '/test_slide_group_window_aggregate_function.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c INT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(\n                a TINYINT,\n                b TIMESTAMP(3),\n                c TIMESTAMP(3),\n                d FLOAT,\n                e INT\n            ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        slide_window = Slide.over(lit(1).hours).every(lit(30).minutes).on(col('rowtime')).alias('w')
        t.window(slide_window).group_by(t.a, col('w')).select(t.a, col('w').start, col('w').end, mean_udaf(t.b), call('max_add', t.b, t.c, 1)).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2018-03-11T02:30, 2018-03-11T03:30, 2.0, 6]', '+I[1, 2018-03-11T03:00, 2018-03-11T04:00, 2.5, 7]', '+I[1, 2018-03-11T03:30, 2018-03-11T04:30, 5.5, 14]', '+I[1, 2018-03-11T04:00, 2018-03-11T05:00, 8.0, 14]', '+I[2, 2018-03-11T02:30, 2018-03-11T03:30, 1.0, 4]', '+I[2, 2018-03-11T03:00, 2018-03-11T04:00, 2.0, 10]', '+I[2, 2018-03-11T03:30, 2018-03-11T04:30, 3.0, 10]', '+I[3, 2018-03-11T03:00, 2018-03-11T04:00, 2.0, 7]', '+I[3, 2018-03-11T02:30, 2018-03-11T03:30, 2.0, 7]'])

    def test_over_window_aggregate_function(self):
        if False:
            for i in range(10):
                print('nop')
        import datetime
        t = self.t_env.from_elements([(1, 2, 3, datetime.datetime(2018, 3, 11, 3, 10, 0, 0)), (3, 2, 1, datetime.datetime(2018, 3, 11, 3, 10, 0, 0)), (2, 1, 2, datetime.datetime(2018, 3, 11, 3, 10, 0, 0)), (1, 3, 1, datetime.datetime(2018, 3, 11, 3, 10, 0, 0)), (1, 8, 5, datetime.datetime(2018, 3, 11, 4, 20, 0, 0)), (2, 3, 6, datetime.datetime(2018, 3, 11, 3, 30, 0, 0))], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TINYINT()), DataTypes.FIELD('b', DataTypes.SMALLINT()), DataTypes.FIELD('c', DataTypes.INT()), DataTypes.FIELD('rowtime', DataTypes.TIMESTAMP(3))]))
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(\n                a TINYINT,\n                b FLOAT,\n                c INT,\n                d FLOAT,\n                e FLOAT,\n                f FLOAT,\n                g FLOAT,\n                h FLOAT,\n                i FLOAT,\n                j FLOAT\n            ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        self.t_env.create_temporary_view('T_test_over_window_aggregate_function', t)
        self.t_env.execute_sql(f"\n            insert into {sink_table}\n            select a,\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN UNBOUNDED preceding AND UNBOUNDED FOLLOWING),\n             max_add(b, c)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN UNBOUNDED preceding AND 0 FOLLOWING),\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING),\n             mean_udaf(c)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND 0 FOLLOWING),\n             mean_udaf(c)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN INTERVAL '20' MINUTE PRECEDING AND UNBOUNDED FOLLOWING),\n             mean_udaf(c)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN INTERVAL '20' MINUTE PRECEDING AND UNBOUNDED FOLLOWING),\n             mean_udaf(c)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN INTERVAL '20' MINUTE PRECEDING AND CURRENT ROW)\n            from T_test_over_window_aggregate_function\n        ").wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 4.3333335, 5, 4.3333335, 3.0, 3.0, 2.5, 4.3333335, 3.0, 2.0]', '+I[1, 4.3333335, 13, 5.5, 3.0, 3.0, 4.3333335, 8.0, 5.0, 5.0]', '+I[1, 4.3333335, 6, 4.3333335, 2.0, 3.0, 2.5, 4.3333335, 3.0, 2.0]', '+I[2, 2.0, 9, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0]', '+I[2, 2.0, 3, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0]', '+I[3, 2.0, 3, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]'])

class StreamPandasUDAFITTests(PyFlinkStreamTableTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(StreamPandasUDAFITTests, cls).setUpClass()
        cls.t_env.create_temporary_system_function('mean_udaf', mean_udaf)
        max_add_min_udaf = udaf(lambda a: a.max() + a.min(), result_type='SMALLINT', func_type='pandas')
        cls.t_env.create_temporary_system_function('max_add_min_udaf', max_add_min_udaf)

    def test_sliding_group_window_over_time(self):
        if False:
            while True:
                i = 10
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2,2018-03-11 03:10:00', '3,3,2,2018-03-11 03:10:00', '2,2,1,2018-03-11 03:10:00', '1,1,3,2018-03-11 03:40:00', '1,1,8,2018-03-11 04:20:00', '2,2,3,2018-03-11 03:30:00']
        source_path = tmp_dir + '/test_sliding_group_window_over_time.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        from pyflink.table.window import Slide
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c SMALLINT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n            CREATE TABLE {sink_table}(a TINYINT, b TIMESTAMP(3), c TIMESTAMP(3), d FLOAT)\n            WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t.window(Slide.over(lit(1).hours).every(lit(30).minutes).on(col('rowtime')).alias('w')).group_by(t.a, t.b, col('w')).select(t.a, col('w').start, col('w').end, mean_udaf(t.c).alias('b')).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2018-03-11T02:30, 2018-03-11T03:30, 2.0]', '+I[1, 2018-03-11T03:00, 2018-03-11T04:00, 2.5]', '+I[1, 2018-03-11T03:30, 2018-03-11T04:30, 5.5]', '+I[1, 2018-03-11T04:00, 2018-03-11T05:00, 8.0]', '+I[2, 2018-03-11T02:30, 2018-03-11T03:30, 1.0]', '+I[2, 2018-03-11T03:00, 2018-03-11T04:00, 2.0]', '+I[2, 2018-03-11T03:30, 2018-03-11T04:30, 3.0]', '+I[3, 2018-03-11T03:00, 2018-03-11T04:00, 2.0]', '+I[3, 2018-03-11T02:30, 2018-03-11T03:30, 2.0]'])
        os.remove(source_path)

    def test_sliding_group_window_over_count(self):
        if False:
            print('Hello World!')
        self.t_env.get_config().set('parallelism.default', '1')
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2,2018-03-11 03:10:00', '3,3,2,2018-03-11 03:10:00', '2,2,1,2018-03-11 03:10:00', '1,1,3,2018-03-11 03:40:00', '1,1,8,2018-03-11 04:20:00', '2,2,3,2018-03-11 03:30:00', '3,3,3,2018-03-11 03:30:00']
        source_path = tmp_dir + '/test_sliding_group_window_over_count.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        from pyflink.table.window import Slide
        self.t_env.get_config().set('pipeline.time-characteristic', 'ProcessingTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c SMALLINT,\n                protime as PROCTIME()\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '%s',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        " % source_path
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, d FLOAT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t.window(Slide.over(row_interval(2)).every(row_interval(1)).on(t.protime).alias('w')).group_by(t.a, t.b, col('w')).select(t.a, mean_udaf(t.c).alias('b')).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2.5]', '+I[1, 5.5]', '+I[2, 2.0]', '+I[3, 2.5]'])
        os.remove(source_path)

    def test_tumbling_group_window_over_time(self):
        if False:
            return 10
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2,2018-03-11 03:10:00', '3,3,2,2018-03-11 03:10:00', '2,2,1,2018-03-11 03:10:00', '1,1,3,2018-03-11 03:40:00', '1,1,8,2018-03-11 04:20:00', '2,2,3,2018-03-11 03:30:00']
        source_path = tmp_dir + '/test_tumbling_group_window_over_time.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        from pyflink.table.window import Tumble
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c SMALLINT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '%s',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        " % source_path
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(\n        a TINYINT, b TIMESTAMP(3), c TIMESTAMP(3), d TIMESTAMP(3), e FLOAT)\n        WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t.window(Tumble.over(lit(1).hours).on(t.rowtime).alias('w')).group_by(t.a, t.b, col('w')).select(t.a, col('w').start, col('w').end, col('w').rowtime, mean_udaf(t.c).alias('b')).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2018-03-11T03:00, 2018-03-11T04:00, 2018-03-11T03:59:59.999, 2.5]', '+I[1, 2018-03-11T04:00, 2018-03-11T05:00, 2018-03-11T04:59:59.999, 8.0]', '+I[2, 2018-03-11T03:00, 2018-03-11T04:00, 2018-03-11T03:59:59.999, 2.0]', '+I[3, 2018-03-11T03:00, 2018-03-11T04:00, 2018-03-11T03:59:59.999, 2.0]'])
        os.remove(source_path)

    def test_tumbling_group_window_over_count(self):
        if False:
            return 10
        self.t_env.get_config().set('parallelism.default', '1')
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2,2018-03-11 03:10:00', '3,3,2,2018-03-11 03:10:00', '2,2,1,2018-03-11 03:10:00', '1,1,3,2018-03-11 03:40:00', '1,1,8,2018-03-11 04:20:00', '2,2,3,2018-03-11 03:30:00', '3,3,3,2018-03-11 03:30:00', '1,1,4,2018-03-11 04:20:00']
        source_path = tmp_dir + '/test_group_window_aggregate_function_over_count.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        from pyflink.table.window import Tumble
        self.t_env.get_config().set('pipeline.time-characteristic', 'ProcessingTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                c SMALLINT,\n                protime as PROCTIME()\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '%s',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        " % source_path
        self.t_env.execute_sql(source_table_ddl)
        t = self.t_env.from_path(source_table)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, d FLOAT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t.window(Tumble.over(row_interval(2)).on(t.protime).alias('w')).group_by(t.a, t.b, col('w')).select(t.a, mean_udaf(t.c).alias('b')).execute_insert(sink_table).wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 2.5]', '+I[1, 6.0]', '+I[2, 2.0]', '+I[3, 2.5]'])
        os.remove(source_path)

    def test_row_time_over_range_window_aggregate_function(self):
        if False:
            while True:
                i = 10
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2013-01-01 03:10:00', '3,2,2013-01-01 03:10:00', '2,1,2013-01-01 03:10:00', '1,5,2013-01-01 03:10:00', '1,8,2013-01-01 04:20:00', '2,3,2013-01-01 03:30:00']
        source_path = tmp_dir + '/test_over_range_window_aggregate_function.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, b FLOAT, c SMALLINT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        self.t_env.execute_sql(f"\n            insert into {sink_table}\n            select a,\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN INTERVAL '20' MINUTE PRECEDING AND CURRENT ROW),\n             max_add_min_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             RANGE BETWEEN INTERVAL '20' MINUTE PRECEDING AND CURRENT ROW)\n            from {source_table}\n        ").wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 3.0, 6]', '+I[1, 3.0, 6]', '+I[1, 8.0, 16]', '+I[2, 1.0, 2]', '+I[2, 2.0, 4]', '+I[3, 2.0, 4]'])
        os.remove(source_path)

    def test_row_time_over_rows_window_aggregate_function(self):
        if False:
            while True:
                i = 10
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['1,1,2013-01-01 03:10:00', '3,2,2013-01-01 03:10:00', '2,1,2013-01-01 03:10:00', '1,5,2013-01-01 03:10:00', '1,8,2013-01-01 04:20:00', '2,3,2013-01-01 03:30:00']
        source_path = tmp_dir + '/test_over_rows_window_aggregate_function.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, b FLOAT, c SMALLINT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        self.t_env.execute_sql(f'\n            insert into {sink_table}\n            select a,\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW),\n             max_add_min_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)\n            from {source_table}\n        ').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 1.0, 2]', '+I[1, 3.0, 6]', '+I[1, 6.5, 13]', '+I[2, 1.0, 2]', '+I[2, 2.0, 4]', '+I[3, 2.0, 4]'])
        os.remove(source_path)

    def test_proc_time_over_rows_window_aggregate_function(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['1,1,2013-01-01 03:10:00', '3,2,2013-01-01 03:10:00', '2,1,2013-01-01 03:10:00', '1,5,2013-01-01 03:10:00', '1,8,2013-01-01 04:20:00', '2,3,2013-01-01 03:30:00']
        source_path = self.tempdir + '/test_proc_time_over_rows_window_aggregate_function.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('parallelism.default', '1')
        self.t_env.get_config().set('pipeline.time-characteristic', 'ProcessingTime')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            create table {source_table}(\n                a TINYINT,\n                b SMALLINT,\n                proctime as PROCTIME()\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '{source_path}',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        sink_table = generate_random_table_name()
        sink_table_ddl = f"\n        CREATE TABLE {sink_table}(a TINYINT, b FLOAT, c SMALLINT) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        self.t_env.execute_sql(f'\n            insert into {sink_table}\n            select a,\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY proctime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW),\n             max_add_min_udaf(b)\n             over (PARTITION BY a ORDER BY proctime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)\n            from {source_table}\n        ').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 1.0, 2]', '+I[1, 3.0, 6]', '+I[1, 6.5, 13]', '+I[2, 1.0, 2]', '+I[2, 2.0, 4]', '+I[3, 2.0, 4]'])

    def test_execute_over_aggregate_from_json_plan(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.tempdir
        data = ['1,1,2013-01-01 03:10:00', '3,2,2013-01-01 03:10:00', '2,1,2013-01-01 03:10:00', '1,5,2013-01-01 03:10:00', '1,8,2013-01-01 04:20:00', '2,3,2013-01-01 03:30:00']
        source_path = tmp_dir + '/test_execute_over_aggregate_from_json_plan.csv'
        sink_path = tmp_dir + '/test_execute_over_aggregate_from_json_plan'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        source_table = generate_random_table_name()
        source_table_ddl = f"\n            CREATE TABLE {source_table} (\n                a TINYINT,\n                b SMALLINT,\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) WITH (\n                'connector' = 'filesystem',\n                'path' = '{source_path}',\n                'format' = 'csv'\n            )\n        "
        self.t_env.execute_sql(source_table_ddl)
        sink_table = generate_random_table_name()
        self.t_env.execute_sql(f"\n            CREATE TABLE {sink_table} (\n                a TINYINT,\n                b FLOAT,\n                c SMALLINT\n            ) WITH (\n                'connector' = 'filesystem',\n                'path' = '{sink_path}',\n                'format' = 'csv'\n            )\n        ")
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        json_plan = self.t_env._j_tenv.compilePlanSql(f'\n        insert into {sink_table}\n            select a,\n             mean_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW),\n             max_add_min_udaf(b)\n             over (PARTITION BY a ORDER BY rowtime\n             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)\n            from {source_table}\n        ')
        from py4j.java_gateway import get_method
        get_method(json_plan.execute(), 'await')()
        import glob
        lines = [line.strip() for file in glob.glob(sink_path + '/*') for line in open(file, 'r')]
        lines.sort()
        self.assertEqual(lines, ['1,1.0,2', '1,3.0,6', '1,6.5,13', '2,1.0,2', '2,2.0,4', '3,2.0,4'])

@udaf(result_type=DataTypes.FLOAT(), func_type='pandas')
def mean_udaf(v):
    if False:
        print('Hello World!')
    return v.mean()

class MaxAdd(AggregateFunction):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.counter = None
        self.counter_sum = 0

    def open(self, function_context):
        if False:
            for i in range(10):
                print('nop')
        mg = function_context.get_metric_group()
        self.counter = mg.add_group('key', 'value').counter('my_counter')
        self.counter_sum = 0

    def get_value(self, accumulator):
        if False:
            while True:
                i = 10
        self.counter.inc(10)
        self.counter_sum += 10
        return accumulator[0]

    def create_accumulator(self):
        if False:
            print('Hello World!')
        return []

    def accumulate(self, accumulator, *args):
        if False:
            while True:
                i = 10
        result = 0
        for arg in args:
            result += arg.max()
        accumulator.append(result)
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)