import datetime
import decimal
from pandas.testing import assert_frame_equal
from pyflink.common import Row
from pyflink.table.types import DataTypes
from pyflink.testing import source_sink_utils
from pyflink.testing.test_case_utils import PyFlinkBatchTableTestCase, PyFlinkStreamTableTestCase

class PandasConversionTestBase(object):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(PandasConversionTestBase, cls).setUpClass()
        cls.data = [(1, 1, 1, 1, True, 1.1, 1.2, 'hello', bytearray(b'aaa'), decimal.Decimal('1000000000000000000.01'), datetime.date(2014, 9, 13), datetime.time(hour=1, minute=0, second=1), datetime.datetime(1970, 1, 1, 0, 0, 0, 123000), ['hello', '中文'], Row(a=1, b='hello', c=datetime.datetime(1970, 1, 1, 0, 0, 0, 123000), d=[1, 2])), (1, 2, 2, 2, False, 2.1, 2.2, 'world', bytearray(b'bbb'), decimal.Decimal('1000000000000000000.02'), datetime.date(2014, 9, 13), datetime.time(hour=1, minute=0, second=1), datetime.datetime(1970, 1, 1, 0, 0, 0, 123000), ['hello', '中文'], Row(a=1, b='hello', c=datetime.datetime(1970, 1, 1, 0, 0, 0, 123000), d=[1, 2]))]
        cls.data_type = DataTypes.ROW([DataTypes.FIELD('f1', DataTypes.TINYINT()), DataTypes.FIELD('f2', DataTypes.SMALLINT()), DataTypes.FIELD('f3', DataTypes.INT()), DataTypes.FIELD('f4', DataTypes.BIGINT()), DataTypes.FIELD('f5', DataTypes.BOOLEAN()), DataTypes.FIELD('f6', DataTypes.FLOAT()), DataTypes.FIELD('f7', DataTypes.DOUBLE()), DataTypes.FIELD('f8', DataTypes.STRING()), DataTypes.FIELD('f9', DataTypes.BYTES()), DataTypes.FIELD('f10', DataTypes.DECIMAL(38, 18)), DataTypes.FIELD('f11', DataTypes.DATE()), DataTypes.FIELD('f12', DataTypes.TIME()), DataTypes.FIELD('f13', DataTypes.TIMESTAMP(3)), DataTypes.FIELD('f14', DataTypes.ARRAY(DataTypes.STRING())), DataTypes.FIELD('f15', DataTypes.ROW([DataTypes.FIELD('a', DataTypes.INT()), DataTypes.FIELD('b', DataTypes.STRING()), DataTypes.FIELD('c', DataTypes.TIMESTAMP(3)), DataTypes.FIELD('d', DataTypes.ARRAY(DataTypes.INT()))]))], False)
        cls.pdf = cls.create_pandas_data_frame()

    @classmethod
    def create_pandas_data_frame(cls):
        if False:
            while True:
                i = 10
        data_dict = {}
        for (j, name) in enumerate(cls.data_type.names):
            data_dict[name] = [cls.data[i][j] for i in range(len(cls.data))]
        import numpy as np
        data_dict['f1'] = np.int8(data_dict['f1'])
        data_dict['f2'] = np.int16(data_dict['f2'])
        data_dict['f3'] = np.int32(data_dict['f3'])
        data_dict['f4'] = np.int64(data_dict['f4'])
        data_dict['f6'] = np.float32(data_dict['f6'])
        data_dict['f7'] = np.float64(data_dict['f7'])
        data_dict['f15'] = [row.as_dict() for row in data_dict['f15']]
        import pandas as pd
        return pd.DataFrame(data=data_dict, index=[2.0, 3.0], columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15'])

class PandasConversionTests(PandasConversionTestBase):

    def test_from_pandas_with_incorrect_schema(self):
        if False:
            print('Hello World!')
        fields = self.data_type.fields.copy()
        (fields[0], fields[7]) = (fields[7], fields[0])
        wrong_schema = DataTypes.ROW(fields)
        with self.assertRaisesRegex(Exception, 'Expected a string.*got int8'):
            self.t_env.from_pandas(self.pdf, schema=wrong_schema)

    def test_from_pandas_with_names(self):
        if False:
            while True:
                i = 10
        pdf = self.pdf.drop(['f10', 'f11', 'f12', 'f13', 'f14', 'f15'], axis=1)
        new_names = list(map(str, range(len(pdf.columns))))
        table = self.t_env.from_pandas(pdf, schema=new_names)
        self.assertEqual(new_names, table.get_schema().get_field_names())
        table = self.t_env.from_pandas(pdf, schema=tuple(new_names))
        self.assertEqual(new_names, table.get_schema().get_field_names())

    def test_from_pandas_with_types(self):
        if False:
            return 10
        new_types = self.data_type.field_types()
        new_types[0] = DataTypes.BIGINT()
        table = self.t_env.from_pandas(self.pdf, schema=new_types)
        self.assertEqual(new_types, table.get_schema().get_field_data_types())
        table = self.t_env.from_pandas(self.pdf, schema=tuple(new_types))
        self.assertEqual(new_types, table.get_schema().get_field_data_types())

class PandasConversionITTests(PandasConversionTestBase):

    def test_from_pandas(self):
        if False:
            return 10
        table = self.t_env.from_pandas(self.pdf, self.data_type, 5)
        self.assertEqual(self.data_type, table.get_schema().to_row_data_type())
        table = table.filter(table.f2 < 2)
        sink_table_ddl = "\n            CREATE TABLE Results(\n            f1 TINYINT,\n            f2 SMALLINT,\n            f3 INT,\n            f4 BIGINT,\n            f5 BOOLEAN,\n            f6 FLOAT,\n            f7 DOUBLE,\n            f8 STRING,\n            f9 BYTES,\n            f10 DECIMAL(38, 18),\n            f11 DATE,\n            f12 TIME,\n            f13 TIMESTAMP(3),\n            f14 ARRAY<STRING>,\n            f15 ROW<a INT, b STRING, c TIMESTAMP(3), d ARRAY<INT>>)\n            WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        table.execute_insert('Results').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 1, 1, 1, true, 1.1, 1.2, hello, [97, 97, 97], 1000000000000000000.010000000000000000, 2014-09-13, 01:00:01, 1970-01-01T00:00:00.123, [hello, 中文], +I[1, hello, 1970-01-01T00:00:00.123, [1, 2]]]'])

    def test_to_pandas(self):
        if False:
            i = 10
            return i + 15
        table = self.t_env.from_pandas(self.pdf, self.data_type)
        result_pdf = table.to_pandas()
        result_pdf.index = self.pdf.index
        self.assertEqual(2, len(result_pdf))
        expected_arrow = self.pdf.to_records(index=False)
        result_arrow = result_pdf.to_records(index=False)
        for r in range(len(expected_arrow)):
            for e in range(len(expected_arrow[r])):
                self.assert_equal_field(expected_arrow[r][e], result_arrow[r][e])

    def test_empty_to_pandas(self):
        if False:
            i = 10
            return i + 15
        table = self.t_env.from_pandas(self.pdf, self.data_type)
        pdf = table.filter(table.f1 < 0).to_pandas()
        self.assertTrue(pdf.empty)

    def test_to_pandas_for_retract_table(self):
        if False:
            print('Hello World!')
        table = self.t_env.from_pandas(self.pdf, self.data_type)
        result_pdf = table.group_by(table.f1).select(table.f2.max.alias('f2')).to_pandas()
        import pandas as pd
        import numpy as np
        assert_frame_equal(result_pdf, pd.DataFrame(data={'f2': np.int16([2])}))
        result_pdf = table.group_by(table.f2).select(table.f1.max.alias('f2')).to_pandas()
        assert_frame_equal(result_pdf, pd.DataFrame(data={'f2': np.int8([1, 1])}))

    def assert_equal_field(self, expected_field, result_field):
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        result_type = type(result_field)
        if result_type == dict:
            self.assertEqual(expected_field.keys(), result_field.keys())
            for key in expected_field:
                self.assert_equal_field(expected_field[key], result_field[key])
        elif result_type == np.ndarray:
            self.assertTrue((expected_field == result_field).all())
        else:
            self.assertTrue(expected_field == result_field)

class BatchPandasConversionTests(PandasConversionTests, PandasConversionITTests, PyFlinkBatchTableTestCase):
    pass

class StreamPandasConversionTests(PandasConversionITTests, PyFlinkStreamTableTestCase):

    def test_to_pandas_with_event_time(self):
        if False:
            for i in range(10):
                print('nop')
        self.t_env.get_config().set('parallelism.default', '1')
        import tempfile
        import os
        tmp_dir = tempfile.gettempdir()
        data = ['2018-03-11 03:10:00', '2018-03-11 03:10:00', '2018-03-11 03:10:00', '2018-03-11 03:40:00', '2018-03-11 04:20:00', '2018-03-11 03:30:00']
        source_path = tmp_dir + '/test_to_pandas_with_event_time.csv'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        self.t_env.get_config().set('pipeline.time-characteristic', 'EventTime')
        source_table = "\n            create table source_table(\n                rowtime TIMESTAMP(3),\n                WATERMARK FOR rowtime AS rowtime - INTERVAL '60' MINUTE\n            ) with(\n                'connector.type' = 'filesystem',\n                'format.type' = 'csv',\n                'connector.path' = '%s',\n                'format.ignore-first-line' = 'false',\n                'format.field-delimiter' = ','\n            )\n        " % source_path
        self.t_env.execute_sql(source_table)
        t = self.t_env.from_path('source_table')
        result_pdf = t.to_pandas()
        import pandas as pd
        os.remove(source_path)
        assert_frame_equal(result_pdf, pd.DataFrame(data={'rowtime': [datetime.datetime(2018, 3, 11, 3, 10), datetime.datetime(2018, 3, 11, 3, 10), datetime.datetime(2018, 3, 11, 3, 10), datetime.datetime(2018, 3, 11, 3, 40), datetime.datetime(2018, 3, 11, 4, 20), datetime.datetime(2018, 3, 11, 3, 30)]}))