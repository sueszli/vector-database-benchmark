from pyflink.common.config_options import ConfigOptions
from pyflink.table import DataTypes
from pyflink.table.schema import Schema
from pyflink.table.table_descriptor import TableDescriptor, FormatDescriptor
from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase

class TableDescriptorTest(PyFlinkStreamTableTestCase):

    def setUp(self):
        if False:
            return 10
        super(TableDescriptorTest, self).setUp()
        self.option_a = ConfigOptions.key('a').boolean_type().no_default_value()
        self.option_b = ConfigOptions.key('b').int_type().no_default_value()
        self.key_format = ConfigOptions.key('key.format').string_type().no_default_value()

    def test_basic(self):
        if False:
            print('Hello World!')
        schema = Schema.new_builder().column('f0', DataTypes.STRING()).column('f1', DataTypes.BIGINT()).primary_key('f0').build()
        descriptor = TableDescriptor.for_connector('test-connector').schema(schema).partitioned_by('f0').comment('Test Comment').build()
        self.assertIsNotNone(descriptor.get_schema())
        self.assertEqual(1, len(descriptor.get_partition_keys()))
        self.assertEqual('f0', descriptor.get_partition_keys()[0])
        self.assertEqual(1, len(descriptor.get_options()))
        self.assertEqual('test-connector', descriptor.get_options().get('connector'))
        self.assertEqual('Test Comment', descriptor.get_comment())

    def test_no_schema(self):
        if False:
            i = 10
            return i + 15
        descriptor = TableDescriptor.for_connector('test-connector').build()
        self.assertIsNone(descriptor.get_schema())

    def test_options(self):
        if False:
            while True:
                i = 10
        descriptor = TableDescriptor.for_connector('test-connector').schema(Schema.new_builder().build()).option(self.option_a, False).option(self.option_b, 42).option('c', 'C').build()
        self.assertEqual(4, len(descriptor.get_options()))
        self.assertEqual('test-connector', descriptor.get_options().get('connector'))
        self.assertEqual('false', descriptor.get_options().get('a'))
        self.assertEqual('42', descriptor.get_options().get('b'))
        self.assertEqual('C', descriptor.get_options().get('c'))

    def test_format_basic(self):
        if False:
            print('Hello World!')
        descriptor = TableDescriptor.for_connector('test-connector').schema(Schema.new_builder().build()).format('json').build()
        self.assertEqual(2, len(descriptor.get_options()))
        self.assertEqual('test-connector', descriptor.get_options().get('connector'))
        self.assertEqual('json', descriptor.get_options().get('format'))

    def test_format_with_format_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        descriptor = TableDescriptor.for_connector('test-connector').schema(Schema.new_builder().build()).format(FormatDescriptor.for_format('test-format').option(self.option_a, True).option(self.option_b, 42).option('c', 'C').build(), self.key_format).build()
        self.assertEqual(5, len(descriptor.get_options()))
        self.assertEqual('test-connector', descriptor.get_options().get('connector'))
        self.assertEqual('test-format', descriptor.get_options().get('key.format'))
        self.assertEqual('true', descriptor.get_options().get('key.test-format.a'))
        self.assertEqual('42', descriptor.get_options().get('key.test-format.b'))
        self.assertEqual('C', descriptor.get_options().get('key.test-format.c'))

    def test_to_string(self):
        if False:
            print('Hello World!')
        schema = Schema.new_builder().column('f0', DataTypes.STRING()).build()
        format_descriptor = FormatDescriptor.for_format('test-format').option(self.option_a, False).build()
        table_descriptor = TableDescriptor.for_connector('test-connector').schema(schema).partitioned_by('f0').option(self.option_a, True).format(format_descriptor).comment('Test Comment').build()
        self.assertEqual('test-format[{a=false}]', str(format_descriptor))
        self.assertEqual("(\n  `f0` STRING\n)\nCOMMENT 'Test Comment'\nPARTITIONED BY (`f0`)\nWITH (\n  'a' = 'true',\n  'connector' = 'test-connector',\n  'test-format.a' = 'false',\n  'format' = 'test-format'\n)", str(table_descriptor))

    def test_execute_insert_to_table_descriptor(self):
        if False:
            while True:
                i = 10
        schema = Schema.new_builder().column('f0', DataTypes.STRING()).build()
        table = self.t_env.from_descriptor(TableDescriptor.for_connector('datagen').option('number-of-rows', '10').schema(schema).build())
        table_result = table.execute_insert(TableDescriptor.for_connector('blackhole').schema(schema).build())
        table_result.collect()

    def test_statement_set_insert_using_table_descriptor(self):
        if False:
            i = 10
            return i + 15
        schema = Schema.new_builder().column('f0', DataTypes.INT()).build()
        source_descriptor = TableDescriptor.for_connector('datagen').schema(schema).option('number-of-rows', '10').build()
        sink_descriptor = TableDescriptor.for_connector('blackhole').schema(schema).build()
        self.t_env.create_temporary_table('T', source_descriptor)
        stmt_set = self.t_env.create_statement_set()
        stmt_set.add_insert(sink_descriptor, self.t_env.from_path('T'))
        stmt_set.execute().wait()
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)