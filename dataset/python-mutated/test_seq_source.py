from pyflink.datastream.connectors.number_seq import NumberSequenceSource
from pyflink.testing.test_case_utils import PyFlinkStreamingTestCase
from pyflink.util.java_utils import load_java_class

class SequenceSourceTests(PyFlinkStreamingTestCase):

    def test_seq_source(self):
        if False:
            for i in range(10):
                print('nop')
        seq_source = NumberSequenceSource(1, 10)
        seq_source_clz = load_java_class('org.apache.flink.api.connector.source.lib.NumberSequenceSource')
        from_field = seq_source_clz.getDeclaredField('from')
        from_field.setAccessible(True)
        self.assertEqual(1, from_field.get(seq_source.get_java_function()))
        to_field = seq_source_clz.getDeclaredField('to')
        to_field.setAccessible(True)
        self.assertEqual(10, to_field.get(seq_source.get_java_function()))